import {
  createErrorObject,
  isErrorObject,
  LogLevelNames,
  ServicesContext,
  type Response,
} from '@node-in-layers/core/index.js'
import { AIMessage, BaseMessage } from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'

import {
  defaultMaxToolRounds,
  executeToolCalls,
  getContent,
  getModel,
  toLangchainTools,
  toMessages,
} from './internal-libs.js'
import {
  ConfigWithLlm,
  LlmConfig,
  LlmNamespace,
  LlmServices,
  LlmStructuredRequest,
  LlmTextRequest,
} from './types.js'

export const create = (
  context: ServicesContext<ConfigWithLlm>
): LlmServices => {
  const _logLangchainCall = (
    config: LlmConfig,
    functionName: string,
    data: Readonly<Record<string, unknown>>,
    crossLayerProps?: Parameters<LlmServices['generateText']>[1]
  ) => {
    if (config.langchainLogging?.enabled !== true) {
      return Promise.resolve()
    }

    const log = context.log.getInnerLogger(functionName, crossLayerProps)
    const logLevel = config.langchainLogging.logLevel ?? LogLevelNames.trace

    return Promise.resolve(
      log[logLevel]('LangChain call lifecycle event', data)
    )
  }

  const _runTools = (
    model: ChatOpenAI,
    config: LlmConfig,
    request: LlmTextRequest,
    messages: BaseMessage[],
    round: number,
    crossLayerProps?: Parameters<LlmServices['generateText']>[1]
  ): Promise<BaseMessage[] | ReturnType<typeof createErrorObject>> => {
    const tools = request.tools
    const invocation = tools
      ? model.bindTools(toLangchainTools({ tools })).invoke(messages)
      : model.invoke(messages)

    return invocation.then(async message => {
      const toolCalls = message.tool_calls ?? []
      await _logLangchainCall(
        config,
        'runTools',
        {
          model: request.model ?? config.defaultModel,
          round,
          toolCallCount: toolCalls.length,
        },
        crossLayerProps
      )

      if (toolCalls.length === 0) {
        return messages.concat([message])
      }

      if (round >= (config.maxToolRounds ?? defaultMaxToolRounds)) {
        return createErrorObject(
          'LLM_TOOL_ROUND_LIMIT_REACHED',
          'The LLM exceeded the configured maximum number of tool rounds.'
        )
      }

      const toolMessages = await executeToolCalls({
        tools: tools ?? {},
        toolCalls,
      })
      if (isErrorObject(toolMessages)) {
        return toolMessages
      }

      return _runTools(
        model,
        config,
        request,
        messages.concat([message]).concat(toolMessages),
        round + 1,
        crossLayerProps
      )
    })
  }

  const generateText = (
    request: LlmTextRequest,
    crossLayerProps?: Parameters<LlmServices['generateText']>[1]
  ) => {
    return Promise.resolve()
      .then(async () => {
        const config = context.config[LlmNamespace.llm]
        if (!config) {
          return createErrorObject(
            'LLM_CONFIGURATION_ERROR',
            'The LLM configuration is missing.'
          )
        }

        const model = getModel({ config, request })
        if (isErrorObject(model)) {
          return model
        }

        await _logLangchainCall(
          config,
          'generateText',
          {
            model: request.model ?? config.defaultModel,
            hasTools: Boolean(request.tools),
          },
          crossLayerProps
        )

        const messages = await _runTools(
          model,
          config,
          request,
          toMessages({ request }),
          0,
          crossLayerProps
        )
        if (isErrorObject(messages)) {
          return messages
        }

        const finalMessage = messages[messages.length - 1]
        if (!AIMessage.isInstance(finalMessage)) {
          return createErrorObject(
            'LLM_INVALID_RESPONSE',
            'The LLM did not return a final assistant message.'
          )
        }

        return getContent({ message: finalMessage })
      })
      .catch(error => {
        return createErrorObject(
          'LLM_PROVIDER_ERROR',
          'The LLM provider request failed.',
          error
        )
      })
  }

  const generateStructured = <T extends object>(
    request: LlmStructuredRequest<T>,
    crossLayerProps?: Parameters<LlmServices['generateText']>[1]
  ): Promise<Response<T>> => {
    return Promise.resolve()
      .then(async () => {
        const config = context.config[LlmNamespace.llm]
        if (!config) {
          return createErrorObject(
            'LLM_CONFIGURATION_ERROR',
            'The LLM configuration is missing.'
          )
        }

        const model = getModel({ config, request })
        if (isErrorObject(model)) {
          return model
        }

        const messages = await _runTools(
          model,
          config,
          request,
          toMessages({ request }),
          0,
          crossLayerProps
        )
        if (isErrorObject(messages)) {
          return messages
        }

        await _logLangchainCall(
          config,
          'generateStructured',
          {
            model: request.model ?? config.defaultModel,
            hasTools: Boolean(request.tools),
          },
          crossLayerProps
        )

        return model
          .withStructuredOutput(request.schema, {
            strict: config.strictStructuredOutput,
          })
          .invoke(messages)
          .then(result => {
            return result as T
          })
          .catch(error => {
            return createErrorObject(
              'LLM_STRUCTURED_OUTPUT_INVALID',
              'The LLM did not return a valid structured response.',
              error
            )
          })
      })
      .catch(error => {
        return createErrorObject(
          'LLM_PROVIDER_ERROR',
          'The LLM provider request failed.',
          error
        )
      }) as Promise<Response<T>>
  }

  return {
    generateText,
    generateStructured,
  }
}
