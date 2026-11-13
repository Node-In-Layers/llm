import { createErrorObject, isErrorObject } from '@node-in-layers/core/index.js'
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'

import { LlmConfig, LlmTextRequest, LlmTools } from './types.js'

export const defaultMaxToolRounds = 5

export type LlmToolCall = Readonly<{
  id?: string
  name: string
  args: unknown
}>

export const toMessages = (
  args: Readonly<{ request: LlmTextRequest }>
): BaseMessage[] => {
  const systemMessages: BaseMessage[] = args.request.systemPrompt
    ? [new SystemMessage(args.request.systemPrompt)]
    : []

  return systemMessages.concat(new HumanMessage(args.request.prompt))
}

export const getContent = (args: Readonly<{ message: AIMessage }>): string => {
  if (typeof args.message.content === 'string') {
    return args.message.content
  }

  return JSON.stringify(args.message.content)
}

export const getModel = (
  args: Readonly<{ config: LlmConfig; request: LlmTextRequest }>
): ChatOpenAI | ReturnType<typeof createErrorObject> => {
  if (typeof args.config.apiKey !== 'string') {
    return createErrorObject(
      'LLM_CONFIGURATION_ERROR',
      'The LLM API key must be resolved by @node-in-layers/secrets before the LLM service is created.'
    )
  }

  return new ChatOpenAI({
    apiKey: args.config.apiKey,
    model: args.request.model ?? args.config.defaultModel,
    temperature: args.request.temperature ?? args.config.temperature,
    maxTokens: args.config.maxTokens,
    timeout: args.config.timeoutMs,
    configuration: args.config.baseUrl
      ? {
          baseURL: args.config.baseUrl,
        }
      : undefined,
  })
}

export const toLangchainTools = (args: Readonly<{ tools: LlmTools }>) => {
  return Object.entries(args.tools).map(([name, tool]) => {
    return {
      name,
      description: tool.description,
      schema: tool.schema,
    }
  })
}

export const executeToolCall = (
  args: Readonly<{ tools: LlmTools; toolCall: LlmToolCall }>
) => {
  const tool = args.tools[args.toolCall.name]
  if (!tool) {
    return Promise.resolve(
      createErrorObject(
        'LLM_TOOL_NOT_FOUND',
        `The model requested an unregistered tool: ${args.toolCall.name}.`
      )
    )
  }

  const parsedArgs = tool.schema.safeParse(args.toolCall.args)
  if (!parsedArgs.success) {
    return Promise.resolve(
      createErrorObject(
        'LLM_TOOL_VALIDATION_FAILED',
        `The model supplied invalid arguments for tool: ${args.toolCall.name}.`,
        parsedArgs.error.flatten()
      )
    )
  }

  return Promise.resolve()
    .then(() => {
      return tool.execute(parsedArgs.data as never)
    })
    .then(result => {
      return new ToolMessage({
        content: JSON.stringify(result ?? null),
        tool_call_id: args.toolCall.id ?? args.toolCall.name,
      })
    })
    .catch(error => {
      return createErrorObject(
        'LLM_TOOL_EXECUTION_FAILED',
        `Tool execution failed: ${args.toolCall.name}.`,
        error
      )
    })
}

export const executeToolCalls = (
  args: Readonly<{
    tools: LlmTools
    toolCalls: readonly LlmToolCall[]
  }>
) => {
  return args.toolCalls.reduce<
    Promise<readonly ToolMessage[] | ReturnType<typeof createErrorObject>>
  >((resultPromise, toolCall) => {
    return resultPromise.then(async result => {
      if (isErrorObject(result)) {
        return result
      }

      const toolResult = await executeToolCall({
        tools: args.tools,
        toolCall,
      })
      if (isErrorObject(toolResult)) {
        return toolResult
      }

      return result.concat([toolResult])
    })
  }, Promise.resolve([]))
}
