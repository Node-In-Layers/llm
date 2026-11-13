import { assert } from 'chai'
import sinon from 'sinon'
import { isErrorObject, LogLevelNames } from '@node-in-layers/core'
import type { ServicesContext } from '@node-in-layers/core'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'
import { z } from 'zod'

import { create } from '../../../src/llm/services.js'
import {
  ConfigWithLlm,
  LlmConfig,
  LlmNamespace,
  OpenAiLlmModel,
} from '../../../src/llm/types.js'
import { defaultMaxToolRounds } from '../../../src/llm/internal-libs.js'

const crossLayerProps = { logging: { ids: [] } }

const createLogger = () => {
  const levels = {
    trace: sinon.stub(),
    debug: sinon.stub(),
    info: sinon.stub(),
    warn: sinon.stub(),
    error: sinon.stub(),
    silent: sinon.stub(),
  }
  const getInnerLogger = sinon.stub().returns(levels)
  return { levels, getInnerLogger }
}

const createContext = (
  config: LlmConfig | undefined,
  getInnerLogger = sinon.stub()
) => {
  const context = {
    config: config ? { [LlmNamespace.llm]: config } : {},
    log: { getInnerLogger },
  }
  return context as ServicesContext<ConfigWithLlm>
}

const withModel = async (
  run: (stubs: {
    invoke: sinon.SinonStub
    boundInvoke: sinon.SinonStub
    bindTools: sinon.SinonStub
    structuredInvoke: sinon.SinonStub
    withStructuredOutput: sinon.SinonStub
  }) => Promise<void>
) => {
  const invoke = sinon.stub(ChatOpenAI.prototype, 'invoke')
  const boundInvoke = sinon.stub()
  const bindTools = sinon
    .stub(ChatOpenAI.prototype, 'bindTools')
    .returns({ invoke: boundInvoke })
  const structuredInvoke = sinon.stub()
  const withStructuredOutput = sinon
    .stub(ChatOpenAI.prototype, 'withStructuredOutput')
    .returns({ invoke: structuredInvoke })

  try {
    await run({
      invoke,
      boundInvoke,
      bindTools,
      structuredInvoke,
      withStructuredOutput,
    })
  } finally {
    sinon.restore()
  }
}

const resolvedConfig = (overrides: Partial<LlmConfig> = {}): LlmConfig => {
  return {
    defaultModel: OpenAiLlmModel.Gpt56Sol,
    apiKey: 'sk-test',
    ...overrides,
  }
}

describe('/src/llm/services.ts', () => {
  describe('#generateText()', () => {
    it('should return a configuration error when the llm config is missing', async () => {
      const llm = create(createContext(undefined))

      const actual = await llm.generateText({ prompt: 'Hello' })

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_CONFIGURATION_ERROR')
      assert.equal(actual.error.message, 'The LLM configuration is missing.')
    })

    it('should return a configuration error when the api key is unresolved', async () => {
      const llm = create(
        createContext({
          defaultModel: OpenAiLlmModel.Gpt56Sol,
          apiKey: { type: 'nil-secret', key: 'llm.apiKey' },
        })
      )

      const actual = await llm.generateText({ prompt: 'Hello' })

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_CONFIGURATION_ERROR')
      assert.equal(
        actual.error.message,
        'The LLM API key must be resolved by @node-in-layers/secrets before the LLM service is created.'
      )
    })

    it('should return the assistant text when logging is off', async () => {
      await withModel(async ({ invoke, bindTools }) => {
        invoke.resolves(new AIMessage('Hello'))
        const { getInnerLogger } = createLogger()
        const llm = create(createContext(resolvedConfig(), getInnerLogger))

        const actual = await llm.generateText({ prompt: 'Hello' })

        const expected = 'Hello'
        assert.equal(actual, expected)
        assert.equal(invoke.calledOnce, true)
        assert.equal(bindTools.called, false)
        assert.equal(getInnerLogger.called, false)
      })
    })

    it('should trace the call at the default log level when logging is enabled', async () => {
      await withModel(async ({ invoke }) => {
        invoke.resolves(new AIMessage('Hello'))
        const { levels, getInnerLogger } = createLogger()
        const llm = create(
          createContext(
            resolvedConfig({
              langchainLogging: { enabled: true },
            }),
            getInnerLogger
          )
        )

        const actual = await llm.generateText(
          { prompt: 'Hello', model: OpenAiLlmModel.Gpt6Sol },
          crossLayerProps
        )

        assert.equal(actual, 'Hello')
        assert.deepEqual(
          getInnerLogger.getCalls().map(call => call.args),
          [
            ['generateText', crossLayerProps],
            ['runTools', crossLayerProps],
          ]
        )
        assert.deepEqual(levels.trace.firstCall.args, [
          'LangChain call lifecycle event',
          { model: OpenAiLlmModel.Gpt6Sol, hasTools: false },
        ])
        assert.deepEqual(levels.trace.secondCall.args, [
          'LangChain call lifecycle event',
          {
            model: OpenAiLlmModel.Gpt6Sol,
            round: 0,
            toolCallCount: 0,
          },
        ])
        assert.equal(levels.info.called, false)
      })
    })

    it('should log at the configured level and use the default model', async () => {
      await withModel(async ({ invoke }) => {
        invoke.resolves(new AIMessage('Hello'))
        const { levels, getInnerLogger } = createLogger()
        const llm = create(
          createContext(
            resolvedConfig({
              langchainLogging: {
                enabled: true,
                logLevel: LogLevelNames.info,
              },
            }),
            getInnerLogger
          )
        )

        const actual = await llm.generateText({ prompt: 'Hello' })

        assert.equal(actual, 'Hello')
        assert.deepEqual(levels.info.firstCall.args[1], {
          model: OpenAiLlmModel.Gpt56Sol,
          hasTools: false,
        })
        assert.deepEqual(levels.info.secondCall.args[1], {
          model: OpenAiLlmModel.Gpt56Sol,
          round: 0,
          toolCallCount: 0,
        })
        assert.equal(levels.trace.called, false)
      })
    })

    it('should skip the logger when langchain logging is disabled', async () => {
      await withModel(async ({ invoke }) => {
        invoke.resolves(new AIMessage('Hello'))
        const { getInnerLogger } = createLogger()
        const llm = create(
          createContext(
            resolvedConfig({
              langchainLogging: { enabled: false },
            }),
            getInnerLogger
          )
        )

        const actual = await llm.generateText({ prompt: 'Hello' })

        assert.equal(actual, 'Hello')
        assert.equal(getInnerLogger.called, false)
      })
    })

    it('should return a provider error when the model request fails', async () => {
      await withModel(async ({ invoke }) => {
        invoke.rejects(new Error('network down'))
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateText({ prompt: 'Hello' })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_PROVIDER_ERROR')
        assert.equal(actual.error.message, 'The LLM provider request failed.')
        assert.equal(actual.error.details, 'network down')
      })
    })

    it('should return an invalid response when the final message is not from the assistant', async () => {
      await withModel(async ({ invoke }) => {
        invoke.resolves(new HumanMessage('not an answer'))
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateText({ prompt: 'Hello' })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_INVALID_RESPONSE')
        assert.equal(
          actual.error.message,
          'The LLM did not return a final assistant message.'
        )
      })
    })

    it('should execute a requested tool and return the final assistant text', async () => {
      await withModel(async ({ boundInvoke, bindTools }) => {
        const lookupSchema = z.object({ accountId: z.string() })
        let received: unknown
        boundInvoke.onCall(0).resolves(
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'call-1',
                name: 'lookup',
                args: { accountId: 'acct-1' },
                type: 'tool_call',
              },
            ],
          })
        )
        boundInvoke.onCall(1).resolves(new AIMessage('The account is open.'))
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateText({
          prompt: 'Look up acct-1',
          tools: {
            lookup: {
              description: 'Look up an account',
              schema: lookupSchema,
              execute: args => {
                received = args
                return { status: 'open' }
              },
            },
          },
        })

        assert.equal(actual, 'The account is open.')
        assert.deepEqual(received, { accountId: 'acct-1' })
        assert.deepEqual(bindTools.firstCall.args[0], [
          {
            name: 'lookup',
            description: 'Look up an account',
            schema: lookupSchema,
          },
        ])
        const continued = boundInvoke.secondCall.args[0]
        assert.equal(continued.length, 3)
        assert.instanceOf(continued[2], ToolMessage)
        assert.equal(continued[2].content, JSON.stringify({ status: 'open' }))
        assert.equal(continued[2].tool_call_id, 'call-1')
      })
    })

    it('should return a tool error when the model calls a tool that was not provided', async () => {
      await withModel(async ({ invoke, bindTools }) => {
        invoke.resolves(
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'call-1',
                name: 'lookup',
                args: {},
                type: 'tool_call',
              },
            ],
          })
        )
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateText({ prompt: 'Look up acct-1' })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_TOOL_NOT_FOUND')
        assert.equal(bindTools.called, false)
      })
    })

    it('should stop when the configured tool round limit is already reached', async () => {
      await withModel(async ({ boundInvoke }) => {
        boundInvoke.resolves(
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'call-1',
                name: 'lookup',
                args: {},
                type: 'tool_call',
              },
            ],
          })
        )
        let executions = 0
        const llm = create(createContext(resolvedConfig({ maxToolRounds: 0 })))

        const actual = await llm.generateText({
          prompt: 'Look up acct-1',
          tools: {
            lookup: {
              description: 'Look up an account',
              schema: z.object({}),
              execute: () => {
                executions += 1
                return { status: 'open' }
              },
            },
          },
        })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_TOOL_ROUND_LIMIT_REACHED')
        assert.equal(
          actual.error.message,
          'The LLM exceeded the configured maximum number of tool rounds.'
        )
        assert.equal(executions, 0)
        assert.equal(boundInvoke.callCount, 1)
      })
    })

    it('should use the default tool round limit when one is not configured', async () => {
      await withModel(async ({ boundInvoke }) => {
        boundInvoke.resolves(
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'call-1',
                name: 'lookup',
                args: {},
                type: 'tool_call',
              },
            ],
          })
        )
        let executions = 0
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateText({
          prompt: 'Look up acct-1',
          tools: {
            lookup: {
              description: 'Look up an account',
              schema: z.object({}),
              execute: () => {
                executions += 1
                return { status: 'open' }
              },
            },
          },
        })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_TOOL_ROUND_LIMIT_REACHED')
        assert.equal(executions, defaultMaxToolRounds)
        assert.equal(boundInvoke.callCount, defaultMaxToolRounds + 1)
      })
    })
  })

  describe('#generateStructured()', () => {
    const schema = z.object({ title: z.string() })

    it('should return a configuration error when the llm config is missing', async () => {
      const llm = create(createContext(undefined))

      const actual = await llm.generateStructured({
        prompt: 'Name it',
        schema,
      })

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_CONFIGURATION_ERROR')
      assert.equal(actual.error.message, 'The LLM configuration is missing.')
    })

    it('should return a configuration error when the api key is unresolved', async () => {
      const llm = create(
        createContext({
          defaultModel: OpenAiLlmModel.Gpt56Sol,
          apiKey: { type: 'nil-secret', key: 'llm.apiKey' },
        })
      )

      const actual = await llm.generateStructured({
        prompt: 'Name it',
        schema,
      })

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_CONFIGURATION_ERROR')
    })

    it('should return the structured value and trace the call by default', async () => {
      await withModel(
        async ({ invoke, structuredInvoke, withStructuredOutput }) => {
          invoke.resolves(new AIMessage('draft'))
          structuredInvoke.resolves({ title: 'Hello' })
          const { levels, getInnerLogger } = createLogger()
          const llm = create(
            createContext(
              resolvedConfig({
                strictStructuredOutput: true,
                langchainLogging: { enabled: true },
              }),
              getInnerLogger
            )
          )

          const actual = await llm.generateStructured(
            { prompt: 'Name it', schema },
            crossLayerProps
          )

          assert.deepEqual(actual, { title: 'Hello' })
          assert.equal(withStructuredOutput.firstCall.args[0], schema)
          assert.deepEqual(withStructuredOutput.firstCall.args[1], {
            strict: true,
          })
          assert.deepEqual(
            getInnerLogger.getCalls().map(call => call.args[0]),
            ['runTools', 'generateStructured']
          )
          assert.deepEqual(levels.trace.secondCall.args[1], {
            model: OpenAiLlmModel.Gpt56Sol,
            hasTools: false,
          })
        }
      )
    })

    it('should run tools before requesting structured output', async () => {
      await withModel(
        async ({ boundInvoke, structuredInvoke, withStructuredOutput }) => {
          boundInvoke.onCall(0).resolves(
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'call-1',
                  name: 'lookup',
                  args: { accountId: 'acct-1' },
                  type: 'tool_call',
                },
              ],
            })
          )
          boundInvoke.onCall(1).resolves(new AIMessage('draft'))
          structuredInvoke.resolves({ title: 'Open' })
          const { getInnerLogger } = createLogger()
          const llm = create(createContext(resolvedConfig(), getInnerLogger))

          const actual = await llm.generateStructured({
            prompt: 'Name the account',
            model: OpenAiLlmModel.Gpt6Sol,
            schema,
            tools: {
              lookup: {
                description: 'Look up an account',
                schema: z.object({ accountId: z.string() }),
                execute: () => ({ status: 'open' }),
              },
            },
          })

          assert.deepEqual(actual, { title: 'Open' })
          assert.deepEqual(withStructuredOutput.firstCall.args[1], {
            strict: undefined,
          })
          assert.equal(getInnerLogger.called, false)
          assert.equal(structuredInvoke.calledOnce, true)
        }
      )
    })

    it('should return a structured-output error when the schema call fails', async () => {
      await withModel(async ({ invoke, structuredInvoke }) => {
        invoke.resolves(new AIMessage('draft'))
        structuredInvoke.rejects(new Error('schema mismatch'))
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateStructured({
          prompt: 'Name it',
          schema,
        })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_STRUCTURED_OUTPUT_INVALID')
        assert.equal(
          actual.error.message,
          'The LLM did not return a valid structured response.'
        )
        assert.equal(actual.error.details, 'schema mismatch')
      })
    })

    it('should return a provider error when the tool loop fails', async () => {
      await withModel(async ({ invoke, withStructuredOutput }) => {
        invoke.rejects(new Error('network down'))
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateStructured({
          prompt: 'Name it',
          schema,
        })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_PROVIDER_ERROR')
        assert.equal(actual.error.details, 'network down')
        assert.equal(withStructuredOutput.called, false)
      })
    })

    it('should return the tool error before requesting structured output', async () => {
      await withModel(async ({ boundInvoke, withStructuredOutput }) => {
        boundInvoke.resolves(
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'call-1',
                name: 'missing',
                args: {},
                type: 'tool_call',
              },
            ],
          })
        )
        const llm = create(createContext(resolvedConfig()))

        const actual = await llm.generateStructured({
          prompt: 'Name it',
          schema,
          tools: {
            lookup: {
              description: 'Look up an account',
              schema: z.object({}),
              execute: () => ({ status: 'open' }),
            },
          },
        })

        assert.equal(isErrorObject(actual), true)
        if (!isErrorObject(actual)) {
          return
        }
        assert.equal(actual.error.code, 'LLM_TOOL_NOT_FOUND')
        assert.equal(withStructuredOutput.called, false)
      })
    })
  })
})
