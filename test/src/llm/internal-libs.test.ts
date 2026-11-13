import { assert } from 'chai'
import { isErrorObject } from '@node-in-layers/core'
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'
import { z } from 'zod'

import {
  defaultMaxToolRounds,
  executeToolCall,
  executeToolCalls,
  getContent,
  getModel,
  toLangchainTools,
  toMessages,
} from '../../../src/llm/internal-libs.js'
import { OpenAiLlmModel } from '../../../src/llm/types.js'

describe('/src/llm/internal-libs.ts', () => {
  describe('#defaultMaxToolRounds', () => {
    it('should default the tool loop to five rounds', () => {
      const actual = defaultMaxToolRounds
      const expected = 5
      assert.equal(actual, expected)
    })
  })

  describe('#toMessages()', () => {
    it('should place a system prompt before the user prompt', () => {
      const input = {
        request: {
          prompt: 'Hello',
          systemPrompt: 'Be brief.',
        },
      }

      const actual = toMessages(input)

      assert.equal(actual.length, 2)
      assert.instanceOf(actual[0], SystemMessage)
      assert.equal(actual[0].content, 'Be brief.')
      assert.instanceOf(actual[1], HumanMessage)
      assert.equal(actual[1].content, 'Hello')
    })

    it('should return only the user prompt when no system prompt is provided', () => {
      const input = {
        request: {
          prompt: 'Hello',
        },
      }

      const actual = toMessages(input)

      assert.equal(actual.length, 1)
      assert.instanceOf(actual[0], HumanMessage)
      assert.equal(actual[0].content, 'Hello')
    })
  })

  describe('#getContent()', () => {
    it('should return string message content unchanged', () => {
      const input = { message: new AIMessage('Hello') }

      const actual = getContent(input)

      const expected = 'Hello'
      assert.equal(actual, expected)
    })

    it('should stringify structured message content', () => {
      const content = [{ type: 'text', text: 'Hello' }]
      const input = { message: new AIMessage({ content }) }

      const actual = getContent(input)

      const expected = JSON.stringify(content)
      assert.equal(actual, expected)
    })
  })

  describe('#getModel()', () => {
    it('should return a configuration error when the api key is still a secret placeholder', () => {
      const input = {
        config: {
          defaultModel: OpenAiLlmModel.Gpt56Sol,
          apiKey: { type: 'nil-secret' as const, key: 'llm.apiKey' },
        },
        request: { prompt: 'Hello' },
      }

      const actual = getModel(input)

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

    it('should build a ChatOpenAI client from request overrides and the configured base url', () => {
      const input = {
        config: {
          defaultModel: OpenAiLlmModel.Gpt56Sol,
          apiKey: 'sk-test',
          baseUrl: 'https://llm-gateway.example.com/v1',
          temperature: 0.4,
          maxTokens: 200,
          timeoutMs: 3000,
        },
        request: {
          prompt: 'Hello',
          model: OpenAiLlmModel.Gpt6Sol,
          temperature: 0.1,
        },
      }

      const actual = getModel(input)

      assert.instanceOf(actual, ChatOpenAI)
      if (isErrorObject(actual)) {
        return
      }
      assert.equal(actual.model, OpenAiLlmModel.Gpt6Sol)
      assert.equal(actual.temperature, 0.1)
      assert.equal(actual.maxTokens, 200)
      assert.equal(actual.timeout, 3000)
      assert.equal(actual.apiKey, 'sk-test')
      assert.equal(
        actual.clientConfig.baseURL,
        'https://llm-gateway.example.com/v1'
      )
    })

    it('should use the configured model and temperature when the request does not override them', () => {
      const input = {
        config: {
          defaultModel: OpenAiLlmModel.Gpt56Sol,
          apiKey: 'sk-test',
          temperature: 0.4,
        },
        request: { prompt: 'Hello' },
      }

      const actual = getModel(input)

      assert.instanceOf(actual, ChatOpenAI)
      if (isErrorObject(actual)) {
        return
      }
      assert.equal(actual.model, OpenAiLlmModel.Gpt56Sol)
      assert.equal(actual.temperature, 0.4)
      assert.equal(actual.clientConfig.baseURL, undefined)
    })
  })

  describe('#toLangchainTools()', () => {
    it('should map tool records to LangChain tool definitions', () => {
      const lookupSchema = z.object({ accountId: z.string() })
      const statusSchema = z.object({ id: z.string() })
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: lookupSchema,
            execute: () => ({ status: 'open' }),
          },
          status: {
            description: 'Read a status',
            schema: statusSchema,
            execute: () => ({ status: 'closed' }),
          },
        },
      }

      const actual = toLangchainTools(input)

      const expected = [
        {
          name: 'lookup',
          description: 'Look up an account',
          schema: lookupSchema,
        },
        {
          name: 'status',
          description: 'Read a status',
          schema: statusSchema,
        },
      ]
      assert.deepEqual(actual, expected)
    })
  })

  describe('#executeToolCall()', () => {
    it('should return a tool-not-found error for an unregistered tool', async () => {
      const input = {
        tools: {},
        toolCall: { name: 'lookup', args: {} },
      }

      const actual = await executeToolCall(input)

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_TOOL_NOT_FOUND')
      assert.equal(
        actual.error.message,
        'The model requested an unregistered tool: lookup.'
      )
    })

    it('should return a validation error when the arguments do not match the schema', async () => {
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: z.object({ accountId: z.string() }),
            execute: () => ({ status: 'open' }),
          },
        },
        toolCall: { name: 'lookup', args: { accountId: 12 } },
      }

      const actual = await executeToolCall(input)

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_TOOL_VALIDATION_FAILED')
      assert.equal(
        actual.error.message,
        'The model supplied invalid arguments for tool: lookup.'
      )
    })

    it('should execute a tool and use the tool name when the call has no id', async () => {
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: z.object({ accountId: z.string() }),
            execute: () => undefined,
          },
        },
        toolCall: { name: 'lookup', args: { accountId: 'acct-1' } },
      }

      const actual = await executeToolCall(input)

      assert.instanceOf(actual, ToolMessage)
      if (isErrorObject(actual)) {
        return
      }
      assert.equal(actual.content, 'null')
      assert.equal(actual.tool_call_id, 'lookup')
    })

    it('should stringify a resolved tool result and keep the call id', async () => {
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: z.object({ accountId: z.string() }),
            execute: () => Promise.resolve({ status: 'open' }),
          },
        },
        toolCall: {
          id: 'call-1',
          name: 'lookup',
          args: { accountId: 'acct-1' },
        },
      }

      const actual = await executeToolCall(input)

      assert.instanceOf(actual, ToolMessage)
      if (isErrorObject(actual)) {
        return
      }
      assert.equal(actual.content, JSON.stringify({ status: 'open' }))
      assert.equal(actual.tool_call_id, 'call-1')
    })

    it('should return an execution error when the tool throws', async () => {
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: z.object({}),
            execute: () => {
              throw new Error('lookup failed')
            },
          },
        },
        toolCall: { id: 'call-1', name: 'lookup', args: {} },
      }

      const actual = await executeToolCall(input)

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_TOOL_EXECUTION_FAILED')
      assert.equal(actual.error.message, 'Tool execution failed: lookup.')
      assert.equal(actual.error.details, 'lookup failed')
    })
  })

  describe('#executeToolCalls()', () => {
    it('should return an empty list when the model requested no tools', async () => {
      const input = { tools: {}, toolCalls: [] }

      const actual = await executeToolCalls(input)

      const expected: readonly ToolMessage[] = []
      assert.deepEqual(actual, expected)
    })

    it('should execute tool calls in order', async () => {
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: z.object({ accountId: z.string() }),
            execute: () => ({ status: 'open' }),
          },
          status: {
            description: 'Read a status',
            schema: z.object({ id: z.string() }),
            execute: () => Promise.resolve({ state: 'ready' }),
          },
        },
        toolCalls: [
          { id: 'call-1', name: 'lookup', args: { accountId: 'acct-1' } },
          { id: 'call-2', name: 'status', args: { id: 'acct-1' } },
        ],
      }

      const actual = await executeToolCalls(input)

      assert.equal(isErrorObject(actual), false)
      if (isErrorObject(actual)) {
        return
      }
      assert.equal(actual.length, 2)
      assert.equal(actual[0].content, JSON.stringify({ status: 'open' }))
      assert.equal(actual[1].content, JSON.stringify({ state: 'ready' }))
    })

    it('should stop after the first failed tool call', async () => {
      let statusCalls = 0
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: z.object({}),
            execute: () => {
              throw new Error('lookup failed')
            },
          },
          status: {
            description: 'Read a status',
            schema: z.object({}),
            execute: () => {
              statusCalls += 1
              return { state: 'ready' }
            },
          },
        },
        toolCalls: [
          { name: 'lookup', args: {} },
          { name: 'status', args: {} },
        ],
      }

      const actual = await executeToolCalls(input)

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_TOOL_EXECUTION_FAILED')
      assert.equal(statusCalls, 0)
    })

    it('should return the later tool error after an earlier tool succeeds', async () => {
      const input = {
        tools: {
          lookup: {
            description: 'Look up an account',
            schema: z.object({}),
            execute: () => ({ status: 'open' }),
          },
          status: {
            description: 'Read a status',
            schema: z.object({ id: z.string() }),
            execute: () => ({ state: 'ready' }),
          },
        },
        toolCalls: [
          { id: 'call-1', name: 'lookup', args: {} },
          { id: 'call-2', name: 'status', args: {} },
        ],
      }

      const actual = await executeToolCalls(input)

      assert.equal(isErrorObject(actual), true)
      if (!isErrorObject(actual)) {
        return
      }
      assert.equal(actual.error.code, 'LLM_TOOL_VALIDATION_FAILED')
    })
  })
})
