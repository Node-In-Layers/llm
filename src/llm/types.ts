import type { XOR } from '@node-in-layers/core'
import {
  Config,
  CrossLayerProps,
  LogLevelNames,
  LayerFunction,
  Response,
} from '@node-in-layers/core'
import type { NilSecretEntry } from '@node-in-layers/secrets'
import type { z } from 'zod'

export enum LlmNamespace {
  llm = '@node-in-layers/llm',
}

export enum OpenAiLlmModel {
  Gpt6Astra = 'gpt-6-astra',
  Gpt61Sol = 'gpt-6.1-sol',
  Gpt61Terra = 'gpt-6.1-terra',
  Gpt6Sol = 'gpt-6-sol',
  Gpt6Luna = 'gpt-6-luna',
  Gpt56Sol = 'gpt-5.6-sol',
  Gpt56Terra = 'gpt-5.6-terra',
  Gpt56Luna = 'gpt-5.6-luna',
  Gpt55 = 'gpt-5.5',
  Gpt54 = 'gpt-5.4',
  Gpt54Mini = 'gpt-5.4-mini',
}

export enum AnthropicLlmModel {
  ClaudeFable52 = 'claude-fable-5-2',
  ClaudeOpus55 = 'claude-opus-5-5',
  ClaudeSonnet55 = 'claude-sonnet-5-5',
  ClaudeHaiku55 = 'claude-haiku-5-5',
}

export enum GoogleLlmModel {
  Gemini38Flash = 'gemini-3.8-flash',
  Gemini36Flash = 'gemini-3.6-flash',
  Gemini35FlashLite = 'gemini-3.5-flash-lite',
  Gemini31ProPreview = 'gemini-3.1-pro-preview',
}

export type LlmModel = XOR<
  XOR<XOR<OpenAiLlmModel, AnthropicLlmModel>, GoogleLlmModel>,
  string
>

export type LlmLangchainLoggingConfig = Readonly<{
  enabled?: boolean
  logLevel?: LogLevelNames
}>

export type LlmConfig = Readonly<{
  defaultModel: LlmModel
  apiKey: string | NilSecretEntry
  baseUrl?: string
  temperature?: number
  maxTokens?: number
  timeoutMs?: number
  maxToolRounds?: number
  strictStructuredOutput?: boolean
  langchainLogging?: LlmLangchainLoggingConfig
}>

export type ConfigWithLlm = Config &
  Readonly<{
    [LlmNamespace.llm]: LlmConfig
  }>

export type LlmTool = Readonly<{
  description: string
  schema: z.ZodType<object>
  execute: (args: never) => Promise<unknown> | unknown
}>

export type LlmTools = Readonly<Record<string, LlmTool>>

export type LlmTextRequest = Readonly<{
  prompt: string
  systemPrompt?: string
  model?: LlmModel
  temperature?: number
  tools?: LlmTools
}>

export type LlmStructuredRequest<T extends object> = LlmTextRequest &
  Readonly<{
    schema: z.ZodType<T>
  }>

export type LlmTextResponse = Response<string>

export type LlmGenerateStructured = <T extends object>(
  request: LlmStructuredRequest<T>,
  crossLayerProps?: CrossLayerProps
) => Promise<Response<T>>

export type LlmServices = Readonly<{
  generateText: LayerFunction<
    (request: LlmTextRequest) => Promise<LlmTextResponse>
  >
  generateStructured: LlmGenerateStructured
}>

export type LlmServicesLayer = Readonly<{
  [LlmNamespace.llm]: LlmServices
}>
