# Node In Layers LLM

Standardized, OpenAI-compatible LLM access for a Node In Layers system.

Configure one default chat model and use it anywhere in the system through
`system.services[LlmNamespace.llm]`. Callers provide a prompt and receive
either a string or a schema-validated JSON value; they do not need to construct
providers, manage credentials, or implement a tool-calling loop.

The package uses LangChain's `ChatOpenAI` adapter. Any model supported by an
OpenAI-compatible endpoint—including a gateway that exposes Claude or Gemini
models—can be selected through that one interface.

## Quick start

Install the package:

```bash
npm install @node-in-layers/llm
```

Add the secrets configuration above the LLM domain so `nil-secret` placeholders
are resolved before the LLM service is created. Never commit credentials to a
config file.

```typescript
import { CoreNamespace, LogFormat, LogLevelNames } from '@node-in-layers/core'
import {
  config as secretsConfig,
  SecretsNamespace,
} from '@node-in-layers/secrets'
import { LlmNamespace, OpenAiLlmModel, llm } from '@node-in-layers/llm'
import { myDomain } from './src/my-domain/index.js'

export default () => ({
  systemName: 'my-system',
  environment: 'base',
  [CoreNamespace.root]: {
    domains: await Promise.all([
      // Resolve secret placeholders before any domain consumes its config.
      secretsConfig,
      llm,
      // Put all domains that use LLM below.
      myDomain,
    ]),
    layerOrder: ['services', 'features'],
    logging: {
      logLevel: LogLevelNames.info,
      logFormat: LogFormat.json,
    },
  },
  [SecretsNamespace.Core]: {},
  [LlmNamespace.llm]: {
    // Required: the default model for this endpoint.
    defaultModel: OpenAiLlmModel.Gpt56Sol,
    // Required: resolved by @node-in-layers/secrets before the LLM loads.
    apiKey: {
      type: 'nil-secret',
      format: 'string',
      key: 'llm.apiKey',
    },
  },
})
```

Load the system as usual, then request text:

```typescript
import { loadSystem } from '@node-in-layers/core'
import { LlmNamespace } from '@node-in-layers/llm'

const system = await loadSystem({ environment: 'dev' })

const answer = await system.services[LlmNamespace.llm].generateText({
  prompt: 'Give one concise fact about the moon.',
})

console.info(answer)
```

## Configuration

The LLM configuration is keyed by `LlmNamespace.llm`, whose value is
`'@node-in-layers/llm'`. It establishes safe system defaults. Per-request
options can override supported model settings without exposing endpoint-specific
objects across the rest of the system.

```typescript
import { CoreNamespace, LogLevelNames } from '@node-in-layers/core'
import {
  config as secretsConfig,
  SecretsNamespace,
} from '@node-in-layers/secrets'
import { LlmNamespace, OpenAiLlmModel, llm } from '@node-in-layers/llm'

const config = {
  [CoreNamespace.root]: {
    domains: [secretsConfig, llm],
  },
  [SecretsNamespace.Core]: {},
  [LlmNamespace.llm]: {
    // Required: the model used unless an individual request overrides it.
    defaultModel: OpenAiLlmModel.Gpt56Sol,

    // Required: a nil-secret placeholder resolved by @node-in-layers/secrets.
    apiKey: {
      type: 'nil-secret',
      format: 'string',
      key: 'llm.apiKey',
    },

    // Optional: OpenAI-compatible endpoint. Omit to use OpenAI's default
    // endpoint. Write non-secret endpoints directly in configuration.
    baseUrl: 'https://llm-gateway.example.com/v1',

    // Optional: sampling default. Lower values are more deterministic.
    temperature: 0,

    // Optional: upper bound for generated tokens. Omit to use the endpoint
    // default.
    maxTokens: 1_000,

    // Optional: fail an unresponsive endpoint request after this many
    // milliseconds.
    timeoutMs: 30_000,

    // Optional: maximum model → tool → model rounds for one request. This
    // prevents a malformed prompt or tool from creating an unbounded loop.
    maxToolRounds: 5,

    // Optional: ask the endpoint to enforce the supplied JSON schema when the
    // selected model supports strict structured output.
    strictStructuredOutput: true,

    // Optional: lifecycle logging around LangChain endpoint calls. It is off
    // by default because prompts and tool results can contain sensitive data.
    langchainLogging: {
      // Optional: defaults to false.
      enabled: false,

      // Optional: defaults to trace when LangChain call logging is enabled.
      logLevel: LogLevelNames.trace,
    },
  },
}
```

The configured endpoint must have the credentials it requires. A missing key or
model produces an `LLM_CONFIGURATION_ERROR` before the request is sent.

The `nil-secret` placeholder is resolved by `@node-in-layers/secrets` before
this package reads its configuration. Configure the secret backend through
`[SecretsNamespace.Core]`; for the built-in local backend, the matching value
is stored under `llm.apiKey` in `secrets.<environment>.json` or
`secrets.<environment>.json5`. `baseUrl` is not a secret and belongs directly
in the LLM configuration.

### Models

`defaultModel` and the request-level `model` accept one of the model enums or a
`string`. The enums are convenience catalogs only: every value is sent as a
model string through the configured OpenAI-compatible endpoint.

```typescript
import {
  AnthropicLlmModel,
  GoogleLlmModel,
  OpenAiLlmModel,
} from '@node-in-layers/llm'

const commonModels = {
  openAi: [
    OpenAiLlmModel.Gpt6Astra,
    OpenAiLlmModel.Gpt61Sol,
    OpenAiLlmModel.Gpt61Terra,
    OpenAiLlmModel.Gpt6Sol,
    OpenAiLlmModel.Gpt6Luna,
    OpenAiLlmModel.Gpt56Sol,
    OpenAiLlmModel.Gpt56Terra,
    OpenAiLlmModel.Gpt56Luna,
    OpenAiLlmModel.Gpt55,
    OpenAiLlmModel.Gpt54,
    OpenAiLlmModel.Gpt54Mini,
  ],
  claude: [
    AnthropicLlmModel.ClaudeFable52,
    AnthropicLlmModel.ClaudeOpus55,
    AnthropicLlmModel.ClaudeSonnet55,
    AnthropicLlmModel.ClaudeHaiku55,
  ],
  google: [
    GoogleLlmModel.Gemini38Flash,
    GoogleLlmModel.Gemini36Flash,
    GoogleLlmModel.Gemini35FlashLite,
    GoogleLlmModel.Gemini31ProPreview,
  ],
}
```

Use a string when an endpoint supports a newer model or a pinned model
snapshot:

```typescript
defaultModel: 'my-provider-chat-model'
```

The package does not authenticate directly with Anthropic or Google in this
version. To use a Claude or Gemini enum value, configure `baseUrl` and `apiKey`
for an OpenAI-compatible endpoint that serves that model.

### LangChain call logging

Set `langchainLogging.enabled` to `true` to emit lifecycle logs around model
requests and tool rounds. Its `logLevel` defaults to `LogLevelNames.trace` and
can be raised to `debug`, `info`, `warn`, or `error` to fit the environment.

The logs identify the model, request phase, selected tool names, elapsed time,
and errors. They intentionally exclude API keys. Prompts, model responses, and
tool results remain subject to the surrounding Node In Layers function tracing,
so production logging must still be configured with the sensitivity of that
data in mind.

## Generate text

`generateText` returns `Promise<Response<string>>`: either the final assistant
message or Node In Layers' standardized `ErrorObject`.

```typescript
import { isErrorObject } from '@node-in-layers/core'

const summary = await system.services[LlmNamespace.llm].generateText({
  systemPrompt: 'You write factual, concise summaries.',
  prompt: 'Summarize the benefits of dependency injection in one sentence.',
})

if (isErrorObject(summary)) {
  console.error(summary.error.code, summary.error.message)
} else {
  console.info(summary)
}
```

Use a request-level model or sampling override only when the default is not
appropriate:

```typescript
import { OpenAiLlmModel } from '@node-in-layers/llm'

const creativeCopy = await system.services[LlmNamespace.llm].generateText({
  prompt: 'Write a short name for a newsletter about software craftsmanship.',
  model: OpenAiLlmModel.Gpt6Sol,
  temperature: 0.8,
})
```

## Generate structured data

`generateStructured<T>` accepts a Zod schema typed as `z.ZodType<T>` and
returns `Promise<Response<T>>`. This gives the caller its domain type after the
value has been validated against the runtime schema.

TypeScript types are erased at runtime, so passing only `T` cannot constrain or
validate an LLM response. Always supply the Zod schema as well. Describe every
field: descriptions materially improve output quality and make the contract
understandable to operators and tools.

```typescript
import { isErrorObject } from '@node-in-layers/core'
import { z } from 'zod'

type BookRecommendation = Readonly<{
  title: string
  author: string
  reason: string
}>

const BookRecommendationSchema: z.ZodType<BookRecommendation> = z.object({
  title: z.string().describe('Recommended book title'),
  author: z.string().describe('Book author'),
  reason: z.string().describe('One-sentence reason for the recommendation'),
})

const recommendation = await system.services[
  LlmNamespace.llm
].generateStructured<BookRecommendation>({
  prompt:
    'Recommend a book about functional programming for an experienced developer.',
  schema: BookRecommendationSchema,
})

if (isErrorObject(recommendation)) {
  console.error(recommendation.error.code, recommendation.error.message)
} else {
  console.info(recommendation.title)
}
```

Structured-output support and JSON Schema restrictions differ by endpoint and
model. Use required properties or nullable properties rather than optional Zod
properties whenever the selected endpoint requires a stricter JSON Schema
subset.

## Tools

Tools are call-scoped because their handlers are executable application code,
not static configuration. Pass a record whose key is the tool name. Every tool
has a description, a Zod argument schema, and an asynchronous handler.

```typescript
import { z } from 'zod'

const result = await system.services[LlmNamespace.llm].generateText({
  prompt: 'What is the current balance for account acct_123?',
  tools: {
    getAccountBalance: {
      description: 'Fetch the current balance for a customer account.',
      schema: z.object({
        accountId: z.string().describe('The customer account identifier'),
      }),
      execute: async ({ accountId }) => {
        return accountsService.getBalance({ accountId })
      },
    },
  },
})
```

When the model requests a registered tool, this package validates the arguments,
executes the handler, sends the result back to the model, and returns the final
answer. It rejects unknown tool names, invalid arguments, handler errors, and
requests that exceed `maxToolRounds`.

Tools also work with `generateStructured`. The package completes tool rounds
first, then obtains the final schema-validated object. This is deliberate:
LangChain does not safely support chaining `bindTools()` and
`withStructuredOutput()` into one runnable.

Treat every tool handler as a security boundary:

- Authenticate and authorize inside the handler.
- Validate IDs and other inputs even though they were checked against the Zod
  schema.
- Do not expose unrestricted filesystem, shell, database, or network access.
- Return the smallest result the model needs; tool results become model context.

## Operational notes

- Provider calls are made in the services layer. `generateText` and
  `generateStructured` are services, so other domains access a stable
  capability rather than a vendor SDK.
- Node In Layers tracing can log service arguments and results. Do not
  place secrets in prompts or tool results, and configure logging appropriately
  for production.
- `langchainLogging` is separate, opt-in lifecycle logging around endpoint
  calls. It defaults to the `trace` level when enabled and never logs API keys.
- Public LLM service methods return Node In Layers `ErrorObject` values rather
  than throwing expected failures. Use `isErrorObject()` before consuming the
  response value. The standardized codes include `LLM_CONFIGURATION_ERROR`,
  `LLM_PROVIDER_ERROR`, `LLM_STRUCTURED_OUTPUT_INVALID`,
  `LLM_TOOL_NOT_FOUND`, `LLM_TOOL_VALIDATION_FAILED`,
  `LLM_TOOL_EXECUTION_FAILED`, and `LLM_TOOL_ROUND_LIMIT_REACHED`.
