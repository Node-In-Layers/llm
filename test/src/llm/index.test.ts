import { assert } from 'chai'

import {
  AnthropicLlmModel,
  GoogleLlmModel,
  LlmNamespace,
  OpenAiLlmModel,
  name,
  services,
} from '../../../src/llm/index.js'

describe('/src/llm/index.ts', () => {
  describe('#name', () => {
    it('should export the llm namespace as the domain name', () => {
      const actual = name
      const expected = LlmNamespace.llm
      assert.equal(actual, expected)
    })
  })

  describe('#services', () => {
    it('should export the service factory', () => {
      const actual = typeof services.create
      const expected = 'function'
      assert.equal(actual, expected)
    })
  })

  describe('model catalogs', () => {
    it('should expose the current model ids', () => {
      assert.equal(OpenAiLlmModel.Gpt6Astra, 'gpt-6-astra')
      assert.equal(OpenAiLlmModel.Gpt61Sol, 'gpt-6.1-sol')
      assert.equal(OpenAiLlmModel.Gpt61Terra, 'gpt-6.1-terra')
      assert.equal(OpenAiLlmModel.Gpt6Sol, 'gpt-6-sol')
      assert.equal(OpenAiLlmModel.Gpt6Luna, 'gpt-6-luna')
      assert.equal(OpenAiLlmModel.Gpt56Sol, 'gpt-5.6-sol')
      assert.equal(OpenAiLlmModel.Gpt56Terra, 'gpt-5.6-terra')
      assert.equal(OpenAiLlmModel.Gpt56Luna, 'gpt-5.6-luna')
      assert.equal(OpenAiLlmModel.Gpt55, 'gpt-5.5')
      assert.equal(OpenAiLlmModel.Gpt54, 'gpt-5.4')
      assert.equal(OpenAiLlmModel.Gpt54Mini, 'gpt-5.4-mini')
      assert.equal(AnthropicLlmModel.ClaudeFable52, 'claude-fable-5-2')
      assert.equal(AnthropicLlmModel.ClaudeOpus55, 'claude-opus-5-5')
      assert.equal(AnthropicLlmModel.ClaudeSonnet55, 'claude-sonnet-5-5')
      assert.equal(AnthropicLlmModel.ClaudeHaiku55, 'claude-haiku-5-5')
      assert.equal(GoogleLlmModel.Gemini38Flash, 'gemini-3.8-flash')
      assert.equal(GoogleLlmModel.Gemini36Flash, 'gemini-3.6-flash')
      assert.equal(GoogleLlmModel.Gemini35FlashLite, 'gemini-3.5-flash-lite')
      assert.equal(GoogleLlmModel.Gemini31ProPreview, 'gemini-3.1-pro-preview')
    })
  })
})
