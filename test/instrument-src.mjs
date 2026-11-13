// Instrument package TypeScript once, then compile it. nyc's own pass runs on
// already compiled JavaScript and counts the coverage preamble as branches.
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'node:url'
import esbuild from 'esbuild'
import { createInstrumenter } from 'istanbul-lib-instrument'

const packageRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..'
)
const srcRoot = path.join(packageRoot, 'src') + path.sep

const instrumenter = createInstrumenter({
  esModules: true,
  compact: true,
  produceSourceMap: false,
  parserPlugins: [
    'typescript',
    'importMeta',
    'topLevelAwait',
    'classProperties',
    'classPrivateProperties',
    'classPrivateMethods',
    'dynamicImport',
    'optionalCatchBinding',
    'objectRestSpread',
    'asyncGenerators',
    'bigInt',
    'numericSeparator',
  ],
})

const filenameFromUrl = url => {
  const parsed = new URL(url)
  if (parsed.protocol !== 'file:') {
    return undefined
  }
  return fileURLToPath(parsed)
}

const compile = (source, filename) => {
  return esbuild.transformSync(source, {
    loader: 'ts',
    format: 'esm',
    target: 'es2022',
    sourcefile: filename,
  }).code
}

export const resolve = async (specifier, context, nextResolve) => {
  try {
    return await nextResolve(specifier, context)
  } catch (error) {
    if (!specifier.endsWith('.js')) {
      throw error
    }
    return nextResolve(`${specifier.slice(0, -3)}.ts`, context)
  }
}

export const load = async (url, context, nextLoad) => {
  const filename = filenameFromUrl(url)
  if (!filename || !filename.endsWith('.ts')) {
    return nextLoad(url, context)
  }

  const source = fs.readFileSync(filename, 'utf8')
  if (!filename.startsWith(srcRoot)) {
    return {
      format: 'module',
      source: compile(source, filename),
      shortCircuit: true,
    }
  }

  return {
    format: 'module',
    source: compile(instrumenter.instrumentSync(source, filename), filename),
    shortCircuit: true,
  }
}
