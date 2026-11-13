'use strict'

const { register } = require('node:module')
const { pathToFileURL } = require('node:url')

register('./instrument-src.mjs', pathToFileURL(__filename))
