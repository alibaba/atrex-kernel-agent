/** DSH projection of AKA's public contribution protocol. No feature-specific code. */
import { open, constants } from 'node:fs/promises'
import http from 'node:http'

export const name = 'aka-contributions'
export const inject = ['tools', 'systemPrompt']
const limit = 1024 * 1024

/** Read the invocation-owned connection without exposing it to the model. */
async function connection() {
  let url = process.env.ATREX_AKA_RUNTIME_URL
  let token = process.env.ATREX_AKA_RUNTIME_TOKEN
  if (!url || !token) {
    const file = await open(process.env.ATREX_AKA_RUNTIME_CONNECTION_FILE,
      constants.O_RDONLY | constants.O_NONBLOCK | constants.O_NOFOLLOW)
    try {
      const stat = await file.stat()
      if (!stat.isFile() || stat.size > 16384) throw new Error('invalid connection file')
      const bytes = Buffer.alloc(16385)
      const { bytesRead } = await file.read(bytes, 0, bytes.length, 0)
      if (bytesRead > 16384) throw new Error('invalid connection file')
      const value = JSON.parse(bytes.subarray(0, bytesRead).toString('utf8'))
      if (Object.keys(value).sort().join(',') !== 'token,url') throw new Error('invalid connection file')
      ;({ url, token } = value)
    } finally { await file.close() }
  }
  const address = new URL(url)
  if (address.protocol !== 'http:' || address.hostname !== '127.0.0.1'
    || !address.port || address.username || address.password || address.pathname !== '/'
    || address.search || address.hash || typeof token !== 'string' || !token) {
    throw new Error('invalid Runtime connection')
  }
  return { address, token }
}

/** Send once and await the owned request; the Supervisor owns job reconciliation. */
async function invoke(payload, signal, accepting) {
  let auth
  try { auth = await connection() }
  catch { throw new Error('AKA Session connection is unavailable; resume through the Supervisor') }
  if (signal.aborted || !accepting()) throw new Error('AKA tool invocation was cancelled before dispatch')
  const body = Buffer.from(JSON.stringify(payload))
  if (body.length > 2 * limit) throw new Error('AKA request exceeds 2 MiB; no request sent')
  return await new Promise((resolve, reject) => {
    const request = http.request(new URL('/v1/contributions/invoke', auth.address), {
      method: 'POST', headers: {
        Authorization: `Bearer ${auth.token}`, 'Content-Type': 'application/json', 'Content-Length': body.length,
      },
    }, response => {
      const chunks = []
      let size = 0
      response.on('data', chunk => {
        size += chunk.length
        if (size > limit) response.destroy(new Error('AKA response exceeds 1 MiB; inspect retained records'))
        else chunks.push(chunk)
      })
      response.on('error', reject)
      response.on('end', () => {
        try {
          const value = JSON.parse(Buffer.concat(chunks).toString('utf8'))
          if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error('invalid response')
          if (response.statusCode !== 200) {
            throw new Error(`AKA Runtime HTTP ${response.statusCode}: ${JSON.stringify(value)}`)
          }
          resolve(value)
        } catch (error) { reject(error) }
      })
    })
    request.on('error', () => reject(new Error('AKA Runtime outcome is unknown; reconcile retained records before another submission')))
    request.end(body)
  })
}

/** Register selected tools and literal, module-owned prompt contributions. */
export function apply(ctx) {
  const catalog = JSON.parse(process.env.ATREX_AKA_CONTRIBUTIONS)
  if (catalog.schema !== 'aka.contributions.v1' || !/^[a-f0-9]{64}$/.test(catalog.identity)
    || !Array.isArray(catalog.tools) || !Array.isArray(catalog.instructions)) {
    throw new Error('unsupported AKA contribution catalog')
  }
  const bindings = new Map()
  const names = new Set()
  for (const tool of catalog.tools) {
    const alias = 'aka_' + tool.id.replace(/[.-]/g, '_')
    if (!/^[a-z][a-z0-9_]{0,63}$/.test(alias) || names.has(alias) || bindings.has(tool.id)) {
      throw new Error('AKA tool names cannot be projected into this DSH catalog')
    }
    names.add(alias)
    bindings.set(tool.id, alias)
  }
  const sections = catalog.instructions.map(instruction => {
    const text = instruction.text.replace(/\{\{tool:([a-z][a-z0-9_.-]*)\}\}/g, (_match, id) => {
      if (!bindings.has(id)) throw new Error('unresolved AKA instruction tool reference')
      return bindings.get(id)
    })
    if (text.includes('{{tool:')) throw new Error('malformed AKA instruction tool reference')
    return { name: 'aka:' + instruction.id, order: 5000, text, interpolate: false }
  })
  let accepting = true
  const pending = new Set()
  ctx.on('dispose', async () => {
    accepting = false
    await Promise.allSettled([...pending])
  })
  for (const tool of catalog.tools) {
    ctx.tools.register({
      name: bindings.get(tool.id), description: tool.description, parameters: tool.input_schema,
      output: {
        schema: tool.output_schema,
        render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
      },
      async execute(args, exec) {
        if (!accepting || exec.signal.aborted) throw new Error('AKA tool invocation was cancelled before dispatch')
        const work = invoke({ catalog_id: catalog.identity, tool: tool.id, arguments: args }, exec.signal, () => accepting)
        pending.add(work)
        // An accepted HTTP request may already own a GPU job. Await its result
        // on cancellation; disconnecting is not evidence that the job stopped.
        try { return await work }
        finally { pending.delete(work) }
      },
    })
  }
  for (const section of sections) ctx.systemPrompt.section(section)
}
