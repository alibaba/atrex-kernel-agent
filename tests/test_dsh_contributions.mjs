import assert from 'node:assert/strict'
import http from 'node:http'
import { mkdtemp, writeFile, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { apply } from '../src/aka/legacy/agent/dsh_bridge.mjs'

function catalog() {
  return {
    schema: 'aka.contributions.v1', identity: 'a'.repeat(64),
    tools: [{ id: 'fixture.echo', description: 'Echo the request', input_schema: { type: 'object' }, output_schema: { type: 'object' } }],
    instructions: [{ id: 'fixture.usage', text: 'Use {{tool:fixture.echo}}; keep {{literal}} unchanged.' }],
  }
}

function context() {
  const tools = [], sections = [], cleanup = []
  return { tools, sections, cleanup, ctx: {
    tools: { register: tool => { tools.push(tool) } },
    systemPrompt: { section: section => { sections.push(section) } },
    on: (name, callback) => { assert.equal(name, 'dispose'); cleanup.push(callback) },
  } }
}

async function withRuntime(handler, run) {
  const server = http.createServer(handler)
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const before = { ...process.env }
  process.env.ATREX_AKA_CONTRIBUTIONS = JSON.stringify(catalog())
  process.env.ATREX_AKA_RUNTIME_URL = `http://127.0.0.1:${server.address().port}`
  process.env.ATREX_AKA_RUNTIME_TOKEN = 'synthetic-capability'
  try { await run() }
  finally {
    process.env = before
    await new Promise(resolve => server.close(resolve))
  }
}

test('one generic plugin projects any selected module and sends its pinned identity', async () => {
  let calls = 0
  await withRuntime((req, res) => {
    calls++
    assert.equal(req.url, '/v1/contributions/invoke')
    assert.equal(req.headers.authorization, 'Bearer synthetic-capability')
    let body = ''
    req.on('data', chunk => { body += chunk })
    req.on('end', () => {
      assert.deepEqual(JSON.parse(body), { catalog_id: 'a'.repeat(64), tool: 'fixture.echo', arguments: { text: 'hello' } })
      res.end(JSON.stringify({ text: 'hello' }))
    })
  }, async () => {
    const host = context()
    apply(host.ctx)
    assert.equal(host.tools[0].name, 'aka_fixture_echo')
    assert.equal(host.sections[0].text, 'Use aka_fixture_echo; keep {{literal}} unchanged.')
    assert.equal(host.sections[0].interpolate, false)
    const value = await host.tools[0].execute({ text: 'hello' }, { signal: new AbortController().signal })
    assert.deepEqual(value, { text: 'hello' })
    assert.deepEqual(host.tools[0].output.render({}, value), [{ type: 'text', text: '{"text":"hello"}' }])
  })
  assert.equal(calls, 1)
})

test('cancellation and unload await an accepted request and close further admission', async () => {
  let release, arrived
  const arrival = new Promise(resolve => { arrived = resolve })
  let calls = 0
  await withRuntime((req, res) => {
    calls++
    req.resume()
    release = () => res.end('{"record_id":"retained"}')
    arrived()
  }, async () => {
    const host = context()
    apply(host.ctx)
    const control = new AbortController()
    const work = host.tools[0].execute({}, { signal: control.signal })
    await arrival
    control.abort()
    let stopped = false
    const closing = host.cleanup[0]().then(() => { stopped = true })
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(stopped, false)
    await assert.rejects(host.tools[0].execute({}, { signal: new AbortController().signal }), /before dispatch/)
    release()
    assert.deepEqual(await work, { record_id: 'retained' })
    await closing
    assert.equal(calls, 1)
  })
})

test('an HTTP policy failure returns once with no retry', async () => {
  let calls = 0
  await withRuntime((req, res) => {
    calls++
    req.resume()
    res.writeHead(503)
    res.end('{"error":{"code":"outcome_unknown"}}')
  }, async () => {
    const host = context()
    apply(host.ctx)
    await assert.rejects(host.tools[0].execute({}, { signal: new AbortController().signal }), /outcome_unknown/)
    assert.equal(calls, 1)
  })
})

test('cancellation during connection preparation prevents dispatch', async () => {
  await withRuntime(() => { assert.fail('cancelled preparation must not dispatch') }, async () => {
    const host = context()
    apply(host.ctx)
    const control = new AbortController()
    const work = host.tools[0].execute({}, { signal: control.signal })
    control.abort()
    await assert.rejects(work, /before dispatch/)
  })
})

test('an invalid Runtime origin is rejected before network dispatch', async () => {
  await withRuntime(() => { assert.fail('network must not be used') }, async () => {
    process.env.ATREX_AKA_RUNTIME_URL = 'https://example.com'
    const host = context()
    apply(host.ctx)
    await assert.rejects(host.tools[0].execute({}, { signal: new AbortController().signal }), /connection is unavailable/)
  })
})

test('the private connection file works and a symlink is rejected before dispatch', async () => {
  let calls = 0
  await withRuntime((req, res) => {
    calls++
    req.resume()
    assert.equal(req.headers.authorization, 'Bearer synthetic-capability')
    res.end('{"record_id":"retained"}')
  }, async () => {
    const root = await mkdtemp(join(tmpdir(), 'aka-bridge-'))
    const file = join(root, 'connection.json')
    try {
      await writeFile(file, JSON.stringify({ url: process.env.ATREX_AKA_RUNTIME_URL, token: process.env.ATREX_AKA_RUNTIME_TOKEN }), { mode: 0o600 })
      delete process.env.ATREX_AKA_RUNTIME_URL
      delete process.env.ATREX_AKA_RUNTIME_TOKEN
      process.env.ATREX_AKA_RUNTIME_CONNECTION_FILE = file
      const host = context()
      apply(host.ctx)
      assert.deepEqual(await host.tools[0].execute({}, { signal: new AbortController().signal }), { record_id: 'retained' })
      const link = join(root, 'link')
      await symlink(file, link)
      process.env.ATREX_AKA_RUNTIME_CONNECTION_FILE = link
      await assert.rejects(host.tools[0].execute({}, { signal: new AbortController().signal }), /connection is unavailable/)
      assert.equal(calls, 1)
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})

test('native-name collisions fail before registering a partial catalog', () => {
  const before = process.env.ATREX_AKA_CONTRIBUTIONS
  const value = catalog()
  value.tools[0].id = 'fixture.echo_value'
  value.tools.push({ ...value.tools[0], id: 'fixture.echo.value' })
  value.instructions[0].text = 'Use {{tool:fixture.echo_value}}.'
  process.env.ATREX_AKA_CONTRIBUTIONS = JSON.stringify(value)
  try {
    const host = context()
    assert.throws(() => apply(host.ctx), /cannot be projected/)
    assert.deepEqual(host.tools, [])
  } finally {
    if (before === undefined) delete process.env.ATREX_AKA_CONTRIBUTIONS
    else process.env.ATREX_AKA_CONTRIBUTIONS = before
  }
})
