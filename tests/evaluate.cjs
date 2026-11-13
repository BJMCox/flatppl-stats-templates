// Evaluate library calls through the engine's real module loader and worker.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const engineDir = process.env.FLATPPL_JS_DIR || path.join(root, '../flatppl-js');
const engine = require(path.resolve(engineDir, 'packages/engine/index.ts'));
const { createWorkerHandler } = require(path.resolve(engineDir, 'packages/engine/worker.ts'));
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const expressions = Array.isArray(request) ? request : null;
const modulePaths = ['main.flatppl', ...['src', 'examples'].flatMap(dir =>
  fs.readdirSync(path.join(root, dir), { recursive: true })
    .filter(name => name.endsWith('.flatppl')).map(name => dir + '/' + name))];
const sources = Object.fromEntries(modulePaths
  .map(name => [path.join(root, name), fs.readFileSync(path.join(root, name), 'utf8')]));
const source = expressions ? 'flatppl_compat = "0.1"\na = load_module("../main.flatppl")\n'
  + expressions.map((expression, i) => `probe${i} = ${expression}`).join('\n') : request.source;
const model = engine.processSource(source, { path: path.join(__dirname, 'query.flatppl'), bundle: { sources } });
const errors = model.diagnostics.filter(d => d.severity === 'error');
if (errors.length) throw new Error(JSON.stringify(errors));
const built = engine.orchestrator.buildDerivations(model.linkedBindings, { moduleRegistry: model.linkedModuleRegistry });
const worker = createWorkerHandler();
worker.handle({ type: 'init', seed: 4711 });
const cache = new Map();
const ctx = {
  ...built, sampleCount: 1, rootSeed: 4711,
  getMeasure(name) {
    if (!cache.has(name)) cache.set(name, engine.materialiser.materialiseMeasure(name, ctx));
    return cache.get(name);
  },
  sendWorker(msg) {
    const reply = worker.handle(msg);
    return reply?.type === 'error' ? Promise.reject(new Error(reply.message)) : Promise.resolve(reply);
  },
};
const names = expressions ? expressions.map((_, i) => `probe${i}`) : request.bindings;
(async () => {
  const values = [];
  for (const name of names) {
    if (built.fixedValues.has(name)) values.push(built.fixedValues.get(name));
    else {
      const m = await ctx.getMeasure(name);
      if (m.value) values.push(m.value);
      else if (m.samples) values.push({ shape: [], data: [m.samples[0]] });
      else throw new Error(`${name} did not produce a numeric value`);
    }
  }
  process.stdout.write(JSON.stringify(values, (_key, value) =>
    ArrayBuffer.isView(value) ? Array.from(value) :
    typeof value === 'number' && !Number.isFinite(value) ? String(value) : value));
})().catch(error => { console.error(error); process.exitCode = 1; });
