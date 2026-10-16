// 重点验证：Agent Plan 网关只认真专属 key —— 两个本机 key 都试一遍
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const raw = fs.readFileSync(path.join(os.homedir(), '.dsh', '.credentials.yaml'), 'utf8');
const get = (n) => raw.match(new RegExp(`^\\s*${n}\\s*:\\s*(\\S+)\\s*$`, 'm'))?.[1] ?? null;
const keys = { ARK_API_KEY: get('ARK_API_KEY'), ARK_CODING_API_KEY: get('ARK_CODING_API_KEY') };

const GATEWAYS = {
  'plan/v3   (Agent Plan) ': 'https://ark.cn-beijing.volces.com/api/plan/v3',
  'v3        (普通方舟)   ': 'https://ark.cn-beijing.volces.com/api/v3',
  'coding/v3 (Coding Plan)': 'https://ark.cn-beijing.volces.com/api/coding/v3',
};

for (const [kw, k] of Object.entries(keys)) {
  if (!k) continue;
  console.log(`\n########## ${kw}   head=${k.slice(0, 9)}… tail=…${k.slice(-5)}`);
  for (const [gwName, gw] of Object.entries(GATEWAYS)) {
    const r = await fetch(`${gw}/chat/completions`, {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + k, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'ark-code-latest', messages: [{ role: 'user', content: 'hi' }], max_tokens: 4 }),
      signal: AbortSignal.timeout(45000),
    }).catch((e) => ({ status: 'ERR', text: async () => e.message }));
    const t = await r.text();
    let note = t;
    try { const j = JSON.parse(t); note = j.error ? `${j.error.code}: ${j.error.message}` : `OK ← ${JSON.stringify(j.choices?.[0]?.message?.content ?? '')}`; } catch {}
    console.log(`  ${gwName}  HTTP ${String(r.status).padEnd(3)}  ${String(note).slice(0, 155)}`);
  }
}
