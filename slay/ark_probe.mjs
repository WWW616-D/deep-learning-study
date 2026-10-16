// 火山方舟 key 诊断脚本：直接读取 ~/.dsh/.credentials.yaml，避免命令行传参出错
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const credPath = path.join(os.homedir(), '.dsh', '.credentials.yaml');
const raw = fs.readFileSync(credPath, 'utf8');
const get = (name) => {
  const m = raw.match(new RegExp(`^\\s*${name}\\s*:\\s*(\\S+)\\s*$`, 'm'));
  return m ? m[1] : null;
};

const keys = {
  ARK_API_KEY: get('ARK_API_KEY'),
  ARK_CODING_API_KEY: get('ARK_CODING_API_KEY'),
};

const describe = (k) => `len=${k.length} head=${k.slice(0, 9)}… tail=…${k.slice(-5)}`;

const bases = {
  'v3        ': 'https://ark.cn-beijing.volces.com/api/v3',
  'coding/v3 ': 'https://ark.cn-beijing.volces.com/api/coding/v3',
};

const timeout = (ms) => AbortSignal.timeout(ms);

async function call(label, url, key, init = {}) {
  try {
    const r = await fetch(url, {
      ...init,
      headers: { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json', ...(init.headers || {}) },
      signal: timeout(30000),
    });
    const t = await r.text();
    let msg = t;
    try {
      const j = JSON.parse(t);
      msg = j.error ? `${j.error.code ?? ''} ${j.error.message ?? ''}`.trim() : JSON.stringify(j).slice(0, 300);
    } catch { /* 保留原文 */ }
    return { label, status: r.status, msg: String(msg).slice(0, 220) };
  } catch (e) {
    return { label, status: 'ERR', msg: e.message };
  }
}

// 从 settings.yaml 里读出 ark-coding provider 配置的模型
const settingsRaw = fs.readFileSync(path.join(os.homedir(), '.dsh', 'settings.yaml'), 'utf8');
const codingModels = [...settingsRaw.matchAll(/^\s*-\s*id:\s*(\S+)/gm)].map((m) => m[1]);

console.log('=== 1) /models 列表鉴权测试 ===');
for (const [kn, k] of Object.entries(keys)) {
  if (!k) { console.log(`${kn}: 未配置`); continue; }
  console.log(`\n${kn}  (${describe(k)})`);
  for (const [bn, base] of Object.entries(bases)) {
    const r = await call(bn, `${base}/models`, k);
    console.log(`  ${bn} HTTP ${r.status}  ${r.msg}`);
  }
}

console.log('\n=== 2) 实际对话请求测试（settings.yaml 中配置的模型） ===');
const uniqueModels = [...new Set(codingModels)];
for (const [kn, k] of Object.entries(keys)) {
  if (!k) continue;
  console.log(`\n# 用 ${kn}`);
  for (const m of uniqueModels) {
    for (const [bn, base] of Object.entries(bases)) {
      const r = await call(`${bn} ${m}`, `${base}/chat/completions`, k, {
        method: 'POST',
        body: JSON.stringify({ model: m, messages: [{ role: 'user', content: 'ping' }], max_tokens: 8, stream: false }),
      });
      const ok = r.status === 200;
      console.log(`  [${ok ? 'OK  ' : 'FAIL'}] ${bn} model=${m}  HTTP ${r.status}  ${ok ? '' : r.msg}`);
    }
  }
}
