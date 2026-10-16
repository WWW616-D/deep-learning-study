// 端到端验证：完全按新 provider 的 compat 语义对 Agent Plan 发一次真实请求
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const raw = fs.readFileSync(path.join(os.homedir(), '.dsh', '.credentials.yaml'), 'utf8');
const key = raw.match(/^\s*ARK_CODING_API_KEY\s*:\s*(\S+)\s*$/m)[1];
const BASE = 'https://ark.cn-beijing.volces.com/api/plan/v3';

// 复刻 settings.yaml 里 ark-agent-plan 的模型清单
const yamlText = fs.readFileSync(path.join(os.homedir(), '.dsh', 'settings.yaml'), 'utf8');
const block = yamlText.slice(yamlText.indexOf('ark-agent-plan:'));
const ids = [...block.matchAll(/^\s+- id:\s*(\S+)/gm)].map((m) => m[1]);
console.log(`从 settings.yaml 读出 ${ids.length} 个模型: ${ids.join(', ')}\n`);

// deepseek 系：thinkingFormat=deepseek + supportsDeveloperRole=false + maxTokensField=max_tokens
const deepseekBody = (model) => ({
  model,
  messages: [{ role: 'system', content: 'You are terse.' }, { role: 'user', content: '用一个词回答：1+1=?' }],
  max_tokens: 32,
  thinking: { type: 'enabled' },
});
// 非 deepseek 系：不做 deepseek 方言
const plainBody = (model) => ({
  model,
  messages: [{ role: 'system', content: 'You are terse.' }, { role: 'user', content: '用一个词回答：1+1=?' }],
  max_tokens: 32,
});

for (const id of ids) {
  const isDeepseek = id.startsWith('deepseek');
  const body = isDeepseek ? deepseekBody(id) : plainBody(id);
  try {
    const r = await fetch(`${BASE}/chat/completions`, {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60000),
    });
    const t = await r.text();
    let note = t;
    try {
      const j = JSON.parse(t);
      note = j.error
        ? `${j.error.code}: ${String(j.error.message).slice(0, 110)}`
        : `✅ 回答=${JSON.stringify(j.choices?.[0]?.message?.content ?? '')} finish=${j.choices?.[0]?.finish_reason} tokens=${j.usage?.total_tokens}`;
    } catch {}
    console.log(`[${r.status === 200 ? '通' : '败'}] HTTP ${String(r.status).padEnd(3)} ${id.padEnd(33)} ${note}`);
  } catch (e) {
    console.log(`[异常] ${id.padEnd(33)} ${e.message}`);
  }
}

console.log('\n=== 反向测试：deepseek 方言打到 GLM 上会怎样（验证我逐模型区分是对的）===');
const r2 = await fetch(`${BASE}/chat/completions`, {
  method: 'POST',
  headers: { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json' },
  body: JSON.stringify(deepseekBody('glm-5-2-260617')),
  signal: AbortSignal.timeout(60000),
});
const t2 = await r2.text();
let n2 = t2;
try { const j = JSON.parse(t2); n2 = j.error ? `${j.error.code}: ${String(j.error.message).slice(0, 160)}` : '✅ 竟然接受 deepseek 方言'; } catch {}
console.log(`  glm-5-2-260617 + thinking:{type:enabled} → HTTP ${r2.status}  ${n2}`);
