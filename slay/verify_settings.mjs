// 校验改写后的 settings.yaml：解析 + 按 llm-pi-ai 的规则自检
import fs from 'node:fs';
import { createRequire } from 'node:module';

// DSH 自带 yaml 解析器，从它的 node_modules 里借
const require = createRequire('C:\\Users\\17944\\.dsh\\profiles\\web\\');
let YAML = null;
for (const mod of ['yaml', 'js-yaml']) {
  try { YAML = require(mod); console.log(`使用 YAML 解析器: ${mod}`); break; } catch (e) { /* 继续找 */ }
}
if (!YAML) { console.error('未找到 yaml 解析器'); process.exit(1); }

const p = 'C:\\Users\\17944\\.dsh\\settings.yaml';
const text = fs.readFileSync(p, 'utf8');
const doc = YAML.parse ? YAML.parse(text) : YAML.load(text);

const COMPLETIONS_ALLOWED = new Set([
  'supportsStore', 'supportsDeveloperRole', 'supportsReasoningEffort', 'supportsUsageInStreaming',
  'maxTokensField', 'requiresToolResultName', 'requiresAssistantAfterToolResult',
  'requiresThinkingAsText', 'requiresReasoningContentOnAssistantMessages', 'thinkingFormat',
  'chatTemplateKwargs', 'supportsStrictMode', 'cacheControlFormat', 'supportsLongCacheRetention',
]);
const WITHHELD = new Set(['openRouterRouting', 'vercelGatewayRouting', 'zaiToolStream',
  'supportsOpenAIGrammarTools', 'sendSessionAffinityHeaders', 'deferredToolsMode', 'sessionAffinityFormat']);

const providers = doc?.['llm-pi-ai']?.providers ?? {};
console.log(`\n=== provider 路由清单（共 ${Object.keys(providers).length}）===`);
let errors = 0;

for (const [route, prof] of Object.entries(providers)) {
  const models = Array.isArray(prof.models) ? prof.models : [];
  const declared = !!(prof.api && prof.baseURL && models.length > 0);
  console.log(`\n■ ${route}  [${prof.displayName ?? '(无 displayName)'}]`);
  console.log(`   api=${prof.api ?? '(继承catalog)'}  baseURL=${prof.baseURL ?? '(继承catalog)'}`);
  console.log(`   apiKeyEnv=${prof.apiKeyEnv ?? '(未认证!)'}  reasoning=${prof.reasoning ?? '-'}  模型数=${models.length}`);

  // 手工声明路由必须有 api + baseURL + 非空 models
  if (prof.api && !prof.baseURL) { console.log('   ❌ 声明了 api 却没有 baseURL'); errors++; }
  if (prof.baseURL && !prof.api && !declared) { console.log('   ⚠️  有 baseURL 无 api —— 需 catalog 认识该路由'); }

  for (const m of models) {
    const issues = [];
    if (!m.id) issues.push('缺少 id');
    if (m.compat) {
      for (const [k, v] of Object.entries(m.compat)) {
        if (WITHHELD.has(k)) issues.push(`compat "${k}" 被 pi-ai 保留，禁止配置`);
        else if (!COMPLETIONS_ALLOWED.has(k)) issues.push(`compat "${k}" 不是 openai-completions 的合法键`);
        if (v === null || v === undefined) issues.push(`compat "${k}" 没有写值（会被拒绝）`);
      }
    }
    if (m.reasoningEfforts) {
      const keys = Object.keys(m.reasoningEfforts);
      if (keys.length === 0) issues.push('reasoningEfforts 空声明');
      const EFFORT_LEVELS = ['off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
      for (const k of keys) if (!EFFORT_LEVELS.includes(k)) issues.push(`reasoningEfforts 档位 "${k}" 非法`);
    }
    const tag = issues.length ? `❌ ${issues.join('; ')}` : '✓';
    if (issues.length) errors++;
    console.log(`     ${tag} ${m.id}${m.contextWindow ? `  ctx=${m.contextWindow}` : ''}${m.maxTokens ? ` max=${m.maxTokens}` : ''}`);
  }
}

console.log(`\n=== 结果：${errors === 0 ? '✅ 全部通过自检' : `❌ 发现 ${errors} 处问题`} ===`);
console.log(`顶层配置分节: ${Object.keys(doc).join(', ')}`);
