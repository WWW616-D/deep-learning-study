// 模拟 arkcli usage plan 输出，验证 dsh-ark-plan 的解析/归一化逻辑
function tryParseJson(text) {
  if (typeof text !== 'string') return null
  const start = text.indexOf('{')
  if (start < 0) return null
  let depth = 0, inStr = false, esc = false, end = -1
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i]
    if (inStr) { if (esc) esc = false; else if (ch === '\\') esc = true; else if (ch === '"') inStr = false; continue }
    if (ch === '"') inStr = true
    else if (ch === '{') depth += 1
    else if (ch === '}') { depth -= 1; if (depth === 0) { end = i; break } }
  }
  if (end < 0) return null
  try { return JSON.parse(text.slice(start, end + 1)) } catch { return null }
}

const PRODUCT = 'coding-plan'
function normalizeUsagePlan(json) {
  const root = json && typeof json === 'object' ? json : {}
  if (root.ok === false && root.error && typeof root.error === 'object') {
    const message = typeof root.error.message === 'string' ? root.error.message : '未登录或未配置'
    return { status: 'auth', subscribed: false, level: null, periods: [], message }
  }
  const source = root.data && typeof root.data === 'object' && !Array.isArray(root.data) ? root.data : root
  const items = Array.isArray(source.items) ? source.items : []
  const item =
    items.find((i) => i && i.product === PRODUCT && i.subscribed === true) ||
    items.find((i) => i && i.product === PRODUCT) ||
    items.find((i) => i && i.subscribed === true) ||
    items[0]
  if (!item || typeof item !== 'object') return { status: 'no-data', subscribed: false, level: null, periods: [], message: '未查询到套餐数据' }
  const subscribed = item.subscribed === true
  if (typeof item.error === 'string' && item.error !== '') return { status: 'auth', subscribed, level: null, periods: [], message: item.error }
  const periods = Array.isArray(item.periods)
    ? item.periods.filter((p) => p && typeof p === 'object' && typeof p.label === 'string' && p.label !== '').map((p) => ({
        label: p.label,
        percent: typeof p.percent === 'number' && Number.isFinite(p.percent) ? Math.max(0, Math.min(100, p.percent)) : null,
        resetAt: typeof p.reset_at === 'string' && p.reset_at !== '' ? p.reset_at : null,
      }))
    : []
  return { status: subscribed ? 'ok' : 'no-data', subscribed, level: typeof source.level === 'string' ? source.level : typeof item.level === 'string' ? item.level : null, periods, message: null }
}

const samples = [
  { name: '未配置/未登录（实测输出）', raw: '{"ok":false,"error":{"type":"error","message":"not configured, run `arkcli config init --profile default` or `arkcli auth login`"}}' },
  { name: '正常（真实形状）', raw: '{"items":[{"product":"coding-plan","subscribed":true,"level":"Pro","periods":[{"label":"session","percent":32.5,"reset_at":"2026-09-03T20:00:00+08:00"},{"label":"weekly","percent":45.1,"reset_at":"2026-09-07T00:00:00+08:00"},{"label":"monthly","percent":60.2,"reset_at":"2026-10-01T00:00:00+08:00"}]}]}' },
  { name: '未登录（items 带 error）', raw: JSON.stringify({ items: [{ product: 'coding-plan', subscribed: false, error: 'refresh_token is invalid, please run arkcli auth login volc-sso' }] }) },
  { name: '无套餐（items 空）', raw: JSON.stringify({ items: [] }) },
  { name: '非 JSON 输出', raw: 'boom: something went wrong' },
]

for (const s of samples) {
  const parsed = tryParseJson(s.raw)
  const result = parsed ? normalizeUsagePlan(parsed) : { status: 'error', message: 'arkcli 输出不是 JSON' }
  console.log('== ' + s.name)
  console.log('   status=' + result.status + ' subscribed=' + result.subscribed + ' level=' + result.level)
  if (result.periods) result.periods.forEach((p) => console.log('   period ' + p.label + ' = ' + p.percent + '% reset=' + p.resetAt))
  if (result.message) console.log('   message=' + String(result.message).slice(0, 90))
}
