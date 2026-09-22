// 用桩环境模拟 DSH 的 __ModuleLoader__，验证 dsh-ark-plan client 模块可加载
const fs = require('fs');

// ---- 最小 React 桩（hooks 返回固定值即可，不真正渲染）----
const hooks = { use: [], handlers: {} };
function useState(init) { return [init, () => {}]; }
function useEffect(fn, deps) { return fn; }
const ReactStub = {
  useState, useEffect, createElement: () => ({}),
};

// ---- 全局桩 ----
const documentStub = {
  createElement: () => ({ style: {}, setAttribute() {}, addEventListener() {}, appendChild() {} }),
  body: { appendChild() {} },
  head: { appendChild() {} },
};
global.window = {
  __ModuleLoader__: null,
};
global.document = documentStub;
global.fetch = () => Promise.resolve({ json: () => Promise.resolve({ status: 'error' }) });

// ---- 模块加载器桩 ----
const modules = { 'react': ReactStub };
global.window.__ModuleLoader__ = {
  load({ id, factory }) {
    const require = (name) => {
      if (!(name in modules)) throw new Error('module not stubbed: ' + name);
      return modules[name];
    };
    const exports = factory(require);
    return { id, exports };
  },
};

const src = fs.readFileSync('D:\\插件\\dsh-ark-plan\\lib\\client.js', 'utf8');
// 执行脚本：脚本体调用 window.__ModuleLoader__.load(...)；我们覆写 load 捕获 factory 的导出
const run = new Function('window', 'document', 'fetch', src);
let loaded = null;
global.window.__ModuleLoader__.load = ({ id, factory }) => {
  const require = (name) => {
    if (!(name in modules)) throw new Error('module not stubbed: ' + name);
    return modules[name];
  };
  loaded = factory(require);
  return loaded;
};
run(global.window, global.document, global.fetch);

console.log('module id: dsh-ark-plan (fixed in script)');
console.log('exports.apply   =', typeof loaded.apply);
console.log('exports.inject  =', JSON.stringify(loaded.inject));
if (typeof loaded.apply === 'function' && Array.isArray(loaded.inject)) {
  console.log('RESULT: client module loads OK');
  process.exit(0);
} else {
  console.log('RESULT: client module shape unexpected');
  process.exit(1);
}
