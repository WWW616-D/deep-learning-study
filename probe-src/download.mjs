import { writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

const out = 'D:\\py\\probe-src';
mkdirSync(out, { recursive: true });

const pkgs = [
  'pi-ark-quota',
  'pi-provider-volcengine-ark',
  '@volcengine/ark-cli',
];

async function latestTarball(pkg) {
  const meta = await (await fetch(`https://registry.npmjs.org/${encodeURIComponent(pkg)}`, { headers: { 'user-agent': 'node-probe' } })).json();
  const version = meta['dist-tags']?.latest;
  const info = meta.versions?.[version];
  if (!info?.dist?.tarball) throw new Error(pkg + ' has no tarball');
  return { version, tarball: info.dist.tarball, desc: info.description ?? '' };
}

for (const pkg of pkgs) {
  try {
    const { version, tarball, desc } = await latestTarball(pkg);
    console.log(`== ${pkg}@${version} :: ${desc}`);
    const res = await fetch(tarball, { headers: { 'user-agent': 'node-probe' } });
    if (!res.ok) { console.log(`   download failed ${res.status}`); continue; }
    const buf = Buffer.from(await res.arrayBuffer());
    const tgz = join(out, pkg.replace(/[/@]/g, '_') + '-' + version + '.tgz');
    writeFileSync(tgz, buf);
    console.log(`   saved ${tgz} (${buf.length} bytes)`);
  } catch (e) {
    console.log(`== ${pkg} ERROR ${e.message}`);
  }
}

for (const f of readdirSync(out)) {
  if (!f.endsWith('.tgz')) continue;
  const dest = join(out, f.replace(/\.tgz$/, ''));
  try {
    execFileSync('tar', ['-xzf', join(out, f), '-C', out]);
    console.log(`extracted ${f}`);
  } catch (e) {
    console.log(`extract failed ${f}: ${String(e.message).slice(0, 120)}`);
  }
}
