import { gunzipSync } from 'node:zlib';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';

const out = 'D:\\py\\probe-src';

function extractTar(tarBuf) {
  const files = [];
  let off = 0;
  while (off + 512 <= tarBuf.length) {
    const header = tarBuf.subarray(off, off + 512);
    // two zero blocks = end
    if (header.every((b) => b === 0)) break;
    const name = header.subarray(0, 100).toString('utf8').replace(/\0.*$/, '');
    const size = parseInt(header.subarray(124, 136).toString('utf8').replace(/\0.*$/, '').trim(), 8) || 0;
    const typeflag = String.fromCharCode(header[156]);
    if (name && typeflag !== '5' && typeflag !== 'x' && typeflag !== 'g') {
      const data = tarBuf.subarray(off + 512, off + 512 + size);
      files.push({ name, data: Buffer.from(data) });
    }
    off += 512 + Math.ceil(size / 512) * 512;
  }
  return files;
}

const tgzs = ['pi-ark-quota-0.3.0.tgz', 'pi-provider-volcengine-ark-0.2.1.tgz', '_volcengine_ark-cli-1.0.25.tgz'];
for (const tgz of tgzs) {
  const buf = gunzipSync(readFileSync(join(out, tgz)));
  const files = extractTar(buf);
  console.log(`== ${tgz}: ${files.length} files`);
  for (const f of files) {
    const dest = join(out, 'pkg', tgz.replace(/\.tgz$/, ''), f.name);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, f.data);
  }
}
console.log('done');
