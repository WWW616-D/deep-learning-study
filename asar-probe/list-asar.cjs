// Minimal asar reader v2 (correct layout): no deps.
const fs = require('fs');
const path = require('path');

function readAsarHeader(file) {
  const fd = fs.openSync(file, 'r');
  const buf = Buffer.alloc(16);
  fs.readSync(fd, buf, 0, 16, 0);
  const strLen = buf.readUInt32LE(12);
  const jsonStart = 16;
  const jsonBuf = Buffer.alloc(strLen);
  fs.readSync(fd, jsonBuf, 0, strLen, jsonStart);
  fs.closeSync(fd);
  const dataStart = jsonStart + strLen;
  const padded = (dataStart + 3) & ~3;
  return { header: JSON.parse(jsonBuf.toString('utf8')), dataStart: padded };
}

function walk(node, prefix, out) {
  if (!node.files) return;
  for (const [name, child] of Object.entries(node.files)) {
    const p = prefix ? `${prefix}/${name}` : name;
    if (child.files) walk(child, p, out);
    else out.push({ path: p, size: child.size, offset: child.offset, unpacked: !!child.unpacked });
  }
}

const asarPath = process.argv[2];
const mode = process.argv[3] || 'list';
const { header, dataStart } = readAsarHeader(asarPath);
const files = [];
walk(header, '', files);

if (mode === 'list') {
  for (const f of files) {
    console.log(`${f.size}\t${f.unpacked ? 'U' : ' '}\t${f.path}`);
  }
  console.error(`TOTAL_FILES=${files.length} DATA_START=${dataStart}`);
} else if (mode === 'extract') {
  const target = process.argv[4];
  const dest = process.argv[5];
  const match = files.filter(f => f.path === target);
  if (!match.length) { console.error('NOT_FOUND'); process.exit(1); }
  const f = match[0];
  const fd = fs.openSync(asarPath, 'r');
  const buf = Buffer.alloc(f.size);
  fs.readSync(fd, buf, 0, f.size, dataStart + Number(f.offset));
  fs.closeSync(fd);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, buf);
  console.error(`EXTRACTED ${target} -> ${dest} (${f.size} bytes)`);
}
