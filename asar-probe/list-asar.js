// Minimal asar reader: list files + optionally extract, no deps.
const fs = require('fs');
const path = require('path');

function readAsarHeader(file) {
  const fd = fs.openSync(file, 'r');
  const buf = Buffer.alloc(16);
  fs.readSync(fd, buf, 0, 16, 0);
  const headerPickleSize = buf.readUInt32LE(0);
  const headerJsonSize = buf.readUInt32LE(4);
  const jsonBuf = Buffer.alloc(headerJsonSize);
  fs.readSync(fd, jsonBuf, 0, headerJsonSize, 8);
  fs.closeSync(fd);
  return JSON.parse(jsonBuf.toString('utf8'));
}

function walk(node, prefix, out) {
  if (!node.files) return;
  for (const [name, child] of Object.entries(node.files)) {
    const p = prefix ? `${prefix}/${name}` : name;
    if (child.files) {
      walk(child, p, out);
    } else {
      out.push({ path: p, size: child.size, offset: child.offset, unpacked: !!child.unpacked });
    }
  }
}

const asarPath = process.argv[2];
const mode = process.argv[3] || 'list';
const header = readAsarHeader(asarPath);
const files = [];
walk(header, '', files);

if (mode === 'list') {
  for (const f of files) {
    console.log(`${f.size}\t${f.unpacked ? 'U' : ' '}\t${f.path}`);
  }
  console.error(`TOTAL_FILES=${files.length}`);
} else if (mode === 'extract') {
  const target = process.argv[4];
  const dest = process.argv[5];
  const match = files.filter(f => f.path === target);
  if (!match.length) { console.error('NOT_FOUND'); process.exit(1); }
  const f = match[0];
  const fd = fs.openSync(asarPath, 'r');
  // data section starts at 8 + headerJsonSize + 4 (pickle trailer for string is 4 bytes of padding? use header offset)
  const dataStart = 8 + header.headerSize; // header.headerSize = pickle size includes the json + padding
  const buf = Buffer.alloc(f.size);
  fs.readSync(fd, buf, 0, f.size, dataStart + f.offset);
  fs.closeSync(fd);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, buf);
  console.error(`EXTRACTED ${target} -> ${dest} (${f.size} bytes)`);
}
