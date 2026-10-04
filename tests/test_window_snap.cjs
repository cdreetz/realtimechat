const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('demos/retro-os/index.html','utf8').split('// ================= snap geometry =================')[1].split('// ================= retro window manager =================')[0];
const geometry = vm.runInNewContext(source + '\nsnapGeometry');
for (const width of [320, 1001, 1658]) {
 for (const layouts of [['left_half','right_half'],['left_third','center_third','right_third'],['left_two_thirds','right_third']]) {
  let edge = 0;
  for (const layout of layouts) {
   const r=geometry(layout,width,777);
   assert.equal(parseInt(r.left),edge);
   edge += parseInt(r.width);
   assert.equal(r.height,'777px');
  }
  assert.equal(edge,width);
 }
}
assert.throws(()=>geometry('bogus',1000,700));
console.log('Snap regions cover the desktop without gaps or overlaps: passed');
