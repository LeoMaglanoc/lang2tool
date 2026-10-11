import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { frameAt, continuous, advance } from '../src/playback.mjs';
const manifest=JSON.parse(fs.readFileSync(new URL('../public/data/manifest.json',import.meta.url)));
const robot=JSON.parse(fs.readFileSync(new URL('../public/assets/robot.json',import.meta.url)));
// Irregular timestamps must interpolate by simulation time, not frame count.
test('irregular timestamps, bounds, and exact endpoint',()=>{
  assert.deepEqual(frameAt([0,.1,.5,1.2],.3),{index:1,next:2,alpha:.49999999999999994});
  assert.equal(frameAt([0,.1,.5,1.2],1.2).index,3);
  assert.equal(frameAt([0,.1,.5,1.2],-1).alpha,0);
  assert.equal(advance(1,.5,2,1.2),1.2);
});
// A reset must never become a fictitious physical sweep in the displayed animation.
test('reset boundaries prevent blending',()=>{
  assert.equal(continuous([0,0,0],[.5,0,0],'running','running'),false);
  assert.equal(continuous([0,0,0],[.01,0,0],'running','reset'),false);
  assert.equal(continuous([0,0,0],[.01,0,0],'running','running'),true);
});
// Test real exported artifacts for complete tool coverage and replay integrity.
test('curated recordings cover six tools and have usable named joints',()=>{
  assert.equal(new Set(manifest.examples.map(e=>e.tool)).size,6);
  assert.equal(manifest.examples.length,8);
  const names=new Set(robot.joints.filter(j=>j.type!=='fixed').map(j=>j.name));
  for(const e of manifest.examples){
    const r=JSON.parse(fs.readFileSync(new URL('../public/'+e.data,import.meta.url)));
    assert.equal(r.times.length,e.frames);
    assert.ok(r.times.every((t,i)=>i===0||t>r.times[i-1]));
    assert.ok(r.jointNames.every(n=>names.has(n)));
    assert.equal(new Set(r.jointNames).size,29);
    for(let i=0;i<r.times.length;i++){
      assert.equal(r.joints[i].length,29);
      assert.ok(r.joints[i].every(Number.isFinite));
      for(const p of [r.tool[i],r.goal[i]]){
        assert.equal(p.length,7);assert.ok(p.every(Number.isFinite));
        assert.ok(Math.abs(Math.hypot(...p.slice(3))-1)<1e-4);
      }
    }
    assert.equal(e.completion,100);
    assert.ok(e.provenance.source.startsWith('experiments/results/'));
  }
});
// Check the deployed binary contract and source-scale bounds, not only loading code.
test('packed meshes retain finite geometry and valid vertex indices',()=>{
  const catalog=JSON.parse(fs.readFileSync(new URL('../public/assets/meshes.json',import.meta.url)));
  for(const info of Object.values(catalog)){
    const bytes=gunzipSync(fs.readFileSync(new URL('../public/'+info.file,import.meta.url)));
    assert.equal(bytes.length,info.vertices*24+info.indices*4);
    for(let i=0;i<info.vertices*24;i+=4)assert.ok(Number.isFinite(bytes.readFloatLE(i)));
    for(let i=info.vertices*24;i<bytes.length;i+=4)assert.ok(bytes.readUInt32LE(i)<info.vertices);
  }
});
