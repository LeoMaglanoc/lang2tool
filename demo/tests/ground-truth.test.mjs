import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import * as T from 'three';
import { connectJoint, jointPose } from '../src/kinematics.mjs';
const read = path => JSON.parse(fs.readFileSync(new URL(path,import.meta.url)));
const manifest=read('../public/data/manifest.json'), graph=read('../public/assets/robot.json'), oracle=read('./ground-truth.json');
// Check the original recording, not only consistency within exported JSON.
test('every exported frame matches original poses, policy-order joints, and timestamps',()=>{
  for(const e of manifest.examples){
    const bytes=fs.readFileSync(new URL('../../'+e.provenance.source,import.meta.url));
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),e.provenance.sha256);
    const payload=JSON.parse(bytes), source=Object.values(payload.sources)[0], r=read('../public/'+e.data);
    const names=payload.runtime_snapshot.policy_interface.policy_joint_lab_names ?? payload.runtime_snapshot.policy_interface.policy_joint_names;
    assert.deepEqual(r.jointNames,names);
    const frames=source.policy_track.frames;
    const trace=read('../../experiments/results/'+e.provenance.run+'/execution/artifacts/'+e.provenance.executionTrial+'/trace.json').episodes[0];
    assert.equal(trace.length,frames.length);
    assert.equal(r.times.length,frames.length-(e.provenance.trimmedTerminalResetFrames??0));
    for(let i=0;i<r.times.length;i++){
      assert.deepEqual(frames[i].tool_pose,trace[i].object_pose);
      assert.deepEqual(frames[i].goal_pose,trace[i].goal_pose);
      assert.deepEqual(frames[i].robot_joint_positions,trace[i].robot_joint_positions);
      for(const [field,key] of [['tool','tool_pose'],['goal','goal_pose'],['joints','robot_joint_positions']])
        assert.deepEqual(r[field][i],frames[i][key].map(x=>Math.round(x*1e6)/1e6));
      assert.equal(r.times[i],Math.round((frames[i].sim_time_sec-frames[0].sim_time_sec)*1e6)/1e6);
      assert.equal(r.goalIndices[i],frames[i].goal_index);
    }
    assert.deepEqual(r.reference,source.reference_track.tool_poses.map(p=>p.map(x=>Math.round(x*1e6)/1e6)));
  }
});
// Exercise the same hierarchy/rotation helpers used by the rendered scene.
test('browser FK agrees with independently computed original URDF transforms',()=>{
  const bytes=fs.readFileSync(new URL('../../'+graph.source,import.meta.url));
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),oracle.urdfSha256);
  let maxError=0, oldFingerError=0;
  for(const c of oracle.cases){
    const e=manifest.examples.find(e=>e.id===c.example), r=read('../public/'+e.data);
    const root=new T.Group();root.position.fromArray(manifest.robotPosition);
    const links=new Map(graph.links.map(l=>[l.name,new T.Group()])), nodes=new Map();
    const children=new Set(graph.joints.map(j=>j.child));
    for(const [name,link] of links)if(!children.has(name))root.add(link);
    for(const j of graph.joints)nodes.set(j.name,connectJoint(links.get(j.parent),links.get(j.child),j));
    function apply(names){
      for(const j of graph.joints)if(j.type!=='fixed')jointPose(nodes.get(j.name),j,r.joints[c.frame][names.indexOf(j.name)]);
      root.updateMatrixWorld(true);
    }
    apply(r.jointNames);
    for(const [name,link] of links){
      const expected=c.worlds[name];
      link.matrixWorld.elements.forEach((v,i)=>{maxError=Math.max(maxError,Math.abs(v-expected[i]));assert.ok(Math.abs(v-expected[i])<2e-6,`${c.example} frame ${c.frame} ${name} element ${i}`);});
    }
    apply(e.provenance.recordedJointNames);
    for(const [name,link] of links)if(/left_.*_DP$/.test(name)){
      const p=new T.Vector3().setFromMatrixPosition(link.matrixWorld), m=c.worlds[name];
      oldFingerError=Math.max(oldFingerError,p.distanceTo(new T.Vector3(m[12],m[13],m[14])));
    }
  }
  console.log(`24 original-frame FK checks: max matrix error ${maxError}; former finger-origin error up to ${(oldFingerError*1000).toFixed(1)} mm`);
});
