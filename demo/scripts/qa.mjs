import fs from 'node:fs';
import * as T from 'three';
const m=JSON.parse(fs.readFileSync('public/data/manifest.json'));
const g=JSON.parse(fs.readFileSync('public/assets/robot.json'));
// Check actual hand placement against recorded tools at several timestamps.
for(const e of m.examples){
 const r=JSON.parse(fs.readFileSync('public/'+e.data)); const results=[];
 for(const frac of [.25,.5,.75]){
  const i=Math.floor((r.times.length-1)*frac);const worlds=new Map();const values=new Map(r.jointNames.map((n,k)=>[n,r.joints[i][k]]));
  const children=new Set(g.joints.map(j=>j.child));for(const l of g.links)if(!children.has(l.name))worlds.set(l.name,new T.Matrix4().makeTranslation(...m.robotPosition));
  for(let iter=0;iter<g.joints.length;iter++)for(const j of g.joints){
   if(!worlds.has(j.parent)||worlds.has(j.child))continue;
   const q=new T.Quaternion().setFromEuler(new T.Euler(...j.rpy,'ZYX'));
   const fixed=new T.Matrix4().compose(new T.Vector3(...j.xyz),q,new T.Vector3(1,1,1));
   const motion=new T.Matrix4().makeRotationAxis(new T.Vector3(...j.axis).normalize(),values.get(j.name)??0);
   worlds.set(j.child,worlds.get(j.parent).clone().multiply(fixed).multiply(motion));
  }
  const tool=new T.Vector3(...r.tool[i].slice(0,3));let distance=Infinity;
  for(const [name,world] of worlds)if(name.startsWith('left_'))distance=Math.min(distance,new T.Vector3().setFromMatrixPosition(world).distanceTo(tool));
  results.push(distance.toFixed(3));
 }
 console.log(e.id,'hand/tool origin distance at 25/50/75% (m)',results.join(' '));
}
