"""Independent stdlib URDF FK oracle; read original files, write demo fixtures only."""
import json, math, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / 'demo'
def read(p): return json.loads(p.read_text())
def identity(): return [[float(i == j) for j in range(4)] for i in range(4)]
def multiply(a,b): return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def rotation(axis, angle):
 n=math.sqrt(sum(x*x for x in axis)); x,y,z=[a/n for a in axis]; c=math.cos(angle);s=math.sin(angle);v=1-c
 m=identity();m[:3]=[[c+x*x*v,x*y*v-z*s,x*z*v+y*s,0],[y*x*v+z*s,c+y*y*v,y*z*v-x*s,0],[z*x*v-y*s,z*y*v+x*s,c+z*z*v,0]];return m
def vec(e, attr, default): return [float(v) for v in (e.get(attr,default) if e is not None else default).split()]
urdf=ROOT/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf'
xml=ET.parse(urdf).getroot();joints=[]
for j in xml.findall('joint'):
 o=j.find('origin');r,p,y=vec(o,'rpy','0 0 0');fixed=multiply(multiply(rotation([0,0,1],y),rotation([0,1,0],p)),rotation([1,0,0],r))
 for k,v in enumerate(vec(o,'xyz','0 0 0')): fixed[k][3]=v
 joints.append((j.get('name'),j.find('parent').get('link'),j.find('child').get('link'),fixed,vec(j.find('axis'),'xyz','1 0 0'),j.get('type')))
roots={l.get('name') for l in xml.findall('link')}-{j[2] for j in joints}
def fk(values):
 worlds={n:identity() for n in roots}
 for m in worlds.values(): m[1][3]=.8
 while len(worlds)<len(xml.findall('link')):
  before=len(worlds)
  for name,parent,child,fixed,axis,kind in joints:
   if parent in worlds and child not in worlds:
    motion=identity() if kind=='fixed' else rotation(axis,values[name])
    worlds[child]=multiply(multiply(worlds[parent],fixed),motion)
  assert len(worlds)>before
 return {n:[m[i][j] for j in range(4) for i in range(4)] for n,m in worlds.items()}
m=read(DEMO/'public/data/manifest.json');cases=[]
for e in m['examples']:
 payload=read(ROOT/e['provenance']['source']);source=next(iter(payload['sources'].values()));frames=source['policy_track']['frames'];interface=payload['runtime_snapshot']['policy_interface'];names=interface.get('policy_joint_lab_names') or interface['policy_joint_names']
 exported=read(DEMO/'public'/e['data']);max_error=0
 for i,t in enumerate(exported['times']):
  f=frames[i]
  for target,key in [('tool','tool_pose'),('goal','goal_pose'),('joints','robot_joint_positions')]:
   max_error=max(max_error,max(abs(a-b) for a,b in zip(exported[target][i],f[key])))
 assert max_error<=.000000500001
 for i in sorted({0,(len(exported['times'])-1)//2,len(exported['times'])-1}):
  f=frames[i];cases.append({'example':e['id'],'frame':i,'time':exported['times'][i],'tool':f['tool_pose'],'goal':f['goal_pose'],'jointNames':names,'joints':f['robot_joint_positions'],'worlds':fk(dict(zip(names,f['robot_joint_positions'])))})
 print(e['id'], 'source rounding max error', max_error, 'corrected joint labels',sum(a!=b for a,b in zip(names,source['metadata']['robot_joint_names'])))
fixture={'oracle':'Original URDF parsed with Python XML; Rz Ry Rx and Rodrigues matrices, independent of Three.js/export graph. Recorded values use policy observation order verified in eval.py and env_lab.py.','urdfSha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),'cases':cases}
(DEMO/'tests/ground-truth.json').write_text(json.dumps(fixture,separators=(',',':'))+'\n')
