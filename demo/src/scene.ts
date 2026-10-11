import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { Graph, Joint, Manifest, MeshInfo, Pose, Recording } from './types';
import { frameAt, continuous } from './playback.mjs';
import { origin, connectJoint, jointPose } from './kinematics.mjs';
const BASE = import.meta.env.BASE_URL;
// Resolve all published resources beneath the GitHub Pages project path.
export async function json<T>(path: string): Promise<T> {
  const response = await fetch(BASE + path);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
  try {return await response.json();}
  catch {throw new Error('The demo data could not load. Please try again.');}
}
export class DemoScene {
  scene = new T.Scene();
  camera = new T.PerspectiveCamera(36, 1, .01, 30);
  renderer: T.WebGLRenderer;
  controls: OrbitControls;
  robot: T.Group | null = null;
  tool: T.Group | null = null;
  toolAsset: string | null = null;
  goal: T.Group | null = null;
  referenceLine: T.Line | null = null;
  executionLine: T.Line | null = null;
  jointNodes = new Map<string, {node: T.Group; joint: Joint}>();
  private meshes: Record<string, MeshInfo> = {};
  private geometries = new Map<string, Promise<T.BufferGeometry>>();
  private target = new T.Mesh(new T.TorusGeometry(.017, .002, 8, 48), new T.MeshBasicMaterial({color: 0xa9814e}));
  private q1 = new T.Quaternion();
  private q2 = new T.Quaternion();
  private recording: Recording | null = null;
  private robotLoad: Promise<void> | null = null;
  mode: 'motion' | 'policy' = 'motion';
  traceVisible = true;
  goalVisible = true;
  // Set a Z-up research scene with a calm studio background and responsive canvas.
  constructor(private container: HTMLElement, manifest: Manifest) {
    this.renderer = new T.WebGLRenderer({antialias: true, alpha: false});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setClearColor(0xf0f2ee);
    this.renderer.outputColorSpace = T.SRGBColorSpace;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.renderer.domElement.setAttribute('aria-label', 'Interactive 3D tool motion. Drag to orbit; scroll to zoom.');
    this.renderer.domElement.setAttribute('role', 'img');
    container.prepend(this.renderer.domElement);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = .25; this.controls.maxDistance = 4;
    this.controls.maxPolarAngle = Math.PI * .48;
    this.scene.add(new T.HemisphereLight(0xffffff, 0x8e9690, 2.5));
    const key = new T.DirectionalLight(0xffffff, 3.5);key.position.set(-1, -2, 3);this.scene.add(key);
    const fill = new T.DirectionalLight(0xffffff, 1.5);fill.position.set(2, 1, 1.5);this.scene.add(fill);
    const table = new T.Mesh(new T.BoxGeometry(.475, .4, .3), new T.MeshStandardMaterial({color: 0xc6c5b7, roughness: .85}));
    table.position.set(0, 0, manifest.tableTop - .15);this.scene.add(table);
    const ground = new T.Mesh(new T.PlaneGeometry(12, 12), new T.MeshStandardMaterial({color: 0xeff0eb, roughness: 1}));
    ground.position.z = -.015;this.scene.add(ground);
    const grid = new T.GridHelper(4, 40, 0xd9ddd6, 0xe3e6df);grid.rotation.x = Math.PI / 2;grid.position.z = -.01;this.scene.add(grid);
    this.target.position.z = manifest.tableTop + .003;this.scene.add(this.target);
    new ResizeObserver(() => this.resize()).observe(container);
    this.resetCamera();this.resize();
  }
  // Fetch the geometry catalog once; individual meshes are cached on first use.
  async init() { this.meshes = await json<Record<string, MeshInfo>>('assets/meshes.json'); }
  // Decode gzip-compressed mesh buffers without a third-party runtime dependency.
  private geometry(id: string): Promise<T.BufferGeometry> {
    if (!this.geometries.has(id)) this.geometries.set(id, (async () => {
      const info = this.meshes[id];if (!info) throw new Error('Mesh is missing from the catalog');
      const response = await fetch(BASE + info.file);
      if (!response.ok || !response.body) throw new Error('Could not load a 3D mesh');
      const bytes = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
      if (bytes.byteLength !== info.vertices * 24 + info.indices * 4) throw new Error('Invalid mesh buffer');
      const geo = new T.BufferGeometry();
      geo.setAttribute('position', new T.BufferAttribute(new Float32Array(bytes, 0, info.vertices * 3), 3));
      geo.setAttribute('normal', new T.BufferAttribute(new Float32Array(bytes, info.vertices * 12, info.vertices * 3), 3));
      geo.setIndex(new T.BufferAttribute(new Uint32Array(bytes, info.vertices * 24, info.indices), 1));
      geo.computeBoundingSphere();return geo;
    })().catch(error => {this.geometries.delete(id);throw error;}));
    return this.geometries.get(id)!;
  }
  // Reconstruct named articulated links and preserve every URDF origin and mesh scale.
  private async graph(path: string, style: 'tool'|'goal'|'robot'): Promise<T.Group> {
    const graph = await json<Graph>(path);const root = new T.Group();const links = new Map<string, T.Group>();
    await Promise.all(graph.links.map(async link => {
      const group = new T.Group();group.name = link.name;links.set(link.name, group);
      for (const visual of link.visuals) {
        let geo: T.BufferGeometry;
        if (visual.mesh) geo = await this.geometry(visual.mesh);
        else if (visual.box) geo = new T.BoxGeometry(...visual.box as [number,number,number]);
        else if (visual.cylinder) {geo = new T.CylinderGeometry(visual.cylinder[0], visual.cylinder[0], visual.cylinder[1], 24);geo.rotateX(Math.PI/2);}
        else geo = new T.SphereGeometry(visual.sphere ?? .01, 16, 16);
        let color = new T.Color(0x24463e);
        if (style === 'robot') color = link.name.startsWith('iiwa') ? new T.Color(0xd5a253) : new T.Color(0xe5e7e2);
        if (style === 'goal') color.set(0xc28642);
        const mat = new T.MeshStandardMaterial({color, roughness: .6, metalness: style === 'robot' ? .15 : .25, transparent: style === 'goal', opacity: style === 'goal' ? .24 : 1, depthWrite: style !== 'goal'});
        const mesh = new T.Mesh(geo, mat);origin(mesh, visual.xyz, visual.rpy);if (visual.scale) mesh.scale.fromArray(visual.scale);group.add(mesh);
      }
    }));
    const children = new Set(graph.joints.map(j => j.child));
    graph.joints.forEach(joint => {
      const parent = links.get(joint.parent), child = links.get(joint.child);
      if (!parent || !child) throw new Error('Robot link mapping is incomplete');
      const moving = connectJoint(parent, child, joint);
      if (style === 'robot' && joint.type !== 'fixed') this.jointNodes.set(joint.name, {node: moving, joint});
    });
    links.forEach((group, name) => {if (!children.has(name)) root.add(group);});
    return root;
  }
  // Load the full robot only when recorded execution is requested.
  async loadRobot(position: number[]) {
    if (this.robot) return;
    if (!this.robotLoad) this.robotLoad = (async () => {
      this.robot = await this.graph('assets/robot.json', 'robot');this.robot.position.fromArray(position);this.robot.visible = false;this.scene.add(this.robot);
    })().catch(error => {this.robotLoad=null;throw error;});
    return this.robotLoad;
  }
  // Replace tools after both models load successfully, disposing only owned materials.
  async setTool(path: string, stillSelected: () => boolean) {
    const [tool, goal] = await Promise.all([this.graph(path, 'tool'), this.graph(path, 'goal')]);
    if (!stillSelected()) {
      this.remove(tool);this.remove(goal);return;
    }
    this.remove(this.tool);this.remove(this.goal);this.tool = tool;this.goal = goal;this.toolAsset=path;this.scene.add(tool, goal);
  }
  // Dispose scene objects while leaving cached shared geometry intact.
  private remove(object: T.Object3D|null) {
    if (!object) return;this.scene.remove(object);
    object.traverse(child => {if (child instanceof T.Mesh) (child.material as T.Material).dispose();});
  }
  // Build reference and actual paths from the selected recording, preserving discontinuities.
  setRecording(recording: Recording, target: number[]) {
    this.recording = recording;
    for (const line of [this.referenceLine, this.executionLine]) {if (line) {this.scene.remove(line);line.geometry.dispose();(line.material as T.Material).dispose();}}
    const path = (poses: Pose[], color: number, events?: (string|null)[]) => {
      const points: T.Vector3[] = [];
      for (let i=1;i<poses.length;i++) if (continuous(poses[i-1], poses[i], events?.[i-1], events?.[i])) points.push(new T.Vector3(...poses[i-1].slice(0,3) as [number,number,number]), new T.Vector3(...poses[i].slice(0,3) as [number,number,number]));
      return new T.LineSegments(new T.BufferGeometry().setFromPoints(points), new T.LineBasicMaterial({color, transparent:true, opacity:.65}));
    };
    this.referenceLine = path(recording.reference, 0xbb884b);this.executionLine = path(recording.tool, 0x3a8270, recording.events);
    this.scene.add(this.referenceLine, this.executionLine);this.target.position.x = target[0];this.target.position.y = target[1];
  }
  // Switch camera framing between the close tool study and full articulated robot.
  resetCamera() {
    if (this.mode === 'motion') {this.controls.target.set(0, -.005, .70);this.camera.position.set(.48, -.66, 1.03);}
    else {this.controls.target.set(0, .25, .70);this.camera.position.set(1.02, -1.12, 1.25);}
    this.controls.update();
  }
  // Offer keyboard/touch-friendly zoom controls in addition to orbit gestures.
  zoom(factor: number) {this.camera.position.sub(this.controls.target).multiplyScalar(factor).add(this.controls.target);this.controls.update();}
  // Apply pose interpolation with shortest-path quaternion SLERP in the original Z-up world.
  private pose(object: T.Group, a: Pose, b: Pose, alpha: number) {
    object.position.set(a[0]+(b[0]-a[0])*alpha,a[1]+(b[1]-a[1])*alpha,a[2]+(b[2]-a[2])*alpha);
    this.q1.fromArray(a,3).normalize();this.q2.fromArray(b,3).normalize();object.quaternion.copy(this.q1).slerp(this.q2, alpha);
  }
  // Synchronize joint samples and actual poses; keep the active goal piecewise constant.
  update(time: number) {
    const r = this.recording;if (!r || !this.tool || !this.goal) return;
    const policy = this.mode === 'policy';
    if (this.robot) this.robot.visible = policy;
    this.goal.visible = policy && this.goalVisible;
    if (this.referenceLine) this.referenceLine.visible = this.traceVisible;
    if (this.executionLine) this.executionLine.visible = this.traceVisible && policy;
    this.target.visible = true;
    if (policy) {
      const {index:i, next:n, alpha} = frameAt(r.times, time);
      const a = continuous(r.tool[i],r.tool[n],r.events[i],r.events[n]) ? alpha : 0;
      this.pose(this.tool,r.tool[i],r.tool[n],a);this.pose(this.goal,r.goal[i],r.goal[i],0);
      r.jointNames.forEach((name,k) => {
        const entry = this.jointNodes.get(name);if (!entry) throw new Error(`Missing robot joint ${name}`);
        const value = r.joints[i][k] + (r.joints[n][k]-r.joints[i][k])*a;
        jointPose(entry.node,entry.joint,value);
      });
    } else {
      const previewDuration = Math.max(3,(r.reference.length-1)*r.referenceInterval);
      const f = Math.min(time / previewDuration * (r.reference.length-1),r.reference.length-1);const i = Math.floor(f),n = Math.min(i+1,r.reference.length-1);
      this.pose(this.tool,r.reference[i],r.reference[n],continuous(r.reference[i],r.reference[n],null,null)? f-i : 0);
    }
  }
  // Fit the drawing buffer to the actual responsive viewport without stretching.
  resize() {const {clientWidth:w,clientHeight:h} = this.container;if (!w || !h) return;this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h,false);}
  // Draw one frame; controls damping continues while playback is paused.
  render() {this.controls.update();this.renderer.render(this.scene,this.camera);}
}
