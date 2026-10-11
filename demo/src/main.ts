import './style.css';
import { DemoScene, json } from './scene';
import { advance } from './playback.mjs';
import type { Manifest, Recording, Example, Tool } from './types';
// Require the application shell to exist before binding controls.
function el<T extends HTMLElement = HTMLElement>(id: string): T {const node=document.getElementById(id);if (!node) throw new Error(`Missing ${id}`);return node as T;}
const icons = [
 '<path d="M13 8h15l4 5-6 3H12L8 12l-4 2 2-5 7-1Z"/><path d="m19 15-2 20h5l1-20"/>',
 '<rect x="6" y="7" width="28" height="12" rx="3"/><path d="M18 19v17h5V19"/>',
 '<rect x="7" y="7" width="27" height="12" rx="1"/><path d="M19 19v17h4V19"/>',
 '<path d="M17 4h7l2 4-2 11h-7L15 8Z"/><path d="M20 19v16l-2 4h4l-1-4V19"/>',
 '<path d="M16 10h9l2 4-2 12h-9l-2-12Z"/><path d="M20 26v9l-2 3h4l-1-3v-9"/>',
 '<rect x="16" y="5" width="9" height="19" rx="3"/><path d="M20 24v11l-2 3h4l-1-3V24"/>'
];
const methodNames: Record<string,string> = {llm_only:'Direct language trajectory',llm_lie:'Language + geometric planner',predefined:'Predefined baseline'};
const exampleNames: Record<string,string> = {llm_only:'Language-generated motion',llm_lie:'Geometrically planned motion',predefined:'Predefined motion'};
const play=el<HTMLButtonElement>('play'), scrub=el<HTMLInputElement>('scrub'), speed=el<HTMLSelectElement>('speed');
let manifest: Manifest, scene: DemoScene, current: Example, recording: Recording;
let time=0, playing=false, duration=1, last=0, serial=0, activeTool='claw_hammer', busy=true;
let mode: 'motion'|'policy'='motion';
const dataCache = new Map<string, Recording>();
// Format precise short trajectories and longer policy recordings without hiding time.
function format(t: number) {return t < 10 ? `${t.toFixed(1)}s` : `${Math.floor(t/60)}:${(t%60).toFixed(1).padStart(4,'0')}`;}
// Keep all controls and assistive labels synchronized with playback state.
function controls() {
  play.setAttribute('aria-label',playing?'Pause':'Play');
  play.innerHTML=playing?'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z" fill="currentColor"/></svg>':'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 11 7-11 7Z" fill="currentColor"/></svg>';
  scrub.max=String(duration);scrub.value=String(time);scrub.setAttribute('aria-valuetext',`${format(time)} of ${format(duration)}`);
  el('time').textContent=format(time);el('duration').textContent=format(duration);
  play.disabled=busy;scrub.disabled=busy;el<HTMLButtonElement>('restart').disabled=busy;
}
// Show recoverable load failures without leaving a misleading active simulation.
function loading(show: boolean, error?: string) {
  busy=show;const overlay=el('loading');overlay.hidden=!show;overlay.classList.toggle('error',Boolean(error));
  el('loading-text').textContent=error??(mode==='policy'?'Loading recorded execution…':'Preparing the scene…');
  el('retry').hidden=!error;el('outcome').hidden=show || !current;controls();
}
// Render optional scientific detail while keeping the main experience concise.
function details(example: Example) {
  const source=`https://github.com/LeoMaglanoc/lang2tool/blob/main/${example.provenance.source}`;
  el('details-content').innerHTML=`<dl><dt>Goal completion</dt><dd>${example.completion.toFixed(0)}%</dd><dt>Position RMSE</dt><dd>${(example.translationRmse*100).toFixed(1)} cm</dd><dt>Orientation RMSE</dt><dd>${example.rotationRmse.toFixed(1)}°</dd><dt>Recorded duration</dt><dd>${example.duration.toFixed(1)} s</dd></dl><p>Run: ${example.provenance.run}<br>Method: ${example.method}</p><p>Instruction shown above is an authored description of this saved example.</p><a href="${source}" target="_blank" rel="noopener noreferrer">View source recording ↗</a>`;
}
// Load a selected pair atomically and discard stale selections before scene mutation.
async function selectExample(id: string, changeTool = false) {
  const token=++serial;playing=false;time=0;loading(true);
  try {
    const example=manifest.examples.find(e=>e.id===id)!;
    let data=dataCache.get(id);if (!data) {data=await json<Recording>(example.data);dataCache.set(id,data);}
    if (token!==serial) return;
    const asset=manifest.tools.find(t=>t.id===example.tool)!.asset;
    if (changeTool || scene.toolAsset!==asset) await scene.setTool(asset,()=>token===serial);
    if (token!==serial) return;
    if (mode==='policy') await scene.loadRobot(manifest.robotPosition);
    if (token!==serial) return;
    current=example;recording=data;scene.mode=mode;scene.setRecording(data,example.target);scene.update(0);
    duration=mode==='motion'? Math.max(3,(data.reference.length-1)*data.referenceInterval) : data.times.at(-1)!;
    const tool=manifest.tools.find(t=>t.id===example.tool)!;
    el('scene-name').textContent=tool.name;
    const left=example.target[0]<-.05?'left':example.target[0]>.05?'right':'center';
    el('instruction').textContent=tool.family==='hammer'?`“Swing the ${tool.name.toLowerCase()} at the ${left} target.”`:`“Twist the ${tool.name.toLowerCase()} at the ${left} target.”`;
    el('method').textContent=methodNames[example.method];el('outcome').textContent=`${example.completion.toFixed(0)}% goals reached`;
    details(example);loading(false);controls();
  } catch(error) {if(token===serial) loading(true,error instanceof Error?error.message:'The recording could not be loaded.');}
}
// Populate only the selected representative examples for this tool.
async function selectTool(tool: Tool) {
  activeTool=tool.id;
  document.querySelectorAll<HTMLButtonElement>('.tool-card').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.tool===tool.id)));
  const select=el<HTMLSelectElement>('example');const examples=manifest.examples.filter(e=>e.tool===tool.id);
  select.replaceChildren(...examples.map(e=>new Option(exampleNames[e.method],e.id)));select.disabled=examples.length<2;
  await selectExample(examples[0].id,true);scene.resetCamera();
}
// Reset time and camera on mode changes, preserving the selected tool/example.
async function setMode(next: 'motion'|'policy') {
  if (next===mode) return;mode=next;scene.mode=mode;
  el('motion-mode').setAttribute('aria-pressed',String(mode==='motion'));el('policy-mode').setAttribute('aria-pressed',String(mode==='policy'));
  el('scene-label').textContent=mode==='motion'?'TOOL MOTION':'RECORDED EXECUTION';
  el('actual-label').textContent=mode==='motion'?'Tool':'Executed';
  el('goal-legend').hidden=mode!=='policy';el('goal-option').hidden=mode!=='policy';
  el('mode-explanation').textContent=mode==='motion'?'Saved tool trajectory. Preview slowed for clarity.':'Recorded arm and hand motion. Amber shows the goal.';
  document.querySelector('.recorded-badge')!.lastChild!.textContent=mode==='motion'?' Precomputed demo':' Recorded policy';
  await selectExample(el<HTMLSelectElement>('example').value);scene.resetCamera();
}
// Initialize real assets and all visitor interactions.
async function init() {
  try {
    manifest=await json<Manifest>('data/manifest.json');scene=new DemoScene(el('viewport'),manifest);await scene.init();
    manifest.tools.forEach((tool,i)=>{
      const button=document.createElement('button');button.className='tool-card';button.dataset.tool=tool.id;button.setAttribute('aria-label',tool.name);button.setAttribute('aria-pressed',String(i===0));
      button.innerHTML=`<svg viewBox="0 0 40 42" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" aria-hidden="true">${icons[i]}</svg><span>${tool.name}</span>`;
      button.onclick=()=>{void selectTool(tool);};el('tools').append(button);
    });
    el<HTMLSelectElement>('example').onchange=event=>{void selectExample((event.target as HTMLSelectElement).value);};
    el('motion-mode').onclick=()=>{void setMode('motion');};el('policy-mode').onclick=()=>{void setMode('policy');};
    play.onclick=()=>{if(time>=duration) time=0;playing=!playing;last=performance.now();controls();};
    el('restart').onclick=()=>{time=0;playing=false;scene.update(time);controls();};
    scrub.oninput=()=>{time=Number(scrub.value);scene.update(time);controls();};
    el<HTMLInputElement>('trace').onchange=event=>{scene.traceVisible=(event.target as HTMLInputElement).checked;scene.update(time);};
    el<HTMLInputElement>('ghost').onchange=event=>{scene.goalVisible=(event.target as HTMLInputElement).checked;scene.update(time);};
    el('reset-camera').onclick=()=>scene.resetCamera();el('zoom-in').onclick=()=>scene.zoom(.85);el('zoom-out').onclick=()=>scene.zoom(1/.85);
    el('retry').onclick=()=>{void selectExample(el<HTMLSelectElement>('example').value,true);};
    document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;controls();}last=performance.now();});
    document.addEventListener('keydown',event=>{if(event.code==='Space' && event.target===document.body && !busy){event.preventDefault();play.click();}});
    el('viewport').addEventListener('webglcontextlost',event=>{event.preventDefault();playing=false;loading(true,'The 3D view lost its graphics connection. Reload to restore it.');el('retry').onclick=()=>location.reload();});
    await selectTool(manifest.tools[0]);
    // Render without tying recorded motion speed to monitor frame rate.
    function tick(now: number) {
      if (playing && !busy && !document.hidden) {time=advance(time,(now-last)/1000,Number(speed.value),duration);if(time>=duration) playing=false;scene.update(time);controls();}
      last=now;scene.render();requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  } catch(error) {loading(true,error instanceof Error?error.message:'WebGL is unavailable. Try a browser with hardware acceleration.');el('retry').onclick=()=>location.reload();}
}
const dialog=el<HTMLDialogElement>('about');
el('about-open').onclick=()=>dialog.showModal();el('about-close').onclick=()=>dialog.close();
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom) dialog.close();}});
void init();
