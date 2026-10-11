# Lang2Tool website demo plan

Date: 2026-10-11
Classification: **Feature** (browser demo and deployment), preceded by this planning document.
Status: **Complete. Demo deployed from `cff0854`; local and live Chrome validation passed.**

## Goal and boundaries

Build a deployable, interactive thesis showcase inside the public
[`LeoMaglanoc/lang2tool`](https://github.com/LeoMaglanoc/lang2tool) repository,
hosted on GitHub Pages. Visitors should understand the progression from language
instructions to geometric tool trajectories to recorded dexterous execution.

- All new code, exports, documentation, commits, and pushes belong to `lang2tool`.
- `../lang2tool_private` is a read-only reference. Do not edit it, run exporters
  that write into it, stage its changes, or push it. It already contains local
  changes; preserve those exactly.
- Prefer assets and recordings already present in the public repository. Consult
  private code to understand behavior, without copying whole private modules or
  publishing private research data by default.
- Use this repository, rather than the separate demo repository proposed in the
  pasted conversation. Use GitHub Pages rather than Sites hosting.
- Keep the existing Python research workflows intact. Visitors need no Python,
  Isaac Lab, GPU, model checkpoint, API key, or backend service.
- Pushing and deploying the completed website are authorized. First deliver this
  plan; website implementation is the next milestone.

## Evidence from the local repositories

The public checkout is on `main`, points to the requested GitHub remote, and was
clean before this document was added. No public `AGENTS.md`, existing frontend
package, or `.github` directory was found during the initial inventory.

| Existing source | Use in the demo |
| --- | --- |
| `laptop/artifacts.py` | Recorded pose/frame schema and timing behavior |
| `laptop/goal_sources_llm_offline.py` | Tool selection, instruction flow, and motion examples |
| `laptop/eval_replay_offline.py` | Source switching and recorded robot playback |
| `geometric_tool_planning/viewer.py` | Scene placement, tool semantics, robot and goal visualization |
| `geometric_tool_planning/` and `llm_runtime/` | Command contracts and deterministic planning reference |
| `assets/urdf/` | Public tool, table, KUKA arm, and ShARPA hand assets |
| `dextoolbench/trajectories/` | Existing predefined tool trajectories |
| `experiments/replay_artifacts.py` | Geometry/execution replay schema `offline_replay_v4` and manifest generation |
| `experiments/results/thesis_main_v3_v4/{geometry,execution}/replay/` | Primary demo catalog: 114 geometry trajectories and 114 matching execution recordings |

### Primary experiment replay catalog (corrected inventory)

The initial inventory inspected the laptop bundle too narrowly. The experiment
replay folders already contain a much broader catalog, including language-derived
motions and corresponding recorded policy execution. Use these as the primary
website data source rather than limiting the demo to two predefined motions.

Both public and private `thesis_main_v3_v4` manifests list the following, and all
listed artifact paths exist in both local checkouts:

| Generation method | Geometry trajectories | Matching execution recordings |
| --- | ---: | ---: |
| `predefined` | 6 | 6 |
| `llm_lie` | 54 | 54 |
| `llm_only` | 54 | 54 |
| **Total** | **114** | **114** |

A full audit of the public execution replay files found 598,294 frames, all with
29 joint values and a 29-name joint mapping, all tracked in Git, and every
execution manifest entry linked to an existing geometry trial. The manifest
reports 69 successful executions and 45 experiment timeouts. This verifies data
availability, not visual fidelity or the correctness of success definitions.

There are 19 entries per tool: `claw_hammer`, `mallet_hammer`,
`cuboid_hammer_v014`, `long_screwdriver`, `short_screwdriver`, and
`cylinder_screwdriver_v3009`. Each tool has one predefined entry and nine target
positions for each language-generation method. Match geometry and execution
using `geometry_trial_id`; expose tool, generation method, and target filters.

The geometry recordings occupy about 5.9 MB and execution replays about 1.07 GB
before compact export. A public `thesis_main_v3_v4_execution_rerun` folder and
several archived runs also contain replay catalogs. Do not count repeated trial
IDs across runs as unique examples or blend runs without provenance. Select the
primary run by comparing recorded metadata and execution results; the experiment
README has older canonical-run wording that must be reconciled with actual data.

The website will publish **representative samples, not the complete catalog**.
Use the full inventory only to choose the strongest examples. Start with one
featured trajectory/execution pair per tool (six primary examples), then add a
small number of strong examples when they help illustrate generation methods,
target variation, swinging, or twisting. Do not require every tool/method/target
combination. Lazy-load only the selected recordings and retain their provenance.

Selection procedure:

1. Check the primary and rerun metadata/results before choosing an experiment
   source. Keep every selected pair associated with its own run and trial ID.
2. Rank within each tool: successful execution first, then goal completion,
   then translation and rotation tracking error from execution raw results.
   Compare both errors explicitly rather than adding incompatible units into an
   arbitrary score. Use runtime and replay length as tie-breakers for presentation.
3. Verify geometry validity, goal transformation/cycle metadata, complete joints,
   and actual visual quality before publishing a selected pair. Inspect apparent
   disagreements between replay summaries and raw execution results.
4. Prefer clear, stable, representative movement rather than a high reported
   score with poor visible tracking. Do not call the selection globally optimal;
   describe it as the best verified representative samples under these criteria.
5. Cover all six tool meshes. Add successful `llm_lie`, `llm_only`, and predefined
   examples across the gallery where useful, without forcing weaker examples to
   complete a matrix.

The current run has successful recordings for five tools: claw hammer 18/19,
mallet hammer 19/19, cuboid hammer 15/19, long screwdriver 6/19, and short
screwdriver 11/19. Cylinder screwdriver has **0/19 successful executions** in this
run. Check other recorded runs for a genuinely successful representative; if none
is available, show its valid tool-only trajectory and, optionally, clearly labeled
best-progress execution. Do not describe an incomplete execution as successful.

### Secondary laptop bundle

The laptop-specific replay file is
`laptop/eval_artifacts/goal_sources_policy_claw_hammer_swing_down_replay.json`
(about 9.1 MB), with schema `offline_replay_v3` and 60 Hz control:

| Source | Tool/task | Reference samples | Policy frames | Recorded duration |
| --- | --- | ---: | ---: | ---: |
| `predefined_swing` | `claw_hammer / swing_down` | 38 | 3,601 | 60.0 s |
| `predefined_twist` | `long_screwdriver / spin_vertical` | 101 | 1,658 | 27.617 s |

Both recordings include 29 joint positions per frame, actual tool poses, goal
poses, and simulation timestamps. Pose order is `[x, y, z, qx, qy, qz, qw]`.
The runtime snapshot includes multiple joint-name lists; their orders differ,
so the exporter must establish which list actually labels each frame's values.
Do not map joints by array position alone.

The laptop `_backup.json` and `_long_backup.json` bundles also contain
`predefined`, `llm_lie[target_a]`, `llm_lie[target_b]`, and `llm_lie[target_c]`
sources. These provide additional candidates; they are not limited to the two
sources in the current laptop bundle.

The public and private artifact helpers and offline entrypoints are identical
in the inspected files. The private viewer adds URDF error diagnostics that
the public viewer currently suppresses. Treat missing meshes as visible failures
in the website, rather than silently showing an incomplete robot.

## Visitor experience

### Mode 1: Language → tool motion

Start with a curated set selected from the experiment catalog, covering all six
tools. Provide a simple tool/example selector and label the generation method and
recorded target for each available example. Display the selected instruction,
structured intent where available,
and animated SE(3) tool motion with a trajectory trace.

- Make example selection the primary interaction.
- Label examples as precomputed motions. Do not imply live LLM inference.
- Only attach a language instruction or tool-call provenance when supported by
  source data. Label authored descriptions as demo descriptions.
- A restricted text-command input is optional after the core experience works.
  If included, show supported phrases and reject unsupported commands clearly.
  Deterministic example matching must not be represented as an LLM.
- Use existing language-derived experiment trajectories; preserve their actual
  generation method. Recover instructions/tool calls from matching raw geometry
  data and generation context where recorded, without inventing missing text or
  matching unrelated Stage 1 language trials by wording alone.

### Mode 2: Tool motion → dexterous execution

Replay matching experiment policy recordings using the actual arm/hand joint states.
Show a solid executed tool, a translucent goal tool, and distinguishable paths.
Explain the difference between the requested trajectory and measured execution.

- Use frame-specific `goal_pose` and `goal_index` to align the active goal with
  execution. Do not stretch the reference sequence across the rollout duration.
- Use the execution manifest's explicit geometry-trial mapping. Only offer
  examples with a matching recording. Changing an instruction must
  never silently play an unrelated policy rollout.
- Preserve recorded successes, failures, and reset events. Avoid interpolating
  through discontinuities as if they were continuous physical motion.
- Display computed tracking metrics only after verifying their definitions and
  units; benchmark percentages are outside the initial demo scope.

### Shared interface and visual direction

Use a restrained research-showcase layout: clear typography, neutral background,
one accent color, and a dominant 3D viewport. Include a short pipeline explanation,
author/project context from verified public sources, and a repository link.

Provide play/pause, restart, timeline scrubbing, speed selection, trajectory and
goal visibility, orbit/zoom, and reset camera. Start paused at a useful pose.
Use accessible labels, keyboard focus, readable legends, and touch-sized controls.

Desktop: viewport with an adjacent compact control panel. Mobile portrait:
viewport above controls with normal page scrolling. Mobile landscape: compact
header and controls beside/below the viewport as space permits, without fixed
heights that hide playback controls. Account for safe-area insets and browser UI.

## Architecture and file layout

Use **Vite + TypeScript + Three.js**, with ordinary HTML/CSS for controls.
Bundle dependencies locally and commit the lockfile. Use browser rendering and
forward kinematics; physics and policy inference are unnecessary for replay.

```text
lang2tool/
  plan.md
  demo/
    package.json
    package-lock.json
    index.html
    vite.config.ts
    tsconfig.json
    src/                   # interface, scene, playback, data loading
    scripts/               # deterministic public-data/asset export
    tests/                 # focused data, transforms, playback tests
    public/
      data/                # manifest and separate recording files
      assets/              # selected optimized meshes and scene metadata
      attribution/         # source/license notes
    README.md
    validation.md          # actual browser results and evidence references
  .github/workflows/pages.yml
```

Keep `node_modules`, `dist`, and temporary export files ignored. CI builds only
`demo/`; publish only `demo/dist/`, never the repository root or backup recordings.

Convert selected OBJ/STL visuals into compact browser assets where worthwhile.
Preserve a robot link/joint hierarchy with joint axes, origins, fixed transforms,
mesh scales, and material information. Use a build-time URDF export or a verified
browser URDF loader; select the simplest option after a one-frame fidelity check.
GLB conversion must preserve articulated link transforms rather than flattening
the robot into one rigid mesh.

## Data fidelity and performance

Create a deterministic export process that reads public files and writes only
under `demo/`. Record source paths, hashes, schema version, units, timestamps,
quaternion order, world convention, robot root transform, and joint-name mapping.
Exclude irrelevant runtime internals and local paths from published files.

1. Validate finite poses, nonzero/normalized quaternions, monotonic timestamps,
   joint dimensions, and all mesh references.
2. Reproduce mesh origins/scales and scene placements from the viewer and source
   assets. Explicitly convert the source world convention for Three.js; apply
   the same conversion to robot, table, tools, goals, and paths.
3. Check the robot base placement (the existing viewer uses `(0, 0.8, 0)`) against
   recorded runtime state, rather than assuming the viewer is authoritative.
4. Verify joint values against exporter/URDF ordering and a known recorded pose.
   Use named mappings and account for any fixed or mimic joints.
5. Interpolate positions linearly and orientations with shortest-path quaternion
   SLERP; sample goals and event indices according to their recorded semantics.
6. Advance using simulation timestamps and elapsed browser time. Pausing,
   scrubbing, speed changes, and tab visibility must not introduce time jumps.
7. Split recordings by source and lazy-load the selected source. Keep full-fidelity
   data initially; reduce precision or sample density only with measured error.
8. Cap pixel ratio, avoid per-frame allocations, dispose replaced GPU resources,
   and use the lowest necessary visual mesh detail for mobile rendering.

Initial performance targets: initial required assets under 5 MB where practical,
each lazy-loaded replay under 5 MB after compact export, and smooth interactive
playback at approximately 30 FPS or better on the validation machine's mobile
viewport. Document measurements; viewport emulation does not establish physical
phone GPU performance. Provide a useful loading/error state and WebGL fallback.

## Implementation milestones

- [x] **0. Plan:** inspect public data and private reference; write this document.
- [x] **1. Select, export, and prove fidelity:** use the audited catalog to rank
  candidates, resolve the cylinder screwdriver's available outcomes, select six
  featured pairs plus useful supplementary examples, and export only selected
  meshes; validate joints/coordinates at first, middle, and last frames. Establish
  actual language-example availability and asset attribution.
- [x] **2. Build tool-motion mode:** responsive shell, real tool meshes, example
  selection, trajectory visualization, camera, and shared playback controls.
- [x] **3. Build execution mode:** articulated robot playback, recorded tool/goal
  overlay, accurate goal synchronization, event handling, and honest mode labels.
- [x] **4. Validate production build:** focused numerical tests and Chrome
  computer-agent interaction at every viewport below; fix observed problems.
- [x] **5. Deploy:** commit logical changes, push to the public repository,
  complete GitHub Pages deployment, and repeat Chrome checks on the live URL.
- [x] **6. Handoff:** provide live URL, revision, reproduction commands,
  verification evidence, and any material limitations.

## Chrome computer-agent validation (required)

Use the Chrome computer-agent surface (`cua` targeting `chrome`) to navigate,
inspect, interact, and capture screenshots. Scripted tests may supplement these
checks, but do not replace the requested Chrome computer-agent validation.
Discover supported viewport/emulation controls from its documentation before
using them. If the Chrome connection or required sizing is unavailable, report
that specific limitation and do not claim the checks passed.

Validate a **production build served under `/lang2tool/`**, then the deployed
URL, at these representative CSS viewport dimensions:

| Layout | Primary | Additional check |
| --- | --- | --- |
| Desktop | 1440 × 900 | 1280 × 720 |
| Mobile portrait | 390 × 844 | 360 × 640 |
| Mobile landscape | 844 × 390 | 640 × 360 |

At each primary viewport, exercise hammer and screwdriver examples in both modes
and all three generation methods. Cover all six tool meshes across the validation
matrix and:

- Verify meshes, articulated fingers, tool alignment, goal overlay, and legend.
- Play, pause, restart, scrub to first/middle/last frame, and change speed.
- Switch sources/modes during playback; ensure state and visible models agree.
- Orbit, zoom, reset the camera, and toggle trace/goal visibility.
- Check all controls remain reachable, text is readable, and no horizontal page
  overflow or overlapping panels occur. Inspect the entire page, not one fold.
- Check portrait-to-landscape resizing retains state and resizes the canvas.
- Use touch emulation where the Chrome surface supports it; identify mouse-only
  emulation explicitly if touch cannot be tested.
- Verify keyboard interaction and focus on desktop, and reduced-motion behavior.
- Inspect available console/network diagnostics for uncaught errors, failed mesh
  loads, incorrect base paths, and asset 404s.
- Reload the page and test the repository link. Test a controlled loading failure
  locally and confirm a helpful recovery/error state.

Save screenshots and a checklist with URL, revision, viewport, test actions,
outcomes, and any browser limitations in `demo/validation.md`. Recheck fixes at
the affected viewport; do not mark an unexecuted check as passed.

## GitHub Pages deployment

Use a GitHub Actions workflow with dependency installation, type checking,
focused tests, production build, Pages artifact upload, and deployment. Give the
deploy job `pages: write` and `id-token: write`, use the `github-pages` environment,
and make deployment depend on the successful build. Deploy `demo/dist` only.
Support pushes to the public default branch and manual dispatch, with deployment
concurrency and path filters appropriate to demo changes.

Set Vite's project base to `/lang2tool/` and resolve data/asset URLs through that
base. Avoid history routes requiring server rewrites; use a single page or hash
state. An expected default URL is `https://leomaglanoc.github.io/lang2tool/`, but
confirm the URL returned by GitHub because account-level domain settings may
affect it. No DNS change or new custom domain is needed for this milestone.

Inspect existing Pages configuration before changing it. The local `gh` CLI is
not authenticated as of planning; use connected GitHub capabilities if they
support the required operations, and separately verify authenticated Git push
and Pages settings access. Never claim a successful deployment from a push alone.
Wait for the deployment result and validate the returned live URL in Chrome.
If authentication blocks deployment, keep the complete tested build/workflow
and report the exact remaining action.

Add local setup/export/build/preview commands to `demo/README.md` and link the
demo from the root README. Keep commits limited to public demo work and avoid
force-pushing. Preserve a known good revision for rollback through a revert.

Official references consulted for the deployment design:

- [GitHub: custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Vite: GitHub Pages deployment and base paths](https://vite.dev/guide/static-deploy.html#github-pages)

## Completion criteria

The task is complete when the live GitHub Pages site loads real public assets,
both modes work with accurate recorded playback and clear provenance, all three
required Chrome layouts have documented passing results, the public repository
contains reproducible source and deployment instructions, and the private
repository remains untouched. Additional arbitrary text planning, click-to-place
targets, live LLM calls, browser policy inference, and a full benchmark dashboard
are follow-up features rather than dependencies for the initial release.

## Delivered implementation

The live demo is [https://leonardo-maglanoc.com/lang2tool/](https://leonardo-maglanoc.com/lang2tool/). Eight curated samples cover all six tools; three claw-hammer examples show the generation methods. The cylinder example comes from the public legacy `thesis_main_v3` run with its own benchmark context; its single terminal reset frame is trimmed and documented in the exported provenance. It does not imply success in the newer primary benchmark.

[Validation and screenshots](demo/validation.md) record the Docker checks, local six-viewport review, and live desktop/portrait/landscape checks. [Reproduction commands](demo/README.md) cover export, build, preview, and deployment. No files in the private repository were changed.
