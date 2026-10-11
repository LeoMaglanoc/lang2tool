# Lang2Tool static demo

A curated, browser-only Three.js thesis showcase. Eight selected geometry/policy
pairs cover six tools. Claw hammer also offers geometric-planner and predefined
baselines. The interface deliberately keeps context, metrics, and source links
behind a short explanation dialog and a recording-details disclosure.

## Docker workflow

Run these from the repository root. No host Node/Python installation is required.
Only the public checkout is mounted; `lang2tool_private` is never used at runtime.

```bash
docker compose -f demo/compose.yaml build
docker compose -f demo/compose.yaml run --rm web npm test
docker compose -f demo/compose.yaml run --rm web npm run build
docker compose -f demo/compose.yaml up -d
```

Open `http://localhost:4173/lang2tool/`. This previews the production output with
the same `/lang2tool/` base as GitHub Pages.

To develop with hot reload instead:

```bash
docker compose -f demo/compose.yaml run --rm -p 127.0.0.1:4173:4173 web npm run dev -- --port 4173
```

To regenerate the curated public assets and check numerical alignment:

```bash
docker compose -f demo/compose.yaml run --rm web npm run export
docker compose -f demo/compose.yaml run --rm web node scripts/qa.mjs
```

The export reads saved public experiment files only and writes under `demo/`.
There is no Isaac Lab, model inference, LLM API, or network dependency in export.
Mesh conversion uses the installed Three.js loaders, and the browser decodes
losslessly gzip-compressed `.mesh` files. Fonts and their notices are vendored;
the font-fetch script is an optional maintenance utility, not a build dependency.

Stop preview with:

```bash
docker compose -f demo/compose.yaml down
```

## Selection and scientific limits

Selection ranks each tool's successful recordings by goal completion, then
translation RMSE, then rotation RMSE. Seven examples come from
`thesis_main_v3_v4`; the cylinder screwdriver sample comes from the older
`old/thesis_main_v3` run because the current run had no successful cylinder
executions. Keep the older run's outcome separate: its benchmark definition is
not evidence that the current run succeeded. Scores are saved trial metrics,
not a new benchmark or a guarantee of visually exact tracking.

The exporter includes the complete recorded sampling of each selected first
trace episode. One trailing post-completion reset frame is removed from the
cylinder sample, explicitly recorded as `trimmedTerminalResetFrames` in provenance.
Internal discontinuities remain visible and are not blended into fictitious
physical movement. Other trace episodes are not exported by the source replay
format. Pose/joint values are rounded to six decimals; meshes retain source
positions, normals, origins, scales, and articulated link structure.

- Tool motion is a saved reference trajectory. The preview takes at least three
  seconds so its movement is readable; that is presentation timing.
- Robot execution uses original simulation timestamps. At 1×, it plays at recorded
  speed. Goal poses are piecewise constant at their recorded frame indices.
- Joint values use the runtime snapshot's 29 policy-observation names. The original
  recorder labels policy-order values with Lab articulation-order names; the
  exporter corrects this mismatch and retains the original labels in provenance.
  The scene stays Z-up, with the source robot base at
  `(0, 0.8, 0)` and table top at `0.53 m`.
- Instruction text is an authored description of the chosen recorded target. It
  does not claim to reproduce an original prompt verbatim or run a live LLM.
- Display colors are chosen for clarity; geometry comes from the public meshes.

Each gallery entry in `public/data/manifest.json` contains trial IDs, source path,
SHA-256, run, generation context, outcome, and saved errors. Individual recordings
are loaded on demand. The eight compact replay files total about 3.9 MB, rather
than publishing the 1.07 GB experiment replay catalog. Robot meshes load only
when execution mode is opened. Mesh sources and hashes are in
`public/assets/meshes.json`; third-party notices are in `public/attribution/`.

## Deploy

`.github/workflows/pages.yml` builds and tests with Docker and publishes only
`demo/dist`. Set repository Settings → Pages → Source to **GitHub Actions**.
Push demo changes to `main` or dispatch the workflow manually. The workflow's
Pages deployment output is the authoritative live URL; project paths use
`/lang2tool/`. No custom-domain or DNS changes are required.

CI uses the committed export, so normal deployment does not need experiment
regeneration. Dependencies are pinned by `package-lock.json`. Revert the demo
commit to roll back; do not force-push shared history.

## Validation

See [validation.md](validation.md) for real Chrome computer-agent checks, visual
review, and device-emulation limitations. The focused Node tests exercise
irregular timestamps, reset boundaries, exported recordings, quaternion norms,
named joints, and packed mesh integrity. Chrome checks cover actual interaction
and responsive appearance rather than treating a successful build as UI proof.

Ground-truth regression checks compare every exported frame with the original raw trace and replay. Regenerate the independent original-URDF FK fixture with `docker compose -f demo/compose.yaml run --rm web python3 scripts/ground-truth.py`, then run `npm test` through Compose as above. The fixture uses Python XML and matrix arithmetic independently of Three.js.
