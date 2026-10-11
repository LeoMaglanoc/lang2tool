# Chrome validation

Date: 2026-10-11. Browser: the user's Chrome, controlled through the Chrome
computer-agent surface. Local target: `http://localhost:4173/lang2tool/`, served
from the production build by Docker. Live deployment checks are recorded below.

## Responsive and visual checks

| CSS viewport | Checked | Result |
| --- | --- | --- |
| 1440 × 900 desktop | All six tools in both modes; all eight samples; first/last frames, playback, speed, paths, camera buttons | Pass |
| 1280 × 720 desktop | Screwdriver execution, endpoint, layout, orbit/reset, explanation and Escape dismissal | Pass |
| 390 × 844 portrait | Hammer preview and long screwdriver execution; play/pause/restart, speed, goal/path toggles, camera, explanation | Pass |
| 360 × 640 portrait | Retained state after resize; screwdriver controls and expanded recording details | Pass |
| 844 × 390 landscape | Hammer preview/execution, quick tool-then-mode switch, readable side panel and normal page scrolling | Pass |
| 640 × 360 landscape | Short screwdriver preview, tool selector, sidebar wrapping and narrow canvas | Pass |

Actual `innerWidth`/`innerHeight` values were checked rather than assuming the
viewport override had applied. No horizontal page overflow was detected. Mobile
portrait stacks the viewport, playback controls, and tool panel. Landscape uses a
side panel and page scrolling; controls are reachable without a nested scroller.

The visual review led to three corrections: closer tool/robot framing, larger tool
labels, and filling the landscape stage height to remove unused blank space below
playback. Typography, restrained color, selected-tool contrast, concise instruction
text, and hidden-by-default explanation/details were inspected in screenshots.
Robot fingers and tool alignment were reviewed at completed hammer and screwdriver
poses, rather than validating only DOM labels or response codes.

## Interaction and recovery

- Verified six tool assets and all selected execution recordings load successfully.
- Verified the three claw-hammer generation variants use distinct matching data.
- Checked play/pause, restart, speed changes, scrubbing with keyboard to endpoints,
  camera zoom/reset, path visibility, and goal overlay visibility.
- Checked immediate tool/mode changes; stale async selections are discarded.
- Verified the native explanation dialog and recording-details disclosure remain
  readable on mobile, and the dialog dismisses with Escape.
- Temporarily removed the **generated local** manifest from `dist`, confirmed a
  helpful loading error and disabled playback, restored it, and used “Try again”
  to recover. No research artifact was changed by this check.
- No uncaught app errors or mesh-loading warnings appeared in Chrome diagnostics
  during successful-load checks. The original `.gz` serving issue was fixed by
  using an explicitly decoded `.mesh` asset format.

## Docker and data verification

The dependency install, complete data/mesh re-export, type check, focused tests,
production build, and preview were rerun inside Docker after the user clarified
that requirement. Initial bootstrap had used host Node/Python; ongoing work and
CI use Docker. The private checkout is not mounted by the demo's Compose file.

Four focused test groups pass: irregular timestamps/endpoints, reset boundaries,
real exported recording/joint/pose integrity, and binary mesh integrity. A separate
forward-kinematics check sampled each selected recording at 25/50/75%: nearest
hand-link origins were within 0.008–0.083 m of the recorded tool origin. This is an
alignment sanity check, not a new physical execution-success measure.

Eight compact replays total approximately 3.9 MB; the largest is approximately
1.35 MB. Twenty-five compressed meshes total approximately 5.93 MB across the
entire gallery/robot; only selected tools load initially and robot meshes load on
entry to execution mode. Fonts are local. Browser use requires no API keys or
research server.

## Verification limits

The browser capability changes CSS viewport size; it does not emulate a physical
phone, its GPU, or touch input. Interactions used Chrome's pointer/keyboard
controls. Hardware-phone performance, pinch gestures, and measured FPS are not
claimed. Reduced-motion behavior was reviewed in source (no autoplay and disabled
CSS animation under the preference), but the OS preference was not toggled.

## Live deployment

- Live URL: [Lang2Tool](https://leonardo-maglanoc.com/lang2tool/).
- Deployed source: `cff08540b0edd06fa965818614b2cbc5fe9043e9`.
- [GitHub Pages workflow run](https://github.com/LeoMaglanoc/lang2tool/actions/runs/38152870692): successful build and deployment, 2m 10s total. CI installed dependencies, ran tests, and built inside Docker.
- Pages uses GitHub Actions and the inherited account domain; HTTPS is enforced.
- Live Chrome checked all six selected execution recordings through their endpoints, including the legacy cylinder sample. Meshes and robot loaded without application errors or warnings.
- Live desktop 1440 × 900: hammer execution endpoint, robot/hand alignment, tool gallery and readable controls; pass.
- Live portrait 390 × 844: long screwdriver execution endpoint, restart/play/pause and mode change, stacked scene and reachable controls; pass.
- Live landscape 844 × 390: predefined hammer trajectory endpoint, method selection and side panel; pass.
- All three live dimensions were read back from the page and had no horizontal overflow. Normal vertical scrolling is intentional on short/portrait viewports.
- The Source link opened the correct public GitHub repository in a new Chrome tab.

Saved live full-page evidence: [desktop](evidence/desktop.jpg), [portrait](evidence/portrait.jpg), [landscape](evidence/landscape.jpg). These are CSS viewport checks with pointer/keyboard input, subject to the verification limits above.


## Pose bug audit and correction (2026-10-11)

The initial joint-integrity and hand-distance checks did not establish the correct
semantic joint order. The demo trusted the replay's `metadata.robot_joint_names`.
The original `dextoolbench/eval.py::_capture_execution_trace_step` instead decodes
joint values from policy observations, while `_robot_joint_names` returns Lab
articulation names. `simtoolreal_lab/tasks/simtoolreal/env_lab.py::_get_observations`
explicitly gathers joints through `policy_joint_indices`. Thus 19 finger labels
were wrong in every selected replay, although arm labels were unchanged.

The exporter now uses the runtime snapshot's `policy_joint_lab_names` (or
`policy_joint_names`) for those values and retains the original labels in
provenance. Original research data and both research implementations are unchanged.
A newly loaded robot starts hidden until the current selection commits its pose,
preventing a late load from appearing in tool-only mode.

Docker verification:

- All 9,335 exported execution frames match the original raw `trace.json` and
  replay tool/goal poses, joint values, goal indices and simulation timestamps.
  Reference trajectories also match. Maximum rounding error is below 0.0000005
  per scalar; no trajectory or tool-pose correction was necessary.
- An independent Python XML/matrix oracle reads the original URDF and computes
  all link transforms at first/middle/last frames for all eight examples.
  The browser's actual hierarchy and joint-rotation helpers agree on 24 poses:
  maximum matrix-element difference is 0.000001572, including rounding.
- Applying the former labels displaced a distal finger-link origin by up to
  124.6 mm relative to the corrected original-URDF calculation at these samples.
- Six Node test groups, TypeScript checks and production build pass.

Chrome visual checks on the corrected local production build inspected all six
execution endpoints, screwdriver start and midpoint, hammer/screwdriver grasp
alignment, scrubbing, rapid execution-to-motion changes and tool selection.
Portrait 390 × 844 and landscape 844 × 390 were reviewed again. Goal ghosts were
hidden for grasp inspection. The late-load race safeguard was also reviewed in
source; timing under an artificially throttled network was not tested.

Evidence: [hammer](evidence/hammer-corrected.jpg),
[screwdriver](evidence/screwdriver-corrected.jpg),
[portrait](evidence/pose-portrait.jpg), [landscape](evidence/pose-landscape.jpg).
The earlier screenshots above show the initial deployment before this correction.
Live deployment verification of the correction is pending.
