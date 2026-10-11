# Chrome validation

Date: 2026-10-11. Browser: the user's Chrome, controlled through the Chrome
computer-agent surface. Local target: `http://localhost:4173/lang2tool/`, served
from the production build by Docker. Deployment checks will be appended after
GitHub Pages finishes.

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

Pending GitHub Pages deployment and live Chrome checks.
