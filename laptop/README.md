# Laptop Offline Viewers

This folder contains CPU-only viewers for laptop demos that do not require Isaac Lab or an NVIDIA GPU.

Included tools:

- `goal_sources_llm_offline.py`: offline LLM trajectory viewer for predefined and Lie tool motions with hammer/screwdriver switching
- `eval_replay_offline.py`: cached replay viewer for the combined predefined hammer+screwdriver artifact
- `export_eval_replay_artifact.py`: exporter for the combined predefined replay artifact
- `eval_artifacts/`: checked-in replay artifacts for laptop demos

The replay viewer now reuses the shared preloaded tool playback stack and switches between the
cached predefined hammer swing and cached predefined screwdriver twist from one source selector.
It also replays the cached robot joint state, visualizes the cached goal pose with the matching
tool mesh, defaults to `1.0x` playback speed, and exposes a wider speed range for faster
inspection. Its default port is `8081`, so it can run next to the offline LLM viewer on `8080`.

## Build the CPU-only image

```bash
docker build -f laptop/Dockerfile -t simtoolreal-laptop .
```

The image pins `torch==2.8.0` from the PyTorch CPU wheel index, so no CUDA runtime is installed.
It also pins `numpy<2` for viewer compatibility and intentionally omits `spatialmath-python` so
the shared Lie compiler uses the same fallback path as the Isaac runtime.
The image also includes `python-dotenv`, so the `openai` backend can load `OPENAI_API_KEY` from `.env`.

## Run the offline LLM viewer


```bash
docker run --rm -it -p 8080:8080 --env-file .env -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 laptop/goal_sources_llm_offline.py \
  --object-name claw_hammer \
  --llm-backend openai
```

The offline LLM viewer preloads `claw_hammer / swing_down` and `long_screwdriver / spin_vertical`.
You can switch tools from the object dropdown or by chat messages such as `switch to the screwdriver`
and `please do predefined motion`.

## Run the cached replay viewer

```bash
docker run --rm -it -p 8081:8081 -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 laptop/eval_replay_offline.py \
  --artifact laptop/eval_artifacts/goal_sources_policy_claw_hammer_swing_down_replay.json
```

## Export the unified goal-source replay artifact

```bash
docker run --rm -it -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 laptop/export_eval_replay_artifact.py \
  --source_kind policy_goal_sources \
  --output laptop/eval_artifacts/goal_sources_policy_claw_hammer_swing_down_replay.json \
  --object_name claw_hammer \
  --task_name swing_down \
  --llm_backend mock \
  --policy_replay_duration_sec 60
```

## Run the experiments dashboard website

```bash
docker run --rm -it -p 8080:8080 -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.results_dashboard \
  --results_dir experiments/results/thesis_main_6obj_v2 \
  --host 0.0.0.0 \
  --port 8080
```

Then open `http://localhost:8080`.

## Notes

- `goal_sources_policy_claw_hammer_swing_down_replay.json` is the preferred laptop replay artifact.
- The replay exporter defaults to `--policy_replay_duration_sec 60`, and the exporter now also
  enforces that cap directly in the replay-capture loop, so no cached source can run longer than
  one minute of sim time apart from one control-step rounding tolerance.
- It contains exactly two replay sources:
  - `predefined_swing` for `claw_hammer / swing_down`
  - `predefined_twist` for `long_screwdriver / spin_vertical`
- Each source stores:
  - its own object/task identity
  - the reference predefined tool motion
  - the capped cached policy rollout with tool pose, goal pose, and robot joints
- The replay viewer now plays back only those two predefined motions and switches the visible tool
  mesh automatically when you switch sources.
- Both laptop viewers now include `Zoom In` / `Zoom Out` buttons for mouse-free camera control.
- The legacy single-source predefined and policy artifacts are still kept for compatibility tests.
- The laptop image now includes the `openai` Python SDK so both `mock` and `openai` backends import cleanly.
- The laptop image also includes `python-dotenv` so the `openai` backend can load `.env`.
- The laptop image pins CPU-only `torch`, which is still required for the shared offline Lie-trajectory compiler.
- For the `openai` backend, place `OPENAI_API_KEY=...` in the repo-root `.env` file and use `--env-file .env`.
- For Docker smoke tests, both laptop entrypoints accept `--startup-only` to initialize and exit cleanly.
