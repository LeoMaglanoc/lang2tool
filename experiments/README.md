# Foundation PC Version

docker compose --env-file .env run --rm -p 8080:8080 isaaclab \
    python3 -m experiments.results_dashboard \
    --results_dir experiments/results/thesis_main_v3_v4 \
    --host 0.0.0.0 \
    --port 8080

docker compose --env-file .env run --rm isaaclab \
    python3 -m experiments.replay_language_offline \
    --experiment_dir experiments/results/thesis_main_v3_v4

docker compose --env-file .env run --rm -p 8084:8084 isaaclab \
    python3 -m experiments.replay_geometry_offline \
    --experiment_dir experiments/results/thesis_main_v3_v4 \
    --port 8084

docker compose --env-file .env run --rm -p 8082:8082 isaaclab \
      python3 -m experiments.replay_execution_offline \
      --experiment_dir experiments/results/thesis_main_v3_v4 \
      --port 8082

docker compose run --rm -p 8083:8083 isaaclab python3 -m experiments.swinging_vis --experiment_dir experiments/results/thesis_main_v3_v4 --port 8083

docker compose run --rm -p 8084:8084 isaaclab python3 -m experiments.twisting_vis --experiment_dir experiments/results/thesis_main_v3_v4 --port 8084

docker compose run --rm -p 8085:8085 isaaclab python3 -m experiments.system_vis --experiment_dir experiments/results/thesis_main_v3_v4 --port 8085

docker compose run --rm -p 8086:8086 isaaclab python3 -m experiments.tool_set_vis --port 8086

# Laptop Version

docker build -f laptop/Dockerfile -t simtoolreal-laptop .

docker run --rm -it --env-file .env -p 8080:8080 \
  -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.results_dashboard \
  --results_dir experiments/results/thesis_main_v3_v4 \
  --host 0.0.0.0 \
  --port 8080

docker run --rm -it --env-file .env \
  -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.replay_language_offline \
  --experiment_dir experiments/results/thesis_main_v3_v4

docker run --rm -it --env-file .env -p 8084:8084 \
  -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.replay_geometry_offline \
  --experiment_dir experiments/results/thesis_main_v3_v4 \
  --port 8084

docker run --rm -it --env-file .env -p 8082:8082 \
  -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.replay_execution_offline \
  --experiment_dir experiments/results/thesis_main_v3_v4 \
  --port 8082

docker run --rm -it --env-file .env -p 8084:8084 \
  -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.twisting_vis \
  --experiment_dir experiments/results/thesis_main_v3_v4 \
  --port 8084

docker run --rm -it --env-file .env -p 8085:8085 \
  -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.system_vis \
  --experiment_dir experiments/results/thesis_main_v3_v4 \
  --port 8085

docker run --rm -it --env-file .env -p 8086:8086 \
  -v "$PWD":/workspace -w /workspace simtoolreal-laptop \
  python3 -m experiments.tool_set_vis \
  --port 8086



EXP=thesis_main_v3_v1; \
  docker compose --env-file .env run --rm -e OPENAI_MODEL=gpt-5.5 isaaclab \
    python3 -m experiments.language_benchmark \
    --results_dir experiments/results \
    --experiment_name "$EXP" \
    --backend openai && \
  docker compose --env-file .env run --rm -e OPENAI_MODEL=gpt-5.5 isaaclab \
    python3 -m experiments.geometry_benchmark \
    --results_dir experiments/results \
    --experiment_name "$EXP" \
    --object_names_csv claw_hammer,mallet_hammer,cuboid_hammer_v014,long_screwdriver,short_screwdriver,cylinder_screwdriver_v3009 \
    --modes_csv predefined,llm_lie,llm_only \
    --llm-backend openai \
    --llm-model gpt-5.5 \
    --target_grid_x 3 \
    --target_grid_y 3 && \
  docker compose --env-file .env run --rm isaaclab \
    python3 -m experiments.execution_benchmark \
    --experiment_dir "experiments/results/$EXP" \
    --config_path pretrained_policy/config.yaml \
    --checkpoint_path pretrained_policy/model.pth \
    --overwrite

EXP=thesis_main_v3_v4; \
  docker compose --env-file .env run --rm isaaclab \
    python3 -m experiments.language_benchmark \
    --results_dir experiments/results \
    --experiment_name "$EXP" \
    --backend openai \
    --model gpt-5.5 && \
  docker compose run --rm isaaclab \
    python3 -m experiments.replay_language_offline \
    --experiment_dir "experiments/results/$EXP"

# Thesis Experiment Pipeline

This directory contains the thesis-facing benchmark pipeline for:

- language grounding
- geometry-only trajectory generation
- fixed-policy execution of frozen geometry artifacts
- static website generation over saved results

## Canonical Benchmark

The canonical Chapter 5 run is `thesis_main_6obj_v3`.

- Benchmark metadata version: `v3`
- Object slice: `claw_hammer`, `mallet_hammer`, `cuboid_hammer_v014`, `long_screwdriver`, `short_screwdriver`, `cylinder_screwdriver_v3009`
- Geometry modes: `predefined`, `llm_lie`, `llm_only`
- Geometry grid: `3 x 3` back-table rectangle, with default x coordinates
  `[-0.15, 0.0, 0.15]` and y coordinates `[-0.13, -0.08, -0.03]`
- Language backend: `openai`
- Thesis-facing execution view: pretrained policy on all three geometry modes
- Optional side evidence: `llm_lie + finetuned` remains supported by the pipeline but is not part of the default thesis website

Older result folders remain readable, but treat pre-`v3` bundles as non-canonical for thesis figures.

## Directory Layout

```text
experiments/results/<experiment_name>/
  metadata.json
  config.json
  language/
    raw/
    summaries/
  geometry/
    raw/
    exemplars/
    replay/
    summaries/
  execution/
    raw/
    replay/
    summaries/
    artifacts/
  website_assets/
```

## Canonical Full Rerun

1. Run the language benchmark:

```bash
docker compose run --rm isaaclab python3 -m experiments.language_benchmark \
  --results_dir experiments/results \
  --experiment_name thesis_main_6obj_v3 \
  --backend openai
```

2. Run the geometry benchmark on the full `3 x 3` target grid:

```bash
docker compose run --rm isaaclab python3 -m experiments.geometry_benchmark \
  --results_dir experiments/results \
  --experiment_name thesis_main_6obj_v3 \
  --object_names_csv claw_hammer,mallet_hammer,cuboid_hammer_v014,long_screwdriver,short_screwdriver,cylinder_screwdriver_v3009 \
  --modes_csv predefined,llm_lie,llm_only \
  --llm-backend openai \
  --llm-model gpt-5.5 \
  --target_grid_x 3 \
  --target_grid_y 3
```

Recompute geometry summaries from the frozen raw trajectory artifacts without making OpenAI calls
or regenerating trajectories:

```bash
docker compose run --rm isaaclab python3 -m experiments.geometry_benchmark \
  --results_dir experiments/results \
  --experiment_name thesis_main_6obj_v3 \
  --recompute-from-raw
```

3. Generate the supplemental target-`a` thesis exemplars:

```bash
docker compose run --rm isaaclab python3 -m experiments.generate_thesis_exemplars \
  --experiment_dir /home/leo/code/simtoolreal/experiments/results/thesis_main_6obj_v3 \
  --backend openai
```

4. Run execution on the frozen geometry artifacts with the pretrained checkpoint:

```bash
docker compose run --rm isaaclab python3 -m experiments.execution_benchmark \
  --experiment_dir experiments/results/thesis_main_6obj_v3 \
  --config_path pretrained_policy/config.yaml \
  --checkpoint_path pretrained_policy/model.pth \
  --overwrite
```

5. Launch the thesis-facing website:

```bash
docker compose run --rm --service-ports isaaclab python3 -m experiments.results_dashboard \
  --results_dir experiments/results/thesis_main_6obj_v3 \
  --host 0.0.0.0 \
  --port 8080
```

## Figure Set

The website is the source of truth for the Chapter 5 figure suite:

- Stage 1: prompt-family language grounding figure
- Stage 2: object-level geometry summary figure
- Stage 2: requested-vs-implied target distribution figure
- Stage 3: execution success/progress figure
- Stage 3: translation RMSE by tool family
- Stage 3 table: mode-level failure attribution

On startup the website regenerates canonical SVG assets under `website_assets/`. The current asset keys are:

- `language_prompt_family_accuracy.svg`
- `geometry_semantic_by_object.svg`
- `geometry_implied_target_topdown.svg`
- `execution_success_progress.svg`
- `execution_translation_rmse_by_family.svg`

## Notes

- `llm_only` uses the matched-input target-conditioned direct-generation contract.
- Language summaries now persist `prompt_family` and `prompt_variant`.
- Geometry summaries now persist semantic target-alignment metrics, plus screwdriver tilt and twist diagnostics.
- Execution summaries now persist mode-level failure attribution in `execution/summaries/aggregate.json`.
