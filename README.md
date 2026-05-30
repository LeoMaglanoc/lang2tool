# Lang2Tool (M.Sc. Thesis Repository)

Thesis-focused extension of SimToolReal for language-conditioned dexterous tool manipulation.

## What This Repo Contributes

- Tool-calling LLM interface that maps natural language to schema-constrained tool intents.
- Deterministic SE(3) trajectory compilation for hammer swinging and screwdriver twisting.
- Benchmark suite for language grounding, geometric validity, and execution success.
- Isaac Lab migration path (`simtoolreal_lab/`) for training and evaluation workflows.
- Reproducible thesis artifacts and manuscript workspace (`experiments/`, `latex/`).

## System Pipeline

1. User provides natural-language instruction.
2. LLM produces a constrained tool call with typed arguments.
3. Runtime validates the call and resolves target semantics.
4. Geometric compiler generates SE(3) trajectory goals.
5. Policy executes goals in simulation.
6. Evaluation pipeline records language/geometry/execution metrics and dashboards.

## Repository Map (Thesis-Relevant)

- `dextoolbench/`: interactive evaluation and benchmark entrypoints (including `llm_lie_trajectory` workflows).
- `llm_runtime/`: schema-constrained LLM tool-calling runtime, validation, and backend adapters.
- `experiments/`: thesis benchmark runners and result processing.
- `geometric_tool_planning/`: geometric motion planning and verification utilities.

## Quickstart

### 1) Build runtime

```bash
docker compose build isaaclab
```

### 2) Download pretrained checkpoint

```bash
docker compose run --rm isaaclab python download_pretrained_policy.py
```

### 3) Run interactive LLM eval with robot

```bash
docker compose run --rm --service-ports isaaclab python3 dextoolbench/eval_llm.py \
  --object_name claw_hammer \
  --task_name swing_down \
  --config_path pretrained_policy/config.yaml \
  --checkpoint_path pretrained_policy/model.pth \
  --llm-backend openai
```

### 4) Run goal-source LLM eval (only tool without robot)

```bash
docker compose run --rm --service-ports isaaclab python3 dextoolbench/eval_goal_sources_llm.py \
  --object_name claw_hammer \
  --task_name swing_down \
  --llm-backend openai
```

### 5) Run language benchmark

```bash
docker compose --env-file .env run --rm isaaclab python3 -m experiments.language_benchmark \
  --results_dir experiments/results \
  --experiment_name thesis_main \
  --backend openai
```

### 6) Run geometry benchmark

```bash
docker compose run --rm isaaclab python3 -m experiments.geometry_benchmark \
  --results_dir experiments/results \
  --experiment_name thesis_main
```

### 7) Run execution benchmark

```bash
docker compose run --rm isaaclab python3 -m experiments.execution_benchmark \
  --experiment_dir experiments/results/thesis_main \
  --config_path pretrained_policy/config.yaml \
  --checkpoint_path pretrained_policy/model.pth
```

### 8) Open dashboard

```bash
docker compose run --rm --service-ports isaaclab python3 -m experiments.results_dashboard \
  --results_dir experiments/results/thesis_main \
  --host 0.0.0.0 \
  --port 8080
```

## Upstream and Scope

This repository started from SimToolReal and now focuses on thesis-specific extensions:
- constrained LLM-to-trajectory control,
- geometric correctness verification,
- execution benchmarking,
- and Isaac Lab migration.

Upstream project: https://github.com/tylerlum/simtoolreal
