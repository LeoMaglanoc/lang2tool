# Isaac Lab image based on Isaac Sim (includes CUDA 12.8 + Blackwell sm_100 support)
FROM nvcr.io/nvidia/isaac-sim:4.5.0

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Blackwell (sm_100) arch — already set in Isaac Sim base, but pin explicitly
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Install system utilities needed for build steps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Isaac Lab from source (pinned release tag).
# The install script looks for Isaac Sim Python at _isaac_sim/python.sh
# (relative to the repo root), so we create a symlink pointing to /isaac-sim.
# TERM=xterm avoids "unknown terminal type" errors from the install script.
RUN git clone --depth 1 --branch v2.1.0 \
    https://github.com/isaac-sim/IsaacLab.git /opt/isaaclab \
    && ln -sf /isaac-sim /opt/isaaclab/_isaac_sim \
    && cd /opt/isaaclab \
    && TERM=xterm ./isaaclab.sh --install

# Add Isaac Sim's Python to PATH so that `python3` and `pip` resolve correctly
# for all subsequent RUN commands and at runtime.
ENV ISACSIM_PYTHON=/opt/isaaclab/_isaac_sim/kit/python/bin
ENV PATH="${ISACSIM_PYTHON}:${PATH}"

# ── Dev tooling (quality gates + test runner) ────────────────────────────────
# yourdfpy: URDF parser used by legacy observation_action_utils_sharpa.py
RUN python3 -m pip install --no-cache-dir \
    pytest pytest-cov coverage ruff black yourdfpy tyro viser "scipy==1.10.1" \
    dash plotly pandas \
    playwright pytest-playwright openai

# Install torchvision matching torch 2.8.0+cu128 so isaaclab_tasks extension imports cleanly.
RUN python3 -m pip install --no-cache-dir "torchvision==0.23.0"

# Install Chromium for Playwright browser e2e tests.
RUN python3 -m playwright install --with-deps chromium

# ── Heavy dependency pre-installation ────────────────────────────────────────
# MUST come BEFORE any COPY of repo source so that this layer stays cached
# when code files change.  Rebuilds after a code edit take <5 s instead of
# >2 min because "pip install --no-deps -e ." (the final step) skips all
# dependency resolution and downloads.
#
# List mirrors setup.py INSTALL_REQUIRES.  torch is the heavy one (~888 MB);
# it is pinned here to the version Isaac Lab v2.1.0 installs so that the
# "already satisfied" path is taken on subsequent rebuilds.
WORKDIR /workspace
RUN python3 -m pip install --no-cache-dir \
    "gym==0.23.1" \
    omegaconf \
    termcolor \
    jinja2 \
    "hydra-core>=1.2" \
    pyvirtualdisplay \
    "urdfpy==0.0.22" \
    "pysdf==0.1.9" \
    "warp-lang==0.10.1" \
    "trimesh==3.23.5" \
    yourdfpy \
    tyro \
    viser \
    dash \
    plotly \
    pandas \
    torch \
    flatdict \
    gymnasium \
    prettytable \
    einops \
    "numpy<2"

# ── rl_games fork ─────────────────────────────────────────────────────────────
# Separate COPY so this layer is only invalidated when rl_games/ itself changes.
COPY rl_games/ /workspace/rl_games/
RUN python3 -m pip install --no-cache-dir -e /workspace/rl_games

# toml is required by isaaclab core __init__.py (not part of isaaclab's own
# setup.py but needed when importing via PYTHONPATH rather than python.sh).
# Placed after the rl_games step so the torch download layer stays cached.
RUN python3 -m pip install --no-cache-dir toml

# Re-pin numpy<2 after rl_games install: rl_games pulls in opencv-python
# which requires numpy>=2, so the rl_games layer upgrades numpy to 2.x.
# Isaac Sim 4.5.0 C extensions require the numpy 1.x ABI (dtype size 96);
# with numpy 2.x they fail at import with "numpy.dtype size changed" errors.
# opencv-python warnings about numpy incompatibility can be safely ignored.
RUN python3 -m pip install --no-cache-dir "numpy<2"

# Re-pin scipy to the Isaac Sim-compatible version to avoid mixed scipy module
# imports between Kit and omni pip_prebundle paths.
RUN python3 -m pip install --no-cache-dir "scipy==1.10.1"

# ── Main source + editable install ────────────────────────────────────────────
# All deps are already installed above; --no-deps skips re-resolution (fast).
# This layer is invalidated on any source change, but the install itself is
# nearly instant because nothing needs to be downloaded.
COPY . /workspace
RUN python3 -m pip install --no-cache-dir --no-deps -e .

# Override Isaac Sim's runheadless.sh entrypoint so we can run arbitrary
# commands (pytest, train_lab.py, etc.) without loading the full Omniverse kit.
ENTRYPOINT []
CMD ["/bin/bash"]
