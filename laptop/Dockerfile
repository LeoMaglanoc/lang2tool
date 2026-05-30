FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0+cpu

RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    matplotlib \
    plotly \
    dash \
    openai \
    python-dotenv \
    tyro \
    termcolor \
    imageio \
    omegaconf \
    hydra-core \
    jinja2 \
    trimesh==3.23.5 \
    urdfpy==0.0.22 \
    viser

CMD ["/bin/bash"]
