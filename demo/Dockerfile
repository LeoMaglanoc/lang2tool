FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 git openssh-client ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/demo
COPY package*.json ./
RUN npm ci
CMD ["npm", "run", "dev"]
