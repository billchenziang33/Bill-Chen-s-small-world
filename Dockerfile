FROM node:22-bookworm

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip python3-venv libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci

COPY backened/requirements.txt ./backened/requirements.txt
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/python -m pip install --upgrade pip \
  && /opt/venv/bin/python -m pip install -r ./backened/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
