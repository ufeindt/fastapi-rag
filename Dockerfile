FROM python:slim

RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app
COPY requirements.lock ./
COPY pyproject.toml ./
COPY README.md ./
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -r requirements.lock
RUN python3 -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1'); model.save('models')"

COPY src .

WORKDIR /tailwind
COPY tailwindcss .
RUN npm install
RUN npm run build

WORKDIR /app
CMD fastapi run main.py
