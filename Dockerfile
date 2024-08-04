FROM python:slim

WORKDIR /app
COPY requirements.lock ./
COPY pyproject.toml ./
COPY README.md ./
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -r requirements.lock
RUN python3 -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1'); model.save('models')"

COPY src .
CMD fastapi run main.py
