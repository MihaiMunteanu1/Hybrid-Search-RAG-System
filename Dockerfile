# The web app: the API, the embedding model, the reranker and the built interface.
# The language model runs in its own container (see docker-compose.yml).

# --- the interface ------------------------------------------------------------
FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- the app ------------------------------------------------------------------
FROM python:3.13-slim
WORKDIR /app

# The CPU build of torch, installed first: the default one from PyPI brings ~2.5 GB of
# CUDA libraries that this image never uses.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The embedding model and the reranker are downloaded into the image, so the container
# starts without network access.
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
import transformers; \
transformers.AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base'); \
SentenceTransformer('intfloat/multilingual-e5-base', device='cpu'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cpu', max_length=512)"

COPY rag/ ./rag/
COPY --from=frontend /frontend/dist ./frontend/dist

ENV HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_DATA=/data \
    RAG_LLM_URL=http://llm:8080 \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000

EXPOSE 8000
CMD ["python", "-m", "rag.api"]
