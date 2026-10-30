# Hugging Face Docker Space: the full stack (Chroma + FastEmbed) on port 7860.
#
# Build context = the repository root, because the COPY paths below are repo-relative.
# On a Space, this file is copied to the Space repo's root as `Dockerfile` (see README.md).
#   Local test:  docker build -f deploy/huggingface/Dockerfile -t acme-genai-hf .
#                docker run -p 7860:7860 --env-file .env acme-genai-hf
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    RETRIEVER=vector \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /app

# Hugging Face requires the app to run as a non-root user with UID 1000.
RUN useradd --create-home --uid 1000 user && chown 1000:1000 /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so cold starts don't download it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && chown -R 1000:1000 /opt/fastembed

COPY --chown=1000:1000 common/ common/
COPY --chown=1000:1000 data/ data/
COPY --chown=1000:1000 projects/02_rag_basics/ projects/02_rag_basics/
COPY --chown=1000:1000 projects/03_langgraph_agent/ projects/03_langgraph_agent/
COPY --chown=1000:1000 showcase/ showcase/
COPY --chown=1000:1000 docs/SHOWCASE_API.md docs/SHOWCASE_API.md

USER 1000
EXPOSE 7860
CMD ["uvicorn", "showcase.server:app", "--host", "0.0.0.0", "--port", "7860"]
