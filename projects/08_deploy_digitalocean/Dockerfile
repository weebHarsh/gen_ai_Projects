# Build from the repo root:  docker build -f projects/08_deploy_digitalocean/Dockerfile -t acme-genai .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 \
    FASTEMBED_CACHE_PATH=/opt/fastembed
WORKDIR /app

COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Bake the embedding model into the image so cold starts don't download 130 MB.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY common/ common/
COPY data/ data/
COPY projects/02_rag_basics/ projects/02_rag_basics/
COPY projects/03_langgraph_agent/ projects/03_langgraph_agent/
COPY projects/08_deploy_digitalocean/app.py projects/08_deploy_digitalocean/app.py

RUN useradd --create-home appuser && chown -R appuser /app /opt/fastembed
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["python", "projects/08_deploy_digitalocean/app.py"]
