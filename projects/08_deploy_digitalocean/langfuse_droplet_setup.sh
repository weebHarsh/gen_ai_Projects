#!/usr/bin/env bash
# Self-host Langfuse (v4, open source) on a DigitalOcean Droplet with Docker Compose.
#
# 1. Create a Droplet: Ubuntu 24.04, at least 4 GB RAM / 2 vCPU (8 GB is comfortable), region BLR1.
#    Docker 1-Click image or plain Ubuntu both work.
# 2. Copy this script up and run it as root:
#       scp langfuse_droplet_setup.sh root@<ip>:~ && ssh root@<ip> 'bash langfuse_droplet_setup.sh'
# 3. Open http://<ip>:3000, sign up, create a project + API keys, put them in your .env:
#       LANGFUSE_BASE_URL=http://<ip>:3000
# For production, put a domain + TLS in front (see the Caddy section at the bottom).
set -euo pipefail

INSTALL_DIR=/opt/langfuse
PUBLIC_IP=$(curl -fsS http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address || hostname -I | awk '{print $1}')

echo "==> Installing Docker (skipped if present)"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi

echo "==> Fetching the official Langfuse docker-compose.yml"
mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR"
curl -fsSLo docker-compose.yml https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml

echo "==> Generating secrets into $INSTALL_DIR/.env (compose reads it automatically)"
if [ ! -f .env ]; then
  rand() { openssl rand -hex "$1"; }
  PG_PASS=$(rand 16); MINIO_PASS=$(rand 16)
  cat > .env <<EOF
NEXTAUTH_URL=http://${PUBLIC_IP}:3000
NEXTAUTH_SECRET=$(rand 32)
SALT=$(rand 32)
ENCRYPTION_KEY=$(rand 32)
POSTGRES_PASSWORD=${PG_PASS}
DATABASE_URL=postgresql://postgres:${PG_PASS}@postgres:5432/postgres
CLICKHOUSE_PASSWORD=$(rand 16)
REDIS_AUTH=$(rand 16)
MINIO_ROOT_PASSWORD=${MINIO_PASS}
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=${MINIO_PASS}
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=${MINIO_PASS}
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=${MINIO_PASS}
EOF
  chmod 600 .env
fi

echo "==> Firewall: only SSH, Langfuse web (3000) and MinIO media (9090)"
if command -v ufw >/dev/null; then
  ufw allow OpenSSH && ufw allow 3000/tcp && ufw allow 9090/tcp && ufw --force enable
fi

echo "==> Starting Langfuse (web, worker, postgres, clickhouse, redis, minio)"
docker compose up -d
echo "Waiting for web to become healthy..."
for _ in $(seq 1 60); do
  curl -fsS "http://localhost:3000/api/public/health" >/dev/null 2>&1 && break
  sleep 5
done
docker compose ps
echo
echo "Langfuse is up: http://${PUBLIC_IP}:3000"
echo "Next: sign up, create a project, copy API keys into your .env with LANGFUSE_BASE_URL=http://${PUBLIC_IP}:3000"

# ---- Production hardening (optional) --------------------------------------------------
# * TLS: point a domain at the Droplet, then
#     apt install -y caddy && echo 'langfuse.example.com { reverse_proxy localhost:3000 }' > /etc/caddy/Caddyfile && systemctl reload caddy
#   and set NEXTAUTH_URL=https://langfuse.example.com in .env, then `docker compose up -d`.
# * Backups: enable Droplet backups or use DO Managed PostgreSQL + Spaces (S3) instead of the bundled containers.
# * Retention: ClickHouse grows with traces. Set data retention in project settings and monitor disk.
# * Pin image tags (langfuse:4 -> a specific version) before upgrading, and read release notes.
