#!/usr/bin/env bash
# Runs on the server in ~/flipside-ai, piped over SSH by .github/workflows/publish-image.yml
# after it has copied docker-compose.yml and ai.env there. The image is pulled with the
# server's own Docker Hub login, which the backend deploy also uses; do not log out here.
set -euo pipefail
cd ~/flipside-ai

# Set one KEY=value line in .env and leave the other lines as they are.
set_env() {
    touch .env
    { grep -v "^$1=" .env || true; printf '%s=%s\n' "$1" "$2"; } > .env.tmp
    mv .env.tmp .env
}

# Join whichever network the backend container is on, so it can call http://ai-service:8000.
# Saved to .env so manual `docker compose` commands in this folder use the same network.
backend="${BACKEND_CONTAINER:-flipside-backend}"
network=$(docker inspect "$backend" \
    --format '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}' 2>/dev/null | head -n 1 || true)
if [ -z "$network" ]; then
    echo "Could not find the Docker network of container '$backend'." >&2
    exit 1
fi
set_env BACKEND_NETWORK "$network"

# The workflow passes the image it just published, by digest (repo@sha256:...), so this
# deploy cannot pull an image another run pushed to the shared tag. Saved to .env so manual
# `docker compose` commands keep using the deployed image. Without it the tag is used.
if [ -n "${AI_IMAGE:-}" ]; then
    set_env AI_IMAGE "$AI_IMAGE"
fi

docker compose pull
docker compose up -d
docker image prune -f

# Healthy means the server answers /health, which starts only after models are preloaded.
# The first deploy also downloads the models, so wait up to 5 minutes.
for _ in $(seq 1 60); do
    status=$(docker inspect flipside-ai --format '{{.State.Health.Status}}')
    if [ "$status" = "healthy" ]; then
        # /health is 200 even when preloading failed; the log is the only sign.
        # Read the log into a variable first: `docker logs | grep -q` fails under pipefail
        # when grep exits early, which would hide the match.
        logs=$(docker logs flipside-ai 2>&1)
        if grep -q "Failed to preload voice models" <<<"$logs"; then
            tail -n 50 <<<"$logs"
            echo "Voice models failed to load (check HF_TOKEN in ai.env)." >&2
            exit 1
        fi
        echo "flipside-ai is healthy on network $network."
        exit 0
    fi
    sleep 5
done

docker logs --tail 50 flipside-ai
echo "flipside-ai did not become healthy (last status: $status)." >&2
exit 1
