#!/usr/bin/env bash
# Runs on the server in ~/flipside-ai, piped over SSH by .github/workflows/publish-image.yml
# after it has copied docker-compose.yml and ai.env there. The image is pulled with the
# server's own Docker Hub login, which the backend deploy also uses; do not log out here.
set -euo pipefail
cd ~/flipside-ai

# Join whichever network the backend container is on, so it can call http://ai-service:8000.
# Saved to .env so manual `docker compose` commands in this folder use the same network.
backend="${BACKEND_CONTAINER:-flipside-backend}"
network=$(docker inspect "$backend" \
    --format '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}' 2>/dev/null | head -n 1 || true)
if [ -z "$network" ]; then
    echo "Could not find the Docker network of container '$backend'." >&2
    exit 1
fi
printf 'BACKEND_NETWORK=%s\n' "$network" > .env

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
