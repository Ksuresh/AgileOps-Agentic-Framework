#!/usr/bin/env bash
set -euo pipefail

# Collect raw Sock Shop runtime artifacts without interpreting them.
# Usage: ./collect_runtime_artifacts.sh <case-id> <repetition> [compose-file]

CASE_ID="${1:?case id required}"
REP="${2:?repetition required}"
COMPOSE_FILE="${3:-docker-compose.yml}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="runtime_validation/artifacts/${CASE_ID}/rep-${REP}/${STAMP}"
mkdir -p "$OUT/logs" "$OUT/inspect"

printf '%s\n' "$STAMP" > "$OUT/collection_started_utc.txt"

docker ps -a --no-trunc > "$OUT/docker_ps.txt" 2>&1 || true
docker compose -f "$COMPOSE_FILE" ps -a > "$OUT/compose_ps.txt" 2>&1 || true
docker stats --no-stream > "$OUT/docker_stats.txt" 2>&1 || true
docker compose -f "$COMPOSE_FILE" config > "$OUT/compose_resolved.yaml" 2>&1 || true

# Capture each compose service separately so a failed service does not prevent
# collection from the remaining services.
while IFS= read -r service; do
  [ -n "$service" ] || continue
  docker compose -f "$COMPOSE_FILE" logs --no-color --timestamps "$service" > "$OUT/logs/${service}.log" 2>&1 || true
  cid="$(docker compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null || true)"
  if [ -n "$cid" ]; then
    docker inspect "$cid" > "$OUT/inspect/${service}.json" 2>&1 || true
  fi
done < <(docker compose -f "$COMPOSE_FILE" config --services 2>/dev/null || true)

END="$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$END" > "$OUT/collection_finished_utc.txt"
printf '%s\n' "$OUT"
