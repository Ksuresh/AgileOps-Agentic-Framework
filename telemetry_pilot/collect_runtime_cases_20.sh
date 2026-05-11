#!/usr/bin/env bash
set -euo pipefail

AAF_REPO="${AAF_REPO:-$HOME/projects/AgileOps-Agentic-Framework}"
SOCK_DIR="${SOCK_DIR:-$HOME/projects/benchmarks/sock-shop/deploy/docker-compose}"
OUT_ROOT="$AAF_REPO/telemetry_pilot/raw_runtime_20"

mkdir -p "$OUT_ROOT"

cd "$SOCK_DIR"

capture_case () {
  case_id="$1"
  case_name="$2"
  out_dir="$OUT_ROOT/${case_id}_${case_name}"
  mkdir -p "$out_dir"

  echo "[CAPTURE] $case_id $case_name"

  docker compose ps > "$out_dir/compose_ps.txt" 2>&1 || true
  docker ps -a > "$out_dir/docker_ps_a.txt" 2>&1 || true
  docker stats --no-stream > "$out_dir/docker_stats.txt" 2>&1 || true

  curl -sS -m 5 http://localhost/ > "$out_dir/curl_root.html" 2> "$out_dir/curl_root.err" || true
  curl -sS -m 5 http://localhost/catalogue > "$out_dir/curl_catalogue.json" 2> "$out_dir/curl_catalogue.err" || true
  curl -sS -m 5 http://localhost/tags > "$out_dir/curl_tags.json" 2> "$out_dir/curl_tags.err" || true

  for svc in edge-router front-end catalogue carts orders payment shipping user queue-master rabbitmq catalogue-db carts-db orders-db user-db; do
    docker compose logs --no-color --tail=120 "$svc" > "$out_dir/${svc}_logs.txt" 2>&1 || true
  done

  for svc in edge-router front-end catalogue carts orders payment shipping user queue-master rabbitmq catalogue-db carts-db orders-db user-db; do
    cid="$(docker compose ps -q "$svc" 2>/dev/null | head -1 || true)"
    if [ -n "$cid" ]; then
      docker inspect "$cid" > "$out_dir/${svc}_inspect.json" 2>&1 || true
    fi
  done
}

restore_all () {
  docker compose up -d >/dev/null 2>&1 || true
  sleep 8
}

echo "[INFO] Sock Shop dir: $SOCK_DIR"
echo "[INFO] Output root: $OUT_ROOT"

restore_all

# T01 baseline
capture_case "T01" "baseline_startup_readiness"

# T02 catalogue stopped
docker compose stop catalogue >/dev/null 2>&1 || true
sleep 6
capture_case "T02" "catalogue_service_stopped"
restore_all

# T03 front-end scaled
docker compose up -d --scale front-end=3 >/dev/null 2>&1 || true
sleep 8
capture_case "T03" "frontend_scaled_three_replicas"
docker compose up -d --scale front-end=1 >/dev/null 2>&1 || true
sleep 6

# T04 carts stopped
docker compose stop carts >/dev/null 2>&1 || true
sleep 6
capture_case "T04" "carts_service_stopped"
restore_all

# T05 orders stopped
docker compose stop orders >/dev/null 2>&1 || true
sleep 6
capture_case "T05" "orders_service_stopped"
restore_all

# T06 front-end restarted repeatedly
docker compose restart front-end >/dev/null 2>&1 || true
sleep 2
docker compose restart front-end >/dev/null 2>&1 || true
sleep 2
docker compose restart front-end >/dev/null 2>&1 || true
sleep 6
capture_case "T06" "frontend_repeated_restarts"
restore_all

# T07 catalogue scaled
docker compose up -d --scale catalogue=3 >/dev/null 2>&1 || true
sleep 8
capture_case "T07" "catalogue_scaled_three_replicas"
docker compose up -d --scale catalogue=1 >/dev/null 2>&1 || true
sleep 6

# T08 carts scaled
docker compose up -d --scale carts=3 >/dev/null 2>&1 || true
sleep 8
capture_case "T08" "carts_scaled_three_replicas"
docker compose up -d --scale carts=1 >/dev/null 2>&1 || true
sleep 6

# T09 orders scaled
docker compose up -d --scale orders=3 >/dev/null 2>&1 || true
sleep 8
capture_case "T09" "orders_scaled_three_replicas"
docker compose up -d --scale orders=1 >/dev/null 2>&1 || true
sleep 6

# T10 front-end CPU stress if stress command exists, otherwise collect load proxy
docker compose exec -T front-end sh -lc 'command -v yes >/dev/null && (yes >/dev/null & echo $! > /tmp/aaf_yes.pid) || true' >/dev/null 2>&1 || true
sleep 8
capture_case "T10" "frontend_cpu_pressure"
docker compose exec -T front-end sh -lc 'test -f /tmp/aaf_yes.pid && kill $(cat /tmp/aaf_yes.pid) || true' >/dev/null 2>&1 || true
restore_all

# T11 catalogue CPU stress
docker compose exec -T catalogue sh -lc 'command -v yes >/dev/null && (yes >/dev/null & echo $! > /tmp/aaf_yes.pid) || true' >/dev/null 2>&1 || true
sleep 8
capture_case "T11" "catalogue_cpu_pressure"
docker compose exec -T catalogue sh -lc 'test -f /tmp/aaf_yes.pid && kill $(cat /tmp/aaf_yes.pid) || true' >/dev/null 2>&1 || true
restore_all

# T12 memory pressure proxy through repeated catalogue calls
for i in $(seq 1 40); do curl -sS -m 2 http://localhost/catalogue >/dev/null 2>&1 || true; done
capture_case "T12" "catalogue_request_pressure"

# T13 config inspection governance check
capture_case "T13" "runtime_config_inspection"

# T14 payment stopped as deployment/service failure proxy
docker compose stop payment >/dev/null 2>&1 || true
sleep 6
capture_case "T14" "payment_service_stopped"
restore_all

# T15 security config inspection
capture_case "T15" "runtime_security_config_inspection"

# T16 image metadata inspection
capture_case "T16" "runtime_image_metadata_inspection"

# T17 release plus reliability: restart front-end and stop catalogue
docker compose restart front-end >/dev/null 2>&1 || true
docker compose stop catalogue >/dev/null 2>&1 || true
sleep 6
capture_case "T17" "release_reliability_mixed"
restore_all

# T18 reliability plus cost: scale front-end and stop carts
docker compose up -d --scale front-end=3 >/dev/null 2>&1 || true
docker compose stop carts >/dev/null 2>&1 || true
sleep 8
capture_case "T18" "reliability_cost_mixed"
docker compose up -d --scale front-end=1 >/dev/null 2>&1 || true
restore_all

# T19 security plus release gate: config/security inspection after restart
docker compose restart front-end >/dev/null 2>&1 || true
sleep 6
capture_case "T19" "security_release_gate_mixed"
restore_all

# T20 multi-domain go/no-go: scale front-end, stop catalogue, restart orders
docker compose up -d --scale front-end=3 >/dev/null 2>&1 || true
docker compose stop catalogue >/dev/null 2>&1 || true
docker compose restart orders >/dev/null 2>&1 || true
sleep 8
capture_case "T20" "multi_domain_go_no_go"
docker compose up -d --scale front-end=1 >/dev/null 2>&1 || true
restore_all

echo "[DONE] Collected runtime artifacts in $OUT_ROOT"
