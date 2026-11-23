#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
if [[ "${1:-}" == "--down" ]]; then
  docker compose stop --timeout 5
else
  docker compose up -d --build --wait
  printf 'Paper Mending Bench http://127.0.0.1:%s/\n' "${PAPERBENCH_PORT:-19882}"
fi
