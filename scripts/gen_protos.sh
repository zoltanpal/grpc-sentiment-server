#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Generating protobufs into $ROOT/src/pb"
OUT="$ROOT/src/pb"

mkdir -p "$OUT"

python -m grpc_tools.protoc \
  -I "$ROOT/src/proto" \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  $(find "$ROOT/src/proto" -name "*.proto")