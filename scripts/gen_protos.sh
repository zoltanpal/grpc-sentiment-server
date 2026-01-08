#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Generating protobufs into $ROOT/src/pb"
OUT="$ROOT/src/pb"

mkdir -p "$OUT"

python -m grpc_tools.protoc \
  -I "$ROOT/protos" \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  "$ROOT/protos/sentiment.proto"

sed -i '' 's/^import sentiment_pb2 as/from pb import sentiment_pb2 as/' src/pb/sentiment_pb2_grpc.py
