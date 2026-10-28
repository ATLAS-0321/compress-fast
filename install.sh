#!/usr/bin/env bash
# Install compress-fast as a systemd user service and print the Hermes config block.
#
#   ./install.sh --upstream http://127.0.0.1:8080 [--port 8091] [--mode hybrid|serve|shadow]
#                [--timeout 90] [--model NAME] [--no-start]
#
# --upstream is the base URL of your OpenAI-compatible model server (llama.cpp, vLLM, SGLang,
# LM Studio, Ollama, ...) without /v1. Re-running the script updates the settings.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="" PORT=8091 MODE=hybrid TIMEOUT=90 MODEL="" START=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream) UPSTREAM="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --no-start) START=0; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$UPSTREAM" ]] || { echo "--upstream is required, e.g. --upstream http://127.0.0.1:8080" >&2; exit 2; }
[[ "$MODE" =~ ^(hybrid|serve|shadow)$ ]] || { echo "--mode must be hybrid, serve or shadow" >&2; exit 2; }
UPSTREAM="${UPSTREAM%/}"; UPSTREAM="${UPSTREAM%/v1}"

python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "python3 >= 3.10 required" >&2; exit 1; }

CONF_DIR="$HOME/.config/compress-fast"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$CONF_DIR" "$UNIT_DIR"
cat > "$CONF_DIR/env" <<EOF
COMPRESS_FAST_UPSTREAM=$UPSTREAM
COMPRESS_FAST_PORT=$PORT
COMPRESS_FAST_MODE=$MODE
COMPRESS_FAST_HYBRID_TIMEOUT=$TIMEOUT
COMPRESS_FAST_MODEL=$MODEL
EOF
chmod 600 "$CONF_DIR/env"

python3 - "$REPO" "$UNIT_DIR/compress-fast.service" <<'EOF'
import sys
from pathlib import Path
repo, dest = sys.argv[1], Path(sys.argv[2])
src = Path(repo, "systemd", "compress-fast.service").read_text()
dest.write_text(src.replace("@REPO@", repo))
EOF

systemctl --user daemon-reload
if [[ $START -eq 1 ]]; then
  systemctl --user enable --now compress-fast.service
  systemctl --user restart compress-fast.service
  sleep 1
  if curl -fsS "http://127.0.0.1:$PORT/compress-fast/stats" >/dev/null 2>&1; then
    echo "compress-fast is running on 127.0.0.1:$PORT (mode $MODE, upstream $UPSTREAM)"
  else
    echo "service did not answer yet; check: journalctl --user -u compress-fast -n 30" >&2
  fi
fi

MODEL_ID="${MODEL:-<the model name your server reports under /v1/models>}"
cat <<EOF

Add this to ~/.hermes/config.yaml (merge into an existing auxiliary: section):

auxiliary:
  compression:
    provider: custom
    model: $MODEL_ID
    base_url: http://127.0.0.1:$PORT/v1
    timeout: 120

Then restart Hermes. Sessions keep the auxiliary settings they started with, so only new
sessions use compress-fast.
EOF
