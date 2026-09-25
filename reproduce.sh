#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="0.9.27"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"

usage() {
    echo "Usage: $0 [all|measure|analysis|check]"
    echo "  all       download data and models, run every GPU and CPU stage, build the report"
    echo "  measure   run every measurement stage without building the report"
    echo "  analysis  rebuild every paper number from saved artifacts on CPU"
    echo "  check     run static analysis and the test suite"
}

load_environment() {
    local profile="${ENV_FILE:-$ROOT/.env}"
    if [[ ! -f "$profile" ]]; then
        if [[ -n "${ENV_FILE:-}" ]]; then
            echo "Environment file does not exist" >&2
            return 1
        fi
        return 0
    fi
    local line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        if [[ ! "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            echo "Environment files must contain KEY=VALUE assignments" >&2
            return 1
        fi
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        if [[ "$value" == \"*\" || "$value" == \'*\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        if [[ ! -v "$key" ]]; then
            export "$key=$value"
        fi
    done <"$profile"
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [[ "$(uv --version)" != "uv $UV_VERSION" ]]; then
        echo "Use uv $UV_VERSION to reproduce this environment" >&2
        return 1
    fi
}

environment() {
    cd "$ROOT"
    uv sync --locked
}

stage() {
    echo "==> entity-lens $*"
    uv run --locked entity-lens "$@"
}

measure() {
    stage download-data
    stage download-models
    stage evaluate-all
    stage experiments
    stage wikipedia
    stage attribute-rates
    stage corpus-download
    stage corpus-scan
}

analysis() {
    stage verify-artifacts
    stage download-data
    stage report
    stage verify-artifacts
}

check() {
    uv run --locked ruff format --check src tests
    uv run --locked ruff check src tests
    uv run --locked mypy
    uv run --locked vulture
    uv run --locked shellcheck reproduce.sh
    uv run --locked pytest --cov --cov-report=term-missing
}

main() {
    case "$MODE" in
        -h|--help) usage; return ;;
        all|measure|analysis|check) ;;
        *) usage >&2; exit 2 ;;
    esac
    load_environment
    ensure_uv
    environment
    case "$MODE" in
        all)
            measure
            stage report
            stage manifest
            ;;
        measure)
            measure
            ;;
        analysis)
            analysis
            ;;
        check)
            check
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main
