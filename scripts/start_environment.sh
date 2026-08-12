#!/usr/bin/env bash
# 启动 ShopSimulator 服务。
#
# 单进程被 GIL 锁死在 ~5 episodes/s（见 docs/environment-notes.md），所以默认起
# 多个独立进程，每个占一个端口。会话状态在进程内，env_idx 只对分配它的进程有效，
# 因此不能用无状态负载均衡——由客户端负责回合内粘连到同一端口。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOPSIM_ROOT="${ROOT}/third_party/ShopSimulator"
SHOP_ENV_ROOT="${SHOPSIM_ROOT}/shop_env"
ENV_DIR="${SHOPSIM_ROOT}/.venv-shopsim"
INDEX_PATH="${SHOP_SEARCH_INDEX:-${SHOP_ENV_ROOT}/search_engine/products.sqlite3}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "ShopSimulator venv is missing. Run: bash scripts/setup_environment.sh" >&2
  exit 1
fi
if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "Search index is missing: ${INDEX_PATH}" >&2
  echo "Run: bash scripts/setup_environment.sh" >&2
  exit 1
fi

# 环境代码的内容锚：起服务前校一次。服务一旦加载进内存，之后改文件不影响这一批数据，
# 所以启动时是唯一有意义的检查点。
python3 "${ROOT}/scripts/hash_environment.py" --quiet

export SHOP_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
export SHOP_ENV_CONFIG="${SHOP_ENV_CONFIG:-${SHOP_ENV_ROOT}/configs/environment.json}"

# reward 权重就在这个文件里。它可以被环境变量指到别处——那样权重就来自一个没被锚住
# 的文件，而锚照样显示"一致"。这是内容锚唯一绕得过去的口子，所以在这儿堵掉。
ANCHORED_CONFIG="${SHOP_ENV_ROOT}/configs/environment.json"
if [[ "$(readlink -f "${SHOP_ENV_CONFIG}")" != "$(readlink -f "${ANCHORED_CONFIG}")" ]]; then
  if [[ -z "${SHOPSIM_ALLOW_UNANCHORED_CONFIG:-}" ]]; then
    echo "SHOP_ENV_CONFIG 指向锚范围之外，reward 权重将不可复现：" >&2
    echo "  当前: ${SHOP_ENV_CONFIG}" >&2
    echo "  锚住: ${ANCHORED_CONFIG}" >&2
    echo "确属有意（例如 reward ablation）则设 SHOPSIM_ALLOW_UNANCHORED_CONFIG=1。" >&2
    exit 1
  fi
  echo "!! reward 配置在锚范围之外：${SHOP_ENV_CONFIG}（已显式放行）" >&2
fi
export SHOP_SEARCH_INDEX="${INDEX_PATH}"
export SHOP_MAX_STEPS="${SHOP_MAX_STEPS:-35}"

# 每进程 slot 数。单进程并发 1 即饱和，slot 只是回合槽位而非吞吐来源，
# 少量余量足够让回合交替时不必等待。
export SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS:-4}"

WORKERS="${SHOPSIM_WORKERS:-8}"
BASE_PORT="${SHOPSIM_BASE_PORT:-5700}"
LOG_DIR="${SHOPSIM_LOG_DIR:-${ROOT}/outputs/environment}"
mkdir -p "${LOG_DIR}"

cd "${SHOP_ENV_ROOT}/shop_env"

if [[ "${WORKERS}" -eq 1 ]]; then
  export SHOPSIM_PORT="${BASE_PORT}"
  echo "starting ShopSimulator: slots=${SHOPSIM_ENV_SLOTS} port=${BASE_PORT}"
  exec "${ENV_DIR}/bin/python" pack_api.py
fi

pids=()
for ((i = 0; i < WORKERS; i++)); do
  port=$((BASE_PORT + i))
  SHOPSIM_PORT="${port}" "${ENV_DIR}/bin/python" pack_api.py \
    >"${LOG_DIR}/worker-${port}.log" 2>&1 &
  pids+=("$!")
done

echo "started ${WORKERS} workers: slots=${SHOPSIM_ENV_SLOTS}/worker" \
  "ports=${BASE_PORT}-$((BASE_PORT + WORKERS - 1))"
echo "logs: ${LOG_DIR}/worker-<port>.log"

# 任一 worker 退出就全部收掉，避免留下半个池子让上游误判容量。
terminate() {
  trap - INT TERM EXIT
  kill "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap terminate INT TERM EXIT

wait -n "${pids[@]}"
echo "a worker exited; shutting down the pool" >&2
