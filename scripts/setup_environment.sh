#!/usr/bin/env bash
# 准备 ShopSimulator 运行环境：独立 venv、解压并校验商品数据、构建 BM25 索引。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOPSIM_ROOT="${ROOT}/third_party/ShopSimulator"
SHOP_ENV_ROOT="${SHOPSIM_ROOT}/shop_env"
ENV_DIR="${SHOPSIM_ROOT}/.venv-shopsim"
COMPRESSED_PRODUCTS="${SHOP_ENV_ROOT}/data/fine_items_eval_train_all.json.gz"
PRODUCTS="${SHOP_ENV_ROOT}/data/items_eval_train.json"
EXPECTED_PRODUCT_SHA256="57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-3.10}"

if [[ ! -d "${SHOP_ENV_ROOT}" ]]; then
  echo "ShopSimulator is missing: ${SHOP_ENV_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${COMPRESSED_PRODUCTS}" ]]; then
  echo "Missing product archive: ${COMPRESSED_PRODUCTS}" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 1
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  uv venv --python "${SHOPSIM_PYTHON}" "${ENV_DIR}"
fi
uv pip install --quiet \
  --python "${ENV_DIR}/bin/python" \
  -r "${SHOP_ENV_ROOT}/requirements.txt"

# 商品数据体积大且解压耗时，已存在则只校验不重做。
if [[ ! -f "${PRODUCTS}" ]]; then
  staging="${PRODUCTS}.preparing"
  trap 'rm -f "${staging}"' EXIT
  gzip -cd "${COMPRESSED_PRODUCTS}" > "${staging}"
  mv "${staging}" "${PRODUCTS}"
  trap - EXIT
fi

actual_sha256="$(sha256sum "${PRODUCTS}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
  echo "Product data SHA-256 mismatch" >&2
  echo "  expected: ${EXPECTED_PRODUCT_SHA256}" >&2
  echo "  actual:   ${actual_sha256}" >&2
  exit 1
fi

cd "${SHOP_ENV_ROOT}"
PYTHONPATH=. "${ENV_DIR}/bin/python" scripts/build_index.py

echo "ShopSimulator is ready."
echo "  product SHA-256: ${actual_sha256}"
echo "  index: ${SHOP_ENV_ROOT}/search_engine/products.sqlite3"
