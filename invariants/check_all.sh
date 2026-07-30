#!/usr/bin/env bash
# 番人4本を1往復でまとめて回す。
#   使い方: bash invariants/check_all.sh [engine|book|all]
#     engine … engine_tieback / engine_golden / size_tieback（src/engine/ を触ったら必須）
#     book   … book_tieback（構成のアンカーを触ったら必須）
#     all    … 全部（既定）
# 全文ログは scratchpad/out_<name>.txt。端末には各本の要約1行だけを出す。
# 1本でも FAIL なら exit 1。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p scratchpad

MODE="${1:-all}"
case "$MODE" in
  engine) NAMES=(engine_tieback engine_golden size_tieback) ;;
  book)   NAMES=(book_tieback) ;;
  all)    NAMES=(engine_tieback engine_golden size_tieback book_tieback) ;;
  *) echo "usage: bash invariants/check_all.sh [engine|book|all]" >&2; exit 2 ;;
esac

rc_all=0
for name in "${NAMES[@]}"; do
  args=""
  [ "$name" = "engine_golden" ] && args="check-run"
  log="scratchpad/out_${name}.txt"
  timeout 1800 "$PY" "invariants/${name}.py" $args >"$log" 2>&1
  rc=$?
  # 各番人は末尾に "===== n/m PASS =====" を出す
  summary=$(grep -E "PASS =====" "$log" | tail -1)
  [ -z "$summary" ] && summary=$(tail -1 "$log")
  if [ $rc -ne 0 ]; then
    rc_all=1
    echo "FAIL  ${name}  (rc=${rc})  ${summary}"
    grep -E "^\s*FAIL" "$log" | head -5
    echo "      → 全文: ${log}"
  else
    echo "PASS  ${name}  ${summary}"
  fi
done

if [ $rc_all -ne 0 ]; then
  echo "===== 番人 FAIL あり。期待値を書き換えるのではなく差分を報告すること ====="
else
  echo "===== 番人 ${#NAMES[@]}/${#NAMES[@]} 全PASS ====="
fi
exit $rc_all
