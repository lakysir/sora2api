#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${PID_FILE:-$APP_DIR/sora2api.pid}"
LOG_FILE="${LOG_FILE:-$APP_DIR/sora2api.out}"

# 优先使用项目内虚拟环境，其次使用系统 python3
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$APP_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

start() {
  if is_running; then
    echo "sora2api 已在运行 (pid=$(cat "$PID_FILE"))"
    return 0
  fi

  cd "$APP_DIR"
  : > "$LOG_FILE"

  # 生产环境：后台运行 + 输出到日志文件
  nohup "$PYTHON_BIN" "$APP_DIR/main.py" >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  sleep 1
  if is_running; then
    echo "sora2api 启动成功 (pid=$(cat "$PID_FILE")), log=$LOG_FILE"
    return 0
  fi

  echo "sora2api 启动失败，请查看日志: $LOG_FILE" >&2
  return 1
}

stop() {
  if ! [[ -f "$PID_FILE" ]]; then
    echo "sora2api 未运行（找不到 pid 文件: $PID_FILE）"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE" || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$PID_FILE"
    echo "pid 文件为空，已清理: $PID_FILE"
    return 0
  fi

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    rm -f "$PID_FILE"
    echo "sora2api 进程不存在，已清理 pid 文件: $PID_FILE"
    return 0
  fi

  echo "正在停止 sora2api (pid=$pid)..."
  kill "$pid" >/dev/null 2>&1 || true

  # 最多等待 30 秒优雅退出
  for _ in $(seq 1 30); do
    if kill -0 "$pid" >/dev/null 2>&1; then
      sleep 1
    else
      break
    fi
  done

  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "优雅停止超时，强制杀进程 (pid=$pid)"
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi

  rm -f "$PID_FILE"
  echo "sora2api 已停止"
}

status() {
  if is_running; then
    echo "sora2api 运行中 (pid=$(cat "$PID_FILE"))"
  else
    echo "sora2api 未运行"
  fi
}

restart() {
  stop
  start
}

usage() {
  cat <<'EOF'
用法:
  ./sora2api_service.sh start|stop|restart|status

可选环境变量:
  PYTHON_BIN=/path/to/python
  PID_FILE=/path/to/sora2api.pid
  LOG_FILE=/path/to/sora2api.out
EOF
}

cmd="${1:-}"
case "$cmd" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  *) usage; exit 1 ;;
esac

