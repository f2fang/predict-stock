#!/bin/bash

APP_DIR="/home/hzieeff/ff_research/new"
APP_FILE="app_stocks.py"
LOG_FILE="$APP_DIR/flask.log"

echo "🔁 正在重启 Flask 应用..."

# 杀掉旧的 Flask 进程（基于脚本名）
pkill -f "$APP_FILE"

# 稍等 2 秒确保彻底退出
sleep 2

# 进入项目目录
cd "$APP_DIR" || exit 1

# 启动 Flask 并重定向日志
nohup python3 "$APP_FILE" > "$LOG_FILE" 2>&1 &

echo "✅ Flask 已重启，日志记录中: $LOG_FILE"

