#!/bin/bash
cd "$(dirname "$0")"
lsof -ti tcp:8004 | xargs kill -9 2>/dev/null   # освободить порт (перезапуск с актуальными правками)
python3 СЕРВЕР.py &
sleep 1
open http://localhost:8004/
wait
