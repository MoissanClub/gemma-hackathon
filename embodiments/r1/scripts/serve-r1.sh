#!/bin/bash
# llama-server on R1 with the harness flags (serve/serve-orin.sh's), from ~/r1-gemma. Stops only the server this script started (pid
# file): R1 is shared, other users' processes are theirs. Usage: serve-r1.sh [extra llama-server flags] | serve-r1.sh stop
B=$HOME/r1-gemma; PID=$B/llama-server.pid
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
[ -f $PID ] && kill "$(cat $PID)" 2>/dev/null && sleep 1; rm -f $PID
[ "${1:-}" = stop ] && { echo stopped; exit 0; }
LAUNCH="$B/llama.cpp/build/bin/llama-server -m $B/models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf --mmproj $B/models/mmproj-BF16.gguf --alias gemma4-e2b --host 127.0.0.1 --port 8080 -ngl 99 --jinja -np 1 -c 4096 --swa-full --cache-ram 0 $*"
echo "$LAUNCH" > $B/server-launch.txt
nohup $LAUNCH > $B/server.log 2>&1 < /dev/null &
echo $! > $PID; echo "server pid $!"
for i in $(seq 1 180); do
  if curl -s http://127.0.0.1:8080/health | grep -q '"ok"'; then echo "server healthy after ${i}s"; exit 0; fi
  kill -0 "$(cat $PID)" 2>/dev/null || break
  sleep 1
done
echo "server not healthy"; tail -20 $B/server.log; exit 1
