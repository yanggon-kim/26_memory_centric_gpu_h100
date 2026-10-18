#!/usr/bin/env bash
# Kill profiled runs whose rocprofv3 crash handler is stuck waiting for children
# (observed after HSA "AQL packet is malformed" aborts). Records each kill.
root=$1
while true; do
  for log in "$root"/m2_profile/*/*/log.txt "$root"/m2_counters/*/*/log.txt; do
    [ -f "$log" ] || continue
    d=$(dirname "$log"); [ -f "$d/done" ] || [ -f "$d/killed_by_watchdog" ] && continue
    if grep -q "rocprofv3 will wait for" "$log" && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt 30 ]; then
      for pid in $(pgrep -f -- "-d $d/rocprof"); do kill -9 "$pid" 2>/dev/null; done
      for pid in $(pgrep -f -- "--output $d/run"); do kill -9 "$pid" 2>/dev/null; done
      echo "$(date +%FT%T) killed hung run $d" | tee "$d/killed_by_watchdog" >> "$root/m2_counters_watchdog.log"
    fi
  done
  sleep 15
done
