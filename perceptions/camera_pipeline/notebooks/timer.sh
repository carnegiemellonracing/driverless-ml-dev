#!/usr/bin/env bash
script="/root/driverless-ml-dev/perceptions/camera_pipeline/notebooks/run_notebooks_simple.sh"   # <- set this

today_utc=$(date -u +%F)                    # e.g., 2025-11-09
target_utc=$(date -d "$today_utc 04:00 -0500" +%s)  # 4:00 EST == 09:00 UTC
now_utc=$(date +%s)

if [ "$now_utc" -ge "$target_utc" ]; then
  echo "It's already past 4:00 AM EST today. Not scheduling."
  exit 0
fi

delay=$(( target_utc - now_utc ))
echo "Will run at $(date -u -d "@$target_utc") UTC (~4:00 AM EST)"
nohup bash -c "sleep $delay; $script" > /root/driverless-ml-dev/perceptions/camera_pipeline/notebooks/script-4am-est.log 2>&1 &
disown
