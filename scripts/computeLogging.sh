#!/bin/bash
# tegrastats_logger.sh
# Logs Jetson utilization to CSV: GPU, CPU, RAM, EMC (memory bandwidth)
#
# USAGE:
#   ./tegrastats_logger.sh [output_file] [interval_ms]
#
#   output_file  - CSV filename (default: stats.csv)
#   interval_ms  - Sampling rate in milliseconds (default: 10)
#
# EXAMPLES:
#   # Basic usage with defaults (stats.csv, 10ms)
#   ./tegrastats_logger.sh
#
#   # Custom file and interval
#   ./tegrastats_logger.sh race_data.csv 10
#
#   # High-frequency logging (5ms)
#   ./tegrastats_logger.sh detailed.csv 5
#
# TYPICAL WORKFLOW:
#   Terminal 1: ./tegrastats_logger.sh race.csv 10
#   Terminal 2: ./your_inference_code
#   Terminal 1: Ctrl+C when done
#
# OUTPUT CSV COLUMNS:
#   timestamp, ram_used_mb, ram_total_mb, cpu0, cpu1, cpu2, cpu3,
#   gpu_util, gpu_freq_mhz, emc_util, emc_freq_mhz

OUTPUT="${1:-stats.csv}"
INTERVAL="${2:-10}"

echo "Logging to: $OUTPUT (interval: ${INTERVAL}ms)"
echo "Press Ctrl+C to stop"

# CSV header
echo "timestamp,ram_used_mb,ram_total_mb,cpu0,cpu1,cpu2,cpu3,gpu_util,gpu_freq_mhz,emc_util,emc_freq_mhz" > "$OUTPUT"

# Parse tegrastats output with awk (faster than multiple grep/cut calls)
tegrastats --interval "$INTERVAL" | while read line; do
    
    timestamp=$(date +%s.%N)
    
    # Single awk call to extract all metrics
    parsed=$(echo "$line" | awk '
    {
        # RAM: "RAM 2477/7850MB"
        if (match($0, /RAM ([0-9]+)\/([0-9]+)/, ram)) {
            ram_used = ram[1]
            ram_total = ram[2]
        }
        
        # CPU: "CPU [12%@1190,8%@1190,10%@1190,9%@1190]"
        if (match($0, /CPU \[([0-9]+)%@[^,]+,([0-9]+)%@[^,]+,([0-9]+)%@[^,]+,([0-9]+)%/, cpu)) {
            cpu0 = cpu[1]
            cpu1 = cpu[2]
            cpu2 = cpu[3]
            cpu3 = cpu[4]
        }
        
        # GPU: "GR3D_FREQ 45%@624"
        if (match($0, /GR3D_FREQ ([0-9]+)%@([0-9]+)/, gpu)) {
            gpu_util = gpu[1]
            gpu_freq = gpu[2]
        }
        
        # EMC: "EMC_FREQ 17%@1600"
        if (match($0, /EMC_FREQ ([0-9]+)%@([0-9]+)/, emc)) {
            emc_util = emc[1]
            emc_freq = emc[2]
        }
        
        # Output comma-separated values (default to 0 if not found)
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
            (ram_used ? ram_used : 0),
            (ram_total ? ram_total : 0),
            (cpu0 ? cpu0 : 0),
            (cpu1 ? cpu1 : 0),
            (cpu2 ? cpu2 : 0),
            (cpu3 ? cpu3 : 0),
            (gpu_util ? gpu_util : 0),
            (gpu_freq ? gpu_freq : 0),
            (emc_util ? emc_util : 0),
            (emc_freq ? emc_freq : 0)
    }')
    
    # Write to CSV
    echo "$timestamp,$parsed" >> "$OUTPUT"
done

echo "Logging stopped. Data: $OUTPUT"