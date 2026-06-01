#!/bin/bash
# v6 自动提交 + 迭代优化
# 提交 → 等结果 → 保存反馈 → 生成下一轮权重 → 重新打包 → 再提交
#
# 用法:
#   bash scripts/auto_submit.sh              # 从 round 1 开始
#   bash scripts/auto_submit.sh 3            # 从 round 3 开始（已有前几轮反馈）
#   bash scripts/auto_submit.sh 1 --once     # 只提交一轮不迭代

TOKEN="7027dacbb2d44548add68a006ea99f8a"
API="http://<PLATFORM_HOST>:8000"
LOG="auto_submit_v6.log"
MAX_ROUNDS=10  # 最多迭代轮数

START_ROUND=${1:-1}
ONCE_MODE=${2:-""}

log() {
    echo "$(date '+%m-%d %H:%M:%S'): $1" | tee -a "$LOG"
}

submit_and_wait() {
    local zip_file="$1"
    local round_num="$2"

    # Submit (retry on rate limit)
    log "--- Round $round_num: Submitting $zip_file ---"
    while true; do
        RESPONSE=$(curl -s -X POST "$API/api/submit" \
            -F "token=$TOKEN" \
            -F "submission=@$zip_file")

        if echo "$RESPONSE" | grep -q "Rate limited"; then
            SECS=$(echo "$RESPONSE" | python3 -c "import sys,json,re; m=re.search(r'(\d+) seconds', json.load(sys.stdin).get('detail','')); print(m.group(1) if m else '3600')" 2>/dev/null)
            log "Round $round_num: Rate limited, sleeping ${SECS}s (~$((SECS/60))min)..."
            sleep "$SECS"
            sleep 10
            continue
        fi

        SUB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('submission_id',''))" 2>/dev/null)
        if [ -n "$SUB_ID" ] && [ "$SUB_ID" != "None" ] && [ "$SUB_ID" != "" ]; then
            log "Round $round_num: Submitted! ID: $SUB_ID"
            break
        fi

        log "Round $round_num: Error: $RESPONSE. Retry in 60s..."
        sleep 60
    done

    # Wait for completion
    log "Round $round_num: Waiting for result..."
    local elapsed=0
    while [ $elapsed -lt 3900 ]; do
        RESULT=$(curl -s "$API/api/result/$SUB_ID?token=$TOKEN")
        STATUS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)

        if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "timeout" ]; then
            echo "$RESULT" | python3 -m json.tool > "feedback_state/result_round${round_num}.json"
            log "Round $round_num: Done! Status=$STATUS"
            
            if [ "$STATUS" = "completed" ]; then
                # 自动保存反馈为迭代格式
                echo "$RESULT" | python3 -c "
import sys, json
from pathlib import Path

data = json.load(sys.stdin)
details = data.get('details', {})
normal_ic = details.get('normal_ic', {})
extreme_ic = details.get('extreme_ic', {})

feedback = {}
for i in range(30):
    ds = f'dataset{i}'
    n = normal_ic.get(ds, {})
    e = extreme_ic.get(ds, {})
    feedback[ds] = {
        'nR5': n.get('Ret5', 0) or 0,
        'nR60': n.get('Ret60', 0) or 0,
        'eR5': e.get('Ret5', 0) or 0,
        'eR60': e.get('Ret60', 0) or 0,
    }

output = {
    'round': $round_num,
    'submission_id': '$SUB_ID',
    'results': feedback,
    'timings': data.get('timings', {}),
}

Path('feedback_state').mkdir(exist_ok=True)
with open('feedback_state/iter_${round_num}.json', 'w') as f:
    json.dump(output, f, indent=2)
print('Feedback saved to feedback_state/iter_${round_num}.json')

# Print summary
import numpy as np
nR5 = np.mean([feedback[f'dataset{i}']['nR5'] for i in range(30)])
nR60 = np.mean([feedback[f'dataset{i}']['nR60'] for i in range(30)])
eR5 = np.mean([feedback[f'dataset{i}']['eR5'] for i in range(30)])
eR60 = np.mean([feedback[f'dataset{i}']['eR60'] for i in range(30)])
print(f'  Mean IC: nR5={nR5:.4f} nR60={nR60:.4f} eR5={eR5:.4f} eR60={eR60:.4f}')
" 2>&1 | tee -a "$LOG"
            fi
            return 0
        fi
        
        log "Round $round_num: status=$STATUS, checking in 30s... (${elapsed}s elapsed)"
        sleep 30
        elapsed=$((elapsed + 30))
    done
    
    log "Round $round_num: Timeout waiting for result!"
    return 1
}

# =============================================================================
# Main loop
# =============================================================================

echo "" >> "$LOG"
log "======== v6 Auto-submit started (round $START_ROUND, max $MAX_ROUNDS) ========"

mkdir -p feedback_state submissions

for round_num in $(seq $START_ROUND $MAX_ROUNDS); do
    log "====== Round $round_num ======"
    
    # 如果不是第一轮，先生成新权重
    if [ $round_num -gt 1 ]; then
        prev=$((round_num - 1))
        if [ -f "feedback_state/iter_${prev}.json" ]; then
            log "Round $round_num: Generating weights from round $prev feedback..."
            python3 scripts/submit_iterate.py next --round $prev 2>&1 | tee -a "$LOG"
        else
            log "Round $round_num: No feedback from round $prev, using current weights"
        fi
    fi
    
    # 打包
    log "Round $round_num: Packing submission..."
    python3 scripts/submit_iterate.py pack --round $round_num 2>&1 | tee -a "$LOG"
    
    SUBMISSION_FILE="submissions/submission_round${round_num}.tar.gz"
    if [ ! -f "$SUBMISSION_FILE" ]; then
        log "Round $round_num: ERROR - Pack failed!"
        break
    fi
    
    # 提交并等待
    submit_and_wait "$SUBMISSION_FILE" "$round_num"
    
    # 只跑一轮模式
    if [ "$ONCE_MODE" = "--once" ]; then
        log "Single round mode, stopping."
        break
    fi
    
    log "Round $round_num complete. Next round in 5s..."
    sleep 5
done

log "======== Auto-submit finished! ========"
log "查看对比: python3 scripts/submit_iterate.py compare"
