#!/bin/bash
# Iterative feedback loop: submit → get IC → update best → try new configs → repeat
# Phase 1: random exploration (LGB>=0.3), Phase 2: hill-climbing
TOKEN="7027dacbb2d44548add68a006ea99f8a"
API="http://118.196.107.100:8000"
LOG="iterative_submit.log"
MAX_ITERATIONS=20

log() {
    echo "$(date '+%m-%d %H:%M:%S'): $1" | tee -a "$LOG"
}

submit_and_wait() {
    local zip_file="$1"
    local result_file="$2"

    log "Submitting $zip_file..."
    while true; do
        RESPONSE=$(curl -s -X POST "$API/api/submit" \
            -F "token=$TOKEN" \
            -F "submission=@$zip_file")

        if echo "$RESPONSE" | grep -q "Rate limited"; then
            SECS=$(echo "$RESPONSE" | python3 -c "import sys,json,re; m=re.search(r'(\d+) seconds', json.load(sys.stdin).get('detail','')); print(m.group(1) if m else '3600')" 2>/dev/null)
            log "Rate limited, sleeping ${SECS}s (~$((SECS/60))min)..."
            sleep "$SECS"
            sleep 10
            continue
        fi

        SUB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('submission_id',''))" 2>/dev/null)
        if [ -n "$SUB_ID" ] && [ "$SUB_ID" != "None" ] && [ "$SUB_ID" != "" ]; then
            log "Submitted! ID: $SUB_ID"
            break
        fi

        log "Error: $RESPONSE. Retry in 60s..."
        sleep 60
    done

    # Wait for completion
    log "Waiting for result..."
    while true; do
        RESULT=$(curl -s "$API/api/result/$SUB_ID?token=$TOKEN")
        STATUS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)

        if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "timeout" ]; then
            echo "$RESULT" | python3 -m json.tool > "$result_file"
            log "Done! Status=$STATUS. Saved $result_file"
            if [ "$STATUS" != "completed" ]; then
                return 1
            fi
            return 0
        fi
        sleep 30
    done
}

# ============================================================
echo "" >> "$LOG"
log "======== Iterative optimization started (max $MAX_ITERATIONS iterations) ========"

# Get current global iteration from state
START_ITER=$(python3 -c "
import json, os
state_path = 'feedback_state/state.json'
if os.path.exists(state_path):
    s = json.load(open(state_path))
    print(s.get('iteration', 0) + 1)
else:
    print(1)
" 2>/dev/null)
log "Resuming from global iteration $START_ITER"

for i in $(seq 1 $MAX_ITERATIONS); do
    ITER=$((START_ITER + i - 1))
    log ""
    log "===== Global Iteration $ITER (loop $i/$MAX_ITERATIONS) ====="

    # Package current weights
    rm -f submission.zip
    zip -j submission.zip factor.py predict.py models_v5/* > /dev/null 2>&1
    log "Packaged submission.zip ($(du -h submission.zip | cut -f1))"

    # Submit and wait
    RESULT_FILE="result_iter_$(printf '%03d' $ITER).json"
    submit_and_wait "submission.zip" "$RESULT_FILE"
    if [ $? -ne 0 ]; then
        log "Submission failed/timeout. Stopping."
        break
    fi

    # Run feedback optimizer: updates state + prepares next weights
    log "Running feedback optimizer..."
    python3 feedback_optimize.py --result "$RESULT_FILE" 2>&1 | tee -a "$LOG"

    log "Global iteration $ITER complete."
done

# Final: revert to historical best weights and submit
log ""
log "===== Finalizing: applying historical best weights ====="
python3 -c "
import json
from pathlib import Path

state = json.load(open('feedback_state/state.json'))
best = state['best_weights']

# Ensure all datasets present
for i in range(30):
    ds = f'dataset{i}'
    if ds not in best:
        best[ds] = {
            'ret5_w_local': 1.0, 'ret5_w_global': 0.0, 'ret5_w_gru': 0.0, 'ret5_w_tf': 0.0,
            'ret60_w_local': 1.0, 'ret60_w_global': 0.0, 'ret60_w_gru': 0.0, 'ret60_w_tf': 0.0,
        }

with open('models_v5/ensemble_weights.json', 'w') as f:
    json.dump(best, f, indent=2)
print('Applied historical best weights.')
" 2>&1 | tee -a "$LOG"

rm -f submission.zip
zip -j submission.zip factor.py predict.py models_v5/* > /dev/null 2>&1
RESULT_FILE="result_final.json"
log "Submitting final (best weights)..."
submit_and_wait "submission.zip" "$RESULT_FILE"

log "======== Iterative optimization complete ========"
log "Check feedback_state/state.json for full history"
