#!/bin/bash
# 等待上次提交冷却后自动提交下一轮，然后轮询结果
# 用法: bash scripts/wait_and_submit.sh <round> [submission_file]
# 例如: bash scripts/wait_and_submit.sh 2
#       bash scripts/wait_and_submit.sh 3 submissions/submission_round3.tar.gz

TOKEN="7027dacbb2d44548add68a006ea99f8a"
API="http://<PLATFORM_HOST>:8000"
ROUND=${1:-2}
SUBMISSION=${2:-"submissions/submission_round${ROUND}.tar.gz"}

echo "=== Round $ROUND 自动提交 ==="
echo "文件: $SUBMISSION"
echo ""

if [ ! -f "$SUBMISSION" ]; then
    echo "ERROR: $SUBMISSION 不存在"
    exit 1
fi

# 尝试提交，遇到 rate limit 自动等待
echo "提交中..."
while true; do
    RESPONSE=$(curl -s -X POST "$API/api/submit" \
        -F "token=$TOKEN" \
        -F "submission=@$SUBMISSION")

    if echo "$RESPONSE" | grep -q "Rate limited"; then
        SECS=$(echo "$RESPONSE" | python3 -c "import sys,json,re; m=re.search(r'(\d+) seconds', json.load(sys.stdin).get('detail','')); print(m.group(1) if m else '3600')" 2>/dev/null)
        echo "Rate limited, 等待 ${SECS}s (~$((SECS/60))min)..."
        sleep "$SECS"
        sleep 5
        continue
    fi

    SUB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('submission_id',''))" 2>/dev/null)
    if [ -n "$SUB_ID" ] && [ "$SUB_ID" != "None" ] && [ "$SUB_ID" != "" ]; then
        echo "✅ 提交成功! ID: $SUB_ID"
        break
    fi

    echo "错误: $RESPONSE"
    echo "60秒后重试..."
    sleep 60
done

# 轮询结果
echo ""
echo "等待结果..."
while true; do
    sleep 30
    RESULT=$(curl -s "$API/api/result/$SUB_ID?token=$TOKEN")
    STATUS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null)
    echo "  $(date '+%H:%M:%S') Status: $STATUS"

    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "timeout" ]; then
        # 保存原始结果
        echo "$RESULT" | python3 -m json.tool > "feedback_state/result_round${ROUND}.json"
        
        # 解析并保存为迭代格式 + 打印摘要
        echo "$RESULT" | python3 -c "
import sys, json, numpy as np

data = json.load(sys.stdin)
status = data.get('status')
print(f'\n=== Round $ROUND 结果: {status} ===')

if status != 'completed':
    print(data.get('error_message', '')[:500])
    sys.exit(0)

timings = data.get('timings', {})
print(f'Factor: {timings.get(\"factor_seconds\",0):.1f}s, Predict: {timings.get(\"predict_seconds\",0):.1f}s')

details = data.get('details', {})
normal_ic = details.get('normal_ic', {})
extreme_ic = details.get('extreme_ic', {})

feedback = {}
all_nR5, all_nR60, all_eR5, all_eR60 = [], [], [], []

print(f'\n{\"Dataset\":<12} {\"nR5\":>8} {\"nR60\":>8} {\"eR5\":>8} {\"eR60\":>8}')
print('-' * 50)

for i in range(30):
    ds = f'dataset{i}'
    n = normal_ic.get(ds, {})
    e = extreme_ic.get(ds, {})
    nR5 = n.get('Ret5', 0) or 0
    nR60 = n.get('Ret60', 0) or 0
    eR5 = e.get('Ret5', 0) or 0
    eR60 = e.get('Ret60', 0) or 0
    feedback[ds] = {'nR5': nR5, 'nR60': nR60, 'eR5': eR5, 'eR60': eR60}
    all_nR5.append(nR5); all_nR60.append(nR60)
    all_eR5.append(eR5); all_eR60.append(eR60)
    print(f'{ds:<12} {nR5:>8.4f} {nR60:>8.4f} {eR5:>8.4f} {eR60:>8.4f}')

print('-' * 50)
mn5, mn60, me5, me60 = np.mean(all_nR5), np.mean(all_nR60), np.mean(all_eR5), np.mean(all_eR60)
print(f'{\"MEAN\":<12} {mn5:>8.4f} {mn60:>8.4f} {me5:>8.4f} {me60:>8.4f}')
print(f'Overall avg: {(mn5+mn60+me5+me60)/4:.4f}')

# 保存迭代格式
output = {'round': $ROUND, 'submission_id': '$SUB_ID', 'results': feedback, 'timings': timings}
with open('feedback_state/iter_${ROUND}.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\n📁 保存: feedback_state/iter_${ROUND}.json')
print(f'下一步: python3 scripts/submit_iterate.py next --round $ROUND')
"
        break
    fi
done
