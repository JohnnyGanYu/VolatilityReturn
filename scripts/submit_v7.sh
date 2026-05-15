#!/bin/bash
# v7 提交脚本
# 用法: ./submit_v7.sh

set -e

TOKEN="7027dacbb2d44548add68a006ea99f8a"
SUBMISSION="submissions/submission_v7.tar.gz"

echo "=== v7 Submission ==="
echo "File: $SUBMISSION"
echo "Size: $(du -h $SUBMISSION | cut -f1)"
echo ""

python3 scripts/submit_to_platform.py --token "$TOKEN" --file "$SUBMISSION"
