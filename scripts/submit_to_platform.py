#!/usr/bin/env python3
"""
自动提交到平台 + 轮询结果。

用法：
    python scripts/submit_to_platform.py --token YOUR_TOKEN --file submissions/submission_round1.tar.gz

    # 只查询某次提交的结果
    python scripts/submit_to_platform.py --token YOUR_TOKEN --query SUBMISSION_ID

    # 查看提交历史
    python scripts/submit_to_platform.py --token YOUR_TOKEN --history
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

BASE_URL = "http://<PLATFORM_HOST>:8000"
POLL_INTERVAL = 30  # 每30秒查询一次结果
MAX_POLL_TIME = 3900  # 最多等65分钟（1小时执行 + 5分钟余量）


def submit(token: str, filepath: str) -> str:
    """提交文件到平台，返回 submission_id。"""
    url = f"{BASE_URL}/api/submit"
    
    if not os.path.exists(filepath):
        print(f"ERROR: 文件不存在: {filepath}")
        sys.exit(1)
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"提交文件: {filepath} ({size_mb:.1f} MB)")
    
    with open(filepath, "rb") as f:
        files = {"submission": (os.path.basename(filepath), f)}
        data = {"token": token}
        
        print("上传中...")
        try:
            resp = requests.post(url, data=data, files=files, timeout=900)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise RuntimeError(f"Upload failed (network): {type(e).__name__}")
    
    if resp.status_code == 200:
        result = resp.json()
        submission_id = result.get("submission_id", "unknown")
        print(f"✅ 提交成功! ID: {submission_id}")
        return submission_id
    else:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        print(f"❌ 提交失败: {detail}")
        raise RuntimeError(f"Submit failed: {detail}")


def query_result(token: str, submission_id: str) -> dict:
    """查询提交结果。网络异常时返回 None。"""
    url = f"{BASE_URL}/api/result/{submission_id}?token={token}"
    try:
        resp = requests.get(url, timeout=30)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"  查询失败 (网络): {type(e).__name__}")
        return None
    
    if resp.status_code == 200:
        return resp.json()
    else:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        print(f"查询失败: {detail}")
        return None


def query_history(token: str):
    """查询提交历史。"""
    url = f"{BASE_URL}/api/submissions?token={token}&page=1&per_page=20"
    resp = requests.get(url, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        submissions = data.get("submissions", [])
        print(f"提交历史 (共 {data.get('total', 0)} 次):")
        print(f"{'ID':<40} {'Status':<12} {'Time'}")
        print("-" * 80)
        for s in submissions:
            print(f"{s['id']:<40} {s['status']:<12} {s.get('created_at', '—')}")
        return data
    else:
        print(f"查询失败: {resp.text}")
        return None


def poll_result(token: str, submission_id: str) -> dict:
    """轮询等待结果。"""
    print(f"\n等待结果 (ID: {submission_id})...")
    print(f"每 {POLL_INTERVAL} 秒查询一次，最多等 {MAX_POLL_TIME//60} 分钟\n")
    
    start_time = time.time()
    
    while time.time() - start_time < MAX_POLL_TIME:
        result = query_result(token, submission_id)
        
        if result is None:
            time.sleep(POLL_INTERVAL)
            continue
        
        status = result.get("status", "unknown")
        elapsed = time.time() - start_time
        
        if status == "pending":
            print(f"  [{elapsed:.0f}s] 排队中...", end="\r")
        elif status == "running":
            print(f"  [{elapsed:.0f}s] 运行中...", end="\r")
        elif status == "completed":
            print(f"\n✅ 运行完成! (耗时 {elapsed:.0f}s)")
            print_result(result)
            save_result(result, submission_id)
            return result
        elif status in ("failed", "timeout"):
            print(f"\n❌ 运行{status}!")
            if result.get("error_message"):
                print(f"错误信息:\n{result['error_message'][:500]}")
            return result
        else:
            print(f"  [{elapsed:.0f}s] 状态: {status}", end="\r")
        
        time.sleep(POLL_INTERVAL)
    
    print(f"\n⚠️ 超过最大等待时间 ({MAX_POLL_TIME//60} 分钟)，请稍后手动查询。")
    return None


def print_result(result: dict):
    """打印评测结果。"""
    details = result.get("details")
    timings = result.get("timings")
    status = result.get("status", "unknown")
    
    print(f"\n状态: {status}")
    
    if not details:
        if result.get("error_message"):
            print(f"错误: {result['error_message'][:500]}")
        return
    
    # 打印耗时
    if timings:
        print(f"\n耗时:")
        print(f"  特征生成: {timings.get('factor_seconds', 0):.1f}s")
        print(f"  信号推理: {timings.get('predict_seconds', 0):.1f}s")
    
    # 打印 IC
    normal_ic = details.get("normal_ic", {})
    extreme_ic = details.get("extreme_ic", {})
    
    if normal_ic or extreme_ic:
        print(f"\n{'Dataset':<12} {'nR5':>8} {'nR60':>8} {'eR5':>8} {'eR60':>8}")
        print("-" * 50)
        
        all_nR5, all_nR60, all_eR5, all_eR60 = [], [], [], []
        
        for i in range(30):
            ds = f"dataset{i}"
            n = normal_ic.get(ds, {})
            e = extreme_ic.get(ds, {})
            
            nR5 = n.get("Ret5", float("nan"))
            nR60 = n.get("Ret60", float("nan"))
            eR5 = e.get("Ret5", float("nan"))
            eR60 = e.get("Ret60", float("nan"))
            
            if nR5 == nR5: all_nR5.append(nR5)
            if nR60 == nR60: all_nR60.append(nR60)
            if eR5 == eR5: all_eR5.append(eR5)
            if eR60 == eR60: all_eR60.append(eR60)
            
            print(f"{ds:<12} {nR5:>8.4f} {nR60:>8.4f} {eR5:>8.4f} {eR60:>8.4f}")
        
        print("-" * 50)
        import numpy as np
        print(f"{'MEAN':<12} {np.mean(all_nR5):>8.4f} {np.mean(all_nR60):>8.4f} "
              f"{np.mean(all_eR5):>8.4f} {np.mean(all_eR60):>8.4f}")


def save_result(result: dict, submission_id: str):
    """保存结果到 feedback_state/ 供迭代使用。"""
    details = result.get("details", {})
    normal_ic = details.get("normal_ic", {})
    extreme_ic = details.get("extreme_ic", {})
    
    # 转换为迭代脚本需要的格式
    feedback = {}
    for i in range(30):
        ds = f"dataset{i}"
        n = normal_ic.get(ds, {})
        e = extreme_ic.get(ds, {})
        feedback[ds] = {
            "nR5": n.get("Ret5", 0),
            "nR60": n.get("Ret60", 0),
            "eR5": e.get("Ret5", 0),
            "eR60": e.get("Ret60", 0),
        }
    
    # 自动保存
    feedback_dir = Path("feedback_state")
    feedback_dir.mkdir(exist_ok=True)
    
    # 找到下一个 iter 编号
    existing = list(feedback_dir.glob("iter_*.json"))
    next_num = len(existing) + 1
    
    output = feedback_dir / f"iter_{next_num}.json"
    save_data = {
        "round": next_num,
        "submission_id": submission_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": feedback,
        "timings": result.get("timings", {}),
    }
    
    with open(output, "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n📁 反馈已自动保存: {output}")
    print(f"   下一步: python scripts/submit_iterate.py next --round {next_num}")


def main():
    parser = argparse.ArgumentParser(description="提交到华东杯评测平台")
    parser.add_argument("--token", required=True, help="参赛 token")
    parser.add_argument("--file", default=None, help="提交包路径 (.tar.gz)")
    parser.add_argument("--query", default=None, help="查询指定 submission ID 的结果")
    parser.add_argument("--history", action="store_true", help="查看提交历史")
    parser.add_argument("--no-poll", action="store_true", help="提交后不等待结果")
    args = parser.parse_args()
    
    if args.history:
        query_history(args.token)
    elif args.query:
        result = query_result(args.token, args.query)
        if result:
            print_result(result)
    elif args.file:
        submission_id = submit(args.token, args.file)
        if not args.no_poll:
            poll_result(args.token, submission_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
