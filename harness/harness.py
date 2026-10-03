#!/usr/bin/env python3
"""Prompt eval harness — run, compare, report.

Runs a JSONL test suite against a prompt on an OpenAI-compatible endpoint,
scores outputs with rubric checks, writes run artifacts, and diffs runs.

CLI:
  harness.py run     --cases cases.jsonl --prompt prompt.txt --out runs/ --label v1
  harness.py compare --baseline runs/2026...-v1 --candidate runs/2026...-v2
  harness.py report  --run runs/2026...-v1

API (import via importlib, never exec()):
  load_cases(path) -> list[dict]
  score_output(text, checks) -> float
  run_suite(...) -> dict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_JUDGE_THRESHOLD = 0.7


# ---------------------------------------------------------------- cases #

def load_cases(path: str | Path, include_holdout: bool = False) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if "id" not in obj or "input" not in obj:
                raise ValueError(f"case missing 'id' or 'input': {obj}")
            obj.setdefault("checks", [])
            if obj.get("holdout") and not include_holdout:
                continue
            cases.append(obj)
    return cases


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- scoring #

def _check_pass(check: dict, text: str, judge_fn=None) -> bool:
    t = check.get("type")
    if t == "regex":
        m = bool(re.search(check["pattern"], text, re.S))
        return m if check.get("must_match", True) else not m
    if t == "must_contain":
        return check["text"] in text
    if t == "must_not_contain":
        return check["text"] not in text
    if t == "must_contain_any":
        return any(x in text for x in check["texts"])
    if t == "must_contain_all":
        return all(x in text for x in check["texts"])
    if t == "json_valid":
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return False
        keys = check.get("require_keys", [])
        return isinstance(obj, dict) and all(k in obj for k in keys)
    if t == "starts_with":
        return text.startswith(check["text"])
    if t == "max_length":
        return len(text) <= int(check["limit"])
    if t == "min_length":
        return len(text) >= int(check["limit"])
    if t == "judge":
        if judge_fn is None:
            return False
        return judge_fn(check["criteria"], text) >= check.get("threshold", DEFAULT_JUDGE_THRESHOLD)
    raise ValueError(f"unknown check type: {t}")


def score_output(text: str, checks: list[dict], judge_fn=None) -> float:
    """Weighted pass ratio in 0..1. Empty checks -> non-emptiness only."""
    if not text:
        return 0.0
    if not checks:
        return 1.0
    total = passed = 0.0
    for c in checks:
        w = float(c.get("weight", 1.0))
        total += w
        if _check_pass(c, text, judge_fn):
            passed += w
    return passed / total if total else 0.0


# ---------------------------------------------------------------- model I/O #

def _resolve_api_key(api_key_env: str, api_key_file: str | None) -> str:
    if api_key_file:
        with open(api_key_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{api_key_env}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise RuntimeError(f"{api_key_env} not found in {api_key_file}")
    val = os.environ.get(api_key_env, "")
    if not val:
        raise RuntimeError(f"env var {api_key_env} is empty")
    return val


def call_model(prompt: str, user: str, model: str, base_url: str,
               api_key: str, timeout: float = 120.0, temperature: float = 0.0) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"] or ""


def _make_judge_fn(judge_model: str | None, model: str, base_url: str,
                   api_key: str, timeout: float):
    def judge(criteria: str, text: str) -> float:
        sys_msg = ("You are a strict grading assistant. Score how well the OUTPUT "
                   "meets the CRITERIA. Reply with ONLY a number from 0.0 to 1.0.")
        user_msg = f"CRITERIA:\n{criteria}\n\nOUTPUT:\n{text}"
        raw = call_model(sys_msg, user_msg, judge_model or model, base_url, api_key, timeout)
        m = re.search(r"0?\.\d+|1\.0|0|1", raw.strip())
        return float(m.group(0)) if m else 0.0
    return judge


# ---------------------------------------------------------------- run #

def run_suite(cases_path: str, prompt_path: str, out_dir: str, label: str,
              model: str, base_url: str, api_key_env: str,
              api_key_file: str | None = None, repeat: int = 3,
              judge_model: str | None = None, temperature: float = 0.0,
              timeout: float = 120.0, include_holdout: bool = False) -> dict:
    cases = load_cases(cases_path, include_holdout)
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    api_key = _resolve_api_key(api_key_env, api_key_file)
    judge_fn = _make_judge_fn(judge_model, model, base_url, api_key, timeout)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_dir) / f"{ts}-{label or 'run'}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        scores, errors = [], []
        for i in range(repeat):
            try:
                out = call_model(prompt, case["input"], model, base_url,
                                 api_key, timeout, temperature)
                scores.append(score_output(out, case["checks"], judge_fn))
            except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError) as e:
                errors.append(f"{type(e).__name__}: {e}")
                scores.append(0.0)
        rec = {
            "id": case["id"],
            "tags": case.get("tags", []),
            "holdout": bool(case.get("holdout")),
            "scores": scores,
            "mean": round(statistics.mean(scores), 4) if scores else 0.0,
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
            "errors": errors,
        }
        results.append(rec)
        print(f"[{case['id']}] mean={rec['mean']:.3f} min={rec['min']:.3f} "
              f"max={rec['max']:.3f}" + (f" errors={len(errors)}" if errors else ""))

    meta = {
        "label": label, "model": model, "judge_model": judge_model,
        "base_url": base_url, "temperature": temperature,
        "repeat": repeat, "include_holdout": include_holdout,
        "cases_path": str(cases_path), "prompt_path": str(prompt_path),
        "cases_sha256": sha256_file(cases_path),
        "prompt_sha256": sha256_file(prompt_path),
        "created": ts, "tool": "prompt-eval-harness 0.1.0",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")

    overall = round(statistics.mean([r["mean"] for r in results]), 4) if results else 0.0
    lines = [f"# Run report — {label or 'run'}", "",
             f"- model: `{model}` | repeat: {repeat} | cases: {len(results)}",
             f"- overall mean score: **{overall}**", "",
             "| case | mean | min | max | errors |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['mean']:.3f} | {r['min']:.3f} | "
                     f"{r['max']:.3f} | {len(r['errors'])} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nrun dir: {run_dir}\noverall mean: {overall}")
    return {"run_dir": str(run_dir), "overall": overall, "results": results}


# ---------------------------------------------------------------- compare #

def compare_runs(baseline_dir: str, candidate_dir: str, threshold: float = 0.05) -> dict:
    def load(d: str) -> dict:
        p = Path(d)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        recs = {}
        with open(p / "results.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                recs[r["id"]] = r
        return {"meta": meta, "recs": recs}

    b, c = load(baseline_dir), load(candidate_dir)
    warnings = []
    if b["meta"]["cases_sha256"] != c["meta"]["cases_sha256"]:
        warnings.append("case sets differ between runs (cases_sha256 mismatch)")
    if b["meta"]["model"] != c["meta"]["model"]:
        warnings.append("models differ between runs — comparison is not apples-to-apples")
    if b["meta"]["prompt_sha256"] == c["meta"]["prompt_sha256"]:
        warnings.append("prompts are identical — no diff expected")

    regressions, improvements, unchanged = [], [], []
    all_ids = sorted(set(b["recs"]) | set(c["recs"]))
    for cid in all_ids:
        br, cr = b["recs"].get(cid), c["recs"].get(cid)
        if br is None or cr is None:
            warnings.append(f"case '{cid}' present in only one run")
            continue
        delta = round(cr["mean"] - br["mean"], 4)
        row = {"id": cid, "baseline": br["mean"], "candidate": cr["mean"], "delta": delta}
        if delta <= -threshold:
            regressions.append(row)
        elif delta >= threshold:
            improvements.append(row)
        else:
            unchanged.append(row)

    out = {"warnings": warnings, "regressions": regressions,
           "improvements": improvements, "unchanged": unchanged, "threshold": threshold}
    print(f"threshold: ±{threshold}")
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"regressions: {len(regressions)} | improvements: {len(improvements)} | "
          f"unchanged: {len(unchanged)}")
    for r in regressions:
        print(f"  REGRESSED {r['id']}: {r['baseline']:.3f} -> {r['candidate']:.3f} "
              f"({r['delta']:+.3f})")
    for r in improvements:
        print(f"  improved  {r['id']}: {r['baseline']:.3f} -> {r['candidate']:.3f} "
              f"({r['delta']:+.3f})")
    return out


# ---------------------------------------------------------------- report #

def report_run(run_dir: str) -> dict:
    p = Path(run_dir)
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in
            (p / "results.jsonl").read_text(encoding="utf-8").splitlines() if l]
    overall = round(statistics.mean([r["mean"] for r in recs]), 4) if recs else 0.0
    by_tag: dict[str, list[float]] = {}
    for r in recs:
        for t in r.get("tags", []) or ["untagged"]:
            by_tag.setdefault(t, []).append(r["mean"])
    print(f"label: {meta.get('label')} | model: {meta.get('model')} | "
          f"repeat: {meta.get('repeat')}")
    print(f"cases: {len(recs)} | overall mean: {overall} | created: {meta.get('created')}")
    for t, vals in sorted(by_tag.items()):
        print(f"  tag [{t}]: mean={statistics.mean(vals):.3f} over {len(vals)} case(s)")
    worst = sorted(recs, key=lambda r: r["mean"])[:3]
    for r in worst:
        print(f"  weak: {r['id']} mean={r['mean']:.3f}")
    return {"meta": meta, "overall": overall, "records": recs}


# ---------------------------------------------------------------- CLI #

def main() -> None:
    ap = argparse.ArgumentParser(description="prompt eval harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a test suite against a prompt")
    r.add_argument("--cases", required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", default="runs")
    r.add_argument("--label", default="")
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", default=DEFAULT_BASE_URL)
    r.add_argument("--api-key-env", required=True)
    r.add_argument("--api-key-file", default=None)
    r.add_argument("--repeat", type=int, default=3)
    r.add_argument("--judge-model", default=None)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--timeout", type=float, default=120.0)
    r.add_argument("--include-holdout", action="store_true")

    c = sub.add_parser("compare", help="diff two runs and flag regressions")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--threshold", type=float, default=0.05)

    p = sub.add_parser("report", help="summarize one run")
    p.add_argument("--run", required=True)

    args = ap.parse_args()
    if args.cmd == "run":
        run_suite(args.cases, args.prompt, args.out, args.label, args.model,
                  args.base_url, args.api_key_env, args.api_key_file,
                  args.repeat, args.judge_model, args.temperature,
                  args.timeout, args.include_holdout)
    elif args.cmd == "compare":
        out = compare_runs(args.baseline, args.candidate, args.threshold)
        sys.exit(2 if out["regressions"] else 0)
    elif args.cmd == "report":
        report_run(args.run)


if __name__ == "__main__":
    main()
