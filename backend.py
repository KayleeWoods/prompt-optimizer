#!/usr/bin/env python3
"""Prompt Optimizer — auto prompt generation + eval loop with web UI.

From a rough task description: generates test cases (JSONL for
prompt-eval-harness) and an initial system prompt, then iterates —
evaluate, revise with failure data, keep the best. Stdlib only.

Run:  python3 backend.py        (listens on 127.0.0.1:8901)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RUNS = ROOT / "runs"

API_BASE = os.environ.get("OPTIMIZER_API_BASE", "https://api.openai.com/v1")
KEY_ENV = os.environ.get("OPTIMIZER_KEY_ENV", "OPENAI_API_KEY")
KEY_FILE = os.path.expanduser(os.environ.get("OPTIMIZER_KEY_FILE", str(ROOT / ".env")))
HARNESS = os.environ.get("PROMPT_EVAL_HARNESS", str(ROOT / "harness" / "harness.py"))
DEFAULT_MODEL = os.environ.get("OPTIMIZER_MODEL", "gpt-4o-mini")
HOST = os.environ.get("OPTIMIZER_HOST", "127.0.0.1")
PORT = int(os.environ.get("OPTIMIZER_PORT", "8901"))

CHECK_TYPES = {
    "regex": ["pattern"], "must_contain": ["text"], "must_not_contain": ["text"],
    "must_contain_any": ["texts"], "must_contain_all": ["texts"],
    "json_valid": ["require_keys"], "starts_with": ["text"],
    "max_length": ["limit"], "min_length": ["limit"],
}

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

_API_KEY_CACHE: str | None = None


def api_key() -> str:
    global _API_KEY_CACHE
    if _API_KEY_CACHE:
        return _API_KEY_CACHE
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(f"{KEY_ENV}="):
                    _API_KEY_CACHE = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return _API_KEY_CACHE
    except OSError:
        pass
    _API_KEY_CACHE = os.environ.get(KEY_ENV, "")
    if not _API_KEY_CACHE:
        raise RuntimeError(f"API key not found: {KEY_ENV} in {KEY_FILE} or env")
    return _API_KEY_CACHE


def llm_chat(model: str, system: str, user: str, temperature: float = 0.2) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key()}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------------- generation #

GEN_CASES_SYS = (
    "You design test suites for LLM system prompts. Output STRICT JSON only: an "
    "array of 3-6 cases. Each case: {\"id\": \"case-N\", \"input\": \"user turn text\", "
    "\"checks\": [{...}]}. Allowed check types and required fields: "
    "regex(pattern), must_contain(text), must_not_contain(text), "
    "must_contain_any(texts:list), must_contain_all(texts:list), "
    "json_valid(require_keys:list, optional), starts_with(text), "
    "max_length(limit:int), min_length(limit:int). Optional \"weight\" per check. "
    "Rules: every case needs >= 1 check that FAILS on a bad answer; cover format, "
    "content, and edge cases; checks must be objectively verifiable. "
    "No prose, no markdown fences, no commentary — JSON array only."
)

GEN_PROMPT_SYS = (
    "You write robust production system prompts. Given a task description, write "
    "the single best system prompt for it: explicit about output format, concrete, "
    "unambiguous, no filler, no disclaimers. Return ONLY the system prompt text — "
    "no markdown fences, no commentary."
)

REVISE_SYS = (
    "You improve system prompts using evaluation failure data. You get: the task, "
    "the current system prompt, and failing cases with the model output they "
    "produced. Diagnose WHY each failure happened (ambiguous instruction, missing "
    "format rule, wrong tone) and rewrite the prompt to fix ALL failures while "
    "preserving what already works. Return ONLY the full revised system prompt "
    "text — no markdown fences, no commentary."
)


def _extract_json_array(text: str) -> list:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON array in LLM output: {text[:200]}")
    return json.loads(text[start:end + 1])


def gen_cases(task: str, n_cases: int, model: str) -> list[dict]:
    user = f"Task description:\n{task}\n\nGenerate exactly {n_cases} cases."
    last_err = None
    for attempt in range(2):
        raw = llm_chat(model, GEN_CASES_SYS, user, temperature=0.2)
        try:
            cases = _extract_json_array(raw)
            validated = []
            seen = set()
            for i, c in enumerate(cases, 1):
                cid = str(c.get("id") or f"case-{i}")
                if cid in seen:
                    cid = f"{cid}-{i}"
                seen.add(cid)
                checks = c.get("checks") or []
                for ck in checks:
                    t = ck.get("type")
                    if t not in CHECK_TYPES:
                        raise ValueError(f"bad check type: {t}")
                    for req in CHECK_TYPES[t]:
                        if req not in ck and not (t == "json_valid"):
                            raise ValueError(f"check {t} missing field: {req}")
                validated.append({"id": cid, "input": str(c["input"]),
                                  "checks": checks, "tags": c.get("tags", [])})
            if not validated:
                raise ValueError("empty cases")
            return validated
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
    raise RuntimeError(f"case generation failed after 2 attempts: {last_err}")


def gen_prompt(task: str, model: str) -> str:
    return llm_chat(model, GEN_PROMPT_SYS,
                    f"Task description:\n{task}", temperature=0.3).strip()


def sample_output(prompt: str, case_input: str, model: str) -> str:
    out = llm_chat(model, prompt, case_input, temperature=0.0)
    return out[:600]


def revise_prompt(task: str, prompt: str, weak: list[dict], model: str) -> str:
    parts = [f"Task:\n{task}", f"\nCurrent system prompt:\n{prompt}",
             "\nFailing cases:"]
    for w in weak:
        parts.append(f"\n--- {w['id']} (mean score {w['mean']:.2f})\n"
                     f"input: {w['input']}\n"
                     f"model output: {w['output']}")
    return llm_chat(model, REVISE_SYS, "\n".join(parts), temperature=0.4).strip()


# ------------------------------------------------------------------- eval #

def run_eval(cases_path: Path, prompt_path: Path, label: str, model: str,
             out_dir: Path, repeat: int) -> tuple[Path, float]:
    cmd = [sys.executable, HARNESS, "run",
           "--cases", str(cases_path), "--prompt", str(prompt_path),
           "--model", model, "--api-key-env", KEY_ENV, "--api-key-file", KEY_FILE,
           "--repeat", str(repeat), "--out", str(out_dir), "--label", label]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if p.returncode != 0:
        raise RuntimeError(f"harness failed (rc={p.returncode}): {p.stderr[-600:]} {p.stdout[-400:]}")
    m = re.search(r"run dir: (.+)", p.stdout)
    om = re.search(r"overall mean: ([0-9.]+)", p.stdout)
    if not m or not om:
        raise RuntimeError(f"harness output unparsable: {p.stdout[-400:]}")
    return Path(m.group(1).strip()), float(om.group(1))


def load_results(run_dir: Path) -> list[dict]:
    recs = []
    with open(run_dir / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


# -------------------------------------------------------------- job loop #

def _log(job: dict, msg: str) -> None:
    job["log"].append(msg)
    print(f"[{job['id']}] {msg}")


def _worker(job_id: str, task: str, rounds: int, n_cases: int,
            model: str, repeat: int) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        job_dir = RUNS / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        _log(job, f"Generating {n_cases} test cases with {model} ...")
        cases = gen_cases(task, n_cases, model)
        cases_path = job_dir / "cases.jsonl"
        cases_path.write_text("".join(json.dumps(c) + "\n" for c in cases),
                              encoding="utf-8")
        _log(job, f"Cases ready: {len(cases)} -> {cases_path}")

        _log(job, "Writing initial system prompt (v1) ...")
        prompt = gen_prompt(task, model)

        best = {"prompt": prompt, "score": -1.0, "label": None, "run_dir": None}
        history = []

        for r in range(1, rounds + 1):
            label = f"r{r}"
            prompt_path = job_dir / f"prompt_{label}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            job["step"] = f"eval {label}"
            _log(job, f"Round {r}/{rounds}: evaluating prompt {label} "
                      f"({len(cases)} cases x {repeat} repeats) ...")
            run_dir, overall = run_eval(cases_path, prompt_path, label,
                                        model, job_dir / "runs", repeat)
            history.append({"round": r, "label": label, "score": overall,
                            "run_dir": str(run_dir)})
            job["history"] = history
            _log(job, f"Round {r}/{rounds}: score {overall:.3f}")
            if overall > best["score"]:
                best = {"prompt": prompt, "score": overall,
                        "label": label, "run_dir": str(run_dir)}
                job["best_score"] = overall
                _log(job, f"New best: {label} ({overall:.3f})")

            if r == rounds:
                break

            recs = load_results(run_dir)
            weak = sorted(recs, key=lambda x: x["mean"])[:3]
            weak = [w for w in weak if w["mean"] < 0.99]
            if not weak:
                _log(job, "All cases near-perfect — stopping early (no failures to fix)")
                break
            job["step"] = f"revise {label}"
            _log(job, f"Revising from {len(weak)} weak case(s): "
                      + ", ".join(w["id"] for w in weak))
            for w in weak:
                w["output"] = sample_output(prompt, w.get("input", "") or
                                            next(c["input"] for c in cases
                                                 if c["id"] == w["id"]), model)
            # reattach input for weak recs
            for w in weak:
                if "input" not in w:
                    w["input"] = next(c["input"] for c in cases if c["id"] == w["id"])
            prompt = revise_prompt(task, prompt, weak, model)
            if not prompt:
                _log(job, "Revision returned empty — keeping current prompt")
                prompt = best["prompt"]

        # finalize
        (job_dir / "best_prompt.txt").write_text(best["prompt"], encoding="utf-8")
        report_src = Path(best["run_dir"]) / "report.md" if best["run_dir"] else None
        report_md = report_src.read_text(encoding="utf-8") if report_src and report_src.exists() else ""
        result = {
            "task": task, "model": model, "rounds": rounds, "repeat": repeat,
            "best": {"label": best["label"], "score": round(best["score"], 4)},
            "best_prompt": best["prompt"],
            "history": [{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in h.items() if k != "run_dir"} for h in history],
            "report_md": report_md,
            "cases_path": str(cases_path),
            "best_prompt_path": str(job_dir / "best_prompt.txt"),
        }
        (job_dir / "result.json").write_text(json.dumps(result, indent=2),
                                             encoding="utf-8")
        job["result"] = result
        job["status"] = "done"
        job["step"] = "done"
        _log(job, f"Done. Best: {best['label']} score {best['score']:.3f} -> "
                  f"{job_dir / 'best_prompt.txt'}")
    except Exception as e:  # noqa: BLE001 — job boundary
        job["status"] = "error"
        job["step"] = "error"
        _log(job, f"ERROR: {type(e).__name__}: {e}")


def start_job(task: str, rounds: int, n_cases: int, model: str, repeat: int) -> str:
    job_id = uuid.uuid4().hex[:8]
    job = {"id": job_id, "status": "queued", "step": "queued",
           "log": [], "history": [], "best_score": None,
           "result": None, "task": task[:120]}
    with JOBS_LOCK:
        JOBS[job_id] = job
    t = threading.Thread(target=_worker,
                         args=(job_id, task, rounds, n_cases, model, repeat),
                         daemon=True)
    t.start()
    return job_id


# ------------------------------------------------------------------- HTTP #

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            body = (STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/health":
            self._json({"ok": True, "model_default": DEFAULT_MODEL,
                        "harness": HARNESS})
        elif path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            job_id = parts[2] if len(parts) > 2 else ""
            job = JOBS.get(job_id)
            if not job:
                self._json({"error": "job not found"}, 404)
                return
            if len(parts) > 3 and parts[3] == "result":
                if not job["result"]:
                    self._json({"error": f"job {job['status']}"}, 409)
                else:
                    self._json(job["result"])
                return
            self._json({k: job[k] for k in
                        ("id", "status", "step", "log", "history",
                         "best_score", "task")})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/api/optimize":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "bad JSON body"}, 400)
            return
        task = (body.get("task") or "").strip()
        if not task:
            self._json({"error": "task is required"}, 400)
            return
        rounds = max(1, min(int(body.get("rounds", 3)), 8))
        n_cases = max(2, min(int(body.get("cases", 5)), 12))
        repeat = max(1, min(int(body.get("repeat", 2)), 5))
        model = (body.get("model") or DEFAULT_MODEL).strip()
        job_id = start_job(task, rounds, n_cases, model, repeat)
        self._json({"job_id": job_id}, 202)


def main() -> None:
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"prompt-optimizer on http://{HOST}:{PORT}  model={DEFAULT_MODEL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
