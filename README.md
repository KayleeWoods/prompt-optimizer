# Prompt Optimizer

Auto prompt optimization loop with a web UI: describe a task roughly → AI generates a system prompt + test cases → a bundled eval harness scores it against real cases → failures feed a revision pass → iterate until the best score. Output: a proven system prompt with an eval report, not a guess.

## How it works

```
task description
   ├─> gen cases  (JSONL, harness-compatible checks)
   ├─> gen prompt v1
   └─> loop N rounds:
         eval(prompt) via harness  ─> score
         revise(prompt, weak cases + outputs) ─> next prompt
       keep best score → best_prompt.txt + result.json
```

Works with any OpenAI-compatible chat endpoint (OpenAI, local gateways, resellers).

## Run

```bash
cp .env.example .env    # add your key
python3 backend.py      # http://127.0.0.1:8901
```

Python 3.10+, stdlib only — no pip installs. The eval harness is bundled under `harness/` (override with `PROMPT_EVAL_HARNESS` if you want your own).

### Config (env or `.env`)

| Var | Default |
|---|---|
| `OPTIMIZER_API_BASE` | `https://api.openai.com/v1` |
| `OPTIMIZER_KEY_ENV` | `OPENAI_API_KEY` |
| `OPTIMIZER_KEY_FILE` | `.env` (repo root) |
| `OPTIMIZER_MODEL` | `gpt-4o-mini` |
| `OPTIMIZER_HOST` / `OPTIMIZER_PORT` | `127.0.0.1` / `8901` |
| `PROMPT_EVAL_HARNESS` | `harness/harness.py` |

### API

- `POST /api/optimize` `{task, rounds?, cases?, model?, repeat?}` → `202 {job_id}`
- `GET /api/jobs/<id>` → status, step, log, history, best_score
- `GET /api/jobs/<id>/result` → best prompt, history, report (409 until done)
- `GET /api/health`

## Test cases

Cases are JSONL with objective checks (regex, must_contain, json_valid, length, LLM judge, ...). Full schema and worked examples: [`harness/case-format.md`](harness/case-format.md). The optimizer generates its own cases from your task description — edit `runs/<job_id>/cases.jsonl` and re-run if you want stricter ones.

## Artifacts

Per job under `runs/<job_id>/`: `cases.jsonl`, `prompt_r*.txt`, `runs/<ts>-r*/` (per-round harness outputs), `best_prompt.txt`, `result.json`.

## Behavior notes

- **Early stop**: if every case scores ~1.0, revision loops skip (nothing to fix).
- **Weak case triage**: bottom-3 cases with mean < 0.99 feed the revision prompt with real model outputs attached.
- **Keep best**: a revision that scores lower never replaces the incumbent — the output is the best-scoring prompt seen, not the last one.
