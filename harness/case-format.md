# Test Case Format — prompt-eval-harness

One JSON object per line (JSONL). UTF-8. Comments allowed: lines starting with `#` are skipped.

## Schema

```json
{"id": "case-01", "input": "User turn text sent after the system prompt", "checks": [{"type": "regex", "pattern": "(?i)json", "weight": 1.0}, {"type": "must_contain", "text": "schema_version"}], "tags": ["core", "format"], "holdout": false}
```

| Field | Required | Meaning |
|:------|:---------|:--------|
| `id` | yes | Stable identifier. Used to join cases across runs. Never reuse an id for a different task. |
| `input` | yes | The user message. The prompt under test is sent as the system message. |
| `checks` | yes | Ordered list of check objects. Empty list = only non-emptiness is measured. |
| `tags` | no | Free-form labels; `report --run` groups summaries by tag. |
| `holdout` | no | If true, `run` skips it unless `--include-holdout`. Keeps clean cases for the final lock-in run. |

## Check Objects

Every check: `{"type": ..., "weight": 1.0}` — weight defaults to 1.0 and is relative within the case.

| type | Extra fields | Passes when |
|:-----|:-------------|:-----------|
| `regex` | `pattern` (required), `must_match` (default true) | `re.search(pattern, output, re.S)` matches (or does not match if `must_match: false`) |
| `must_contain` | `text` | Case-sensitive substring present |
| `must_not_contain` | `text` | Case-sensitive substring absent |
| `must_contain_any` | `texts` (list) | At least one substring present |
| `must_contain_all` | `texts` (list) | All substrings present |
| `json_valid` | `require_keys` (list, optional) | Output parses as JSON; all `require_keys` present at top level |
| `starts_with` | `text` | Output begins with `text` |
| `max_length` | `limit` | `len(output) <= limit` |
| `min_length` | `limit` | `len(output) >= limit` |
| `judge` | `criteria` (natural-language rubric) | LLM judge scores >= judge threshold (default 0.7). Uses `--judge-model` if set, else the run model. |

## Case score

- Case score = sum(weight of passed checks) / sum(all weights), range 0..1.
- Empty `checks` list: score is 1.0 when output is non-empty, else 0.0.
- A case with a failed API call scores 0.0 and is flagged `error` in `results.jsonl` — never silently averaged in.

## Suite tips

- 5-15 cases catches most regressions; more is better only if each adds a distinct behavior.
- Every case needs at least one check that would fail on a bad output — a suite where everything passes both prompts measures nothing.
- Split 20% into `holdout: true` cases and only run them before locking a version.
- Regex checks use Python `re` with DOTALL; escape literal dots, anchor when the position matters.
- `judge` checks cost one extra model call each per repeat — keep them few and specific.

## Worked example (3 cases)

```json
{"id": "format-json", "input": "Return the user record as JSON only.", "checks": [{"type": "json_valid", "require_keys": ["name", "id"], "weight": 2.0}, {"type": "regex", "pattern": "^\\{.*\\}$", "weight": 1.0}], "tags": ["format"]}
{"id": "no-apology", "input": "Explain how rate limiting works.", "checks": [{"type": "must_not_contain", "text": "I'm sorry", "weight": 1.0}, {"type": "min_length", "limit": 100, "weight": 1.0}], "tags": ["style"]}
{"id": "judged-clarity", "input": "Explain CAP theorem to a junior dev.", "checks": [{"type": "judge", "criteria": "Clear, accurate, uses one concrete example, no jargon without explanation.", "weight": 2.0}], "tags": ["quality"], "holdout": true}
```
