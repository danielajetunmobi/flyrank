# Agent instructions

Before any task in this repo: **read `skills/README.md`** — it is the router.
Find the task in its table and load exactly **one** skill (plus `skills/flyrank/flyrank-data/SKILL.md`
whenever the task touches the data). Do not load every skill; keep context small.

Ground rules for this repo:
- Search the repo before assuming something is missing or not implemented.
- One task per conversation; finish and verify before starting the next.
- Never commit datasets (CI blocks them). Never print private data, client names, or raw queries.
- The intern validates your output — end each task by running the notebook top to bottom.
- Then run `python work/tools/check_claims.py`. It fails if any number written in prose does not
  appear in the output of a cell that ran. Analysis done in a scratch script is fine; a number
  that reaches the notebook without being recomputed there is not. Every violation found so far
  was a real defect — stale figures, transposed digits, and one claim about a column that does
  not exist. Fix the number or compute it; only add to `ALLOWED` with a written reason.
