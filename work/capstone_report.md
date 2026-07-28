# Capstone Report — Growth / Recovery / Momentum Prediction

- **Author:** danielajetunmobi
- **Lane:** Freestyle — Growth / Recovery / Momentum Prediction
- **Repo:** https://github.com/danielajetunmobi/flyrank
- **Date:** 2026-07-28 (living document — updated as each assignment lands)

> **Status:** sections 1, 2 and 8 reflect completed work (ML-02 → ML-05). Sections 3–7
> are scaffolded with what is currently known and marked **NOT YET DONE** where the
> assignment that produces them (ML-06 → ML-10) has not been reached. Nothing below
> claims a result that has not been computed.

---

## 1. Problem framing

**Decision supported.** Which pages a senior SEO specialist should review first, because
they show early signs of decline, recovery, or unexpected momentum.

**Unit of analysis.** One row = one content item, evaluated at a fixed decision-point date.
Built from `fact_content_daily_performance` (daily grain, aggregated to one row per page);
the 30k-row starter CSV has no daily granularity and cannot express a future window at all.

**Output.** A ranked, explained flag list. A model probability per page doubles as the sort
key for the review queue.

**Action a human takes.** The model never acts. It flags a page with the reasons it was
flagged; a specialist reads those, then accepts or declines before anything happens to the page.

**Cost of a wrong call — asymmetric.**
- *False flag*: a specialist spends review time on a page that did not need it. Recoverable.
- *Missed signal*: a real decline or recovery is never surfaced, and the window to act passes.
Not recoverable.

The missed signal is the more expensive error, so **recall matters here, not just precision at
the top of the list**. This drove the metric choice in section 5.

**Why ML rather than a fixed rule.** A single threshold cannot separate a real 20% drop on a
high-traffic page from meaningless noise on a page that went from 2 impressions to 1 — the
starter data contains `trend_pct` values as extreme as 44,900%, produced entirely by tiny
denominators. Weighing percentage change *together with* prior volume, position, age and
freshness is the part a hand-written rule does badly. Verified separately: even *defining*
"already declining" is design-sensitive — a 30-vs-30-day window and a 45-vs-45-day window
disagree on **21.9%** of pages.

*Source: `work/notebooks/w01_research_question.ipynb`, `w02_ml_task_framing.ipynb`.*

---

## 2. Data safety

**Data used.** `FlyRank/internship-warehouse` (gated Hugging Face release, build
`v20260703`): `fact_content_daily_performance`, `dim_content`, `dim_clients`. Access token
supplied at runtime via a gitignored `.env` or `getpass` — never written into a cell, since
this repo is public.

**Deliberately excluded, and why.**

| Excluded | Reason |
|---|---|
| `future_change_pct`, `future_impressions`, and the three target columns | The label, or the window it is computed from |
| All of `fact_content_query_90d` | Its fixed 90-day window (2026-04-02 → 2026-06-30) **overlaps this lane's label window** — any column from it leaks the future |
| `last_optimized_date`, `optimization_eligible_date` | 87.8% missing; sparsity + naming suggest they populate only when FlyRank's own system acted on a page — the product-decision-as-feature trap. Excluded until independently verified |
| `provider_used`, `model_used` | Marked "not a model feature" in the data dictionary |
| `ai_copilot`, `ai_claude`, `ai_meta`, `ai_other` | 100.0% zero across every content item in the window — zero variance |
| `sessions_ai`, `ai_chatgpt/perplexity/gemini`, `sessions_referral/social/paid` | Traffic channels with no plausible link to a GSC-impression-based label (sparsity was secondary) |
| `client_hash_id`, `content_hash_id`, `keyword_hash_id`, `url_hash_id` | Pseudonyms — grouping, joining and splitting only, never features |

**Leakage risks considered.** Timeline verified: every feature is drawn from
2025-12-31 → 2026-03-30; the label window opens 2026-03-31; no overlap. Three explicit
attacks were run and behave as they should — injecting the label-generating ratio drives AUC
to 0.92, injecting a raw future count leaks weakly, and a random split inflates the score by
+0.169 over a client-grouped split. Feature importances were inspected for a suspiciously
dominant term; the largest (`char_count` / `word_count`, opposite signs) is **multicollinearity**
(r = 0.934), not a leak.

**Client-identifying content:** none. Swept the notebooks for URLs and raw queries — clean.
The only identifiers that appear are pseudonymous `content_*` / `client_*` hashes in `df.head()`
output, which `DATA_USE.md` explicitly permits, shown to demonstrate grain rather than as a
row-level dump.

*Source: `work/notebooks/w03_data_contract.ipynb`, `w03_feature_leakage_check.ipynb`.*

---

## 3. Baseline

**NOT YET DONE** — produced by ML-07 (`w04_baseline_score.ipynb`).

Planned: a transparent rule score in the spirit of the reference pipeline's
`stale_visible_page` / `declining_with_demand` reason codes, scored on the *same*
client-grouped split and the *same* metric as any model, so the comparison is fair.

---

## 4. Model / analysis

**Target definition.** Three binary targets, each defined only on the population where the
question is meaningful, rather than one blurred multi-class label:

| Target | Population | Positive when |
|---|---|---|
| `future_decline` | was **not** already declining | future change ≤ −20% |
| `future_recovery` | **was** already declining | future change ≥ +20% |
| `future_momentum` | was **not** already declining | future change ≥ +20% |

**Windows.** Features from the trailing 90 days before the decision point; "already declining"
from a 30-vs-30-day split of that window, matching FlyRank's own `trend_pct` convention rather
than an invented one; label from the 30 days after. Future change compares the future 30-day
daily rate against the **recent 30-day** daily rate, not a 90-day average — verified that using
the 90-day average hides real recoveries behind stale history (recovery rate 18.3% → 28.2%).

**Feature list.** 25 features: impression-weighted average position, logged heavy-tailed counts
(impressions, clicks, sum-position, GA4 sessions, engaged sessions, scroll events, search volume,
backlinks), `prior_trend_pct`, content age and freshness, `ctr` / `engagement_rate` / `scroll_rate`,
`days_with_impressions` / `days_with_sessions`, `word_count` / `char_count` / `category_count`, and
five `has_*` missingness flags. Categorical encoding (`main_intent`, `content_type`,
`competition_level`) is deferred to the modelling stage.

**Missing-value policy.** Flag first, then fill — never a blind zero where zero is a meaningful
extreme. `gsc_avg_position` is median-filled behind `has_position_data` because 0 would read as
*better than rank 1*; `ctr` is safely zero-filled because 56.8% of tracked pages genuinely have
0 CTR.

**Model choice:** **NOT YET DONE** — ML-08 (`w05_model.ipynb`). Only a logistic-regression probe
has been run so far, as a leakage harness rather than as a candidate model.

*Source: `work/notebooks/w02_ml_task_framing.ipynb`, `w03_feature_leakage_check.ipynb`.*

---

## 5. Evaluation

**Split.** `GroupShuffleSplit` on `client_hash_id` — whole clients held out, so no client's
pages appear on both sides. Justified empirically, not by assertion: the same features on a
random row split score **+0.169 AUC higher**, and that gap is memorised client structure, not skill.

**Metric.** Precision@50, matching a specialist's realistic weekly review capacity, reported
next to the base rate and recall (because section 1 established that missed signals cost more
than false flags).

**Result so far — the two metrics disagree, and that is the headline.**

| Metric | Value | Verdict |
|---|---|---|
| AUC (whole ranking) | 0.426 | **worse** than the 0.500 chance level |
| Precision@50 (top of queue) | 0.580 vs. 33.7% base rate | **1.72x** better than random |
| Recall@50 | 0.66% | catches 29 of 4,403 real declines |

Both numbers are correct; they measure different things. Across the full ranking of 13,078
held-out pages the ordering is slightly inverted. But the **top 50** — the only part a specialist
ever reads — is genuinely enriched: 29 real declines where random selection returns ~17, roughly
3.6 standard deviations above chance at this base rate.

Precision@50 is the metric that governs, because section 1 committed to it *before* any result
was seen, for the concrete reason that it matches a specialist's weekly review capacity. Had this
work reported AUC alone it would have concluded "no signal, stop" — which would have been wrong.

The counterweight is recall. Section 1 argued a missed decline costs more than a false flag; a
queue that surfaces 0.66% of real declines is precise at the top but barely dents the problem it
was built for. Reconciling those two facts is a live question for ML-09.

**Why the ranking as a whole is sub-chance — three open hypotheses, none yet tested.** These are
signal-audit questions, so they are recorded here and answered in ML-06 rather than asserted now:

1. **The relationship flips between clients** — a pattern holding in the training clients may
   point the wrong way in held-out ones. *Test:* fit per client, compare coefficient signs. The
   +0.170 random-vs-grouped gap is at least consistent with this.
2. **The label carries little page-level signal** — *test:* correlate `prior_trend_pct` with
   `future_change_pct`, and compare decline rates across prior-trend buckets. A flat decline rate
   regardless of prior behaviour would mean the target is near-coin-flip by construction.
3. **Cohort selection** — the cohort keeps only pages with `trend_recent_impr > 0`, i.e. active in
   the last 30 days. That filter may preferentially select pages having an unusually strong month,
   and unusual months end. *Test:* re-compute the base rate with the filter relaxed; compare
   established vs. newly-active pages.

**No number is quoted for any of these**, because no cell in this repo computes them yet.
Producing them is ML-06's opening task.

**Error analysis:** **NOT YET DONE** — ML-09 (`w06_validation_audit.ipynb`).

*Source: `work/notebooks/w03_feature_leakage_check.ipynb`.*

---

## 6. Interpretation

**NOT YET DONE** — ML-06 (`w04_signal_audit.ipynb`) and ML-09.

What is known: feature coefficients are dominated by `char_count` (−1.42) and `word_count`
(+1.23), which is a collinearity artefact (r = 0.934) rather than a finding about content
length. No feature shows a suspicious standalone dominance.

The honest headline so far is **mixed, and metric-dependent**: no page-level signal is
demonstrable across the ranking as a whole (sub-chance AUC), but the top of the queue is enriched
1.72x over the base rate. Whether that top-end lift is a small real effect or an artefact of how
the cohort was selected is unresolved — see the three hypotheses in section 5. Per the lane guide,
a well-understood "no effect" is a valid result, and so is a well-understood "small effect, only
at the top"; neither can be claimed from a single split at a single decision point.

---

## 7. Recommendation

**NOT YET DONE** — ML-10 (`w07_action_playbook.ipynb`).

No ranked recommendations can honestly be issued yet: no model has beaten a baseline, and no
baseline has been built. Publishing a queue now would be presenting noise as decision support.

---

## 8. Reproducibility

**Environment.** `pip install -r requirements.txt` (pandas, numpy, scikit-learn, matplotlib,
reportlab, duckdb, huggingface_hub) plus `python-dotenv` for local `.env` loading.

**Data access.** Requires a Hugging Face account with access to the gated
`FlyRank/internship-warehouse` dataset (request access, accept terms — instant). Supply a READ
token either as `HF_TOKEN` in a local `.env` (gitignored) or at the `getpass` prompt. Never in a cell.

**Seeds.** `random_state=42` throughout — `GroupShuffleSplit`, `train_test_split`, and
`LogisticRegression`.

**Run order.** `w01_research_question` → `w02_ml_task_framing` → `w03_data_contract` →
`w03_feature_leakage_check`. Each is self-contained and re-downloads what it needs.

**Known reproducibility gaps (honest list):**
1. **The computed feature vector is not cached** — `work/outputs/` is empty, so the aggregation,
   joins and feature engineering are recomputed on every run. (The underlying Parquet files *are*
   cached automatically by `huggingface_hub`, so this costs seconds of recompute, not a repeat
   download.) Caching is deliberately **deferred until after ML-06**: hypothesis 3 in section 5
   proposes relaxing the `trend_recent_impr > 0` cohort filter, and a cache written now would
   bake that filter in — a later session could load it and test the hypothesis against
   already-filtered data. `work/outputs/*.parquet` is gitignored and CI hard-fails on committed
   Parquet, so caching will be safe once the cohort definition is settled.
2. **The remote host is intermittently unreliable.** Repeated `ZSTD Decompression failure`
   errors on `hf://` reads during development; a re-run may need retries.
3. **Single decision point.** Every number rests on 2026-03-31. No walk-forward across multiple
   decision dates, so none of these figures is known to be stable month to month.
4. **Only one of three targets is built.** `future_recovery` and `future_momentum` are defined
   but have never been modelled or evaluated.

---

> **Claims checklist.** All statements above are framed as observed / measured / directional /
> decision-support. No causal claim is made — nothing here shows that refreshing a page *causes*
> recovery, which would require an experiment this data cannot support. No claim is made about
> Google's ranking algorithm. No client-identifying details appear. Every number cited is
> reproduced by a cell in the linked notebook rather than quoted from memory.
