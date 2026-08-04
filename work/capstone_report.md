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
extreme. `has_*` indicators are computed before any fill, so "unknown" and "genuinely zero" stay
distinguishable; `ctr` is safely zero-filled because 56.8% of tracked pages genuinely have 0 CTR.

**Position is zero-based.** `gsc_avg_position` follows GSC's bulk-export convention where **0 is
the top rank**, so average position is `SUM(sum_position)/SUM(impressions) + 1`
([Google's reference](https://support.google.com/webmasters/answer/12917991)). An earlier draft
applied the starter CSV's rule that `0` means *missing* and filtered those rows out — discarding
each page's best days for 53.4% of the cohort. ML-05 contains the six-check experiment that caught
it.

**Model choice:** **NOT YET DONE** — ML-08 (`w05_model.ipynb`). Only a logistic-regression probe
has been run so far, as a leakage harness rather than as a candidate model.

*Source: `work/notebooks/w02_ml_task_framing.ipynb`, `w03_feature_leakage_check.ipynb`.*

---

## 5. Evaluation

**Split.** `GroupShuffleSplit` on `client_hash_id` — whole clients held out, so no client's
pages appear on both sides. Justified empirically, not by assertion: the same features on a
random row split score **+0.169 AUC higher**, and that gap is memorised client structure, not skill.

**Metric.** Precision@K on a **per-client** queue, **K = 100**, monthly — reported next to the
base rate and recall, averaged over repeated splits rather than one.

**Both parameters are decided, not assumed.** FlyRank's public pricing sells *"Monthly SEO audits —
up to 100 pages scanned"* per account (100 → 250 → 500 → unlimited by tier), with dedicated account
managers per brand. Work is organised by account, so the queue is. K = 100 is the entry tier, the
most conservative defensible choice. Monthly matches the audit cadence and aligns with the 30-day
label window, so there is no sliding-horizon problem.

Earlier drafts used a *global* queue at K = 50 on a *weekly* cadence. All three were wrong: K = 50
sits below even the smallest tier, and the weekly reading came from a lecture's repeated *"this
week"*, which describes the team's working rhythm rather than the client deliverable. `w02` §3
carries the capacity arithmetic showing per-client scoping is worth roughly two orders of magnitude
of recall at the same K.

Residual uncertainty, stated plainly: this rests on public pricing, not internal process
documentation. "Pages scanned" in an audit may be a wider funnel than pages actually refreshed —
content credits bound the latter. Our output is a review queue, so the audit figure is the right
analogue. If that operational number differs, K moves; the per-client scoping does not.

**Result: no demonstrable signal.** Ten grouped splits, identical pipeline, only the seed changed:

| | mean | range | spread |
|---|---|---|---|
| test base rate | 0.391 | 0.259 – 0.515 | 0.256 |
| **AUC** | **0.502** | 0.457 – 0.549 | 0.092 |
| Precision@20 | 0.440 | 0.300 – 0.650 | 0.350 |
| **Precision@50** | **0.408** | 0.240 – 0.660 | 0.420 |

AUC averages **0.502** against a chance level of 0.500 and beats chance on **5 of 10** seeds.
Precision@50 averages 0.408 against a mean base rate of 0.391 — a lift of **1.04x**, i.e. none.

> ⚠️ **An earlier draft of this report claimed a 1.72x lift and a sub-chance AUC of 0.426.** Both
> came from a single split (`random_state=42`), whose Precision@50 of 0.860 falls *outside* the
> entire seed 0–9 range. Every conclusion built on it — the "two metrics disagree" framing, the
> "3.6 standard deviations" — was an artefact of one favourable draw. It is recorded here rather
> than quietly deleted, because the mistake is the lesson: **a single grouped split cannot support
> a claim either way on this dataset.**

**Per-client evaluation at K = 100 — the decision-matching metric.** Same ten splits, ranking
*within* each held-out client rather than in one global pile:

| | Precision@100 | Recall@100 |
|---|---|---|
| one global queue | 0.407 | **1.1%** |
| per client | 0.445 | **48.4%** |
| base rate | 0.391 | — |

**Recall improves ~46x.** A global queue cannot cover 42 clients at any realistic capacity; a
per-client queue reaches roughly half of all real declines at the entry audit tier. The queue
design is now defensible.

**Precision does not follow.** Per-client Precision@100 is 0.445 against a 0.391 base rate — a
lift of **1.14x**, beating its own base rate on 8 of 10 splits. Re-scoping fixes *deployment
viability*; it does not create discriminative power. An earlier note speculated that per-client
ranking might repair the sub-chance AUC on its own — it does not. Scope was a capacity error; the
flat ranking is a separate, unresolved signal problem.


**Why the variance is so large.** `GroupShuffleSplit(test_size=0.2)` holds out 20% of *clients*,
not of pages, and client sizes run from 711 to 24,418 pages. The realised test set ranges from
**1,339 to 50,516 pages** — 1.2% to 43.7% of the cohort. Swapping one large client moves the base
rate by up to 26 points before any model is involved. The split, not the features, is the dominant
source of variance.

**Consequences for the rest of the track.** Report `GroupKFold` or repeated-seed mean ± range,
never a single split. Any baseline-versus-model comparison in ML-07/ML-08 must run on the *same*
repeated splits, or the difference measured will be seed noise.

**Reproducibility.** These figures are stable only because the source query carries
`ORDER BY content_hash_id`. Without it DuckDB may return rows in any order, so a fixed seed still
produced different splits between runs — Precision@20's mean drifted across 0.435, 0.440 and 0.445
on three runs of identical code. Two consecutive runs now match byte for byte.

**Open hypotheses for ML-06.** Five, each falsifiable, none yet tested: client-level sign flips
(partly answered — client choice dominates); weak page-level label signal; cohort selection via the
`trend_recent_impr > 0` filter; wrong evaluation scope (global versus per-client queue); and whether
**CTR is usable in this release at all** — positions 1–3 measure 0.40% CTR against the data
dictionary's documented ≈2.78%, flat at every volume floor, and the starter CSV reproduces the same
flat curve, so it is not a pseudonymization artefact.

**This is a legitimate result, not a failure.** The lane guide is explicit that a well-understood
"no effect" is valid. What was not legitimate was reporting one seed as though it were the answer.

**Error analysis:** **NOT YET DONE** — ML-09 (`w06_validation_audit.ipynb`).

*Source: `work/notebooks/w03_feature_leakage_check.ipynb`.*

---

## 6. Interpretation

**NOT YET DONE** — ML-06 (`w04_signal_audit.ipynb`) and ML-09.

What is known: feature coefficients are dominated by `char_count` (−1.42) and `word_count`
(+1.23), which is a collinearity artefact (r = 0.934) rather than a finding about content
length. No feature shows a suspicious standalone dominance.

The honest headline so far is a **negative result on ranking, alongside a solved design problem**.
Across ten grouped splits this feature set cannot rank declining pages better than chance (mean AUC
0.502; per-client Precision@100 lift 1.14x). Re-scoping the queue per client raised recall from 1.1%
to 48.4% — that part is settled and the queue design is defensible. What sits inside the queue is
not: the ordering is still no better than random. Whether the cause is the features, the label, the
cohort, or unusable CTR is unresolved and is ML-06's work.

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
