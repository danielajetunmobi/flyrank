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

**A leak the ML-05 hunt missed.** `dim_content` is an **export-time snapshot**, not a
point-in-time record: `content_updated_date` falls *after* the decision point for **77.3%** of
cohort pages at D1, and a single bulk date (2026-05-20) accounts for 204,409 items
(39.3% of all content). `days_since_last_update` was therefore leaky and has been removed.
The ML-05 hunt verified the timeline for the *fact table* windows and never asked the same question
of the dimension table. A related consequence: an `is_deleted`/`is_published` filter added during
ML-05 was itself selection on the outcome window — excluding pages by their July status when making
a March decision — and has been reverted.

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

**Target definition.** **Amended 2026-08-05 after ML-06.** A single continuous target replaces the
three binary labels:

```
target = log( future_daily_rate / baseline_daily_rate )
```

The original design used three binary targets — `future_decline`, `future_recovery`,
`future_momentum` — each defined on the population where it was meaningful. That is superseded for
four reasons: the ±20% threshold was inherited rather than derived; binarising discarded magnitude,
so a 21% dip and a 95% collapse became the same label while −19% and −21% became opposite classes;
the cohort was split three ways when one regression can use every row; and the threshold created a
definitional artefact where `future_decline` scored exactly 0.00% for already-declining pages by
construction. Sign now carries what the three labels carried. The log makes a target running
−100% to +94,550% symmetric about zero.

This fixes the label's *design*. It does not fix its *denominator*: ML-06 showed any ratio against
`trend_recent_impr` inherits a 79.0-point phantom gradient against 12.6 observed. `baseline_daily_rate`
must therefore be an independent baseline rather than the window used to select the cohort. Both
corrections land in ML-07.

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

**Metric.** **Amended 2026-08-05.** The target is now continuous, so a *base rate* — the share of
the positive class — is undefined, and so is ROC-AUC. Both are replaced:

- **Primary: Spearman correlation** between predicted and actual change. The task is ranking, and
  rank correlation measures it directly without inventing a cut-off.
- **Queue metric: Precision@K** on a **per-client** queue, **K = 100**, monthly — unchanged, except
  that the cut defining "actually declined" is now an explicit *evaluation* parameter rather than a
  property of the label.

The decision remains binary — review or don't — so a threshold still exists; it has moved from
training to evaluation. **Every figure in this section was measured under the superseded binary
label** and is kept as a description of that design, not of the current one. Re-measurement is
ML-07's first task.

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

**Result: the probe finds no signal, and the signal audit explains why.**

ML-05's logistic-regression probe exists to test the leakage harness, not to select a model — that
is ML-08's job. Within that scope, over ten grouped splits:

| | mean | range |
|---|---|---|
| test base rate | 0.413 | 0.259 – 0.539 |
| AUC | **0.502** | 0.465 – 0.534 |
| Precision@50 | 0.404 | 0.260 – 0.620 |

AUC averages 0.502 against a chance level of 0.500, beating chance on 5 of 10 seeds.
**This is a statement about a linear probe, not about the problem** — the reference pipeline's own
comparison puts logistic regression last of three (Precision@50 0.400 against a random forest's
0.740), so a tree may find structure this cannot. ML-08 settles that.

> ⚠️ **An earlier draft claimed a 1.72x lift from a single split** (`random_state=42`), whose
> Precision@50 fell outside the entire seed 0–9 range. It is recorded rather than deleted: a single
> grouped split cannot support a claim either way on this dataset.

**Queue scope is settled, and it matters — but less than an earlier draft claimed.** Ranking
*within* each client rather than in one global pile lifts pooled recall from **1.25% to 3.76%** at
K=100, roughly 3x. Precision does not follow — per-client Precision@100 is 0.420 against a 0.413
base rate, a lift of **1.02x**.

> ⚠️ **Corrected.** This previously read "1.2% to 48.5%", a ~39x claim. The 48.5% was an unweighted
> mean of per-client recall rates set against a pooled global rate — not the same quantity, and
> unstable enough that one seed read 0.819 off five clients. Both sides are pooled now.

**Recall stays low because capacity binds, not because the model is weak.** With a median 422
declines per client, a 100-page audit cannot catch more than 100 however well it ranks; the
perfect-model pooled ceiling is 4.2% at K=100. A better model in ML-08 buys precision, not coverage.
This strengthens the `w01` asymmetry: most declines are missed regardless, so the reviewed pages
must be the right ones.

**Variance, and why single splits are worthless here.** `GroupShuffleSplit` holds out 20% of
*clients*, not pages, and client sizes span 711 to 24,418. Realised test sets range from 1,339 to
50,516 pages. Any baseline-versus-model comparison in ML-07/ML-08 must run on the *same* repeated
splits, or the difference measured is seed noise. Figures are stable only because the source query
carries `ORDER BY content_hash_id`.

**Two feature groups contribute nothing.** Dropping the eight `dim_content` columns that describe
July 2026 state moves AUC by +0.002; adding a point-in-time reconstruction of freshness moves it
-0.002.

**Error analysis:** **NOT YET DONE** — ML-09 (`w06_validation_audit.ipynb`).

*Source: `work/notebooks/w03_feature_leakage_check.ipynb`.*

---

## 6. Interpretation

**The label is dominated by the arithmetic of its own denominator.** This is the central finding,
established in ML-06 on two decision points and then stress-tested with a simulation.

**Two artefacts, not one — and the larger is the label's own definition.**

`future_decline` was defined as `(~was_declining) & (future_change_pct <= -20)`. The first clause
scores every already-declining page as False. `was_declining` and `peak_ratio` both compare the same
recent window against the same older history, so the exclusion is not spread evenly across the peak
bands — it lands almost entirely in the low ones, which is precisely where the reported gradient
begins:

| peak_ratio | % `was_declining` | as published | exclusion removed |
|---|---|---|---|
| below own norm | **86.9%** | 7.91% | **55.3%** |
| 1.0–1.5× | 9.2% | 49.65% | 54.7% |
| >2.0× | **0.0%** | 53.36% | 53.36% |
| **spread** | | **46.2 pts** | **5.3 pts** |

Dropping the clause collapses the gradient from **46.2 to 5.3 points** at D1 and **66.9 to 7.9** at
D2 — the exclusion accounts for **89%** and **88%**. `Spearman(peak_ratio, continuous target) =
−0.044`. The "7x gradient" reported earlier is about 1.1x, and it points the other way: pages above
their own norm decline at 53.4% against 55.3% below it.

The **second** artefact is real and independent: `future_change_pct` divides by `trend_recent_impr`,
the same quantity that sets peak ratio, so selecting on a high recent window and measuring change
*from* it manufactures gradient before any behaviour enters. The null simulation establishes this on
exclusion-free data. **67.9% of the D1 cohort sits above its own norm** — a property of the cohort
filter that holds regardless of label.

**This is a correction to an earlier version of this section**, which reported the 7x gradient as the
central finding and attributed all of it to the denominator. The denominator artefact is real but
secondary; the definitional one dominates. Both are recorded rather than replaced.

**It is an artefact, not a behaviour — and an earlier draft of this report got that wrong.** The
first version called it *mean reversion*, which asserts that pages return to normal after an unusual
month. That is testable, and it fails. Replacing each page's real future window with a random 30-day
window from its **own history** — preserving level and volatility, destroying any information about
what happened next — produces a gradient of **79.0 points**, against **12.6 points** observed on the same pages —
pure arithmetic is **6.3x steeper than reality**. The observed pattern is not even monotonic: the
most extreme band declines *least* (38.5%) where the null predicts it should decline *most*
(87.9%).

Real pages decay far *less* than chance predicts, which matches the autocorrelation evidence:
consecutive 30-day means correlate at **r ≈ 0.89**, so the recent window is a good estimate of a
page's level rather than noise. Persistent levels are exactly why observed decay undershoots the null.

The simulation scores `chg <= -0.20` on every page and never applies the exclusion, so its 12.6-point
observed spread was never comparable with Test 3's 46.2 — a point an earlier draft explained away as
sparse pages suffering the artefact more than dense ones. The exclusion-free numbers are 5.3 points
on the full cohort and 12.6 on the dense subset. Both are small; the gap is cohort, not behaviour.

**Why the distinction decides ML-07.** If genuine reversion drove this, no label redefinition would
escape it. Because it is arithmetic, a label built on an independent baseline — or on a fitted trend
rather than a ratio of two windows — escapes it cleanly.

**This explains the rest.** Prior trend does not predict future decline — the rate is flat at
52.88%–55.56% across the entire eligible trend range on D1. A near-coin-flip target is exactly
what a chance-level probe would produce, with no feature at fault.

**CTR levels are not usable; the CTR–position *ordering* partly is.** Weighted CTR for positions
1–3 measures 0.30% (D1) and 0.39% (D2) against the ≈2.78% documented in `docs/data-dictionary.md` —
7–9x below FlyRank's own figure, flat across rank bands, and reproduced in the starter CSV, so not a
warehouse artefact. Absolute CTR and any threshold built on it are therefore unreliable.

But rank-based correlation between position and CTR is **−0.234** on the starter CSV and **−0.217**
on the warehouse (−0.239 above 1,000 impressions) — consistent, correctly signed, and roughly **3x
stronger than the Pearson value of −0.080** recorded in the Week 1 notebook. Heavy tails
(`ctr` p99/p50 ≈ 105) make Pearson understate it. So position does inform CTR *ordering*; it is the
absolute level that cannot be trusted. Weighted CTR looks flat because large pages dominate it,
while Spearman weights every page equally.

**FlyRank's `page_one_decay_risk` flag does not select at-risk pages** against this label: lift
1.01 at D1 and 0.91 at D2 — at the second decision point it flags pages that decline
*less* than average. Read carefully, this condemns the pairing rather than the rule: the flag
encodes slow editorial decay while our label captures fast statistical reversion. A rule failing
against a compromised label is weak evidence against the rule.

**A freshness relationship does exist, but bounded.** On the subset where the update date is exact,
decline rises with staleness to about 180 days and then flattens — MIXED, matching the shape the
Week 4 lecture found in its own example. Enough to justify a freshness *rule*; far too weak to rank on.

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
