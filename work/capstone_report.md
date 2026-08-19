# Capstone Report — Growth / Recovery / Momentum Prediction

- **Author:** danielajetunmobi
- **Lane:** Freestyle — Growth / Recovery / Momentum Prediction
- **Repo:** https://github.com/danielajetunmobi/flyrank
- **Date:** 2026-08-19 (living document — updated as each assignment lands)

> **Status:** sections 1–6, 8 and 9 reflect completed work (ML-02 → ML-09). **Section 7 is the
> only one still marked NOT YET DONE**, because ML-10 (`w07_action_playbook.ipynb`) has not been
> reached. Nothing below claims a result that has not been computed.

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
disagree on **22.3%** of pages at a −20% cut, and order them at only **Spearman +0.565**.

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
to 0.92, injecting a raw future count leaks weakly, and a random row split inflates the score by
**+0.091** AUC over a client-grouped one, on the same ten seeds. Feature importances were inspected
for a suspiciously dominant term; the largest (`char_count` / `word_count`, opposite signs) is
**multicollinearity** — they correlate at **r = 0.997** — not a leak.

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

**Missingness follows `content_type`, so imputation is not neutral.** Four keyword columns —
`search_volume`, `cpc`, `competition`, `backlinks` — run from **0% to 100% missing** across the three
content types: present for every page of one type, absent for every page of another. `char_count` and
`word_count` span 46 points, GA4 columns 15.6 by `main_intent` (29.8 by `content_type` at D2).

A median fill on those columns therefore stamps the content type into the feature: every page of the
missing type receives an identical value that a tree splits on immediately. The model would appear to
use search volume while actually using content type. The `has_*` flags record *that* a value was
missing, which is honest, but the filled column still carries the same fact in continuous disguise.
ML-07 either drops those four columns or keeps them unimputed and lets the model handle NaN natively.

## 3. Baseline

**The rule.** A page is worth reviewing if it is old enough that decay is the likely story, it is
among the pages that matter for its own client, and it still has something left to lose:

```
gate:  content_age_days >= 180
       impr_90d >= that client's own median
       slip <= 0.5              (a page that already lost half is not preventable)
score: content_age_days
```

Only three fields are permitted. ML-07 measured availability by client and content type and found six
of 36 clients have **no** GA4 data at all, seven have **no** backlinks data, and one content type has
none of the keyword enrichment — a rule reading those would rank data availability rather than page
health. Of the four universally available fields, `content_updated_date` leaks (ML-06 Finding 1),
leaving `impr_90d`, `avg_position` and `content_created_date`.

**Result: the gate is the baseline; the ranking is not.** Scored on ten `GroupShuffleSplit` seeds at
`test_size=0.2` — the same protocol any model must use:

| | mean | range |
|---|---|---|
| pool base rate | 0.5245 | 0.3180 – 0.7435 |
| `P@100`, the rule | **0.5041** | 0.3265 – 0.6709 |
| `P@100`, random order in the gate | **0.5432** | 0.3597 – 0.6690 |

The rule's ordering is **−0.0391 below random** and beats it on only 2 of 10 seeds. The gate is worth
having — it lifts the decline rate from the cohort's 0.4228 to 0.5245 — but ranking by age inside it
transfers badly to unseen clients, because the base rate itself swings from 0.318 to 0.744 depending
which clients are held out.

**The number to beat is the gate with pages in arbitrary order**, not the rule's 0.5041 — shipping a
ranking that underperforms shuffling would be indefensible. ML-07 estimated that bar at 0.5432 from a
single shuffle per seed; **ML-08 re-estimated it at 0.5132** over 20 draws per seed, with a per-seed
standard deviation of 0.0178. The gate is unchanged; only the estimate of random ordering inside it
improved.

**Three earlier versions are kept in the notebook rather than replaced.** v1 (`slip × impr_90d`)
scored `P@100` 0.9427 and looked far stronger — but its pool was pre-selected to 80.4% decline, its
persistence null returned **1.0000**, meaning it detected pages that had already fallen rather than
predicting, and six of its top twenty had already lost 93–96% of their traffic. Had that number
reached this report unexamined, ML-08 would have been chasing a bar that never existed.

*Source: `work/notebooks/w04_baseline_score.ipynb`.*

---

## 4. Model / analysis

**Target definition.** One continuous target:

```
target = asinh(future_daily_rate) - asinh(baseline_daily_rate)
```

Sign is direction, magnitude is size — a 21% dip and a 95% collapse are no longer the same event.
`asinh(x) = log(x + sqrt(x^2+1))` equals 0 at zero and converges to `log(2x)` for large x, so it
behaves like a log ratio for healthy pages while staying finite when one reaches zero, which a ratio
cannot represent at all — and that is **7.4% of the D1 cohort**, 13.7% at D2. Agreement with the log
ratio where both are defined is Spearman **+0.9546** (D1) and **+0.9009** (D2).

It also deflates percentage swings at trivial volume, which a ratio exaggerates. The same 50% drop at
four sizes:

| before/day | after/day | log ratio | `asinh` |
|---|---|---|---|
| 0.1 | 0.05 | −0.6931 | **−0.0499** |
| 1.0 | 0.50 | −0.6931 | −0.4002 |
| 10.0 | 5.00 | −0.6931 | −0.6858 |
| 100.0 | 50.00 | −0.6931 | **−0.6931** |

A ratio calls all four the same −50%. Given a cohort whose median dead page ran 0.13 impressions a
day, that deflation matters.

**Dead pages get a second, separate treatment.** `went_to_zero AND prior_rate >= 1 impr/day` is
reported as its own flag — **1,334 pages at D1 over 34 clients**. 87% of pages reaching zero were
already carrying under one impression a day, so an ungated flag would bury the 136 that lost ten or
more.

The flag is **ranked by prior daily rate rather than merely listed**, because it does not reliably
fit the audit budget: a median of 10 pages per client, but a tail reaching 195 (D1) and 663 (D2),
with **4 clients at D1 and 9 at D2 exceeding K = 100**. Ranked, an overrun surfaces the largest
losses; unranked, it would be least usable for the clients worst affected.

This replaced three binary targets — `future_decline`, `future_recovery`, `future_momentum` — and
then a log ratio. The reasoning for each step is in the revision log (§9).

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
`competition_level`) is deferred to the modelling stage. **Settled in ML-08:** `main_intent` and
`competition_level` are unusable — two clients have none of either, so including them would rank
data coverage rather than page health — and `content_type`, the one that is universally present,
moves `P@100` from 0.7563 to **0.7562**. No categorical is in the final feature set.

**Missing-value policy.** Flag first, then fill — never a blind zero where zero is a meaningful
extreme. `has_*` indicators are computed before any fill, so "unknown" and "genuinely zero" stay
distinguishable; `ctr` is safely zero-filled because 56.8% of tracked pages genuinely have 0 CTR.

**Position is zero-based.** `gsc_avg_position` follows GSC's bulk-export convention where **0 is
the top rank**, so average position is `SUM(sum_position)/SUM(impressions) + 1`
([Google's reference](https://support.google.com/webmasters/answer/12917991)). The starter CSV uses
the opposite rule — `0` means *missing* — and applying that here filters out each page's best days
for 53.4% of the cohort. ML-05 contains the six-check experiment that established which convention
this warehouse follows.

**Model choice: ridge regression on four features.** Four algorithms — ridge, logistic, gradient
boosting as regressor and as classifier — each on two feature sets, scored on the same ten
`GroupShuffleSplit` seeds the baseline uses.

| model | spearman | P@20 | P@100 | recall@100 | AUC | R² |
|---|---|---|---|---|---|---|
| **ridge FULL** | **0.5302** | **0.7828** | **0.7563** | 0.1293 | 0.7584 | 0.0026 |
| logistic FULL | 0.5152 | 0.7814 | 0.7500 | 0.1293 | 0.7519 | — |
| gbm_reg FULL | 0.4945 | 0.7477 | 0.7216 | 0.1182 | 0.7380 | −0.0020 |
| gbm_clf FULL | 0.4843 | 0.7769 | 0.7298 | 0.1266 | 0.7487 | — |
| logistic SAFE | 0.0553 | 0.5442 | 0.5466 | 0.0930 | 0.5436 | — |
| **baseline, gate + random order** | **−0.0014** | 0.5094 | **0.5129** | 0.0839 | 0.5000 | — |

**The four features are `peak_ratio`, `prior_trend`, `impr_90d` and `avg_position`, and effectively
it is one.** `peak_ratio` holds **0.6034** permutation importance against ≤0.0073 for every other
feature.

**ML-08 shipped five; ML-09's ablation removed one.** `content_age_days` scored **−0.0645** on
permutation importance — shuffling it *improved* held-out fit — and dropping it outright gains on
every metric: spearman **0.5302 → 0.5546**, `P@100` 0.7563 → **0.7612**, AUC 0.7584 → **0.7693**. It
correlates −0.1981 with the target and is collinear enough with `peak_ratio` to split its
coefficient. The table above reports the five-feature model as ML-08 measured it; the shipped model
is the four-feature one.

**All four survivors are window aggregates.** `impr_90d` sums impressions over 90 days;
`avg_position` is impression-weighted, `SUM(sum_position)/SUM(impressions) + 1`; `peak_ratio` and
`prior_trend` are each ratios of two such sums. `content_age_days` was the only point-in-time value
in the set — a date difference, nothing aggregated — and it is the one validation removed. Every
aggregate is computed by summing then dividing, never by averaging per-day rates, which is the
correction that fixed `avg_position` in ML-04.

**Twenty candidate features were examined, not five.** Every column in `dim_content` is accounted for:
four identifiers, two pipeline-metadata fields, and twenty candidates each either in the model or
excluded with a measured reason — availability (six clients have no GA4 at all), timing (both
optimisation dates sit after the decision point for **100.0%** of populated rows), or no measured gain.

**Correlation with the target barely predicts whether a feature helps.** `prior_trend` correlates
**+0.5438** and contributes 0.0038. `url_char_count` correlates **−0.2225** and costs **−0.0185**.
`peak_ratio` absorbs what the others carry.

**Ridge over logistic on the primary metric.** The gap is +0.0063 on `P@100` — inside seed noise — but
**+0.0150** on rank agreement, which is the metric this project declared primary. Logistic trains on a
binarised label and discards magnitude by construction. Ridge's R² of 0.0026 makes its predicted
*value* unfit to display: **ship the rank, never the number**.

**Boosting placed third, and the notebook records why rather than leaving it implicit** — R² is ~0 for
every model, so a squared-error objective spends its capacity on the unpredictable part. A ranking
objective is the principled follow-up, not a hyperparameter sweep.

**One model serves the whole lane.** Sorting the same predictions two ways and splitting on prior
state gives three queues: decline **+0.2111** over random, momentum **+0.1734**, recovery **+0.0962**.
The three-label design would have needed three models.

*Source: `work/notebooks/w02_ml_task_framing.ipynb`, `w03_feature_leakage_check.ipynb`.*

---

## 5. Evaluation

**Split.** `GroupShuffleSplit` on `client_hash_id` — whole clients held out, so no client's
pages appear on both sides. Justified empirically, not by assertion: the same features on a
random row split score **+0.091 AUC higher** on every one of ten seeds, and that gap is memorised client structure, not skill.

**Metric.**

- **Primary: Spearman correlation** between predicted and actual change. The task is ranking, and
  rank correlation measures it directly without inventing a cut-off.
- **Queue: Precision@K** on a **per-client** queue, **K = 100**, monthly, **pooled** across clients —
  with the cut defining "actually declined" stated as an evaluation parameter, not hidden in the label.

A continuous target has no positive class, so neither a base rate nor ROC-AUC is defined. The
decision a specialist takes is still binary, so a threshold still exists — it has moved from training
to evaluation only.

> **Every figure in this section was measured under the superseded binary label.** They are accurate
> for the design that was tested and kept on that basis; re-measurement is ML-07's first task.

**Both parameters are decided, not assumed.** FlyRank's public pricing sells *"Monthly SEO audits —
up to 100 pages scanned"* per account (100 → 250 → 500 → unlimited by tier), with dedicated account
managers per brand. Work is organised by account, so the queue is. K = 100 is the entry tier, the
most conservative defensible choice. Monthly matches the audit cadence and aligns with the 30-day
label window, so there is no sliding-horizon problem.

`w02` §3 carries the capacity arithmetic: per-client scoping is worth **25–30x** the recall of a
global queue at the same K, and that ratio holds whether decline is defined at −10%, −20%, −30% or
−50%. Section 9 records the earlier global / K = 50 / weekly assumptions and why each was wrong.

> ⚠️ **Corrected by FlyRank's answer (2026-08-19).** Both paragraphs above are withdrawn in part.
> FlyRank confirms the queue is generated **separately per client** and is **customized rather than a
> global top 50** — so Decision 1 stands, now on the client's word rather than on inference from
> pricing copy. The rest does not:
>
> - **Cadence is weekly**, not monthly. Section 9's revision-log row recording "weekly → monthly" as
>   a correction was itself the error; the earlier draft had it right.
> - **K is a configurable per-client budget.** The automated workflow can process **thousands of
>   pages per day**, so the audit tier bounds neither the queue nor the review, and K = 100 has no
>   special status.
> - **The 30-day label must not be tied to the queue cadence.** FlyRank asks for several K values and
>   several outcome windows tested independently. The "no sliding-horizon problem" convenience above
>   is exactly what they rejected.
> - **Report macro client metrics** — the mean of per-client rates — as the headline, with pooled
>   alongside. This reverses the emphasis this report uses throughout. It does **not** reinstate the
>   withdrawn 48.5%, which compared a macro numerator against a pooled baseline and is wrong under
>   either convention.
>
> The **25–30x** figure is a like-for-like pooled comparison at one K and survives as stated. What
> does not survive is reading it as a capacity argument.

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

**Queue scope is settled.** Ranking *within* each client rather than in one global pile lifts pooled
recall from **1.25% to 3.76%** at K = 100, roughly 3x. Precision does not follow — per-client
Precision@100 is 0.420 against a 0.413 base rate, a lift of **1.02x**.

**Recall stays low because capacity binds, not because the model is weak.** With a median 422
declines per client, a 100-page audit cannot catch more than 100 however well it ranks; the
perfect-model pooled ceiling is 4.2% at K=100. A better model in ML-08 buys precision, not coverage.
This strengthens the `w01` asymmetry: most declines are missed regardless, so the reviewed pages
must be the right ones.

> ⚠️ **Withdrawn by FlyRank's answer (2026-08-19).** Capacity does not bind. The workflow processes
> thousands of pages per day and K is a per-client budget, so the 4.2% ceiling describes a K chosen
> from public pricing copy rather than a limit of the design. The measured **3.76%** stands as what
> this probe reached *at K = 100*; it is not a ceiling. What survives untouched is the `w01`
> asymmetry's other half — a missed decline costs more than a false flag — which argued for ranking
> quality independently of coverage. The replacement ceiling is a K sweep, owed in ML-10.

**Variance, and why single splits are worthless here.** `GroupShuffleSplit` holds out 20% of
*clients*, not pages, and client sizes span 711 to 24,418. Realised test sets range from 815 to
47,661 pages. Any baseline-versus-model comparison in ML-07/ML-08 must run on the *same* repeated
splits, or the difference measured is seed noise. Figures are stable only because the source query
carries `ORDER BY content_hash_id`.

**Two feature groups contribute nothing.** Dropping the eight `dim_content` columns that describe
July 2026 state moves AUC by +0.002; adding a point-in-time reconstruction of freshness moves it
-0.002.

### Where the model is wrong (ML-08 §4; extended in ML-09)

**It leans on one feature, and one of the five actively hurts.** `peak_ratio` **0.6034** permutation
importance, `avg_position` 0.0073, `impr_90d` 0.0061, `prior_trend` 0.0038, `content_age_days`
**−0.0645**. A feature correlating −0.1981 with the target degrades held-out fit — collinear with
`peak_ratio` and splitting its coefficient.

**Performance varies more across clients than across anything else.** Per-client `P@100` on one
held-out split spans **0.458 to 0.970**. The worst client took **72** picks, so it is not a
sample-size artefact — the two clients scoring 0.500 and 0.600 took 6 and 5 picks and are.

**Precision is flat across the confident picks and collapses on the marginal ones.** By decile of
predicted target: **0.896, 0.908, 0.922, 0.961** — then **0.442** in the last band. The model is
reliable exactly where it is confident, and near-random at the edge of what it will pick. For a queue
truncated at K that is the right shape of failure, and it argues for surfacing the score's confidence
rather than presenting all 100 picks as equivalent.

**The ordering is monotonic across the signed outcome**, which no threshold metric could show. Mean
predicted percentile by what actually happened: large fall **0.3330**, small fall 0.4221, stagnant
**0.5012**, small rise 0.5153, large rise **0.6648**. Pages that fell hard are **9x** likelier to sit
in the predicted-worst decile than the best; pages that rose hard, **12x** the reverse.

**Stagnant pages need no special handling** — they sit at 0.5012 and reach an extreme **16.97%** of the
time against 25.42% for large fallers. The `w01` "negative class" was unnecessary.

**About a fifth of the gain is arithmetic.** A null replacing each page's future with a random window
from its own history leaves the FULL models only **+0.04** above the null's own base rate, against
**+0.21** on the real future. The rest is genuine prediction — the first result in this project to
survive that test.

**The advantage transfers to an unseen month essentially intact.** Trained at D1, scored once at D2:
**+0.1291** over random, against **+0.2436** at D1 — **53%** retained. That 53% was read as decay until
ML-09 walked four decision points forward and found it is not. Holding the test month fixed and moving
only the training date — the controlled version of the question — a **92-day-old** model scores
**0.1238** where a 31-day-old one scores **0.1242**, a gap of **0.0009**. What falls between D1 and D2
is the headroom, not the skill: D2's pool declines at 0.7873 against 0.5131, so the most any ranker
could gain fell from 0.4869 to **0.2131**. Measured against what was available, capture stays in a
**33.5% - 58.3%** band, mean **47.4%**, and the last month is the **highest** of the three forward
steps at **0.5829**. Fresher is still directionally better — the within-month correlation between
staleness and lift is **-0.7541** — but the largest penalty measured anywhere is **0.0194**, against a
**0.2415** to **0.1242** swing driven purely by which month is scored.

**The base rate is not stationary, and no split in this project could see that.** It runs
**0.3169 → 0.5131 → 0.6146 → 0.7869** across four monthly decision points. Every grouped split — all
ten seeds, every stability range quoted in this report — resamples clients at D1 and therefore holds
the base rate fixed at 0.5131. Those splits measure variation between clients; variation over time was
invisible to them by construction.

**All three items ML-09 owed are now closed** (`w06_validation_audit.ipynb`). The 0.458 client is a
capacity artefact — 72 pages against 100 slots, so the queue took every page and precision equalled
its base rate of 0.4583. The three failure cases are all **recoveries**: pages that fell to roughly
half their baseline and then rose above it, which the model extrapolated downward. And the ranking
objective does **not** close boosting's gap — a rank-transformed target buys **+0.0004** spearman,
leaving boosting **0.0168** behind ridge, so ML-06's objective-mismatch hypothesis is withdrawn.

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
D2 — the exclusion accounts for **89%** and **88%**. `Spearman(peak_ratio, asinh target)` is
**−0.0665** (D1) and **−0.1505** (D2) — weak on both dates and not stable between them. The "7x
gradient" reported earlier is about 1.1x, and it points the other way: pages above their own norm
decline at 53.4% against 55.3% below it.

The **second** artefact is real and independent: `future_change_pct` divides by `trend_recent_impr`,
the same quantity that sets peak ratio, so selecting on a high recent window and measuring change
*from* it manufactures gradient before any behaviour enters. The null simulation establishes this on
exclusion-free data. **67.9% of the D1 cohort sits above its own norm** — a property of the cohort
filter that holds regardless of label.

**The artefact is arithmetic, not behaviour.** *Mean reversion* would assert that pages return to
normal after an unusual month. That is testable, and it fails. Replacing each page's real future window with a random 30-day
window from its **own history** — preserving level and volatility, destroying any information about
what happened next — produces a gradient of **79.0 points**, against **12.6 points** observed on the same pages —
pure arithmetic is **6.3x steeper than reality**. The observed pattern is not even monotonic: the
most extreme band declines *least* (38.5%) where the null predicts it should decline *most*
(87.9%).

Real pages decay far *less* than chance predicts, which matches the persistence in the series:
consecutive 30-day means correlate at **Pearson 0.799 / Spearman 0.823** across the 70,013 dense
pages, so the recent window is a reasonable estimate of a page's level rather than noise. Persistent
levels are exactly why observed decay undershoots the null.

The simulation scores `chg <= -0.20` on every page and never applies the exclusion, so its 12.6-point
observed spread was never comparable with Test 3's 46.2. The exclusion-free numbers are 5.3 points
on the full cohort and 12.6 on the dense subset. Both are small; the gap is cohort, not behaviour.

**Why the distinction decides ML-07.** If genuine reversion drove this, no label redefinition would
escape it. Because it is arithmetic, a label built on an independent baseline — or on a fitted trend
rather than a ratio of two windows — escapes it cleanly.

**This explains the rest.** Prior trend does not predict future decline — the rate is flat at
52.88%–55.56% across the entire eligible trend range on D1. A near-coin-flip target is exactly
what a chance-level probe would produce, with no feature at fault.

**CTR is correctly measured and simply compressed** — corrected against FlyRank's own research.
Weighted CTR for positions 1–3 measures 0.30% (D1) and 0.39% (D2). An earlier version of this report
read that as **7–9x below FlyRank's own figure**, comparing against the ≈2.78% in
`docs/data-dictionary.md`.

FlyRank's March 2026 paper (`docs/flyrank-seo-research-march-2026.pdf`, Finding #3) publishes weighted
CTR by position tier and says these figures *"replace the older per-row average that produced
impossible values above 100%"*:

| tier | ours (D1 / D2) | FlyRank research |
|---|---|---|
| top 3 | 0.30% / 0.39% | 0.423% |
| page 1 (4–10) | 0.33% / 0.34% | **0.339%** |
| striking (11–20) | 0.32% / 0.32% | **0.325%** |
| page 3–5 (21–50) | 0.16% / 0.15% | **0.163%** |

Three of four bands agree to two decimal places, and the top-3 band is the thinnest stratum on both
sides. **The measurement was right; the reference was wrong**, and the dictionary concedes as much in
its own note — its 2.78% comes from a slice with median volume ~53 impressions/90d, "where one click
moves CTR by ~1.9pp", which is precisely the thin per-row average the research paper replaced.

ML-06 Test 7 had already ruled out a broken denominator: no page on either date has more clicks than
impressions, and the pooled rate (0.305%) sits below the mean of per-page rates (0.421%). That test
was right that the data was sound; it could not tell that the *benchmark* was the problem, because
the benchmark was not in the warehouse.

**The finding that survives is compression.** The first twenty positions span **0.10 percentage
points**, in our data and in FlyRank's. A rule keyed to CTR level cannot separate pages that a
0.10-point spread cannot separate — which is why `low_ctr_visible_page` has little to work with here.
CTR stays out of the model on measured redundancy rather than on doubt: `Spearman(ctr, target)` is
**−0.0189**, and adding it costs **−0.0136** AUC because `avg_position` already carries the ordering.

But rank-based correlation between position and CTR is **−0.288** across the cohort and **−0.231**
above 1,000 impressions — correctly signed, stable across that filter, and about **3.6x stronger
than the Pearson value of −0.080** recorded in the Week 1 notebook
(`notebooks/01_first_look_and_discovery.ipynb`, outside this workspace). Heavy tails
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

**Pooling across clients destroyed the signal, and that is the most useful thing ML-06 found.**

Every verdict above is computed on the pooled cohort. Rerun per client — the slice the signal-audit
skill asks for and this project had skipped — the picture changes:

| | D1 (17 clients ≥ 200 pages) | D2 (29 clients) |
|---|---|---|
| `Spearman(prior_trend, target)`, median client | **−0.116** | **−0.220** |
| clients with \|ρ\| > 0.2 | **8 of 17** | **17 of 29** |
| `Spearman(peak_ratio, target)`, median client | **−0.239** | **−0.231** |
| clients with \|ρ\| > 0.2 | **9 of 17** | **20 of 29** |

Pooled, `Spearman(peak_ratio, target)` is **−0.0665** at D1. The median client shows **−0.239** —
three and a half times stronger. Clients differ in traffic scale, seasonality and content mix, so a
relationship holding *inside* each one flattens when they are stacked.

**Then an adversarial check overturned it.** Both signals contain `trend_recent_impr`, and so does
the target's baseline — the same shared term that made Test 3's gradient 89% definitional. Replacing
each page's future with a random window from its own history, on 27,801 dense pages across 8 clients,
gives a median per-client ρ of **−0.6225** (`prior_trend`) and **−0.6608** (`peak_ratio`), with every
client above 0.2. Observed is **−0.0306** and **−0.0888**. A null carrying no information about the
future produces seven to twenty times more correlation than reality.

Shuffling the target inside each client collapses ρ to −0.009, so the observed values are not
sampling noise. They are the arithmetic. Grouping by client does not remove it — the shared term sits
inside each page.

**What survives:** clients genuinely differ from one another, and per-client correlations are not
noise. **What does not:** the reading of that as a predictive relationship pooling had hidden.

**For ML-07:** the per-client *queue* stands on capacity grounds, untouched. Per-client *modelling*
has no evidence behind it. The precondition for asking the question again is the one already on the
table — give the target a baseline that does not appear in the features. Until `trend_recent_impr` is
out of the denominator, no correlation between these signals and this target means anything at any
grouping.

**The cohort filter inflates the decline rate, and `asinh` makes that fixable.** Relaxing
`trend_recent_impr > 0 AND trend_baseline_impr > 0` and scoring every active page shows the cohort is
**12.6 points more decline-heavy at D1** (65.8% against 53.2%) and 8.3 at D2. Both filters push the
same way: `rec30 = 0` pages decline at **exactly 0.0%** — a page at zero cannot fall further, and 85%
stay there — while `base30 = 0` pages decline at 46.6%, still below the cohort's 65.8%. The margin
moves with the date, so no constant corrects it.

That filter existed because a *ratio* needs a non-zero denominator. `asinh` does not, so ML-07 can
readmit the 26,868 pages excluded at D1 — precisely the ones that already lost everything.

**GA4 engagement carries no page-level signal.** `engaged_rate` correlates with the target at
**+0.005** (D1) and **+0.009** (D2); `pages_per_session` and `scroll_per_session` flip sign between
decision points. Unlike prior trend, a per-client rerun does not rescue it: median ρ of +0.037 and
+0.015, with 0 of 13 and 2 of 24 clients above \|ρ\| = 0.2. Dropping these features costs nothing and
removes 13.9% missingness plus a category leak. `search_volume`, `cpc` and `competition` are left
untested on purpose — absent for an entire content type, any correlation would measure the type.

### What FlyRank's own research says about this work

`docs/flyrank-seo-research-march-2026.pdf` analyses 341,701 content pieces across 57 brands — roughly
1.7x this warehouse. It was in the repository throughout and was not read until ML-08 was finished.
Reading it corrected one finding, confirmed four, and exposed a contradiction between FlyRank's own
two documents.

**A contradiction we could not have resolved from inside the data.** The research paper defines
`trend_direction` as *"Up: >10% growth. Down: >10% decline. Stable: within +/-10%"*. The data
dictionary defines the same field as `up > +20%; down < −20%`. **The two documents disagree by a
factor of two**, and this project followed the dictionary throughout ML-03 to ML-06, describing ±20%
as "FlyRank's own convention".

Neither choice was verifiable against the warehouse, because a threshold is a definition rather than a
measurement. What resolved it was abandoning thresholds: the continuous target has no cut to get
wrong. That was adopted for other reasons entirely — and this is the clearest evidence that the
redesign was worth its cost.

**Their ML appendix contains the same error this project spent five weeks on.** A random forest
predicts `health_score`, and `Average Position` scores **43** on feature importance with `Impressions`
at **32**. But `health_score` is *defined* as impressions (30 pts) + position (30 pts) + CTR (20 pts)
+ scroll depth (20 pts). **Sixty percent of the target is built from the two features that top its own
importance ranking.**

To their credit the paper says so plainly — *"the target itself is partly constructed from some of
these inputs, so importance is descriptive rather than causal"* — and confines the ML work to a
labelled appendix. This project made the same class of mistake three times (Test 3, the simulation,
Test 5) and needed a null simulation to see it. That two independent efforts on the same data hit the
identical trap is the strongest argument for making the null a standing check rather than an
occasional one.

**Four findings confirmed independently.**

| our finding | their finding |
|---|---|
| `search_volume` carries no signal — Spearman **+0.0124** | raw SV→impressions **0.0083**, log-scaled **−0.0419**, "both are weak" |
| `page_one_decay_risk` does not select at-risk pages (lift 1.01 / 0.91) | Myth #3 **REVERSED** — "flags mark leverage, not failure"; flagged pages score *higher* because flags need visibility to trigger |
| `content_age_days` negatively related to future traffic — Spearman **−0.1981** | "Content age is the strongest negative signal in this model" |
| CTR compressed across the first twenty positions | 0.423% → 0.325%, a 0.10-point span |

The `page_one_decay_risk` result is the most useful. ML-06 marked it FALSE and treated that as a
finding about the rule. Their Myth #3 explains *why*: optimisation flags can only fire on pages with
enough traffic to diagnose, so flagged pages are systematically more visible than unflagged ones.
**The flag was never a decline predictor and was not built to be one.** Our verdict stands; the
interpretation improves.

**Where this project's evidence standard goes further.** Their growth model reports **71%** holdout
accuracy without the base rate beside it — and their own split is 74.8K growing against 45.6K
declining, so predicting "growing" every time scores about 62%. The building-baselines skill in this
repository is explicit that precision must be reported next to the base rate, and that is exactly the
gap. They use one holdout; this project uses ten client-grouped splits, reports the random-order bar
next to every model, and runs a null on the headline result.

None of that makes their paper wrong — it is a public portfolio study, not a model evaluation, and it
says so. But it does mean the reference pipeline's numbers should not be used as a target to beat.

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
1. **The computed feature vector is still not cached**, so the aggregation, joins and feature
   engineering are recomputed on every run. (The underlying Parquet files *are* cached
   automatically by `huggingface_hub`, so this costs seconds of recompute, not a repeat download.)
   Caching was **deferred until after ML-06** because hypothesis 3 proposed relaxing the
   `trend_recent_impr > 0` cohort filter and a cache written first would have baked that filter in.
   **That condition has now expired** — ML-06 Test 8 closed hypothesis 3 (the filter inflates the
   decline rate by 12.6 points at D1) and ML-07 froze the gate — so the reason for deferring is
   gone and the work simply has not been done. `work/outputs/*.parquet` is gitignored and CI
   hard-fails on committed Parquet, so caching is safe to add whenever it is wanted.
2. **The remote host is intermittently unreliable.** Repeated `ZSTD Decompression failure`
   errors on `hf://` reads during development; a re-run may need retries.
3. **Most figures rest on one decision point, but no longer all of them.** The headline numbers are
   measured at 2026-03-31. ML-08 added a second scored date and ML-09 §8 walks **four** decision
   points, which is the most the warehouse allows. What that walk found is not reassuring and is
   reported in section 5: the decline rate runs **0.3169 → 0.5131 → 0.6146 → 0.7869** across four
   months, so any single-date figure quoted without its base rate is not comparable month to month.
4. **`future_recovery` and `future_momentum` are not modelled, and no longer exist to model.** Both
   were binary targets under the design replaced on 08-05 by the single continuous `asinh`
   difference (section 2, and the revision log in section 9). Direction is read off the sign of one
   target rather than from three separate models, so there is nothing outstanding here — this item
   is retained only because earlier drafts listed it as a gap.

---

> **Claims checklist.** All statements above are framed as observed / measured / directional /
> decision-support. No causal claim is made — nothing here shows that refreshing a page *causes*
> recovery, which would require an experiment this data cannot support. No claim is made about
> Google's ranking algorithm. No client-identifying details appear. Every number cited is
> reproduced by a cell in the linked notebook rather than quoted from memory.

---

## 9. Revision log

Sections 1–8 state the current position. This is how it got there, kept so the reasoning can be
audited rather than taken on trust. Nothing above was deleted to make room for it — superseded
figures are still present and still labelled with the design they measured.

### Design changes

| date | change | why |
|---|---|---|
| 08-05 | three binary labels → one continuous target | the ±20% cut was inherited, not derived; binarising made a 21% dip and a 95% collapse the same label while −19% and −21% became opposite classes; the cohort was split three ways; `future_decline` was false *by construction* for already-declining pages |
| 08-05 | AUC → Spearman, base rate dropped | a base rate is the share of the positive class, and a continuous target has none; the same kills AUC. The threshold moves from training to evaluation only |
| 08-06 | log ratio → `asinh` difference | `log(0) = −∞`, and 7.4% of the D1 cohort reaches zero. `asinh` agrees at Spearman +0.9546 / +0.9009 where both are defined and stays finite where the ratio does not |
| 08-06 | dead pages get their own ranked flag | 87% of pages reaching zero already carried under 1 impression/day; the other 1,334 (D1) need surfacing without competing for queue position |
| — | global queue at K = 50, weekly → **per-client at K = 100, monthly** | K = 50 is below the smallest audit tier; "weekly" came from a lecture's turn of phrase describing the team's rhythm, not the client deliverable |
| 08-19 | per-client at K = 100, monthly → **per-client at configurable K, weekly** | FlyRank answered directly. The row above got the scope right and the cadence wrong: "weekly" was the client deliverable after all. K = 100 came from an audit tier that bounds nothing — the workflow processes thousands of pages per day |
| 08-19 | pooled metrics → **macro client metrics as the headline**, pooled retained | FlyRank asks for macro. The 48.5% withdrawal below stands on its own terms — macro numerator against pooled baseline — but the inference that pooling was therefore the right convention does not |
| 08-19 | 30-day outcome window tied to the queue cadence → **swept independently** | tying the label to the rebuild interval was a convenience ("no sliding-horizon problem"), and FlyRank rejected it: several K values and several outcome windows, tested separately |

### Findings that were withdrawn

| claim | what it actually is | how it was caught |
|---|---|---|
| **1.72x lift** over base rate | seed noise | measured on a single split (`random_state=42`) whose Precision@50 fell outside the entire seed 0–9 range |
| per-client scope lifts recall **1.2% → 48.5%**, ~39x | **1.25% → 3.76%**, ~3x | the 48.5% was an unweighted mean of per-client rates set against a *pooled* global rate; one seed read 0.819 off five clients |
| the decline gradient is **mean reversion** | an arithmetic artefact | a null replacing each page's future with a random window from its own history produced **79.0** points of gradient against **12.6** observed |
| Test 3's **7x gradient** is the central finding | **89% the label's `~was_declining` exclusion** | dropping the clause collapsed the spread 46.2 → 5.3 pts (D1) and 66.9 → 7.9 (D2) |
| Test 3 and the simulation differ because sparse pages suffer more | they were never the same quantity | the simulation never applies the exclusion, so 12.6 was never comparable with 46.2 |
| consecutive 30-day means correlate at **r ≈ 0.89** | Pearson **0.799**, Spearman 0.823 | carried in from other work; ML-06 contained no autocorrelation code until it was added |
| `Spearman(peak_ratio, target)` = **−0.044** | **−0.0665** (D1), **−0.1505** (D2) | came from a scratch script, measured against the superseded log target |
| the model **retains 53%** of its advantage on an unseen month | 53% is a ratio between two different **ceilings**, not a decay rate | ML-09 walked four decision points forward, then held the test month fixed and moved only the training date. Over a 31-to-92-day span of staleness the lift moves **0.0009**; over the same walk the base rate moves 0.3169 to 0.7869 and the lift moves 0.2415 to 0.1242 |
| the walk's own lift figures, first run | drifted **0.2449 → 0.2424** between identical runs | `cohort_at` returned rows in DuckDB's aggregation order; `queue_precision` breaks score ties positionally and the random baseline assigns draws positionally, so the bar moved every run. **This is a recurrence** — the same bug is already in the data-corrections table below, fixed once in the source query, then reintroduced by a new helper that did not carry the `ORDER BY` forward. Caught because two cells that should have agreed disagreed; fixed, then verified by running the notebook twice and diffing |

### Data corrections

| correction | effect |
|---|---|
| GSC position is **zero-based** — `SUM(sum_position)/SUM(impressions) + 1` | the previous gate discarded 53.4% of pages' best days |
| `avg_position` weighted by impressions, not a mean of daily means | a page's busy days now count for more than its quiet ones |
| `dim_content` is an **export-time snapshot**, not point-in-time | `days_since_last_update` leaked the future for 77.3% of D1 pages and was removed |
| the `is_deleted` / `is_published` filter was itself outcome-window selection | reverted — it judged a March decision by July status |
| 30-day future window divided by 30, not 31 or 90 | an off-by-one moved the label 49.62% → 50.95%; the 90-day error moved recovery 18.3% → 28.2% |
| `ORDER BY content_hash_id` added to the source query | without it, fixed-seed splits differed between runs |

### What FlyRank confirmed, and what it overturned (2026-08-19)

Most of this project's operating parameters were inferred from public pricing copy because no
authoritative answer was available. One arrived. It is recorded here in full because several sections
above were written against the guesses, and because two of the guesses were right for the wrong
reason while one reversal turned out to have been a mistake.

| FlyRank's statement | Effect |
|---|---|
| queues are generated separately for each client | **confirms** Decision 1, which had rested on inference from account-based pricing |
| customized rather than limited to a global top 50 | **confirms** dropping the global K = 50 queue |
| rebuilt **weekly** | **overturns** the monthly cadence — and re-reverses a change this log recorded as a correction |
| the automated workflow can process **thousands of pages per day** | **overturns** "capacity binds, not ranking quality", the 4.2% ceiling, and K = 100 as a defensible fixed choice |
| treat K as a **configurable per-client budget** | K becomes a parameter to sweep, not a constant |
| report **macro** client metrics | **reverses** this report's pooled-first convention |
| test several K values and outcome windows rather than tying a 30-day label to the queue cadence | **overturns** the label-window justification and defines new work |

**What this does not touch.** Every model comparison was run at one K against one label, so the
ordering of models is unaffected — ridge still leads, `SAFE` still fails to transfer, the ML-07 rule
still ranks below random. The leakage findings, the shared-denominator artefact, the null simulations
and the walk-forward staleness result are all independent of K and cadence. What moves is everything
phrased as *coverage* or *ceiling*, and the reporting convention.

**Owed in ML-10, and now specified rather than guessed:** a sweep over K, a sweep over outcome
window, macro metrics reported beside pooled, and a weekly-spaced rebuild of the walk-forward.

### Status claims corrected

A sweep of the notebooks for promises made and not kept, run after ML-09. Most held: ML-06's Tests
5–10 delivered every check it named, `recall@100` is reported throughout ML-08 as its section 1
demanded, and ML-09 closed all three items the capstone said it owed. These did not.

| stale claim | what was actually true |
|---|---|
| status header: "sections 1, 2 and 8 reflect completed work (ML-02 → ML-05)… ML-06 → ML-10 has not been reached" | ML-06 → ML-09 were all complete; only section 7 was outstanding |
| gap 3: "no walk-forward across multiple decision dates" | ML-08 scored a second date and ML-09 §8 walks four |
| gap 4: "only one of three targets is built… `future_recovery` and `future_momentum` have never been modelled" | both stopped existing on 08-05 when three binary targets became one continuous one — contradicting this document's own section 2 |
| gap 1: "`work/outputs/` is empty" | it holds `baseline_action_score.csv`. The *substance* stands — no feature cache — but the reason given for deferring it expired when ML-06 Test 8 closed hypothesis 3 |
| section 2: categorical encoding "deferred to the modelling stage" | ML-08 settled it; two of the three are unusable and the third moves `P@100` by −0.0001 |
| ML-06 carried an unfilled template line duplicated below its own heading | removed |

**Two methodological notes.** Several withdrawn findings share a cause: analysis run in a scratch
script and then described in prose, rather than recomputed in the notebook that quotes it. Exploring
outside the notebook is fine; anything that survives into it has to be computed there. And the stale
claims above share a different one — they were all written as *forward-looking* statements ("deferred
until…", "has not been reached") that no step in the process ever revisits. A promise dated to a
milestone needs re-reading when that milestone lands; nothing here did that automatically.
