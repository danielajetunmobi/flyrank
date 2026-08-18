"""What each code cell does and why, for the code hand-off PDF.

Keyed by (notebook stem, cell index). A cell with no entry is rendered with a
generated summary and flagged in the document, so nothing is silently skipped.
"""

EXPLAIN = {
 ("w01_research_question", 6):
   "Loads the anonymised starter CSV and prints the numbers that justify the lane: the split of "
   "pages by trend_direction, how many are old enough to support a prior-and-future window, and "
   "how many are visible enough to be worth reviewing. The trend_pct range was added later, when "
   "ML-03 was found citing a figure this notebook had never printed. It shows the measure running "
   "from -100% to +44,900% -- bounded below, unbounded above -- which is the asymmetry argument "
   "for using asinh rather than a raw percentage.",

 ("w02_ml_task_framing", 9):
   "Builds the first real cohort from the daily fact table: one row per content item at the "
   "decision point, with a 90-day prior window and a 30-day future window. This fixes the grain "
   "of the problem, and every later notebook inherits it.",
 ("w02_ml_task_framing", 11):
   "The window sanity check, comparing a 30-vs-30 day trend split against a 45-vs-45 split on the "
   "same pages. It originally reported only how often the two disagree at a -20% cut; the "
   "continuous agreement (Spearman +0.565) was added later, because a threshold-specific "
   "disagreement rate says more about the threshold than about the windows.",
 ("w02_ml_task_framing", 12):
   "Computes prior trend and the three original binary labels and prints their base rates. Kept "
   "after the labels were superseded, because these are the numbers the amendment argues against.",
 ("w02_ml_task_framing", 13):
   "The capacity arithmetic behind the per-client queue: how many declines exist, how many a "
   "perfect model could catch at each audit tier, and how a per-client queue compares with one "
   "global pile. It originally averaged per-client rates against a pooled global rate -- two "
   "different quantities -- and now pools both sides.",
 ("w02_ml_task_framing", 17):
   "Tests the rejected 90-day baseline against the adopted 30-day one, so the choice rests on a "
   "measured difference rather than an assertion.",

 ("w03_data_contract", 6):
   "Opens the warehouse over DuckDB and verifies the fact table: row counts, date coverage, and "
   "the grain claim that one row is one page-day. The contract begins by proving the table is "
   "what the documentation says it is.",
 ("w03_data_contract", 8):
   "Checks GA4 availability across the portfolio. This is where the first evidence appears that "
   "engagement data is far sparser than the column list suggests.",
 ("w03_data_contract", 10):
   "Verifies the GSC columns and establishes that position is zero-based, so average position is "
   "SUM(sum_position)/SUM(impressions) + 1. An earlier draft applied the starter CSV's opposite "
   "convention and discarded the best days of half the cohort.",
 ("w03_data_contract", 12):
   "Sweeps the remaining fact-table columns for missingness and zero-inflation, so ML-05's feature "
   "vector starts from measured availability rather than assumption.",
 ("w03_data_contract", 14):
   "The page-day sweep, added after ML-09 found it missing. Every other check in this notebook -- "
   "including the one directly above -- groups by content and sums over 90 days, which is the right "
   "grain for a feature and the wrong grain for finding a broken row: one impossible day disappears "
   "the moment it is summed. This queries the fact table without aggregating, reports each numeric "
   "column's maximum against its own 99th percentile, and separately tests bounds that real data "
   "cannot violate. It runs on all seven months rather than the five this notebook's cohort needs, "
   "because a data contract describes the table rather than one cohort's view of it -- on five "
   "months it reported ga4_pageviews at 4,964 against 64,750 across the warehouse.",
 ("w03_data_contract", 17):
   "Profiles dim_content: what each column means, how often it is populated, and which fields are "
   "export-time snapshots rather than point-in-time records. The snapshot property found here is "
   "what later makes content_updated_date unusable.",
 ("w03_data_contract", 19):
   "Profiles dim_clients, including gsc_data_start, which determines whether a client's history "
   "actually covers the prior window a decision point requires.",
 ("w03_data_contract", 21):
   "Examines FlyRank's own optimisation flags to decide whether they can be features. They cannot: "
   "a flag encodes a decision someone already made, so they are recorded as baselines to beat.",

 ("w03_feature_leakage_check", 2):
   "Builds the ML-05 cohort and feature vector -- fact-table windows, the dim_content join, "
   "derived rates and availability flags. The comment block records why days_since_last_update was "
   "removed and why features are split by provenance into windowed fact columns and July-2026 "
   "snapshot columns.",
 ("w03_feature_leakage_check", 4):
   "Six checks on the zero-based position question, run because the starter CSV and the warehouse "
   "disagree about what a position of 0 means. Establishes that 0 is the top rank here.",
 ("w03_feature_leakage_check", 9):
   "Defines fit_scaled, the standardise-then-fit helper every probe in this notebook uses, and "
   "sets up the honest and snapshot feature lists.",
 ("w03_feature_leakage_check", 12):
   "Defines precision_at_k and runs the first single-split probe. The 1.72x lift it produced was "
   "later withdrawn: one grouped split cannot support a claim on this data, which the next cell "
   "demonstrates directly.",
 ("w03_feature_leakage_check", 15):
   "Runs the same probe over ten grouped splits and reports mean, range and spread. This is the "
   "cell that retired the single-split result -- AUC averages 0.502 across seeds ranging 0.465 to "
   "0.534.",
 ("w03_feature_leakage_check", 19):
   "Measures what the eight dim_content snapshot columns contribute, and what a point-in-time "
   "reconstruction of freshness contributes. Both move AUC by less than 0.01.",
 ("w03_feature_leakage_check", 21):
   "Restricts the feature set to fact-table columns only, isolating the snapshot columns' effect "
   "rather than inferring it.",
 ("w03_feature_leakage_check", 24):
   "Builds days_since_update_pit, an honest point-in-time freshness feature. Only 22.8% of pages "
   "have a visible pre-decision update, so most rows fall back to creation date.",
 ("w03_feature_leakage_check", 27):
   "Compares a global queue against a per-client queue at K=100. It originally averaged per-client "
   "rates against a pooled global figure and reported a 39x gain; both sides are pooled now and "
   "the gain is about 3x.",
 ("w03_feature_leakage_check", 30):
   "Confirms the scaler is fitted on training data only -- a small check that catches a common and "
   "otherwise invisible form of leakage.",
 ("w03_feature_leakage_check", 33):
   "Attack 1: injects future_change_pct, the exact quantity the label is computed from. AUC goes "
   "to 0.919, which is the confession a leaky feature is supposed to produce.",
 ("w03_feature_leakage_check", 35):
   "Attack 2: injects a raw future impression count. It leaks only weakly -- the first sign that a "
   "partial overlap with the label does not announce itself loudly.",
 ("w03_feature_leakage_check", 37):
   "Attack 3: swaps the client-grouped split for a random row split. The score rises by +0.091 AUC "
   "on every one of ten seeds. That gap is memorised client structure, and it is why every later "
   "comparison uses grouped splits.",
 ("w03_feature_leakage_check", 39):
   "States the feature timeline in one line: the latest date any feature can reach is the day "
   "before the decision point.",

 ("w04_signal_audit", 2):
   "Defines build_cohort and runs it at two decision points, so every verdict in the notebook can "
   "be checked twice. All the ML-05 corrections are folded in here.",
 ("w04_signal_audit", 4):
   "Defines bucket_table and the verdict helper, including the sample-size floor the "
   "signal-audit skill requires: no verdict from a bucket under 50 rows.",
 ("w04_signal_audit", 7):
   "Describes every numeric field in the cohort. The heavy tails visible here are what force log "
   "scaling and rank-based correlation in the tests that follow.",
 ("w04_signal_audit", 8):
   "Draws density curves with a normal-fit overlay on log-scaled axes, so the shape of each "
   "distribution is visible rather than asserted.",
 ("w04_signal_audit", 11):
   "Measures what the cohort filter removes. A third of active pages never enter the study, and "
   "the group removed is not random -- it includes every page that fell to zero.",
 ("w04_signal_audit", 14):
   "Counts pages reaching zero, measures how much traffic they carried, and compares a log ratio "
   "against an asinh difference. This cell settles the target's form: 7.4% of pages reach zero, "
   "and asinh is finite for all of them.",
 ("w04_signal_audit", 18):
   "Runs the three headline signal tests -- prior trend, CTR by rank, and peak ratio -- side by "
   "side on both decision points.",
 ("w04_signal_audit", 21):
   "Decomposes Test 3 by removing the label's own exclusion rule. The gradient collapses from 46.2 "
   "points to 5.3, which is how that finding turned out to be 89% definitional.",
 ("w04_signal_audit", 23):
   "Reports the arithmetic bound on peak_ratio: because the recent window sits inside the 90-day "
   "window that divides it, the ratio cannot exceed 3.",
 ("w04_signal_audit", 26):
   "The null simulation. Each page's real future is replaced with a random 30-day window from its "
   "own history, preserving level and volatility while destroying all information about what "
   "happened next. The null produces 79.0 points of gradient against 12.6 observed.",
 ("w04_signal_audit", 30):
   "Tests FlyRank's page_one_decay_risk rule by decomposing it into its two conditions rather than "
   "only testing the pair.",
 ("w04_signal_audit", 33):
   "Reruns the pooled verdicts client by client -- the slice the signal-audit skill asks for and "
   "the notebook had skipped.",
 ("w04_signal_audit", 35):
   "Tests whether missingness follows a category. It does: four keyword columns are absent for an "
   "entire content type, so filling them injects the content type into the features.",
 ("w04_signal_audit", 37):
   "Checks whether the CTR ratio can exceed 100%, the trap the skill warns about when numerator "
   "and denominator come from different systems. It cannot, so the CTR gap was never a broken "
   "denominator.",
 ("w04_signal_audit", 41):
   "Hypothesis 3's relaxed-filter comparison, computable only once asinh removed the need for a "
   "non-zero denominator. The filter inflates the decline rate by 12.6 points at D1.",
 ("w04_signal_audit", 44):
   "Signal-tests GA4 engagement, the one data source never previously tested, pooled and per "
   "client.",
 ("w04_signal_audit", 47):
   "The adversarial check on Test 5, using the same null that broke Test 3. It shows the "
   "per-client correlations are the shared-denominator artefact rather than a within-client "
   "relationship.",

 ("w04_baseline_score", 2):
   "ML-07's setup and the baseline-selection test: four candidate denominators for the target, "
   "each scored by whether a randomised future can reproduce its correlation.",
 ("w04_baseline_score", 4):
   "Runs the candidate comparison and prints the artefact share -- what the null produces relative "
   "to what is observed. Candidate C, the days 30-90 window, inverts the ratio.",
 ("w04_baseline_score", 7):
   "Rebuilds the cohort on the settled target. Because asinh needs no non-zero denominator, the "
   "two trend filters are gone and the cohort roughly doubles.",
 ("w04_baseline_score", 9):
   "The availability check: for every candidate field, what share of pages have it, per client and "
   "per content type. Six clients have no GA4 at all; seven have no backlinks data.",
 ("w04_baseline_score", 12):
   "Codes the rule and its reason codes -- slipping, worth saving, mature -- so every scored page "
   "carries why it scored.",
 ("w04_baseline_score", 14):
   "Evaluates the rule against three nulls and writes the queue CSV. The persistence null returns "
   "1.0000, which is how the rule turned out to describe the present rather than predict the "
   "future.",
 ("w04_baseline_score", 17):
   "Pulls the top twenty for hand review -- the step the building-baselines skill says is where "
   "bad logic shows itself.",
 ("w04_baseline_score", 20):
   "The leakage check on the rule's inputs, plus a direct measurement of which of the three "
   "permitted fields actually predicts the target.",
 ("w04_baseline_score", 23):
   "The revised rule and the comparison across all three versions, including the per-client "
   "coverage each achieves.",
 ("w04_baseline_score", 26):
   "Scores the frozen baseline on ten client-grouped splits -- the same protocol any model must "
   "use, so ML-08's comparison is fair.",

 ("w05_model", 2):
   "ML-08's setup: cohort, target and the frozen ML-07 gate, rebuilt here so the notebook stands "
   "alone. The printed totals must match ML-07 exactly, or the two notebooks have drifted.",
 ("w05_model", 8):
   "Defines the model set and the pooled per-client precision helper, then trains all eight "
   "configurations on ten grouped splits.",
 ("w05_model", 10):
   "Establishes the random-order bar by averaging twenty shuffles per seed, which is what ML-07's "
   "single-shuffle estimate should have been.",
 ("w05_model", 13):
   "The full metric suite: rank agreement, precision at 20 and 100, recall, AUC and R-squared. The "
   "Spearman column is the metric ML-03 declared primary and ML-08 had not computed.",
 ("w05_model", 16):
   "The segmented-model test: for each enrichment group, restrict to clients with more than 95% "
   "coverage and ask whether the extra columns help there.",
 ("w05_model", 18):
   "Tests the three remaining dim_content date columns. Both optimisation dates postdate the "
   "decision point for 100% of populated rows, making them a record of the intervention rather "
   "than a predictor of it.",
 ("w05_model", 21):
   "Retests the rejected features on bagging and boosting, because every earlier rejection was "
   "decided by a linear model and coefficient-splitting is a linear-model problem.",
 ("w05_model", 25):
   "Tests the categorical columns. Only content_type is universally available, and it adds nothing "
   "on top of the existing features.",
 ("w05_model", 28):
   "Tests CTR and the log transform. CTR proves redundant with position; the log transform "
   "improves AUC and costs precision at K, so it is not adopted.",
 ("w05_model", 31):
   "The lane test: one model, three queues. Sorting the same predictions two ways and splitting on "
   "prior state produces decline, momentum and recovery.",
 ("w05_model", 34):
   "Tests the last two reference-pipeline features and checks the ML-02 claim that stagnant pages "
   "sort to the middle of the ranking on their own.",
 ("w05_model", 37):
   "Error analysis: coefficients, permutation importance on held-out clients, precision by decile "
   "and by client, and three concrete wrong picks.",
 ("w05_model", 39):
   "The touched-once D2 holdout. The model is trained at D1 and scored once on a month it has "
   "never seen.",

 ("w06_validation_audit", 2):
   "ML-09's setup, rebuilt independently and checked against ML-08's printed totals, so the two "
   "notebooks can be compared without trusting a shared file.",
 ("w06_validation_audit", 4):
   "Runs FlyRank's own freshness bucketing on our warehouse twice: once taking "
   "content_updated_date at face value, once restricted to updates verifiably before the decision "
   "point.",
 ("w06_validation_audit", 7):
   "Scores every client across all ten splits, so per-client performance can be examined against "
   "client properties rather than anecdotally.",
 ("w06_validation_audit", 8):
   "Tests three structural explanations for the per-client spread before allowing 'clients differ' "
   "to stand. Twelve clients have fewer pages than the queue has slots, which makes their lift "
   "zero by construction.",
 ("w06_validation_audit", 13):
   "The leakage audit, opening with a positive control: inject the answer and confirm the harness "
   "detects it. Then timeline, ablation, population selection and product flags.",
 ("w06_validation_audit", 16):
   "The GA4 segment test ML-08 skipped. Only six clients have usable GA4 once actual sessions are "
   "required, which is too few to hold out.",
 ("w06_validation_audit", 19):
   "A placeholder. Section 4 of ML-09 is the claim-rewrite exercise, which computes nothing -- the "
   "figures it quotes are produced by cells in ML-06 and cited rather than recalculated. The cell "
   "prints that fact rather than sitting empty, so a reader does not wonder whether a computation "
   "was lost.",
 ("w06_validation_audit", 21):
   "Reads the three failure cases individually and tests whether a rank-transformed target closes "
   "boosting's gap. It does not.",
 ("w06_validation_audit", 28):
   "Two late corrections, both to work already committed. It shows that ga4_data_available has three "
   "states rather than two -- 58.30% FALSE, 37.65% NULL, 4.04% TRUE -- so only about one page-day in "
   "twenty-five has tracking on, and the flag separates 'tracked but idle' from 'not tracked' for "
   "just 1.40% of flag-true rows. Then it sweeps the raw fact table for extremes, which nothing in "
   "this project had done: every distribution was computed on the cohort, which is already 90-day "
   "sums, so a single impossible day disappears once summed. Four GA4 columns exceed 10,000x their "
   "own 99th percentile. The cell also carries two of its own bugs in the record -- a 'tracked but "
   "idle' figure computed for flag states where it means nothing, and a seconds-to-hours conversion "
   "that followed whichever column topped the ranking and so labelled a session count as hours.",
 ("w06_validation_audit", 25):
   "The hyperparameter check the project never ran. Ridge gets its alpha chosen by leave-one-out "
   "inside the training fold; boosting gets a small grid scored on an inner grouped split carved "
   "out of the training clients. Tuning inside the fold is what stops the held-out clients from "
   "influencing the choice. Ridge is unmoved -- +0.0000 on every metric, across alphas spanning "
   "five orders of magnitude -- while boosting gains +0.0120 and still trails. What the tuner asks "
   "for matters more than the gain: every seed picked the lower learning rate and most picked "
   "shallower trees, so the search is trying to make the ensemble more linear.",
}
