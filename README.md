# Credit Policy Lab

## Question

Which loans should a lender approve, and what does that do to profit?

This project uses historical LendingClub loans to build a default model, turns it into an approval policy that maximises profit, and shows how robust that policy is to out-of-time drift and to stress in funding cost and credit losses. It answers a lender's question, not a modelling-contest question: the headline numbers are money, not just AUC.

## Notebooks

- `credit_policy_lab.ipynb`: full version (SQL cohort in DuckDB, own WoE scorecard, LightGBM, calibration, PSI, reason codes, caching, stress test, memo). 
- `credit_policy_simple.ipynb`: 22-cell short version of the same pipeline, easier to read and extend.

## Approach

1. **Cohort (DuckDB SQL).** 36-month loans issued 2012 onward. `bad = 1` for Charged Off / Default, `0` for Fully Paid. A maturity filter drops issue months where more than 1% of loans are still unresolved, so late vintages are not flattered by right-censoring.
2. **Out-of-time splits.** Train 2012-2013, validation 2014, test 2015, split by issue date. No random split.
3. **Features from an allowlist only.** Origination-time applicant fields plus SQL window features (income percentile among same-purpose peers, DTI against a trailing peer average). `int_rate`, `grade`, `sub_grade` and `installment` are never features: they are the incumbent lender's own risk price (and `installment / loan_amnt` is a function of `int_rate` at a fixed term). Leakage assertions run at every stage.
4. **Scorecard.** Own Weight of Evidence binning, IV-based selection, L2 logistic regression, points scaling (base 600, PDO 40).
5. **LightGBM challenger.** Monotone constraints where the direction of risk is unambiguous, early stopping on validation.
6. **Calibration.** Isotonic regression fitted on validation; knots saved as JSON. No resampling, because the profit layer needs real probabilities.
7. **Evaluation.** AUC, KS, Gini, Brier, bootstrap intervals, lift, PSI, against the incumbent's `int_rate` as a baseline score.
8. **Economics and policy.** Expected profit per loan from calibrated PD, contractual yield, prepayment and loss-given-default parameters estimated on train, funding cost and opex. Approve when expected profit beats a hurdle chosen on validation; backtest once on test with realised cash-flow P&L, against approve-all and the incumbent grade policy.
9. **Stress test.** Funding cost and PD / loss multipliers with decisions held fixed, including the breakeven PD multiplier.
10. **Credit committee memo** generated from computed results.

Every threshold, hurdle and hyperparameter is chosen on validation. The test set is only used for final evaluation.

## How to run

The data file is the public LendingClub accepted-loans export, `accepted_2007_to_2018Q4.csv.gz`, from the Kaggle dataset *Lending Club Loan Data* (kaggle.com/datasets/wordsforthewise/lending-club). Download it by hand; the notebook does not download anything.

**Google Colab**
1. In Google Drive, create `MyDrive/credit_policy_lab/data/` and upload `accepted_2007_to_2018Q4.csv.gz` there.
2. Open `credit_policy_lab.ipynb` in Colab and choose *Runtime > Run all*. Allow the Drive mount when asked.
3. Missing packages are installed automatically. A CPU runtime is enough.

**Local**
```
pip install -r requirements.txt
```
Put the file at `credit_policy_lab/data/accepted_2007_to_2018Q4.csv.gz` next to the notebook (the unzipped `.csv` also works), then run the notebook top to bottom in Jupyter.

**Runtime.** The first run reads the full CSV once (a few minutes) and trains the models. Later runs load the cached parquet files and saved models and finish in a few minutes on a Colab CPU. Set `DEMO_MODE = True` in the first cell for a quick smoke run on a 150k-loan sample; set `FORCE_RETRAIN = True` to retrain the models. Changing any config value retrains exactly the artifacts that depend on it (tracked in `models/manifest.json`).

Outputs: figures in `credit_policy_lab/reports/figures/`, headline numbers in `credit_policy_lab/reports/results_summary.json` and `.md`.

## Results

The table below is written by the notebook from `reports/results_summary.md` when it is run from this folder. It is empty until the notebook has been run on the real data.

<!-- results:start -->
| Metric | Value |
|---|---|
| Status | Not yet run on the real data |
<!-- results:end -->

## Limitations

- US consumer loans, not MSME credit. The method transfers; the numbers do not.
- Reject inference: outcomes exist only for loans the original lender approved, so the policy can only be tightened, never loosened, and rejected applicants are invisible.
- Time value of money is ignored in profit; funding cost and opex are assumptions.
- Final loan status only, with no monthly performance data, so no roll rates or months-on-book curves.
- The expected-value model is simplified: one average loss-per-dollar parameter for bad loans and one prepayment factor for good loans.
- Origination features only; no behavioural or bureau-trend data.
