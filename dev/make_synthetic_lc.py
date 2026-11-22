"""Write a synthetic file shaped like the LendingClub accepted-loans export, for testing the notebook only.

The numbers it produces are meaningless. Column names and formats match the real file:
term as " 36 months", issue_d and earliest_cr_line as "Mon-YYYY", int_rate and revol_util as floats,
emp_length as "10+ years" / "< 1 year", realistic cash-flow columns, a mix of statuses including
unresolved late-vintage loans, and two summary rows at the end with no issue date.

Usage: python dev/make_synthetic_lc.py OUTPUT_PATH [N_ROWS]
"""
import sys

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2019-03-01")


def months_between(a, b):
    """Whole months from a to b."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def main(out_path, n=500_000, seed=7):
    rng = np.random.default_rng(seed)

    # issue months 2007-06 .. 2018-12 with volume growing until 2016, like the real book
    months = pd.date_range("2007-06-01", "2018-12-01", freq="MS")
    t = np.arange(len(months))
    weights = np.exp(0.045 * np.minimum(t, 110))
    issue = months[rng.choice(len(months), size=n, p=weights / weights.sum())]
    issue_year = issue.year.to_numpy()

    term = np.where(rng.random(n) < 0.72, 36, 60)

    fico_low = np.clip(np.floor(rng.normal(700, 30, n) / 5) * 5, 660, 845)
    annual_inc = np.round(np.exp(rng.normal(np.log(65_000), 0.55, n)), -2)
    annual_inc[rng.random(n) < 0.0002] = np.nan
    annual_inc[rng.random(n) < 0.0003] = 0.0
    dti = np.round(np.clip(rng.gamma(4.0, 4.5, n), 0, 45), 2)
    dti[rng.random(n) < 0.0005] = np.nan
    dti[rng.random(n) < 0.0002] = -1.0
    delinq = rng.poisson(0.3, n).astype(float)
    inq = rng.poisson(0.7, n).astype(float)
    open_acc = (rng.poisson(10, n) + 1).astype(float)
    pub_rec = rng.poisson(0.2, n).astype(float)
    total_acc = open_acc + rng.poisson(13, n)
    revol_bal = np.round(np.exp(rng.normal(np.log(12_000), 0.9, n)))
    revol_util = np.round(np.clip(rng.beta(2.2, 2.0, n) * 105, 0, None), 1)
    revol_util[rng.random(n) < 0.001] = np.nan
    revol_util[rng.random(n) < 0.0005] = 180.0
    loan_amnt = np.clip(np.round(np.exp(rng.normal(np.log(13_000), 0.55, n)) / 25) * 25, 1000, 40000)

    emp_levels = np.array(["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                           "6 years", "7 years", "8 years", "9 years", "10+ years", None], dtype=object)
    emp_p = np.array([8, 7, 9, 8, 6, 6, 5, 5, 5, 4, 31, 6], dtype=float)
    emp_length = rng.choice(emp_levels, size=n, p=emp_p / emp_p.sum())

    home = rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], size=n, p=[0.49, 0.395, 0.1145, 0.0005])
    verif = rng.choice(["Not Verified", "Source Verified", "Verified"], size=n, p=[0.32, 0.38, 0.30])
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
                "medical", "small_business", "car", "moving", "vacation", "house", "wedding"]
    purpose_p = np.array([58, 22, 6.5, 5.5, 2, 1.2, 1.1, 1.0, 0.8, 0.7, 0.6, 0.6])
    purpose = rng.choice(purposes, size=n, p=purpose_p / purpose_p.sum())

    cr_years = np.clip(rng.gamma(4.0, 4.0, n), 1, 50)
    earliest = issue - pd.to_timedelta(np.round(cr_years * 365.25), unit="D")
    earliest = earliest.to_period("M").to_timestamp()

    # true default risk: origination features, a vintage effect and a hidden part
    hidden = rng.normal(0, 0.8, n)
    vintage = np.select([issue_year >= 2016, issue_year == 2015, issue_year == 2014], [0.35, 0.30, 0.10], 0.0)
    logit = (-2.55 - 0.028 * (fico_low + 2 - 700) + 0.035 * (np.nan_to_num(dti, nan=18) - 18)
             - 0.45 * (np.log1p(np.nan_to_num(annual_inc, nan=65_000)) - 11) + 0.12 * inq + 0.15 * delinq
             + 0.15 * pub_rec + 0.008 * (np.nan_to_num(revol_util, nan=50) - 50)
             + 0.30 * (purpose == "small_business") + 0.15 * (home == "RENT")
             + 0.45 * (term == 60) + vintage + hidden)
    pd_true = 1 / (1 + np.exp(-logit))
    bad = rng.random(n) < pd_true

    # incumbent lender's grade: sees most of the observable risk and part of the hidden part
    lender_score = logit - vintage + 0.5 * rng.normal(0, 0.8, n) - 0.35 * hidden
    cuts = np.quantile(lender_score, [0.17, 0.46, 0.74, 0.89, 0.96, 0.99])
    grade_idx = np.searchsorted(cuts, lender_score)
    grades = np.array(list("ABCDEFG"))[grade_idx]
    sub_grade = np.char.add(grades.astype(str), (rng.integers(1, 6, n)).astype(str))
    base_rate = np.array([7.0, 10.8, 14.0, 17.6, 21.0, 24.5, 27.8])[grade_idx]
    int_rate = np.round(base_rate + rng.normal(0, 0.9, n) + 0.3 * (term == 60), 2)

    r = int_rate / 100 / 12
    funded = loan_amnt.copy()
    installment = np.round(funded * r / (1 - (1 + r) ** (-term)), 2)

    # status as of the snapshot date
    m_since = np.array([months_between(d, SNAPSHOT) for d in issue])
    p_unres = np.where(m_since >= term + 3, 0.004, np.where(m_since >= term, 0.03, 1 - 0.7 * m_since / term))
    unresolved = rng.random(n) < np.clip(p_unres, 0, 1)
    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    status[bad & (rng.random(n) < 0.005)] = "Default"
    early = (issue_year <= 2010) & (rng.random(n) < 0.05)
    status[early] = np.where(bad[early], "Does not meet the credit policy. Status:Charged Off",
                             "Does not meet the credit policy. Status:Fully Paid")
    unres_status = rng.choice(["Current", "Late (31-120 days)", "In Grace Period", "Late (16-30 days)"],
                              size=n, p=[0.90, 0.04, 0.03, 0.03])
    status[unresolved] = unres_status[unresolved]

    # cash flows
    contractual_int = installment * term - funded
    k_bad = np.floor(rng.uniform(0.1, 0.95, n) * term)
    bal_k = funded * ((1 + r) ** term - (1 + r) ** k_bad) / ((1 + r) ** term - 1)
    prepay_frac = np.clip(rng.beta(6.0, 1.5, n), 0.2, 1.0)
    k_cur = np.clip(m_since, 0, term - 1)
    bal_cur = funded * ((1 + r) ** term - (1 + r) ** k_cur) / ((1 + r) ** term - 1)

    is_bad_status = np.isin(status, ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"])
    is_unres = unresolved

    prncp = np.where(is_bad_status, funded - bal_k, funded)
    intr = np.where(is_bad_status, installment * k_bad - (funded - bal_k), contractual_int * prepay_frac)
    prncp = np.where(is_unres, funded - bal_cur, prncp)
    intr = np.where(is_unres, installment * k_cur - (funded - bal_cur), intr)
    late_fee = np.where(rng.random(n) < np.where(is_bad_status, 0.3, 0.03), rng.uniform(15, 50, n), 0.0)
    rec = np.where(is_bad_status & (rng.random(n) < 0.6), bal_k * rng.beta(1, 6, n), 0.0)
    rec_fee = rec * 0.18
    total_pymnt = prncp + intr + late_fee + rec
    noisy = rng.random(n) < 0.005
    total_pymnt = np.where(noisy, total_pymnt + rng.uniform(-50, 50, n), total_pymnt)

    df = pd.DataFrame({
        "id": (np.arange(n) + 1_000_000).astype(str),
        "member_id": np.nan,
        "loan_amnt": loan_amnt,
        "funded_amnt": funded,
        "funded_amnt_inv": funded,
        "term": np.where(term == 36, " 36 months", " 60 months"),
        "int_rate": int_rate,
        "installment": installment,
        "grade": grades,
        "sub_grade": sub_grade,
        "emp_title": "Analyst",
        "emp_length": emp_length,
        "home_ownership": home,
        "annual_inc": annual_inc,
        "verification_status": verif,
        "issue_d": issue.strftime("%b-%Y"),
        "loan_status": status,
        "purpose": purpose,
        "zip_code": "123xx",
        "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], size=n),
        "dti": dti,
        "delinq_2yrs": delinq,
        "earliest_cr_line": earliest.strftime("%b-%Y"),
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        "inq_last_6mths": inq,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "out_prncp": np.where(is_unres, bal_cur, 0.0).round(2),
        "total_pymnt": total_pymnt.round(2),
        "total_rec_prncp": prncp.round(2),
        "total_rec_int": intr.round(2),
        "total_rec_late_fee": late_fee.round(2),
        "recoveries": rec.round(2),
        "collection_recovery_fee": rec_fee.round(2),
        "last_pymnt_d": "Feb-2019",
        "policy_code": 1.0,
    })
    footer = pd.DataFrame({c: [np.nan, np.nan] for c in df.columns})
    footer["id"] = ["Total amount funded in policy code 1: 1234567", "Total amount funded in policy code 2: 7654321"]
    df = pd.concat([df, footer], ignore_index=True)
    df.to_csv(out_path, index=False, compression="gzip" if str(out_path).endswith(".gz") else None)
    print(f"wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    out = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
    main(out, rows)
