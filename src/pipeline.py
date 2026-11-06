"""
Card fraud detection, evaluated the way a fraud operation runs.

A fraud team can review only so many alerts. So the questions are: of the
transactions the team has time to look at, how many are fraud, how much
money (euros) does that stop, and where should the alert threshold sit once review
costs are counted? Accuracy is meaningless at 0.17% fraud: flagging nothing
is 99.83% accurate.

    python -m src.pipeline
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "creditcard.csv"
SOURCE_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
RESULTS, DOCS = ROOT / "results", ROOT / "docs"
SEED = 13

# Operations assumptions, stated so they can be argued with.
REVIEW_COST = 4.0        # analyst time to review one alert, euros (the data is European cardholders)
CHARGEBACK_FEE = 15.0    # what a missed fraud costs on top of the amount itself
CAPACITY = [0.001, 0.002, 0.005]  # share of transactions the team can review


def load() -> tuple[pd.DataFrame, dict]:
    if not DATA.exists():
        import urllib.request
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, DATA)
    raw = pd.read_csv(DATA)
    df = raw.drop_duplicates().sort_values("Time").reset_index(drop=True)
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    cut1, cut2 = df["Time"].quantile([0.6, 0.8])
    df["split"] = np.where(df["Time"] <= cut1, "train", np.where(df["Time"] <= cut2, "val", "test"))
    info = {"raw_rows": len(raw), "duplicates_removed": len(raw) - len(df), "fraud": int(df["Class"].sum()),
            "fraud_rate": float(df["Class"].mean()), "hours": float(df["Time"].max() / 3600),
            "splits": {s: {"transactions": int((df.split == s).sum()), "fraud": int(df.loc[df.split == s, "Class"].sum()),
                           "fraud_rate": float(df.loc[df.split == s, "Class"].mean())} for s in ("train", "val", "test")},
            "median_amount": {"fraud": float(df.loc[df.Class == 1, "Amount"].median()),
                              "legit": float(df.loc[df.Class == 0, "Amount"].median())},
            "share_of_fraud_under_10": float((df.loc[df.Class == 1, "Amount"] < 10).mean())}
    return df, info


FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]


def models() -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.1, max_iter=3000)),
        "gradient_boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                                            l2_regularization=1.0, class_weight="balanced",
                                                            random_state=SEED),
        # No labels at all: scores how unusual a transaction looks.
        "isolation_forest": IsolationForest(n_estimators=300, contamination="auto", random_state=SEED),
    }


def scores(name, model, X_train, y_train, X):
    if name == "isolation_forest":
        model.fit(X_train[y_train == 0])  # learn what normal looks like
        return -model.score_samples(X)    # higher = more unusual
    model.fit(X_train, y_train)
    return model.predict_proba(X)[:, 1]


def at_capacity(score, y, amount, share) -> dict:
    k = max(1, int(round(share * len(score))))
    top = np.argsort(-score)[:k]
    caught = y[top] == 1
    return {"capacity": share, "alerts": k, "fraud_caught": int(caught.sum()), "precision": float(caught.mean()),
            "recall": float(caught.sum() / y.sum()), "dollars_caught": float(amount[top][caught].sum()),
            "dollar_recall": float(amount[top][caught].sum() / amount[y == 1].sum())}


def net_savings(score, y, amount, threshold) -> dict:
    flag = score >= threshold
    caught = flag & (y == 1)
    saved = (amount[caught] + CHARGEBACK_FEE).sum()
    cost = flag.sum() * REVIEW_COST
    return {"threshold": float(threshold), "alerts": int(flag.sum()), "fraud_caught": int(caught.sum()),
            "saved": float(saved), "review_cost": float(cost), "net": float(saved - cost)}


def best_threshold(score, y, amount) -> float:
    """Pick the threshold that maximizes net savings: on validation data only."""
    cands = np.unique(np.quantile(score, np.linspace(0.95, 0.99995, 400)))
    return float(max(cands, key=lambda t: net_savings(score, y, amount, t)["net"]))


def main():
    df, info = load()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, ytr = tr[FEATURES], tr["Class"].to_numpy()
    results = {"data": info, "assumptions": {"review_cost": REVIEW_COST, "chargeback_fee": CHARGEBACK_FEE}, "models": {}}

    test_scores = {}
    for name, m in models().items():
        s_val = scores(name, m, Xtr, ytr, va[FEATURES])
        s_test = m.predict_proba(te[FEATURES])[:, 1] if name != "isolation_forest" else -m.score_samples(te[FEATURES])
        test_scores[name] = s_test
        y, amt = te["Class"].to_numpy(), te["Amount"].to_numpy()
        thr = best_threshold(s_val, va["Class"].to_numpy(), va["Amount"].to_numpy())
        results["models"][name] = {
            "pr_auc": float(average_precision_score(y, s_test)), "roc_auc": float(roc_auc_score(y, s_test)),
            "at_capacity": [at_capacity(s_test, y, amt, c) for c in CAPACITY],
            "cost_threshold_on_test": net_savings(s_test, y, amt, thr),
        }

    # A rule a team might run without a model: review the largest transactions.
    y, amt = te["Class"].to_numpy(), te["Amount"].to_numpy()
    results["rule_largest_amounts"] = {"pr_auc": float(average_precision_score(y, amt)),
                                       "at_capacity": [at_capacity(amt, y, amt, c) for c in CAPACITY]}
    results["no_review"] = {"fraud_losses": float(amt[y == 1].sum() + CHARGEBACK_FEE * y.sum())}

    # Hour-of-day pattern (relative hours: the data doesn't say when the clock starts).
    by_hour = df.groupby("hour").agg(transactions=("Class", "size"), fraud=("Class", "sum"))
    by_hour["fraud_rate"] = by_hour["fraud"] / by_hour["transactions"]
    results["by_hour"] = by_hour.reset_index().to_dict(orient="records")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=1))
    pd.DataFrame({"time": te["Time"], "amount": te["Amount"], "is_fraud": te["Class"],
                  **{f"score_{k}": v for k, v in test_scores.items()}}).to_csv(RESULTS / "test_scores.csv", index=False)
    figures(results, te, test_scores)
    report(results)


def figures(r, te, test_scores):
    DOCS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"gradient_boosting": "#1f3a5f", "logistic": "#6b8fb8", "isolation_forest": "#c08a2b", "rule": "#9aa5ae"}
    y, amt = te["Class"].to_numpy(), te["Amount"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    shares = np.linspace(0.0002, 0.01, 60)
    for name, s in list(test_scores.items()) + [("rule", amt)]:
        rec = [at_capacity(s, y, amt, c)["recall"] for c in shares]
        ax.plot(shares * 100, rec, lw=2.2, color=colors[name], label=name.replace("_", " "))
    for c in CAPACITY:
        ax.axvline(c * 100, color="#d0d5db", lw=1, zorder=0)
    ax.set(xlabel="Share of transactions the team reviews (%)", ylabel="Share of fraud caught",
           title="Fraud caught for a given review workload (test period)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(DOCS / "capacity.png", dpi=160); plt.close(fig)

    bh = pd.DataFrame(r["by_hour"])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(bh["hour"], bh["fraud_rate"] * 100, color="#1f3a5f")
    ax.set(xlabel="Hour of day (relative: the data doesn't say when hour 0 is)", ylabel="Fraud rate (%)",
           title="Fraud rate by hour: highest when legitimate volume is lowest")
    fig.tight_layout(); fig.savefig(DOCS / "by_hour.png", dpi=160); plt.close(fig)


def report(r):
    d = r["data"]
    print(f"{d['raw_rows']:,} rows, {d['duplicates_removed']} duplicates removed, {d['fraud']} fraud ({d['fraud_rate']:.3%})")
    for s, v in d["splits"].items():
        print(f"  {s:<5} {v['transactions']:>7,} txns  {v['fraud']:>3} fraud  {v['fraud_rate']:.3%}")
    for name, m in list(r["models"].items()) + [("rule_largest_amounts", r["rule_largest_amounts"])]:
        caps = "  ".join(f"@{c['capacity']:.1%}: {c['fraud_caught']} caught, prec {c['precision']:.0%}, €{c['dollars_caught']:,.0f}"
                         for c in m["at_capacity"])
        print(f"{name:<22} PR-AUC {m['pr_auc']:.3f}  {caps}")
        if "cost_threshold_on_test" in m:
            c = m["cost_threshold_on_test"]
            print(f"{'':<22} cost threshold: {c['alerts']} alerts, {c['fraud_caught']} caught, saved €{c['saved']:,.0f}, "
                  f"review €{c['review_cost']:,.0f}, net €{c['net']:,.0f}")
    print(f"no review: losses €{r['no_review']['fraud_losses']:,.0f}")


if __name__ == "__main__":
    main()
