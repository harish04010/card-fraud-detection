"""
Exports the test-period scores to site/data.json for the alert-queue simulator.

For each model it precomputes, at every review capacity from 5 to 2,000
alerts, how many frauds are caught and how many euros they were worth, so the
page can recompute net savings instantly for any review cost.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"score_logistic": "Logistic regression", "score_gradient_boosting": "Gradient boosting",
          "score_isolation_forest": "Isolation forest (anomaly detection)"}


def export() -> str:
    r = json.loads((ROOT / "results" / "results.json").read_text())
    t = pd.read_csv(ROOT / "results" / "test_scores.csv")
    caps = sorted(set(np.unique(np.round(np.geomspace(5, 2000, 120)).astype(int)).tolist()))
    curves = {}
    for col, label in {**MODELS, "amount": "Rule: review the largest amounts"}.items():
        order = t.sort_values(col, ascending=False)
        fraud = order["is_fraud"].to_numpy()
        amt = np.where(fraud == 1, order["amount"].to_numpy(), 0.0)
        cf, ca = np.cumsum(fraud), np.cumsum(amt)
        curves[col] = {"label": label, "caught": [int(cf[k - 1]) for k in caps],
                       "amount": [round(float(ca[k - 1]), 2) for k in caps]}
    top = t.sort_values("score_logistic", ascending=False).head(60)
    queue = [{"hour": int((row.time // 3600) % 24), "amount": round(float(row.amount), 2),
              "score": round(float(row.score_logistic), 4), "fraud": int(row.is_fraud)} for row in top.itertuples()]
    frauds = t[t.is_fraud == 1]
    data = {
        "caps": caps, "curves": curves, "queue": queue,
        "test": {"transactions": int(len(t)), "fraud": int(t.is_fraud.sum()),
                 "hours": round(float((t.time.max() - t.time.min()) / 3600), 1),
                 "fraud_amount": round(float(frauds.amount.sum()), 2)},
        "assumptions": r["assumptions"], "no_review": r["no_review"]["fraud_losses"],
        "metrics": {k: {"pr_auc": v["pr_auc"], "roc_auc": v["roc_auc"], "chosen": v["cost_threshold_on_test"]}
                    for k, v in r["models"].items()},
        "data": {k: r["data"][k] for k in ("raw_rows", "duplicates_removed", "fraud", "fraud_rate", "median_amount", "share_of_fraud_under_10")},
        "by_hour": [{"h": int(h["hour"]), "rate": h["fraud_rate"], "n": h["transactions"]} for h in r["by_hour"]],
    }
    out = ROOT / "site" / "data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    return f"{out}: {out.stat().st_size:,} bytes"


if __name__ == "__main__":
    print(export())
