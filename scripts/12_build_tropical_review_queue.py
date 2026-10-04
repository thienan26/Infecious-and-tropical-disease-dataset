


#kqua: reports/scope/tropical_review_queue.csv
from pathlib import Path
import pandas as pd


scope = pd.read_parquet(
    "reports/scope/"
    "disease_scope.parquet"
)


unknown = scope[
    scope[
        "tropical_status"
    ]
    == "unknown"
].copy()


unknown = unknown.sort_values(
    "candidate_cases",
    ascending=False,
)


unknown[
    [
        "concept_id",
        "disease",
        "candidate_cases",
        "raw_mentions",
    ]
].to_csv(
    "reports/scope/"
    "tropical_review_queue.csv",
    index=False,
    encoding="utf-8-sig",
)


print(
    "Unknown tropical status:",
    len(unknown)
)

print()

print(
    unknown[
        [
            "concept_id",
            "disease",
            "candidate_cases",
        ]
    ]
    .head(100)
    .to_string(
        index=False
    )
)