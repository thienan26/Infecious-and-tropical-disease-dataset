# MultiCaRe thực sự chứa disease gì

from pathlib import Path
import json
import pandas as pd


CATALOG = pd.read_parquet(
    "reports/scope/"
    "disease_catalog.parquet"
)


scope = CATALOG[
    (
        CATALOG[
            "candidate_cases"
        ]
        > 0
    )
    &
    (
        CATALOG[
            "is_infectious"
        ]
        == True
    )
].copy()


scope = scope.sort_values(
    [
        "candidate_cases",
        "raw_mentions",
    ],
    ascending=False,
)


# Lists do not display nicely
# in Excel, so serialize them.
scope[
    "tropical_sources"
] = scope[
    "source_id"
].apply(
    lambda value:
        json.dumps(
            value
            if isinstance(
                value,
                list,
            )
            else [],
            ensure_ascii=False,
        )
)


columns = [
    "concept_id",
    "disease",

    "is_infectious",
    "infectious_source",

    "is_tropical",
    "tropical_status",
    "tropical_sources",

    "raw_mentions",
    "candidate_cases",
    "text_candidate_cases",
    "metadata_candidate_cases",
]


scope[
    columns
].to_csv(
    "reports/scope/"
    "disease_scope.csv",
    index=False,
    encoding="utf-8-sig",
)


scope.to_parquet(
    "reports/scope/"
    "disease_scope.parquet",
    index=False,
)


print(
    "Diseases in project scope:",
    len(scope)
)

print()

print(
    scope[
        [
            "concept_id",
            "disease",
            "tropical_status",
            "candidate_cases",
        ]
    ]
    .head(100)
    .to_string(
        index=False
    )
)