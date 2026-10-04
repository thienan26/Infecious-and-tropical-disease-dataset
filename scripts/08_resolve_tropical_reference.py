from collections import defaultdict
from pathlib import Path
import json
import re
import pandas as pd


KB_PATH = Path(
    "taxonomy/disease_kb_base.parquet"
)

SEED_PATH = Path(
    "taxonomy/references/"
    "tropical_seed.csv"
)

OUTPUT = Path(
    "taxonomy/"
    "tropical_reference.parquet"
)

UNRESOLVED = Path(
    "reports/scope/"
    "tropical_unresolved.csv"
)


def normalize_name(value):

    value = (
        str(value)
        .lower()
        .strip()
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


kb = pd.read_parquet(
    KB_PATH
)

seed = pd.read_csv(
    SEED_PATH
)


alias_index = defaultdict(
    set
)


for row in kb.itertuples():

    names = [
        row.disease,
        *row.aliases,
    ]

    for name in names:

        key = normalize_name(name)

        if key:
            alias_index[
                key
            ].add(
                row.concept_id
            )


resolved = []
unresolved = []


for row in seed.itertuples():

    query = normalize_name(
        row.query_name
    )

    candidates = sorted(
        alias_index.get(
            query,
            set(),
        )
    )

    if len(candidates) != 1:

        unresolved.append({
            "query_name":
                row.query_name,

            "source_id":
                row.source_id,

            "evidence_type":
                row.evidence_type,

            "candidate_ids":
                json.dumps(
                    candidates
                ),

            "reason":
                "NO_MATCH"
                if len(candidates) == 0
                else "AMBIGUOUS_MATCH",
        })

        continue

    cid = candidates[0]

    disease = kb.loc[
        kb["concept_id"] == cid,
        "disease",
    ].iloc[0]

    resolved.append({
        "concept_id":
            cid,

        "disease":
            disease,

        "is_tropical":
            True,

        "source_id":
            row.source_id,

        "evidence_type":
            row.evidence_type,

        "query_name":
            row.query_name,
    })


resolved_df = pd.DataFrame(
    resolved
)

unresolved_df = pd.DataFrame(
    unresolved
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

UNRESOLVED.parent.mkdir(
    parents=True,
    exist_ok=True,
)


resolved_df.to_parquet(
    OUTPUT,
    index=False,
)

unresolved_df.to_csv(
    UNRESOLVED,
    index=False,
    encoding="utf-8-sig",
)


print(
    "Resolved:",
    len(resolved_df)
)

print(
    "Unresolved:",
    len(unresolved_df)
)


if len(unresolved_df):

    print()
    print(
        unresolved_df.to_string(
            index=False
        )
    )