from pathlib import Path
import pandas as pd


BASE = pd.read_parquet(
    "taxonomy/"
    "disease_kb_base.parquet"
)

TROPICAL = pd.read_parquet(
    "taxonomy/"
    "tropical_reference.parquet"
)


# A concept can be supported
# by several sources.
tropical_grouped = (
    TROPICAL
    .groupby(
        "concept_id",
        as_index=False,
    )
    .agg({
        "source_id":
            lambda x:
                sorted(set(x)),

        "evidence_type":
            lambda x:
                sorted(set(x)),
    })
)


tropical_grouped[
    "is_tropical"
] = True


kb = BASE.merge(
    tropical_grouped,
    on="concept_id",
    how="left",
)


kb[
    "tropical_status"
] = "unknown"


mask = (
    kb["is_tropical"]
    == True
)


kb.loc[
    mask,
    "tropical_status",
] = "tropical"


kb[
    "is_tropical"
] = kb[
    "is_tropical"
].astype("boolean")


kb.to_parquet(
    "taxonomy/"
    "disease_knowledge_base.parquet",
    index=False,
)


print(
    "All concepts:",
    len(kb)
)

print(
    "Infectious:",
    int(
        kb[
            "is_infectious"
        ].sum()
    )
)

print(
    "Known tropical:",
    int(
        (
            kb[
                "tropical_status"
            ]
            == "tropical"
        ).sum()
    )
)