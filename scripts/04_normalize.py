from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


RAW = Path(
    "data/raw/multicare"
)

OUT = Path(
    "data/normalized"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


def as_list(value):
    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple),
    ):
        return list(value)

    return [value]


# ---------------------
# ARTICLES
# ---------------------

raw_metadata = (
    pq.read_table(
        RAW / "metadata.parquet"
    )
    .to_pylist()
)

articles = []

for source in raw_metadata:

    nested = source.get(
        "article_metadata"
    )

    rows = (
        nested
        if isinstance(nested, list)
        else [nested]
        if isinstance(nested, dict)
        else [source]
    )

    for row in rows:

        if row is None:
            continue

        article_id = (
            source.get("article_id")
            or row.get("article_id")
            or row.get("pmcid")
        )

        if not article_id:
            continue

        articles.append({
            "article_id":
                str(article_id),

            "title":
                row.get("title") or "",

            "keywords":
                as_list(
                    row.get("keywords")
                ),

            "mesh_terms":
                as_list(
                    row.get("mesh_terms")
                ),
        })


articles_df = (
    pd.DataFrame(articles)
    .drop_duplicates(
        subset=["article_id"]
    )
)

articles_df.to_parquet(
    OUT / "articles.parquet",
    index=False,
)


# ---------------------
# CASES
# ---------------------

raw_cases = (
    pq.read_table(
        RAW / "cases.parquet"
    )
    .to_pylist()
)

cases = []

for source in raw_cases:

    nested = source.get(
        "cases"
    )

    rows = (
        nested
        if isinstance(nested, list)
        else [nested]
        if isinstance(nested, dict)
        else [source]
    )

    for row in rows:

        if row is None:
            continue

        case_id = row.get(
            "case_id"
        )

        text = (
            row.get("case_text")
            or ""
        )

        article_id = (
            source.get("article_id")
            or row.get("article_id")
        )

        if not case_id:
            continue

        cases.append({
            "case_id":
                str(case_id),

            "article_id":
                str(article_id),

            "raw_case_text":
                text,
        })


cases_df = (
    pd.DataFrame(cases)
    .drop_duplicates(
        subset=["case_id"]
    )
)

cases_df.to_parquet(
    OUT / "cases.parquet",
    index=False,
)


print(
    "Articles:",
    len(articles_df),
)

print(
    "Cases:",
    len(cases_df),
)

print(
    "Saved to:",
    OUT,
)