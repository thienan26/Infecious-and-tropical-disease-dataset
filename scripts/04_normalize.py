from pathlib import Path
import hashlib

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
    exist_ok=True,
)


# =========================================================
# HELPERS
# =========================================================

def normalize_id(value):
    """
    Normalize IDs without accidentally
    turning None into the string 'None'.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def normalize_text(value):
    """
    Keep original textual content while
    normalizing null values.
    """

    if value is None:
        return ""

    return str(value)


def as_list(value):

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            str(x)
            for x in value
            if x is not None
        ]

    # Pandas / NumPy / Arrow
    # may expose array-like objects.
    if hasattr(
        value,
        "tolist",
    ):
        try:
            converted = value.tolist()

            if isinstance(
                converted,
                list,
            ):
                return [
                    str(x)
                    for x in converted
                    if x is not None
                ]
        except Exception:
            pass

    value = str(value).strip()

    if not value:
        return []

    return [value]


def sha256_text(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# =========================================================
# ARTICLES
# =========================================================

print(
    "Normalizing articles..."
)

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

    if isinstance(
        nested,
        list,
    ):
        rows = nested

    elif isinstance(
        nested,
        dict,
    ):
        rows = [nested]

    else:
        rows = [source]


    for row in rows:

        if row is None:
            continue

        article_id = normalize_id(
            source.get(
                "article_id"
            )
            or row.get(
                "article_id"
            )
            or row.get(
                "pmcid"
            )
        )

        if article_id is None:
            continue


        title = normalize_text(
            row.get(
                "title"
            )
        )


        articles.append({
            "article_id":
                article_id,

            "title":
                title,

            "keywords":
                as_list(
                    row.get(
                        "keywords"
                    )
                ),

            "mesh_terms":
                as_list(
                    row.get(
                        "mesh_terms"
                    )
                ),
        })


articles_df = pd.DataFrame(
    articles
)


if articles_df.empty:

    raise RuntimeError(
        "No articles were normalized."
    )


# ---------------------------------------------------------
# Deal with duplicate article IDs carefully.
# ---------------------------------------------------------

article_conflicts = []


for article_id, group in (
    articles_df.groupby(
        "article_id",
        sort=False,
    )
):

    titles = {
        value.strip()
        for value in group[
            "title"
        ].fillna("")
        if value.strip()
    }

    if len(titles) > 1:

        article_conflicts.append(
            article_id
        )


if article_conflicts:

    raise RuntimeError(
        "Conflicting duplicate article IDs: "
        f"{article_conflicts[:20]}"
    )


# Exact/logically equivalent duplicates
# can now safely collapse.
articles_df = (
    articles_df
    .drop_duplicates(
        subset=[
            "article_id"
        ]
    )
    .sort_values(
        "article_id"
    )
    .reset_index(
        drop=True
    )
)


articles_df.to_parquet(
    OUT / "articles.parquet",
    index=False,
)


# =========================================================
# CASES
# =========================================================

print(
    "Normalizing cases..."
)

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

    if isinstance(
        nested,
        list,
    ):
        rows = nested

    elif isinstance(
        nested,
        dict,
    ):
        rows = [nested]

    else:
        rows = [source]


    for row in rows:

        if row is None:
            continue


        case_id = normalize_id(
            row.get(
                "case_id"
            )
        )


        article_id = normalize_id(
            source.get(
                "article_id"
            )
            or row.get(
                "article_id"
            )
        )


        raw_case_text = normalize_text(
            row.get(
                "case_text"
            )
        )


        if case_id is None:
            continue


        cases.append({
            "case_id":
                case_id,

            "article_id":
                article_id,

            "raw_case_text":
                raw_case_text,

            "raw_text_sha256":
                sha256_text(
                    raw_case_text
                ),

            "raw_text_chars":
                len(
                    raw_case_text
                ),
        })


cases_df = pd.DataFrame(
    cases
)


if cases_df.empty:

    raise RuntimeError(
        "No cases were normalized."
    )


# =========================================================
# CASE-ID CONFLICT CHECK
# =========================================================

conflicting_cases = []


for case_id, group in (
    cases_df.groupby(
        "case_id",
        sort=False,
    )
):

    text_hashes = set(
        group[
            "raw_text_sha256"
        ]
    )

    article_ids = set(
        group[
            "article_id"
        ].dropna()
    )


    # Same case ID pointing to different
    # text is a hard data-integrity error.
    if len(
        text_hashes
    ) > 1:

        conflicting_cases.append({
            "case_id":
                case_id,

            "reason":
                "DIFFERENT_TEXT",
        })

        continue


    # Same case ID attached to different
    # articles is also suspicious.
    if len(
        article_ids
    ) > 1:

        conflicting_cases.append({
            "case_id":
                case_id,

            "reason":
                "DIFFERENT_ARTICLE",
        })


if conflicting_cases:

    print()

    print(
        "Conflicting case IDs:"
    )

    print(
        pd.DataFrame(
            conflicting_cases
        )
        .head(30)
        .to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Case ID conflicts found."
    )


# Now duplicate collapse is safe.
cases_df = (
    cases_df
    .drop_duplicates(
        subset=[
            "case_id"
        ]
    )
    .sort_values(
        "case_id"
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# FOREIGN-KEY CHECK
# =========================================================

known_articles = set(
    articles_df[
        "article_id"
    ]
)


case_articles = set(
    cases_df[
        "article_id"
    ]
    .dropna()
)


missing_articles = (
    case_articles
    - known_articles
)


if missing_articles:

    print()

    print(
        "WARNING:"
    )

    print(
        len(
            missing_articles
        ),
        "case article IDs were not found "
        "in articles.parquet"
    )

    print(
        "Examples:",
        sorted(
            missing_articles
        )[:20],
    )


# =========================================================
# SAVE
# =========================================================

cases_df.to_parquet(
    OUT / "cases.parquet",
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

print()
print(
    "=" * 60
)

print(
    "NORMALIZATION COMPLETE"
)

print(
    "=" * 60
)

print(
    "Articles:",
    len(
        articles_df
    ),
)

print(
    "Cases:",
    len(
        cases_df
    ),
)

print(
    "Cases with article_id:",
    int(
        cases_df[
            "article_id"
        ].notna().sum()
    ),
)

print(
    "Cases with non-empty text:",
    int(
        (
            cases_df[
                "raw_text_chars"
            ]
            > 0
        ).sum()
    ),
)

print(
    "Unique case IDs:",
    cases_df[
        "case_id"
    ].nunique(),
)

print(
    "Duplicate case IDs:",
    int(
        cases_df[
            "case_id"
        ].duplicated().sum()
    ),
)

print()
print(
    "Saved:",
    OUT,
)