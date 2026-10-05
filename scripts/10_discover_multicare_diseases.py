from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from matching_utils import (
    as_list,
    build_matcher,
    find_matches,
)


# =========================================================
# LOAD DATA
# =========================================================

KB = pd.read_parquet(
    "taxonomy/"
    "disease_knowledge_base.parquet"
)

CASES = pd.read_parquet(
    "data/normalized/"
    "cases.parquet"
)

ARTICLES = pd.read_parquet(
    "data/normalized/"
    "articles.parquet"
)


# =========================================================
# BUILD THE SAME MATCHER USED BY STEP 15
# =========================================================

matcher = build_matcher(
    KB
)


# =========================================================
# ARTICLE LOOKUP
# =========================================================

article_lookup = {}


for row in ARTICLES.itertuples(
    index=False
):

    article_lookup[
        str(row.article_id)
    ] = {
        "title":
            row.title
            if isinstance(
                row.title,
                str,
            )
            else "",

        "keywords":
            as_list(
                row.keywords
            ),

        "mesh_terms":
            as_list(
                row.mesh_terms
            ),
    }


# =========================================================
# COUNTERS
# =========================================================

raw_mentions = Counter()

candidate_cases = Counter()

text_case_counts = Counter()

metadata_case_counts = Counter()


# =========================================================
# DISCOVER
# =========================================================

for row in tqdm(
    CASES.itertuples(
        index=False
    ),
    total=len(CASES),
):

    case_text = (
        row.raw_case_text
        if isinstance(
            row.raw_case_text,
            str,
        )
        else ""
    )


    article_id = (
        str(row.article_id)
        if row.article_id
        is not None
        else None
    )


    article = article_lookup.get(
        article_id,
        {
            "title": "",
            "keywords": [],
            "mesh_terms": [],
        },
    )


    # -----------------------------------------------------
    # RAW CASE TEXT
    # -----------------------------------------------------

    text_matches = find_matches(
        matcher,
        case_text,
        field="raw_case_text",
    )


    # -----------------------------------------------------
    # ARTICLE TITLE
    # -----------------------------------------------------

    metadata_matches = []


    metadata_matches.extend(
        find_matches(
            matcher,
            article[
                "title"
            ],
            field="article_title",
        )
    )


    # -----------------------------------------------------
    # ARTICLE KEYWORDS
    # -----------------------------------------------------

    for index, keyword in enumerate(
        article[
            "keywords"
        ]
    ):

        if keyword is None:
            continue


        metadata_matches.extend(
            find_matches(
                matcher,
                str(keyword),
                field="article_keyword",
                field_item_index=index,
            )
        )


    # -----------------------------------------------------
    # MeSH TERMS
    # -----------------------------------------------------

    for index, mesh in enumerate(
        article[
            "mesh_terms"
        ]
    ):

        if mesh is None:
            continue


        metadata_matches.extend(
            find_matches(
                matcher,
                str(mesh),
                field="article_mesh",
                field_item_index=index,
            )
        )


    # =====================================================
    # COUNT OCCURRENCES
    # =====================================================

    text_concepts = [
        match[
            "concept_id"
        ]
        for match
        in text_matches
    ]


    metadata_concepts = [
        match[
            "concept_id"
        ]
        for match
        in metadata_matches
    ]


    raw_mentions.update(
        text_concepts
    )

    raw_mentions.update(
        metadata_concepts
    )


    # =====================================================
    # COUNT UNIQUE CASES PER CONCEPT
    # =====================================================

    text_unique = set(
        text_concepts
    )


    metadata_unique = set(
        metadata_concepts
    )


    all_unique = (
        text_unique
        |
        metadata_unique
    )


    candidate_cases.update(
        all_unique
    )


    text_case_counts.update(
        text_unique
    )


    metadata_case_counts.update(
        metadata_unique
    )


# =========================================================
# BUILD CATALOG
# =========================================================

catalog = KB.copy()


catalog[
    "raw_mentions"
] = (
    catalog[
        "concept_id"
    ]
    .map(
        raw_mentions
    )
    .fillna(0)
    .astype(int)
)


catalog[
    "candidate_cases"
] = (
    catalog[
        "concept_id"
    ]
    .map(
        candidate_cases
    )
    .fillna(0)
    .astype(int)
)


catalog[
    "text_candidate_cases"
] = (
    catalog[
        "concept_id"
    ]
    .map(
        text_case_counts
    )
    .fillna(0)
    .astype(int)
)


catalog[
    "metadata_candidate_cases"
] = (
    catalog[
        "concept_id"
    ]
    .map(
        metadata_case_counts
    )
    .fillna(0)
    .astype(int)
)


# =========================================================
# SAVE
# =========================================================

OUTPUT = Path(
    "reports/scope/"
    "disease_catalog.parquet"
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


catalog.to_parquet(
    OUTPUT,
    index=False,
)


print()
print(
    "Observed concepts:",
    int(
        (
            catalog[
                "candidate_cases"
            ]
            > 0
        ).sum()
    )
)


print(
    "Observed infectious:",
    int(
        (
            (
                catalog[
                    "candidate_cases"
                ]
                > 0
            )
            &
            (
                catalog[
                    "is_infectious"
                ]
                == True
            )
        ).sum()
    )
)


print()
print(
    "Saved:",
    OUTPUT,
)

print(
    "FINAL STATUS: PASS"
)