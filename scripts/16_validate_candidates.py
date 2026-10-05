from pathlib import Path
import json
import sys

import pandas as pd


# =========================================================
# PATHS
# =========================================================

SCOPE_PATH = Path(
    "reports/scope/"
    "disease_scope.parquet"
)

MATCHES_PATH = Path(
    "data/candidates/"
    "candidate_matches.parquet"
)

CANDIDATES_PATH = Path(
    "data/candidates/"
    "candidates.parquet"
)

INVENTORY_PATH = Path(
    "data/candidates/"
    "candidate_case_inventory.parquet"
)

CASES_PATH = Path(
    "data/normalized/"
    "cases.parquet"
)

ARTICLES_PATH = Path(
    "data/normalized/"
    "articles.parquet"
)

IMAGES_PATH = Path(
    "data/normalized/"
    "images.parquet"
)


REPORT_ROOT = Path(
    "reports/candidates"
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


VALIDATION_OUTPUT = (
    REPORT_ROOT /
    "candidate_validation.json"
)

COUNT_COMPARISON_OUTPUT = (
    REPORT_ROOT /
    "scope_count_comparison.csv"
)

ALIAS_DIAGNOSTICS_OUTPUT = (
    REPORT_ROOT /
    "alias_diagnostics.csv"
)

METADATA_REPORT_OUTPUT = (
    REPORT_ROOT /
    "metadata_only_by_disease.csv"
)


# =========================================================
# HELPERS
# =========================================================

def normalize_id(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    return value or None


def as_list(value):

    if value is None:
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return list(value)

    if hasattr(
        value,
        "tolist",
    ):

        try:

            result = value.tolist()

            if isinstance(
                result,
                list,
            ):
                return result

        except Exception:
            pass

    return [value]


def normalized_list(value):

    return sorted(
        str(x)
        for x in as_list(value)
        if x is not None
    )


def record_error(
    errors,
    code,
    message,
    count=None,
):

    item = {
        "code":
            code,

        "message":
            message,
    }

    if count is not None:

        item[
            "count"
        ] = int(
            count
        )

    errors.append(
        item
    )


def record_warning(
    warnings,
    code,
    message,
    count=None,
):

    item = {
        "code":
            code,

        "message":
            message,
    }

    if count is not None:

        item[
            "count"
        ] = int(
            count
        )

    warnings.append(
        item
    )


# =========================================================
# LOAD
# =========================================================

print(
    "Loading candidate artifacts..."
)


scope = pd.read_parquet(
    SCOPE_PATH
)

matches = pd.read_parquet(
    MATCHES_PATH
)

candidates = pd.read_parquet(
    CANDIDATES_PATH
)

inventory = pd.read_parquet(
    INVENTORY_PATH
)

cases = pd.read_parquet(
    CASES_PATH
)

articles = pd.read_parquet(
    ARTICLES_PATH
)

images = pd.read_parquet(
    IMAGES_PATH
)


errors = []
warnings = []


# =========================================================
# REQUIRED SCHEMAS
# =========================================================

required = {

    "scope": {
        "concept_id",
        "disease",
        "candidate_cases",
        "text_candidate_cases",
        "metadata_candidate_cases",
    },

    "matches": {
        "case_id",
        "article_id",
        "concept_id",
        "disease",
        "field",
        "alias",
        "matched_text",
        "start",
        "end",
    },

    "candidates": {
        "case_id",
        "article_id",
        "concept_id",
        "disease",

        "match_count",
        "raw_text_match_count",
        "metadata_match_count",

        "has_raw_text_match",
        "has_title_match",
        "has_keyword_match",
        "has_mesh_match",

        "candidate_reason",

        "image_count",
        "image_ids",
    },

    "inventory": {
        "case_id",
        "article_id",
        "concept_id",
        "raw_case_text",
    },

    "cases": {
        "case_id",
        "article_id",
        "raw_case_text",
    },

    "articles": {
        "article_id",
        "title",
        "keywords",
        "mesh_terms",
    },

    "images": {
        "image_id",
        "case_id",
    },
}


for name, df in [
    ("scope", scope),
    ("matches", matches),
    ("candidates", candidates),
    ("inventory", inventory),
    ("cases", cases),
    ("articles", articles),
    ("images", images),
]:

    missing = (
        required[name]
        -
        set(
            df.columns
        )
    )

    if missing:

        record_error(
            errors,
            "MISSING_COLUMNS",
            (
                f"{name} missing "
                f"columns: {missing}"
            ),
        )


if errors:

    print(
        "Schema validation failed."
    )

    for error in errors:

        print(
            "[FAIL]",
            error["message"],
        )

    sys.exit(1)


# =========================================================
# NORMALIZE IDS
# =========================================================

for df in [
    scope,
    matches,
    candidates,
    inventory,
]:

    df[
        "concept_id"
    ] = df[
        "concept_id"
    ].map(
        normalize_id
    )


for df in [
    matches,
    candidates,
    inventory,
    cases,
    images,
]:

    df[
        "case_id"
    ] = df[
        "case_id"
    ].map(
        normalize_id
    )


for df in [
    matches,
    candidates,
    inventory,
    cases,
    articles,
]:

    if (
        "article_id"
        in df.columns
    ):

        df[
            "article_id"
        ] = df[
            "article_id"
        ].map(
            normalize_id
        )


images[
    "image_id"
] = images[
    "image_id"
].map(
    normalize_id
)


# =========================================================
# LOOKUPS
# =========================================================

scope_ids = set(
    scope[
        "concept_id"
    ]
)


case_lookup = (
    cases
    .drop_duplicates(
        "case_id"
    )
    .set_index(
        "case_id"
    )
)


article_lookup = (
    articles
    .drop_duplicates(
        "article_id"
    )
    .set_index(
        "article_id"
    )
)


candidate_keys = set(
    zip(
        candidates[
            "case_id"
        ],
        candidates[
            "concept_id"
        ],
    )
)


match_keys = set(
    zip(
        matches[
            "case_id"
        ],
        matches[
            "concept_id"
        ],
    )
)


inventory_keys = set(
    zip(
        inventory[
            "case_id"
        ],
        inventory[
            "concept_id"
        ],
    )
)


# =========================================================
# CHECK 1
# CANDIDATE KEY UNIQUENESS
# =========================================================

duplicate_candidates = (
    candidates.duplicated(
        [
            "case_id",
            "concept_id",
        ]
    )
)


if duplicate_candidates.any():

    record_error(
        errors,
        "DUPLICATE_CANDIDATE_KEY",
        (
            "Duplicate "
            "(case_id, concept_id)"
        ),
        duplicate_candidates.sum(),
    )


# Exact duplicate match rows should
# never occur.
match_key_columns = [
    "case_id",
    "concept_id",
    "field",
    "field_item_index",
    "alias",
    "start",
    "end",
]


match_key_columns = [
    x
    for x in match_key_columns
    if x in matches.columns
]


duplicate_matches = (
    matches.duplicated(
        match_key_columns
    )
)


if duplicate_matches.any():

    record_error(
        errors,
        "DUPLICATE_MATCH",
        "Duplicate exact match rows",
        duplicate_matches.sum(),
    )


# =========================================================
# CHECK 2
# ALL CONCEPTS MUST BE IN SCOPE
# =========================================================

outside_scope = (
    set(
        candidates[
            "concept_id"
        ]
    )
    -
    scope_ids
)


if outside_scope:

    record_error(
        errors,
        "CONCEPT_OUTSIDE_SCOPE",
        (
            "Candidate concepts "
            "outside disease scope"
        ),
        len(
            outside_scope
        ),
    )


# =========================================================
# CHECK 3
# MATCH ↔ CANDIDATE RELATIONSHIP
# =========================================================

candidate_without_match = (
    candidate_keys
    -
    match_keys
)


match_without_candidate = (
    match_keys
    -
    candidate_keys
)


if candidate_without_match:

    record_error(
        errors,
        "CANDIDATE_WITHOUT_MATCH",
        (
            "Candidate rows without "
            "matching evidence row"
        ),
        len(
            candidate_without_match
        ),
    )


if match_without_candidate:

    record_error(
        errors,
        "MATCH_WITHOUT_CANDIDATE",
        (
            "Match rows without "
            "candidate aggregate row"
        ),
        len(
            match_without_candidate
        ),
    )


# =========================================================
# CHECK 4
# INVENTORY ↔ CANDIDATE
# =========================================================

if (
    candidate_keys
    !=
    inventory_keys
):

    record_error(
        errors,
        "INVENTORY_KEY_MISMATCH",
        (
            "candidate inventory keys "
            "do not equal candidate keys"
        ),
    )


# =========================================================
# CHECK 5
# CASE / ARTICLE RELATIONSHIP
# =========================================================

known_cases = set(
    cases[
        "case_id"
    ]
)


orphan_candidate_cases = (
    set(
        candidates[
            "case_id"
    ]
    )
    -
    known_cases
)


if orphan_candidate_cases:

    record_error(
        errors,
        "UNKNOWN_CASE_ID",
        "Candidate case not in cases.parquet",
        len(
            orphan_candidate_cases
        ),
    )


article_mismatches = 0


for row in candidates.itertuples(
    index=False
):

    case_id = row.case_id

    if (
        case_id
        not in case_lookup.index
    ):
        continue


    expected_article = normalize_id(
        case_lookup.loc[
            case_id,
            "article_id",
        ]
    )


    actual_article = normalize_id(
        row.article_id
    )


    if (
        expected_article
        !=
        actual_article
    ):

        article_mismatches += 1


if article_mismatches:

    record_error(
        errors,
        "CASE_ARTICLE_MISMATCH",
        (
            "Candidate article_id does "
            "not match case article_id"
        ),
        article_mismatches,
    )


# =========================================================
# CHECK 6
# EXACT MATCH OFFSETS
# =========================================================

print(
    "Checking exact match offsets..."
)


allowed_fields = {
    "raw_case_text",
    "article_title",
    "article_keyword",
    "article_mesh",
}


invalid_fields = (
    set(
        matches[
            "field"
        ]
    )
    -
    allowed_fields
)


if invalid_fields:

    record_error(
        errors,
        "INVALID_MATCH_FIELD",
        (
            f"Unknown match fields: "
            f"{invalid_fields}"
        ),
    )


offset_errors = []


for row in matches.itertuples(
    index=False
):

    source_text = None


    # ---------------------------------------------
    # RAW CASE TEXT
    # ---------------------------------------------

    if (
        row.field
        ==
        "raw_case_text"
    ):

        if (
            row.case_id
            not in case_lookup.index
        ):

            continue


        source_text = (
            case_lookup.loc[
                row.case_id,
                "raw_case_text",
            ]
        )


    # ---------------------------------------------
    # ARTICLE METADATA
    # ---------------------------------------------

    else:

        if (
            row.article_id
            not in article_lookup.index
        ):

            offset_errors.append({
                "case_id":
                    row.case_id,

                "concept_id":
                    row.concept_id,

                "reason":
                    "ARTICLE_NOT_FOUND",
            })

            continue


        article = article_lookup.loc[
            row.article_id
        ]


        if (
            row.field
            ==
            "article_title"
        ):

            source_text = (
                article[
                    "title"
                ]
            )


        elif (
            row.field
            ==
            "article_keyword"
        ):

            values = as_list(
                article[
                    "keywords"
                ]
            )

            index = getattr(
                row,
                "field_item_index",
                None,
            )


            if (
                index is None
                or pd.isna(index)
            ):

                offset_errors.append({
                    "case_id":
                        row.case_id,

                    "concept_id":
                        row.concept_id,

                    "reason":
                        "MISSING_KEYWORD_INDEX",
                })

                continue


            index = int(
                index
            )


            if (
                index < 0
                or index >= len(
                    values
                )
            ):

                offset_errors.append({
                    "case_id":
                        row.case_id,

                    "concept_id":
                        row.concept_id,

                    "reason":
                        "INVALID_KEYWORD_INDEX",
                })

                continue


            source_text = str(
                values[
                    index
                ]
            )


        elif (
            row.field
            ==
            "article_mesh"
        ):

            values = as_list(
                article[
                    "mesh_terms"
                ]
            )

            index = getattr(
                row,
                "field_item_index",
                None,
            )


            if (
                index is None
                or pd.isna(index)
            ):

                offset_errors.append({
                    "case_id":
                        row.case_id,

                    "concept_id":
                        row.concept_id,

                    "reason":
                        "MISSING_MESH_INDEX",
                })

                continue


            index = int(
                index
            )


            if (
                index < 0
                or index >= len(
                    values
                )
            ):

                offset_errors.append({
                    "case_id":
                        row.case_id,

                    "concept_id":
                        row.concept_id,

                    "reason":
                        "INVALID_MESH_INDEX",
                })

                continue


            source_text = str(
                values[
                    index
                ]
            )


    if not isinstance(
        source_text,
        str,
    ):

        source_text = str(
            source_text
            or ""
        )


    start = int(
        row.start
    )

    end = int(
        row.end
    )


    if (
        start < 0
        or end <= start
        or end > len(
            source_text
        )
    ):

        offset_errors.append({
            "case_id":
                row.case_id,

            "concept_id":
                row.concept_id,

            "reason":
                "OFFSET_OUT_OF_BOUNDS",
        })

        continue


    actual_text = source_text[
        start:end
    ]


    if (
        actual_text
        !=
        row.matched_text
    ):

        offset_errors.append({
            "case_id":
                row.case_id,

            "concept_id":
                row.concept_id,

            "reason":
                "MATCHED_TEXT_MISMATCH",
        })

        continue


    if (
        actual_text.lower()
        !=
        str(
            row.alias
        ).lower()
    ):

        offset_errors.append({
            "case_id":
                row.case_id,

            "concept_id":
                row.concept_id,

            "reason":
                "ALIAS_OFFSET_MISMATCH",
        })


if offset_errors:

    pd.DataFrame(
        offset_errors
    ).to_csv(
        REPORT_ROOT /
        "offset_errors.csv",
        index=False,
    )


    record_error(
        errors,
        "INVALID_OFFSETS",
        (
            "Disease match offsets "
            "are invalid"
        ),
        len(
            offset_errors
        ),
    )


# =========================================================
# CHECK 7
# AGGREGATED MATCH COUNTS
# =========================================================

print(
    "Checking candidate aggregations..."
)


aggregation_errors = 0


for (
    case_id,
    concept_id,
), group in matches.groupby(
    [
        "case_id",
        "concept_id",
    ],
    sort=False,
):


    candidate = candidates[
        (
            candidates[
                "case_id"
            ]
            ==
            case_id
        )
        &
        (
            candidates[
                "concept_id"
            ]
            ==
            concept_id
        )
    ]


    if len(
        candidate
    ) != 1:

        aggregation_errors += 1
        continue


    candidate = candidate.iloc[
        0
    ]


    total = len(
        group
    )


    raw_count = int(
        (
            group[
                "field"
            ]
            ==
            "raw_case_text"
        ).sum()
    )


    metadata_count = (
        total
        -
        raw_count
    )


    fields = set(
        group[
            "field"
        ]
    )


    has_raw = (
        "raw_case_text"
        in fields
    )

    has_title = (
        "article_title"
        in fields
    )

    has_keyword = (
        "article_keyword"
        in fields
    )

    has_mesh = (
        "article_mesh"
        in fields
    )


    has_metadata = (
        has_title
        or has_keyword
        or has_mesh
    )


    if (
        has_raw
        and has_metadata
    ):

        expected_reason = (
            "RAW_TEXT_AND_METADATA"
        )

    elif has_raw:

        expected_reason = (
            "RAW_TEXT"
        )

    else:

        expected_reason = (
            "METADATA_ONLY"
        )


    checks = [

        int(
            candidate[
                "match_count"
            ]
        )
        ==
        total,

        int(
            candidate[
                "raw_text_match_count"
            ]
        )
        ==
        raw_count,

        int(
            candidate[
                "metadata_match_count"
            ]
        )
        ==
        metadata_count,

        bool(
            candidate[
                "has_raw_text_match"
            ]
        )
        ==
        has_raw,

        bool(
            candidate[
                "has_title_match"
            ]
        )
        ==
        has_title,

        bool(
            candidate[
                "has_keyword_match"
            ]
        )
        ==
        has_keyword,

        bool(
            candidate[
                "has_mesh_match"
            ]
        )
        ==
        has_mesh,

        candidate[
            "candidate_reason"
        ]
        ==
        expected_reason,
    ]


    if not all(
        checks
    ):

        aggregation_errors += 1


if aggregation_errors:

    record_error(
        errors,
        "AGGREGATION_MISMATCH",
        (
            "Candidate aggregation "
            "does not match exact "
            "match table"
        ),
        aggregation_errors,
    )


# =========================================================
# CHECK 8
# IMAGE RELATIONSHIPS
# =========================================================

print(
    "Checking image relationships..."
)


actual_images = {}


for case_id, group in images.groupby(
    "case_id",
    sort=False,
):

    actual_images[
        str(
            case_id
        )
    ] = sorted(
        str(x)
        for x in group[
            "image_id"
        ].dropna()
    )


image_errors = 0


for row in candidates.itertuples(
    index=False
):

    actual = actual_images.get(
        str(
            row.case_id
        ),
        [],
    )


    expected = normalized_list(
        row.image_ids
    )


    if (
        int(
            row.image_count
        )
        !=
        len(
            actual
        )
    ):

        image_errors += 1

        continue


    if expected != actual:

        image_errors += 1


if image_errors:

    record_error(
        errors,
        "IMAGE_RELATIONSHIP_MISMATCH",
        (
            "Candidate image summary "
            "does not match "
            "images.parquet"
        ),
        image_errors,
    )


# =========================================================
# CHECK 9
# STEP 10 / SCOPE COUNTS VS STEP 15
# =========================================================

print(
    "Comparing candidate counts "
    "against disease scope..."
)


derived = (
    candidates.groupby(
        [
            "concept_id",
            "disease",
        ],
        as_index=False,
    )
    .agg(
        derived_candidate_cases=(
            "case_id",
            "nunique",
        ),

        derived_text_candidate_cases=(
            "has_raw_text_match",
            "sum",
        ),
    )
)


metadata_counts = (
    candidates[
        (
            candidates[
                "has_title_match"
            ]
        )
        |
        (
            candidates[
                "has_keyword_match"
            ]
        )
        |
        (
            candidates[
                "has_mesh_match"
            ]
        )
    ]
    .groupby(
        "concept_id"
    )[
        "case_id"
    ]
    .nunique()
)


derived[
    "derived_metadata_candidate_cases"
] = (
    derived[
        "concept_id"
    ]
    .map(
        metadata_counts
    )
    .fillna(0)
    .astype(int)
)


comparison = (
    scope[
        [
            "concept_id",
            "disease",
            "candidate_cases",
            "text_candidate_cases",
            "metadata_candidate_cases",
        ]
    ]
    .merge(
        derived,
        on=[
            "concept_id",
            "disease",
        ],
        how="left",
    )
)


for column in [
    "derived_candidate_cases",
    "derived_text_candidate_cases",
    "derived_metadata_candidate_cases",
]:

    comparison[
        column
    ] = (
        comparison[
            column
        ]
        .fillna(0)
        .astype(int)
    )


comparison[
    "candidate_diff"
] = (
    comparison[
        "derived_candidate_cases"
    ]
    -
    comparison[
        "candidate_cases"
    ]
)


comparison[
    "text_diff"
] = (
    comparison[
        "derived_text_candidate_cases"
    ]
    -
    comparison[
        "text_candidate_cases"
    ]
)


comparison[
    "metadata_diff"
] = (
    comparison[
        "derived_metadata_candidate_cases"
    ]
    -
    comparison[
        "metadata_candidate_cases"
    ]
)


comparison.to_csv(
    COUNT_COMPARISON_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


count_mismatches = comparison[
    (
        comparison[
            "candidate_diff"
        ]
        != 0
    )
    |
    (
        comparison[
            "text_diff"
        ]
        != 0
    )
    |
    (
        comparison[
            "metadata_diff"
        ]
        != 0
    )
]


if len(
    count_mismatches
):

    record_error(
        errors,
        "SCOPE_COUNT_MISMATCH",
        (
            "Step 15 candidate counts "
            "do not reproduce Step 10/"
            "11 scope counts"
        ),
        len(
            count_mismatches
        ),
    )


# =========================================================
# DIAGNOSTIC REPORT:
# ALIASES
# =========================================================

print(
    "Building alias diagnostics..."
)


alias_diagnostics = (
    matches.groupby(
        "alias",
        as_index=False,
    )
    .agg(
        occurrences=(
            "alias",
            "size",
        ),

        unique_cases=(
            "case_id",
            "nunique",
        ),

        unique_concepts=(
            "concept_id",
            "nunique",
        ),
    )
)


alias_diagnostics[
    "alias_length"
] = alias_diagnostics[
    "alias"
].str.len()


alias_fields = (
    matches.groupby(
        "alias"
    )[
        "field"
    ]
    .apply(
        lambda x:
            "|".join(
                sorted(
                    set(
                        x
                    )
                )
            )
    )
)


alias_diagnostics[
    "fields"
] = alias_diagnostics[
    "alias"
].map(
    alias_fields
)


alias_diagnostics[
    "ambiguous_alias"
] = (
    alias_diagnostics[
        "unique_concepts"
    ]
    > 1
)


alias_diagnostics[
    "short_alias"
] = (
    alias_diagnostics[
        "alias_length"
    ]
    <= 5
)


alias_diagnostics = (
    alias_diagnostics
    .sort_values(
        [
            "ambiguous_alias",
            "occurrences",
        ],
        ascending=[
            False,
            False,
        ],
    )
)


alias_diagnostics.to_csv(
    ALIAS_DIAGNOSTICS_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


ambiguous_aliases = int(
    alias_diagnostics[
        "ambiguous_alias"
    ].sum()
)


if ambiguous_aliases:

    record_warning(
        warnings,
        "AMBIGUOUS_ALIASES",
        (
            "Aliases map to more than "
            "one disease concept. "
            "Review alias_diagnostics.csv"
        ),
        ambiguous_aliases,
    )


# =========================================================
# DIAGNOSTIC REPORT:
# METADATA-ONLY
# =========================================================

disease_counts = (
    candidates.groupby(
        [
            "concept_id",
            "disease",
        ],
        as_index=False,
    )
    .agg(
        candidates=(
            "case_id",
            "nunique",
        ),
    )
)


metadata_only = (
    candidates[
        candidates[
            "candidate_reason"
        ]
        ==
        "METADATA_ONLY"
    ]
    .groupby(
        "concept_id"
    )[
        "case_id"
    ]
    .nunique()
)


disease_counts[
    "metadata_only"
] = (
    disease_counts[
        "concept_id"
    ]
    .map(
        metadata_only
    )
    .fillna(0)
    .astype(int)
)


disease_counts[
    "metadata_only_fraction"
] = (
    disease_counts[
        "metadata_only"
    ]
    /
    disease_counts[
        "candidates"
    ]
)


disease_counts = (
    disease_counts
    .sort_values(
        "metadata_only_fraction",
        ascending=False,
    )
)


disease_counts.to_csv(
    METADATA_REPORT_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# FINAL SUMMARY
# =========================================================

summary = {

    "scope_diseases":
        int(
            len(
                scope
            )
        ),

    "candidate_diseases":
        int(
            candidates[
                "concept_id"
            ].nunique()
        ),

    "candidate_cases":
        int(
            candidates[
                "case_id"
            ].nunique()
        ),

    "case_disease_hypotheses":
        int(
            len(
                candidates
            )
        ),

    "match_occurrences":
        int(
            len(
                matches
            )
        ),

    "metadata_only_hypotheses":
        int(
            (
                candidates[
                    "candidate_reason"
                ]
                ==
                "METADATA_ONLY"
            ).sum()
        ),

    "cases_with_images":
        int(
            candidates.loc[
                candidates[
                    "image_count"
                ]
                > 0,
                "case_id",
            ].nunique()
        ),

    "ambiguous_aliases":
        ambiguous_aliases,

    "hard_errors":
        len(
            errors
        ),

    "warnings":
        len(
            warnings
        ),
}


status = (
    "PASS"
    if not errors
    else "FAIL"
)


report = {
    "status":
        status,

    "summary":
        summary,

    "errors":
        errors,

    "warnings":
        warnings,
}


with VALIDATION_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print(
    "=" * 65
)

print(
    "CANDIDATE VALIDATION"
)

print(
    "=" * 65
)


for key, value in (
    summary.items()
):

    print(
        f"{key}: {value}"
    )


print()


for warning in warnings:

    print(
        "[WARNING]",
        warning[
            "message"
        ],
    )


for error in errors:

    print(
        "[FAIL]",
        error[
            "message"
        ],
    )


print()
print(
    "Reports:"
)

print(
    " -",
    VALIDATION_OUTPUT,
)

print(
    " -",
    COUNT_COMPARISON_OUTPUT,
)

print(
    " -",
    ALIAS_DIAGNOSTICS_OUTPUT,
)

print(
    " -",
    METADATA_REPORT_OUTPUT,
)

print()

print(
    "FINAL STATUS:",
    status,
)


if errors:

    sys.exit(1)