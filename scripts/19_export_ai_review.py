from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


# =========================================================
# PATHS
# =========================================================

CONFIG_PATH = Path(
    "configs/ai_review.yaml"
)

PRE_AI_PATH = Path(
    "data/pre_ai/"
    "pre_ai_eligible.parquet"
)

ROUTING_PATH = Path(
    "data/pre_ai/"
    "routing.parquet"
)

CASES_PATH = Path(
    "data/normalized/"
    "cases.parquet"
)

CANDIDATES_PATH = Path(
    "data/candidates/"
    "candidates.parquet"
)


OUTPUT_DIR = Path(
    "data/ai_review/input"
)

REPORT_DIR = Path(
    "reports/ai_review"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REVIEW_INPUT_OUTPUT = (
    OUTPUT_DIR /
    "ai_review_input.parquet"
)

RETAINED_DISEASES_OUTPUT = (
    OUTPUT_DIR /
    "retained_diseases.parquet"
)

EXCLUDED_DISEASES_OUTPUT = (
    OUTPUT_DIR /
    "excluded_diseases.parquet"
)

SCOPE_FILTERED_CASES_OUTPUT = (
    OUTPUT_DIR /
    "scope_filtered_cases.parquet"
)


DISEASE_FREQUENCY_REPORT = (
    REPORT_DIR /
    "disease_frequency_all.csv"
)

SUMMARY_OUTPUT = (
    REPORT_DIR /
    "step19_summary.json"
)


# =========================================================
# HELPERS
# =========================================================

def normalize_id(value):

    if value is None:
        return None

    try:

        if pd.isna(
            value
        ):
            return None

    except Exception:
        pass

    value = str(
        value
    ).strip()

    return value or None


def as_python_list(value):

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(
            value
        )

    if isinstance(
        value,
        set,
    ):
        return sorted(
            value
        )

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

    return [
        value
    ]


def require_columns(
    dataframe,
    required,
    name,
):

    missing = (
        set(
            required
        )
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{name} missing columns: "
            f"{sorted(missing)}"
        )


def sha256_text(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =========================================================
# CONFIG
# =========================================================

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(
        f
    )


VERSION = str(
    config[
        "version"
    ]
)


MIN_CASES_PER_DISEASE = int(
    config[
        "disease_frequency_filter"
    ][
        "min_candidate_cases_per_disease"
    ]
)


INCLUDE_IMAGE_PROVENANCE = bool(
    config[
        "export"
    ].get(
        "include_image_provenance",
        True,
    )
)


BLIND_REVIEW = bool(
    config[
        "export"
    ].get(
        "blind_review",
        True,
    )
)


if not BLIND_REVIEW:

    raise RuntimeError(
        "Step 19 requires blind_review=true."
    )


# =========================================================
# LOAD
# =========================================================

print(
    "Loading Step 19 inputs..."
)


eligible = pd.read_parquet(
    PRE_AI_PATH
)

routing = pd.read_parquet(
    ROUTING_PATH
)

cases = pd.read_parquet(
    CASES_PATH
)

candidates = pd.read_parquet(
    CANDIDATES_PATH
)


# =========================================================
# SCHEMA
# =========================================================

require_columns(
    eligible,
    [
        "case_id",
        "article_id",
        "pre_ai_eligible",
        "decodable_image_count",
        "decodable_files",
        "decodable_source_image_ids",
    ],
    "pre_ai_eligible",
)


require_columns(
    routing,
    [
        "case_id",
        "pre_ai_eligible",
    ],
    "routing",
)


require_columns(
    cases,
    [
        "case_id",
        "article_id",
        "raw_case_text",
    ],
    "cases",
)


require_columns(
    candidates,
    [
        "case_id",
        "concept_id",
        "disease",
    ],
    "candidates",
)


# =========================================================
# NORMALIZE IDs
# =========================================================

for dataframe in [
    eligible,
    routing,
    cases,
    candidates,
]:

    if (
        "case_id"
        in dataframe.columns
    ):

        dataframe[
            "case_id"
        ] = dataframe[
            "case_id"
        ].map(
            normalize_id
        )


candidates[
    "concept_id"
] = candidates[
    "concept_id"
].map(
    normalize_id
)


# =========================================================
# STEP 18 POPULATION VALIDATION
# =========================================================

if eligible[
    "case_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate case_id in "
        "pre_ai_eligible.parquet."
    )


if not (
    eligible[
        "pre_ai_eligible"
    ]
    ==
    True
).all():

    raise RuntimeError(
        "pre_ai_eligible.parquet contains "
        "non-eligible cases."
    )


routing_eligible = routing[
    routing[
        "pre_ai_eligible"
    ]
    ==
    True
]


if set(
    eligible[
        "case_id"
    ]
) != set(
    routing_eligible[
        "case_id"
    ]
):

    raise RuntimeError(
        "pre_ai_eligible.parquet does not "
        "match routing.parquet."
    )


PRE_AI_CASE_COUNT = int(
    len(
        eligible
    )
)


print(
    "Pre-AI eligible cases:",
    PRE_AI_CASE_COUNT,
)


# =========================================================
# CASE TABLE
# =========================================================

if cases[
    "case_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate case_id in cases.parquet."
    )


case_lookup = (
    cases.set_index(
        "case_id"
    )
)


eligible_case_ids = set(
    eligible[
        "case_id"
    ]
)


missing_cases = (
    eligible_case_ids
    -
    set(
        case_lookup.index
    )
)


if missing_cases:

    raise RuntimeError(
        "Eligible cases missing from "
        "cases.parquet: "
        f"{sorted(missing_cases)[:20]}"
    )


# =========================================================
# CANDIDATES INSIDE PRE-AI ELIGIBLE POPULATION
# =========================================================

eligible_candidates = (
    candidates[
        candidates[
            "case_id"
        ].isin(
            eligible_case_ids
        )
    ]
    .copy()
)


if eligible_candidates.empty:

    raise RuntimeError(
        "No candidates found for "
        "pre-AI eligible cases."
    )


# Remove accidental duplicate case × concept rows
# for counting purposes.
candidate_case_concepts = (
    eligible_candidates[
        [
            "case_id",
            "concept_id",
            "disease",
        ]
    ]
    .drop_duplicates(
        [
            "case_id",
            "concept_id",
        ]
    )
    .copy()
)


# =========================================================
# VALIDATE CONCEPT → DISEASE NAME
# =========================================================

name_counts = (
    candidate_case_concepts
    .groupby(
        "concept_id"
    )[
        "disease"
    ]
    .nunique(
        dropna=True
    )
)


bad_name_concepts = (
    name_counts[
        name_counts
        >
        1
    ]
)


if len(
    bad_name_concepts
) > 0:

    raise RuntimeError(
        "Some concept_id values map to "
        "multiple disease names: "
        f"{bad_name_concepts.index.tolist()[:20]}"
    )


disease_name_lookup = (
    candidate_case_concepts
    .drop_duplicates(
        "concept_id"
    )
    .set_index(
        "concept_id"
    )[
        "disease"
    ]
    .to_dict()
)


# =========================================================
# DISEASE FREQUENCY
#
# IMPORTANT:
#
# Frequency means:
#
# COUNT(DISTINCT case_id)
#
# NOT number of raw text occurrences.
# =========================================================

print()
print(
    "Calculating candidate disease "
    "frequencies..."
)


disease_frequency = (
    candidate_case_concepts
    .groupby(
        "concept_id"
    )[
        "case_id"
    ]
    .nunique()
    .rename(
        "candidate_case_count"
    )
    .reset_index()
)


disease_frequency[
    "disease"
] = disease_frequency[
    "concept_id"
].map(
    disease_name_lookup
)


disease_frequency[
    "keep_for_ai_review"
] = (
    disease_frequency[
        "candidate_case_count"
    ]
    >=
    MIN_CASES_PER_DISEASE
)


disease_frequency[
    "threshold"
] = (
    MIN_CASES_PER_DISEASE
)


disease_frequency = (
    disease_frequency[
        [
            "concept_id",
            "disease",
            "candidate_case_count",
            "threshold",
            "keep_for_ai_review",
        ]
    ]
    .sort_values(
        [
            "candidate_case_count",
            "concept_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# RETAINED / EXCLUDED DISEASES
# =========================================================

retained_diseases = (
    disease_frequency[
        disease_frequency[
            "keep_for_ai_review"
        ]
        ==
        True
    ]
    .copy()
)


excluded_diseases = (
    disease_frequency[
        disease_frequency[
            "keep_for_ai_review"
        ]
        ==
        False
    ]
    .copy()
)


retained_concept_ids = set(
    retained_diseases[
        "concept_id"
    ]
)


excluded_concept_ids = set(
    excluded_diseases[
        "concept_id"
    ]
)


print(
    "Candidate disease concepts before filter:",
    len(
        disease_frequency
    ),
)


print(
    "Retained disease concepts:",
    len(
        retained_diseases
    ),
)


print(
    "Excluded disease concepts:",
    len(
        excluded_diseases
    ),
)


# =========================================================
# FILTER CANDIDATE TARGETS
# =========================================================

retained_candidate_rows = (
    candidate_case_concepts[
        candidate_case_concepts[
            "concept_id"
        ].isin(
            retained_concept_ids
        )
    ]
    .copy()
)


review_case_ids = set(
    retained_candidate_rows[
        "case_id"
    ]
    .unique()
)


# =========================================================
# CASES REMOVED BY DISEASE-FREQUENCY FILTER
# =========================================================

scope_filtered_case_ids = (
    eligible_case_ids
    -
    review_case_ids
)


scope_filtered_cases = (
    eligible[
        eligible[
            "case_id"
        ].isin(
            scope_filtered_case_ids
        )
    ][
        [
            "case_id",
            "article_id",
        ]
    ]
    .copy()
)


scope_filtered_cases[
    "scope_filter_reason"
] = (
    "NO_CANDIDATE_DISEASE_WITH_"
    f"AT_LEAST_{MIN_CASES_PER_DISEASE}_CASES"
)


scope_filtered_cases = (
    scope_filtered_cases
    .sort_values(
        "case_id"
    )
    .reset_index(
        drop=True
    )
)


print(
    "Cases retained for AI review:",
    len(
        review_case_ids
    ),
)


print(
    "Cases removed by disease-frequency filter:",
    len(
        scope_filtered_case_ids
    ),
)


# =========================================================
# REVIEW POPULATION
# =========================================================

review_population = (
    eligible[
        eligible[
            "case_id"
        ].isin(
            review_case_ids
        )
    ]
    .copy()
)


review_population = (
    review_population
    .sort_values(
        "case_id"
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# RETAINED CANDIDATE TARGET MAP
#
# IMPORTANT:
#
# Only retained >= threshold diseases are exposed
# to ChatGPT / Claude.
# =========================================================

candidate_map = {}


for case_id, group in (
    retained_candidate_rows
    .groupby(
        "case_id",
        sort=False,
    )
):

    targets = (
        group[
            [
                "concept_id",
                "disease",
            ]
        ]
        .drop_duplicates(
            "concept_id"
        )
        .sort_values(
            [
                "concept_id",
                "disease",
            ]
        )
    )


    candidate_map[
        case_id
    ] = [
        {
            "concept_id":
                normalize_id(
                    row.concept_id
                ),

            "disease":
                str(
                    row.disease
                ),
        }

        for row
        in targets.itertuples(
            index=False
        )
    ]


# =========================================================
# BUILD BLIND AI REVIEW INPUT
# =========================================================

print()
print(
    "Building blind AI review Parquet..."
)


output_rows = []


for input_row_number, row in enumerate(
    review_population.itertuples(
        index=False
    ),
    start=1,
):

    case_id = (
        row.case_id
    )


    case = (
        case_lookup.loc[
            case_id
        ]
    )


    raw_case_text = (
        case[
            "raw_case_text"
        ]
    )


    if not isinstance(
        raw_case_text,
        str,
    ):

        raw_case_text = ""


    raw_case_text = (
        raw_case_text.strip()
    )


    if not raw_case_text:

        raise RuntimeError(
            "Review case has empty "
            "raw_case_text: "
            f"{case_id}"
        )


    candidate_targets = (
        candidate_map.get(
            case_id,
            []
        )
    )


    if not candidate_targets:

        raise RuntimeError(
            "Review case has no retained "
            "candidate targets: "
            f"{case_id}"
        )


    decodable_files = (
        as_python_list(
            row.decodable_files
        )
    )


    source_image_ids = (
        as_python_list(
            row.decodable_source_image_ids
        )
    )


    output_row = {

        # -----------------------------------------
        # INPUT IDENTITY
        # -----------------------------------------

        "review_input_version":
            VERSION,

        "input_row_number":
            int(
                input_row_number
            ),

        "case_id":
            case_id,

        "article_id":
            normalize_id(
                case[
                    "article_id"
                ]
            ),

        # -----------------------------------------
        # RAW CLINICAL TEXT
        # -----------------------------------------

        "raw_text_sha256":
            sha256_text(
                raw_case_text
            ),

        "raw_text_chars":
            int(
                len(
                    raw_case_text
                )
            ),

        "raw_case_text":
            raw_case_text,

        # -----------------------------------------
        # RETAINED CANDIDATE TARGETS ONLY
        # -----------------------------------------

        "candidate_target_count":
            int(
                len(
                    candidate_targets
                )
            ),

        "candidate_targets_json":
            json.dumps(
                candidate_targets,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            ),
    }


    # ---------------------------------------------
    # IMAGE PROVENANCE
    #
    # Provenance only.
    # NOT diagnosis evidence.
    # ---------------------------------------------

    if INCLUDE_IMAGE_PROVENANCE:

        output_row.update({

            "decodable_image_count":
                int(
                    row.decodable_image_count
                ),

            "decodable_files_json":
                json.dumps(
                    decodable_files,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                ),

            "source_image_ids_json":
                json.dumps(
                    source_image_ids,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                ),
        })


    output_rows.append(
        output_row
    )


review_input = pd.DataFrame(
    output_rows
)


# =========================================================
# BLINDNESS CHECK
# =========================================================

forbidden_columns = {

    "routing_status",
    "routing_reason",
    "routing_stage",

    "confirmed_targets",
    "confirmed_target_names",

    "high_confidence_targets",
    "high_confidence_target_names",

    "silver_target_concept_id",
    "silver_target_disease",

    "diagnosis_status",
    "high_confidence_confirmed",

    "has_contradictory_assertions",

    "raw_scope_diseases",
    "raw_scope_disease_names",
}


leaked_columns = (
    set(
        review_input.columns
    )
    &
    forbidden_columns
)


if leaked_columns:

    raise RuntimeError(
        "Blind review input leaked "
        "deterministic information: "
        f"{sorted(leaked_columns)}"
    )


# =========================================================
# REVIEW INPUT INTEGRITY
# =========================================================

if review_input[
    "case_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate case_id in "
        "AI review input."
    )


if len(
    review_input
) != len(
    review_case_ids
):

    raise RuntimeError(
        "AI review row count mismatch."
    )


if set(
    review_input[
        "case_id"
    ]
) != review_case_ids:

    raise RuntimeError(
        "AI review population mismatch."
    )


if (
    review_input[
        "candidate_target_count"
    ]
    <=
    0
).any():

    raise RuntimeError(
        "AI review case with zero "
        "retained candidate targets."
    )


if INCLUDE_IMAGE_PROVENANCE:

    if (
        review_input[
            "decodable_image_count"
        ]
        <=
        0
    ).any():

        raise RuntimeError(
            "AI review case without "
            "decodable image."
        )


# =========================================================
# VERIFY ALL EXPOSED TARGETS ARE RETAINED
# =========================================================

for row in review_input.itertuples(
    index=False
):

    targets = json.loads(
        row.candidate_targets_json
    )


    for target in targets:

        concept_id = (
            target[
                "concept_id"
            ]
        )


        if (
            concept_id
            not in
            retained_concept_ids
        ):

            raise RuntimeError(
                "Excluded disease leaked into "
                "AI review candidate list: "
                f"{concept_id}"
            )


# =========================================================
# SAVE
# =========================================================

review_input.to_parquet(
    REVIEW_INPUT_OUTPUT,
    index=False,
)


retained_diseases.to_parquet(
    RETAINED_DISEASES_OUTPUT,
    index=False,
)


excluded_diseases.to_parquet(
    EXCLUDED_DISEASES_OUTPUT,
    index=False,
)


scope_filtered_cases.to_parquet(
    SCOPE_FILTERED_CASES_OUTPUT,
    index=False,
)


disease_frequency.to_csv(
    DISEASE_FREQUENCY_REPORT,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# SHA256
# =========================================================

review_file_sha256 = (
    sha256_file(
        REVIEW_INPUT_OUTPUT
    )
)


# =========================================================
# SUMMARY
# =========================================================

summary = {

    "version":
        VERSION,

    "frequency_definition":
        "COUNT(DISTINCT case_id) per disease "
        "within pre_ai_eligible",

    "minimum_candidate_cases_per_disease":
        int(
            MIN_CASES_PER_DISEASE
        ),

    "pre_ai_eligible_cases":
        int(
            PRE_AI_CASE_COUNT
        ),

    "candidate_disease_concepts_before_filter":
        int(
            len(
                disease_frequency
            )
        ),

    "retained_disease_concepts":
        int(
            len(
                retained_diseases
            )
        ),

    "excluded_disease_concepts":
        int(
            len(
                excluded_diseases
            )
        ),

    "ai_review_cases":
        int(
            len(
                review_input
            )
        ),

    "scope_filtered_cases":
        int(
            len(
                scope_filtered_cases
            )
        ),

    "coverage_fraction":
        float(
            len(
                review_input
            )
            /
            PRE_AI_CASE_COUNT
        ),

    "raw_text_chars_total":
        int(
            review_input[
                "raw_text_chars"
            ].sum()
        ),

    "raw_text_chars_median":
        float(
            review_input[
                "raw_text_chars"
            ].median()
        ),

    "raw_text_chars_max":
        int(
            review_input[
                "raw_text_chars"
            ].max()
        ),

    "review_input_sha256":
        review_file_sha256,

    "blind_review":
        True,

    "routing_information_exported":
        False,

    "deterministic_diagnosis_exported":
        False,

    "article_title_exported":
        False,

    "keywords_exported":
        False,

    "mesh_exported":
        False,

    "image_captions_exported":
        False,
}


with SUMMARY_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =========================================================
# PRINT
# =========================================================

print()
print(
    "=" * 72
)

print(
    "STEP 19 DISEASE-FREQUENCY FILTER "
    "+ BLIND AI REVIEW EXPORT"
)

print(
    "=" * 72
)


print(
    "Pre-AI eligible cases:",
    PRE_AI_CASE_COUNT
)


print(
    "Candidate disease concepts before filter:",
    len(
        disease_frequency
    ),
)


print(
    "Minimum cases per disease:",
    MIN_CASES_PER_DISEASE
)


print(
    "Retained disease concepts:",
    len(
        retained_diseases
    ),
)


print(
    "Excluded disease concepts:",
    len(
        excluded_diseases
    ),
)


print()
print(
    "Cases retained for AI review:",
    len(
        review_input
    ),
)


print(
    "Cases removed by scope filter:",
    len(
        scope_filtered_cases
    ),
)


print(
    "Coverage:",
    f"{len(review_input) / PRE_AI_CASE_COUNT:.2%}"
)


print()
print(
    "AI review input:"
)


print(
    REVIEW_INPUT_OUTPUT
)


print(
    "SHA256:",
    review_file_sha256
)


print()
print(
    "Audit outputs:"
)


print(
    " -",
    RETAINED_DISEASES_OUTPUT
)


print(
    " -",
    EXCLUDED_DISEASES_OUTPUT
)


print(
    " -",
    SCOPE_FILTERED_CASES_OUTPUT
)


print(
    " -",
    DISEASE_FREQUENCY_REPORT
)


print(
    " -",
    SUMMARY_OUTPUT
)


print()
print(
    "BLIND REVIEW CHECK: PASS"
)


print(
    "FINAL STATUS: PASS"
)


"""
STEP 19 DISEASE-FREQUENCY FILTER + BLIND AI REVIEW EXPORT
========================================================================
Pre-AI eligible cases: 7557
Candidate disease concepts before filter: 316
Minimum cases per disease: 50
Retained disease concepts: 52
Excluded disease concepts: 264

Cases retained for AI review: 6861
Cases removed by scope filter: 696
Coverage: 90.79%

AI review input:
data\ai_review\input\ai_review_input.parquet
SHA256: aac90214727de24e727117753fb0991ef10d3c331158a28ec66710d7f12e5478

Audit outputs:
 - data\ai_review\input\retained_diseases.parquet
 - data\ai_review\input\excluded_diseases.parquet
 - data\ai_review\input\scope_filtered_cases.parquet
 - reports\ai_review\disease_frequency_all.csv
 - reports\ai_review\step19_summary.json
"""