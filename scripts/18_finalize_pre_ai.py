from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, features
from tqdm import tqdm


# =========================================================
# PATHS
# =========================================================

CONFIG_PATH = Path(
    "configs/pre_ai.yaml"
)

VALIDATION_PATH = Path(
    "reports/candidates/"
    "candidate_validation.json"
)

KB_PATH = Path(
    "taxonomy/"
    "disease_knowledge_base.parquet"
)

SCOPE_PATH = Path(
    "reports/scope/"
    "disease_scope.parquet"
)

CASES_PATH = Path(
    "data/normalized/"
    "cases.parquet"
)

IMAGES_PATH = Path(
    "data/normalized/"
    "images.parquet"
)

CANDIDATES_PATH = Path(
    "data/candidates/"
    "candidates.parquet"
)

MATCHES_PATH = Path(
    "data/candidates/"
    "candidate_matches.parquet"
)

LABELS_PATH = Path(
    "data/screening/"
    "labels.parquet"
)


OUTPUT_DIR = Path(
    "data/pre_ai"
)

REPORT_DIR = Path(
    "reports/pre_ai"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ROUTING_OUTPUT = (
    OUTPUT_DIR /
    "routing.parquet"
)

REJECT_OUTPUT = (
    OUTPUT_DIR /
    "pre_ai_rejected.parquet"
)

ELIGIBLE_OUTPUT = (
    OUTPUT_DIR /
    "pre_ai_eligible.parquet"
)

SILVER_OUTPUT = (
    OUTPUT_DIR /
    "silver_accept.parquet"
)

AI_REVIEW_OUTPUT = (
    OUTPUT_DIR /
    "ai_review_queue.parquet"
)

IMAGE_AUDIT_OUTPUT = (
    OUTPUT_DIR /
    "physical_image_audit.parquet"
)

IMAGE_CASE_OUTPUT = (
    OUTPUT_DIR /
    "physical_image_case_summary.parquet"
)


SUMMARY_OUTPUT = (
    REPORT_DIR /
    "step18_summary.json"
)

REJECT_REASONS_OUTPUT = (
    REPORT_DIR /
    "reject_reasons.csv"
)

AI_REASONS_OUTPUT = (
    REPORT_DIR /
    "ai_review_reasons.csv"
)

MULTI_SCOPE_OUTPUT = (
    REPORT_DIR /
    "multiple_scope_mentions.csv"
)

IMAGE_FAILURES_OUTPUT = (
    REPORT_DIR /
    "image_failures.csv"
)

IMAGE_IDENTITY_OUTPUT = (
    REPORT_DIR /
    "image_identity_diagnostics.json"
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

    value = str(
        value
    ).strip()

    return value or None


def normalize_text(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(
        value
    ).strip()

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
        return list(
            value
        )

    if hasattr(
        value,
        "tolist",
    ):
        try:

            result = (
                value.tolist()
            )

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


def safe_int(value):

    if value is None:
        return None

    try:

        if pd.isna(
            value
        ):
            return None

    except Exception:
        pass

    try:

        return int(
            value
        )

    except Exception:

        return None


def require_columns(
    dataframe,
    required_columns,
    table_name,
):

    missing = (
        set(
            required_columns
        )
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{table_name} missing columns: "
            f"{sorted(missing)}"
        )


def unique_non_null(
    values,
):

    output = set()

    for value in values:

        normalized = (
            normalize_id(
                value
            )
        )

        if normalized is not None:

            output.add(
                normalized
            )

    return sorted(
        output
    )


# =========================================================
# CLI
# =========================================================

parser = argparse.ArgumentParser(
    description=(
        "Step 18 final pre-AI gate."
    )
)


parser.add_argument(
    "--image-root",
    required=True,
    help=(
        "Directory containing already "
        "extracted MultiCaRe images."
    ),
)


args = parser.parse_args()


IMAGE_ROOT = Path(
    args.image_root
).resolve()


if not IMAGE_ROOT.exists():

    raise RuntimeError(
        "Image root does not exist: "
        f"{IMAGE_ROOT}"
    )


if not IMAGE_ROOT.is_dir():

    raise RuntimeError(
        "Image root is not a directory: "
        f"{IMAGE_ROOT}"
    )


print(
    "Image root:",
    IMAGE_ROOT,
)


# =========================================================
# STEP 16 VALIDATION
# =========================================================

if VALIDATION_PATH.exists():

    with VALIDATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        validation = json.load(
            f
        )


    validation_status = (
        validation.get(
            "status"
        )
        or
        validation.get(
            "final_status"
        )
    )


    if (
        validation_status is not None
        and
        str(
            validation_status
        ).upper()
        !=
        "PASS"
    ):

        raise RuntimeError(
            "Step 16 validation "
            "is not PASS: "
            f"{validation_status}"
        )


    hard_errors = (
        validation.get(
            "hard_errors"
        )
    )


    if isinstance(
        hard_errors,
        list,
    ):

        hard_error_count = len(
            hard_errors
        )

    elif hard_errors is None:

        hard_error_count = None

    else:

        try:

            hard_error_count = int(
                hard_errors
            )

        except Exception:

            hard_error_count = None


    if (
        hard_error_count is not None
        and
        hard_error_count > 0
    ):

        raise RuntimeError(
            "Step 16 validation contains "
            "hard errors."
        )


    print(
        "Step 16 validation: PASS"
    )

else:

    print(
        "WARNING: Step 16 validation JSON "
        "not found. Continuing with internal "
        "Step 18 integrity checks."
    )


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


RULE_VERSION = str(
    config[
        "version"
    ]
)


rules = (
    config[
        "rules"
    ]
)


MAX_SCOPE_DISEASES_FOR_SILVER = int(
    rules.get(
        "max_scope_diseases_for_silver",
        1,
    )
)


# =========================================================
# LOAD INPUTS
# =========================================================

print()
print(
    "Loading Step 18 inputs..."
)


kb = pd.read_parquet(
    KB_PATH
)

scope = pd.read_parquet(
    SCOPE_PATH
)

cases = pd.read_parquet(
    CASES_PATH
)

images = pd.read_parquet(
    IMAGES_PATH
)

candidates = pd.read_parquet(
    CANDIDATES_PATH
)

matches = pd.read_parquet(
    MATCHES_PATH
)

labels = pd.read_parquet(
    LABELS_PATH
)


# =========================================================
# SCHEMA CHECKS
# =========================================================

require_columns(
    kb,
    [
        "concept_id",
        "disease",
        "parents",
    ],
    "disease_knowledge_base",
)


require_columns(
    scope,
    [
        "concept_id",
        "disease",
    ],
    "disease_scope",
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


# Actual Step 14 schema.
require_columns(
    images,
    [
        "image_id",
        "file",
        "case_id",
        "article_id",
        "source_image_id",
        "file_size",
    ],
    "images",
)


require_columns(
    candidates,
    [
        "case_id",
        "article_id",
        "concept_id",
        "disease",
    ],
    "candidates",
)


require_columns(
    matches,
    [
        "case_id",
        "concept_id",
        "field",
        "alias",
    ],
    "candidate_matches",
)


require_columns(
    labels,
    [
        "case_id",
        "concept_id",
        "diagnosis_status",
        "high_confidence_confirmed",
        "contradictory_assertions",
    ],
    "labels",
)


# =========================================================
# NORMALIZE IDS / FILENAMES
# =========================================================

for dataframe in [
    kb,
    scope,
    candidates,
    matches,
    labels,
]:

    dataframe[
        "concept_id"
    ] = dataframe[
        "concept_id"
    ].map(
        normalize_id
    )


for dataframe in [
    cases,
    images,
    candidates,
    matches,
    labels,
]:

    dataframe[
        "case_id"
    ] = dataframe[
        "case_id"
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


images[
    "source_image_id"
] = images[
    "source_image_id"
].map(
    normalize_id
)


images[
    "file"
] = images[
    "file"
].map(
    normalize_text
)


# =========================================================
# BASIC INTEGRITY
# =========================================================

if cases[
    "case_id"
].duplicated().any():

    duplicate_ids = (
        cases.loc[
            cases[
                "case_id"
            ].duplicated(
                keep=False
            ),
            "case_id",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "Duplicate case_id in cases.parquet: "
        f"{duplicate_ids[:20]}"
    )


if candidates[
    [
        "case_id",
        "concept_id",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate case_id × concept_id "
        "in candidates.parquet."
    )


# =========================================================
# IMAGE IDENTITY DIAGNOSTICS
#
# IMPORTANT:
#
# image_id is NOT assumed to be unique.
#
# Physical final-image identity for Step 18:
#
#     case_id + file
#
# All image_id/source_image_id values are retained
# as provenance.
# =========================================================

duplicate_image_id_rows = int(
    images[
        "image_id"
    ]
    .duplicated(
        keep=False
    )
    .sum()
)


duplicate_case_file_rows = int(
    images[
        [
            "case_id",
            "file",
        ]
    ]
    .duplicated(
        keep=False
    )
    .sum()
)


print()
print(
    "Rows with duplicated image_id:",
    duplicate_image_id_rows,
)


print(
    "Rows participating in duplicate "
    "case_id + file groups:",
    duplicate_case_file_rows,
)


# =========================================================
# COMPLETE CANDIDATE POPULATION
# =========================================================

all_candidate_case_ids = sorted(
    set(
        candidates[
            "case_id"
        ]
        .dropna()
        .tolist()
    )
)


all_candidate_case_set = set(
    all_candidate_case_ids
)


print()
print(
    "Candidate cases:",
    len(
        all_candidate_case_ids
    ),
)


case_lookup = cases.set_index(
    "case_id"
)


missing_cases = (
    all_candidate_case_set
    -
    set(
        case_lookup.index
    )
)


if missing_cases:

    raise RuntimeError(
        "Candidate cases missing from "
        "cases.parquet: "
        f"{sorted(missing_cases)[:20]}"
    )


# =========================================================
# ONTOLOGY
# =========================================================

print(
    "Building ontology graph..."
)


parent_map = {}


for row in kb.itertuples(
    index=False
):

    concept_id = normalize_id(
        row.concept_id
    )


    parents = {
        normalize_id(
            parent
        )
        for parent
        in as_list(
            row.parents
        )
        if normalize_id(
            parent
        )
        is not None
    }


    parent_map[
        concept_id
    ] = parents


@lru_cache(
    maxsize=None
)
def ancestors(
    concept_id,
):

    output = set()

    stack = list(
        parent_map.get(
            concept_id,
            set(),
        )
    )

    seen = set()


    while stack:

        parent = stack.pop()


        if parent in seen:

            continue


        seen.add(
            parent
        )

        output.add(
            parent
        )


        stack.extend(
            parent_map.get(
                parent,
                set(),
            )
        )


    return frozenset(
        output
    )


def most_specific(
    concept_ids,
):

    ids = {
        normalize_id(
            concept_id
        )
        for concept_id
        in concept_ids
        if normalize_id(
            concept_id
        )
        is not None
    }


    if not ids:

        return []


    ancestor_ids = set()


    for concept_id in ids:

        ancestor_ids.update(
            ancestors(
                concept_id
            )
        )


    return sorted(
        ids
        -
        ancestor_ids
    )


disease_lookup = dict(
    zip(
        kb[
            "concept_id"
        ],
        kb[
            "disease"
        ],
    )
)


def disease_names(
    concept_ids,
):

    return [
        disease_lookup.get(
            concept_id,
            concept_id,
        )
        for concept_id
        in concept_ids
    ]


# =========================================================
# DISEASE SCOPE
# =========================================================

scope_ids = set(
    scope[
        "concept_id"
    ]
    .dropna()
    .tolist()
)


print(
    "Disease scope concepts:",
    len(
        scope_ids
    ),
)


# =========================================================
# DETECT AMBIGUOUS ALIASES
#
# Derived directly from candidate_matches.
# =========================================================

scope_alias_rows = matches[
    matches[
        "concept_id"
    ].isin(
        scope_ids
    )
][
    [
        "alias",
        "concept_id",
    ]
].copy()


scope_alias_rows[
    "alias_normalized"
] = (
    scope_alias_rows[
        "alias"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


alias_concept_counts = (
    scope_alias_rows
    .groupby(
        "alias_normalized"
    )[
        "concept_id"
    ]
    .nunique()
)


ambiguous_aliases = set(
    alias_concept_counts[
        alias_concept_counts
        >
        1
    ].index
)


print(
    "Ambiguous aliases:",
    len(
        ambiguous_aliases
    ),
)


# =========================================================
# RAW-TEXT DISEASE MENTIONS
#
# ONLY:
#   field == raw_case_text
#
# Never:
#   article_title
#   keyword
#   MeSH
# =========================================================

print()
print(
    "Building raw-text disease inventory..."
)


raw_scope_matches_all = matches[
    (
        matches[
            "case_id"
        ].isin(
            all_candidate_case_set
        )
    )
    &
    (
        matches[
            "field"
        ]
        ==
        "raw_case_text"
    )
    &
    (
        matches[
            "concept_id"
        ].isin(
            scope_ids
        )
    )
].copy()


raw_scope_matches_all[
    "alias_normalized"
] = (
    raw_scope_matches_all[
        "alias"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


raw_scope_matches_all[
    "ambiguous_alias"
] = (
    raw_scope_matches_all[
        "alias_normalized"
    ]
    .isin(
        ambiguous_aliases
    )
)


if rules.get(
    "ignore_ambiguous_aliases_for_scope_count",
    True,
):

    raw_scope_matches_for_count = (
        raw_scope_matches_all[
            ~raw_scope_matches_all[
                "ambiguous_alias"
            ]
        ]
        .copy()
    )

else:

    raw_scope_matches_for_count = (
        raw_scope_matches_all.copy()
    )


def build_scope_map(
    dataframe,
):

    result = {}


    for case_id, group in (
        dataframe.groupby(
            "case_id",
            sort=False,
        )
    ):

        concept_ids_raw = sorted(
            set(
                group[
                    "concept_id"
                ]
                .dropna()
                .tolist()
            )
        )


        collapsed = most_specific(
            concept_ids_raw
        )


        result[
            case_id
        ] = {

            "raw":
                concept_ids_raw,

            "collapsed":
                collapsed,

            "match_count":
                int(
                    len(
                        group
                    )
                ),
        }


    return result


raw_scope_all_map = build_scope_map(
    raw_scope_matches_all
)


raw_scope_count_map = build_scope_map(
    raw_scope_matches_for_count
)


ambiguous_cases = set(
    raw_scope_matches_all.loc[
        raw_scope_matches_all[
            "ambiguous_alias"
        ],
        "case_id",
    ]
    .dropna()
    .tolist()
)


# =========================================================
# CANDIDATE IMAGE METADATA
# =========================================================

candidate_image_rows = images[
    images[
        "case_id"
    ].isin(
        all_candidate_case_set
    )
].copy()


# A final physical image requires a non-empty `file`.
candidate_image_rows_with_file = (
    candidate_image_rows[
        candidate_image_rows[
            "file"
        ].notna()
    ]
    .copy()
)


# =========================================================
# CASE IMAGE COUNTS
#
# Do NOT use image_id as identity.
# =========================================================

image_count_map = (
    candidate_image_rows_with_file
    .groupby(
        "case_id"
    )[
        "file"
    ]
    .nunique()
    .to_dict()
)


image_metadata_row_count_map = (
    candidate_image_rows
    .groupby(
        "case_id"
    )
    .size()
    .to_dict()
)


source_figure_count_map = (
    candidate_image_rows
    .groupby(
        "case_id"
    )[
        "source_image_id"
    ]
    .nunique()
    .to_dict()
)


# =========================================================
# BUILD UNIQUE PHYSICAL IMAGE RECORDS
#
# Grain:
#
#   case_id + file
#
# Multiple metadata rows mapping to the same physical
# file are consolidated here, while retaining provenance.
# =========================================================

physical_image_records = []


for (
    case_id,
    filename,
), group in (
    candidate_image_rows_with_file
    .groupby(
        [
            "case_id",
            "file",
        ],
        sort=False,
        dropna=False,
    )
):

    image_ids = unique_non_null(
        group[
            "image_id"
        ].tolist()
    )


    source_image_ids = unique_non_null(
        group[
            "source_image_id"
        ].tolist()
    )


    article_ids = unique_non_null(
        group[
            "article_id"
        ].tolist()
    )


    metadata_sizes = sorted(
        {
            value
            for value
            in (
                safe_int(
                    item
                )
                for item
                in group[
                    "file_size"
                ].tolist()
            )
            if value is not None
        }
    )


    image_key = (
        str(
            case_id
        )
        +
        "::"
        +
        str(
            filename
        )
    )


    physical_image_records.append({

        "image_key":
            image_key,

        "case_id":
            case_id,

        "file":
            filename,

        "image_ids":
            image_ids,

        "source_image_ids":
            source_image_ids,

        "article_ids":
            article_ids,

        "metadata_file_sizes":
            metadata_sizes,

        "metadata_row_count":
            int(
                len(
                    group
                )
            ),
    })


physical_image_table = pd.DataFrame(
    physical_image_records
)


if physical_image_table[
    "image_key"
].duplicated().any():

    raise RuntimeError(
        "Internal error: duplicate image_key "
        "after case_id + file grouping."
    )


print(
    "Unique candidate physical images:",
    len(
        physical_image_table
    ),
)


# =========================================================
# STRUCTURAL HARD GATES
#
# R01:
#   raw_case_text must exist
#
# R02:
#   at least one final image filename must exist
# =========================================================

print()
print(
    "Applying structural hard gates..."
)


structural_map = {}

structural_survivors = set()


for case_id in all_candidate_case_ids:

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


    has_raw_case_text = bool(
        raw_case_text.strip()
    )


    image_count = int(
        image_count_map.get(
            case_id,
            0,
        )
    )


    metadata_row_count = int(
        image_metadata_row_count_map.get(
            case_id,
            0,
        )
    )


    source_figure_count = int(
        source_figure_count_map.get(
            case_id,
            0,
        )
    )


    failed_rules = []


    if (
        rules.get(
            "require_raw_case_text",
            True,
        )
        and
        not has_raw_case_text
    ):

        failed_rules.append(
            "NO_RAW_CASE_TEXT"
        )


    if (
        rules.get(
            "require_source_image",
            True,
        )
        and
        image_count
        ==
        0
    ):

        failed_rules.append(
            "NO_SOURCE_IMAGE"
        )


    structural_map[
        case_id
    ] = {

        "has_raw_case_text":
            has_raw_case_text,

        "image_metadata_row_count":
            metadata_row_count,

        "image_count":
            image_count,

        "source_figure_count":
            source_figure_count,

        "failed_rules":
            failed_rules,
    }


    if not failed_rules:

        structural_survivors.add(
            case_id
        )


print(
    "Passed structural gates:",
    len(
        structural_survivors
    ),
)


# =========================================================
# PHYSICAL IMAGES TO VALIDATE
# =========================================================

physical_images_to_validate = (
    physical_image_table[
        physical_image_table[
            "case_id"
        ].isin(
            structural_survivors
        )
    ]
    .copy()
)


needed_filenames = set(
    physical_images_to_validate[
        "file"
    ]
    .dropna()
    .astype(str)
    .map(
        lambda value:
            Path(
                value
            ).name.lower()
    )
)


print()
print(
    "Unique physical images to validate:",
    len(
        physical_images_to_validate
    ),
)


print(
    "Unique filenames needed:",
    len(
        needed_filenames
    ),
)


# =========================================================
# WEBP SUPPORT CHECK
# =========================================================

has_webp_support = bool(
    features.check(
        "webp"
    )
)


print(
    "Pillow WebP support:",
    has_webp_support
)


if (
    any(
        filename.endswith(
            ".webp"
        )
        for filename
        in needed_filenames
    )
    and
    not has_webp_support
):

    raise RuntimeError(
        "Pillow does not have WebP support. "
        "Valid dataset images would be falsely "
        "reported as decode failures."
    )


# =========================================================
# INDEX PHYSICAL FILES
#
# Match actual files using images.parquet column `file`.
# =========================================================

print()
print(
    "Indexing extracted image directory..."
)


basename_index = defaultdict(
    list
)


physical_files_scanned = 0

relevant_physical_files = 0


for path in tqdm(
    IMAGE_ROOT.rglob("*")
):

    if not path.is_file():

        continue


    physical_files_scanned += 1


    basename = (
        path.name
        .strip()
        .lower()
    )


    if basename not in needed_filenames:

        continue


    basename_index[
        basename
    ].append(
        path
    )


    relevant_physical_files += 1


print(
    "Physical files scanned:",
    physical_files_scanned
)


print(
    "Relevant physical files indexed:",
    relevant_physical_files
)


# =========================================================
# PHYSICAL FILE RESOLUTION
# =========================================================

def locate_image(
    filename,
    metadata_sizes,
):

    if filename is None:

        return (
            None,
            "NO_FILENAME",
        )


    basename = (
        Path(
            str(
                filename
            ).strip()
        )
        .name
        .lower()
    )


    found = (
        basename_index.get(
            basename,
            [],
        )
    )


    if len(
        found
    ) == 0:

        return (
            None,
            "NOT_FOUND",
        )


    if len(
        found
    ) == 1:

        return (
            found[
                0
            ],
            "UNIQUE_BASENAME",
        )


    # If the same basename exists in several directories,
    # try resolving it using the metadata byte size.
    metadata_sizes = set(
        metadata_sizes
        or []
    )


    if metadata_sizes:

        size_matches = [
            path
            for path
            in found
            if int(
                path.stat().st_size
            )
            in metadata_sizes
        ]


        if len(
            size_matches
        ) == 1:

            return (
                size_matches[
                    0
                ],
                "BASENAME_SIZE_MATCH",
            )


    return (
        None,
        "AMBIGUOUS_BASENAME",
    )


# =========================================================
# IMAGE DECODE
# =========================================================

def validate_image(
    path,
):

    if path is None:

        return {
            "file_exists":
                False,

            "actual_file_size":
                None,

            "decode_ok":
                False,

            "width":
                None,

            "height":
                None,

            "image_format":
                None,

            "error_reason":
                "FILE_NOT_FOUND",
        }


    if not path.exists():

        return {
            "file_exists":
                False,

            "actual_file_size":
                None,

            "decode_ok":
                False,

            "width":
                None,

            "height":
                None,

            "image_format":
                None,

            "error_reason":
                "FILE_NOT_FOUND",
        }


    file_size = int(
        path.stat().st_size
    )


    if file_size <= 0:

        return {
            "file_exists":
                True,

            "actual_file_size":
                file_size,

            "decode_ok":
                False,

            "width":
                None,

            "height":
                None,

            "image_format":
                None,

            "error_reason":
                "EMPTY_FILE",
        }


    try:

        # Structural integrity check.
        with Image.open(
            path
        ) as image:

            image.verify()


        # Reopen and actually decode pixels.
        with Image.open(
            path
        ) as image:

            image.load()

            width, height = (
                image.size
            )

            image_format = (
                image.format
            )


        if (
            width <= 0
            or
            height <= 0
        ):

            return {
                "file_exists":
                    True,

                "actual_file_size":
                    file_size,

                "decode_ok":
                    False,

                "width":
                    int(
                        width
                    ),

                "height":
                    int(
                        height
                    ),

                "image_format":
                    image_format,

                "error_reason":
                    "INVALID_DIMENSIONS",
            }


        return {
            "file_exists":
                True,

            "actual_file_size":
                file_size,

            "decode_ok":
                True,

            "width":
                int(
                    width
                ),

            "height":
                int(
                    height
                ),

            "image_format":
                image_format,

            "error_reason":
                None,
        }


    except Exception as exc:

        return {
            "file_exists":
                True,

            "actual_file_size":
                file_size,

            "decode_ok":
                False,

            "width":
                None,

            "height":
                None,

            "image_format":
                None,

            "error_reason":
                (
                    type(
                        exc
                    ).__name__
                    +
                    ": "
                    +
                    str(
                        exc
                    )[:300]
                ),
        }


# =========================================================
# PHYSICAL IMAGE VALIDATION
# =========================================================

print()
print(
    "Validating physical images..."
)


decode_cache = {}

image_audit_rows = []


for row in tqdm(
    physical_images_to_validate.itertuples(
        index=False
    ),
    total=len(
        physical_images_to_validate
    ),
):

    path, resolution_method = (
        locate_image(
            row.file,
            row.metadata_file_sizes,
        )
    )


    # Avoid decoding the same physical path repeatedly
    # if it is referenced by several cases.
    cache_key = (
        str(
            path
        )
        if path is not None
        else None
    )


    if (
        cache_key is not None
        and
        cache_key in decode_cache
    ):

        validation_result = dict(
            decode_cache[
                cache_key
            ]
        )

    else:

        validation_result = (
            validate_image(
                path
            )
        )


        if cache_key is not None:

            decode_cache[
                cache_key
            ] = dict(
                validation_result
            )


    actual_file_size = (
        validation_result[
            "actual_file_size"
        ]
    )


    metadata_sizes = list(
        row.metadata_file_sizes
    )


    if (
        actual_file_size is not None
        and
        metadata_sizes
    ):

        size_matches_metadata = (
            actual_file_size
            in
            set(
                metadata_sizes
            )
        )

    else:

        size_matches_metadata = None


    image_audit_rows.append({

        "image_key":
            row.image_key,

        "case_id":
            row.case_id,

        "file":
            row.file,

        "image_ids":
            list(
                row.image_ids
            ),

        "source_image_ids":
            list(
                row.source_image_ids
            ),

        "article_ids":
            list(
                row.article_ids
            ),

        "metadata_row_count":
            int(
                row.metadata_row_count
            ),

        "metadata_file_sizes":
            metadata_sizes,

        "resolved_path":
            (
                str(
                    path
                )
                if path is not None
                else None
            ),

        "resolution_method":
            resolution_method,

        **validation_result,

        "size_matches_metadata":
            size_matches_metadata,

        "rule_version":
            RULE_VERSION,
    })


image_audit = pd.DataFrame(
    image_audit_rows
)


if image_audit[
    "image_key"
].duplicated().any():

    raise RuntimeError(
        "Duplicate image_key in "
        "physical_image_audit."
    )


image_audit.to_parquet(
    IMAGE_AUDIT_OUTPUT,
    index=False,
)


# =========================================================
# CASE-LEVEL PHYSICAL IMAGE SUMMARY
# =========================================================

physical_summary_map = {}


for case_id in sorted(
    structural_survivors
):

    group = image_audit[
        image_audit[
            "case_id"
        ]
        ==
        case_id
    ]


    decodable = group[
        group[
            "decode_ok"
        ]
        ==
        True
    ]


    all_image_ids = set()

    all_source_image_ids = set()


    for values in decodable[
        "image_ids"
    ].tolist():

        for value in as_list(
            values
        ):

            if value is not None:

                all_image_ids.add(
                    str(
                        value
                    )
                )


    for values in decodable[
        "source_image_ids"
    ].tolist():

        for value in as_list(
            values
        ):

            if value is not None:

                all_source_image_ids.add(
                    str(
                        value
                    )
                )


    physical_summary_map[
        case_id
    ] = {

        "physical_found_count":
            int(
                group[
                    "file_exists"
                ].sum()
            ),

        "decodable_image_count":
            int(
                len(
                    decodable
                )
            ),

        "missing_image_count":
            int(
                (
                    group[
                        "file_exists"
                    ]
                    ==
                    False
                ).sum()
            ),

        "decode_failed_count":
            int(
                (
                    (
                        group[
                            "file_exists"
                        ]
                        ==
                        True
                    )
                    &
                    (
                        group[
                            "decode_ok"
                        ]
                        ==
                        False
                    )
                ).sum()
            ),

        "size_mismatch_count":
            int(
                (
                    group[
                        "size_matches_metadata"
                    ]
                    ==
                    False
                ).sum()
            ),

        "decodable_image_keys":
            sorted(
                decodable[
                    "image_key"
                ]
                .dropna()
                .unique()
                .tolist()
            ),

        "decodable_files":
            sorted(
                decodable[
                    "file"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),

        "decodable_image_ids":
            sorted(
                all_image_ids
            ),

        "decodable_source_image_ids":
            sorted(
                all_source_image_ids
            ),
    }


physical_summary = pd.DataFrame(
    [
        {
            "case_id":
                case_id,

            **values,
        }

        for case_id, values
        in physical_summary_map.items()
    ]
)


physical_summary.to_parquet(
    IMAGE_CASE_OUTPUT,
    index=False,
)


# =========================================================
# CANDIDATE DISEASE MAP
# =========================================================

candidate_map = (
    candidates
    .groupby(
        "case_id"
    )[
        "concept_id"
    ]
    .apply(
        lambda values:
            sorted(
                set(
                    values
                    .dropna()
                    .tolist()
                )
            )
    )
    .to_dict()
)


# =========================================================
# STEP 17 LABEL MAP
# =========================================================

label_groups = {

    case_id:
        group.copy()

    for case_id, group
    in labels.groupby(
        "case_id",
        sort=False,
    )
}


# =========================================================
# FINAL ROUTING
# =========================================================

print()
print(
    "Building final pre-AI routing..."
)


routing_rows = []


for case_id in tqdm(
    all_candidate_case_ids
):

    case = (
        case_lookup.loc[
            case_id
        ]
    )


    structural = (
        structural_map[
            case_id
        ]
    )


    raw_all_info = (
        raw_scope_all_map.get(
            case_id,
            {
                "raw": [],
                "collapsed": [],
                "match_count": 0,
            },
        )
    )


    raw_count_info = (
        raw_scope_count_map.get(
            case_id,
            {
                "raw": [],
                "collapsed": [],
                "match_count": 0,
            },
        )
    )


    raw_scope_diseases_all = (
        raw_all_info[
            "collapsed"
        ]
    )


    raw_scope_diseases = (
        raw_count_info[
            "collapsed"
        ]
    )


    raw_scope_disease_count = len(
        raw_scope_diseases
    )


    has_multiple_scope_mentions = (
        raw_scope_disease_count
        >
        MAX_SCOPE_DISEASES_FOR_SILVER
    )


    physical = (
        physical_summary_map.get(
            case_id
        )
    )


    confirmed_targets = []

    high_confidence_targets = []

    contradiction = False

    silver_target = None


    # =====================================================
    # HARD GATE 1:
    # RAW CASE TEXT REQUIRED
    # =====================================================

    if (
        rules.get(
            "require_raw_case_text",
            True,
        )
        and
        not structural[
            "has_raw_case_text"
        ]
    ):

        routing_status = (
            "PRE_AI_REJECT"
        )

        routing_reason = (
            "NO_RAW_CASE_TEXT"
        )

        routing_stage = (
            "STRUCTURAL_GATE"
        )


    # =====================================================
    # HARD GATE 2:
    # SOURCE IMAGE REQUIRED
    # =====================================================

    elif (
        rules.get(
            "require_source_image",
            True,
        )
        and
        structural[
            "image_count"
        ]
        ==
        0
    ):

        routing_status = (
            "PRE_AI_REJECT"
        )

        routing_reason = (
            "NO_SOURCE_IMAGE"
        )

        routing_stage = (
            "STRUCTURAL_GATE"
        )


    # =====================================================
    # HARD GATE 3:
    # >=1 PHYSICALLY DECODABLE IMAGE REQUIRED
    # =====================================================

    elif (
        rules.get(
            "require_decodable_image",
            True,
        )
        and
        (
            physical is None
            or
            physical[
                "decodable_image_count"
            ]
            ==
            0
        )
    ):

        routing_status = (
            "PRE_AI_REJECT"
        )

        routing_reason = (
            "NO_DECODABLE_SOURCE_IMAGE"
        )

        routing_stage = (
            "PHYSICAL_IMAGE_GATE"
        )


    # =====================================================
    # DIAGNOSIS ROUTING
    # =====================================================

    else:

        group = (
            label_groups.get(
                case_id
            )
        )


        if (
            group is None
            or
            group.empty
        ):

            raise RuntimeError(
                "Case passed structural/physical "
                "gates but has no Step 17 labels: "
                f"{case_id}"
            )


        # -------------------------------------------------
        # ALL deterministic confirmed targets
        # -------------------------------------------------

        confirmed_group = group[
            group[
                "diagnosis_status"
            ]
            ==
            "confirmed"
        ]


        confirmed_targets = (
            most_specific(
                confirmed_group[
                    "concept_id"
                ]
                .dropna()
                .tolist()
            )
        )


        # -------------------------------------------------
        # HIGH-CONFIDENCE confirmed targets
        # -------------------------------------------------

        high_group = group[
            group[
                "high_confidence_confirmed"
            ]
            ==
            True
        ]


        high_confidence_targets = (
            most_specific(
                high_group[
                    "concept_id"
                ]
                .dropna()
                .tolist()
            )
        )


        # -------------------------------------------------
        # CONTRADICTION
        # -------------------------------------------------

        contradiction = bool(
            group[
                "contradictory_assertions"
            ]
            .fillna(
                False
            )
            .any()
        )


        # =================================================
        # HARD DIAGNOSIS GATE
        #
        # Only multiple HIGH-CONFIDENCE confirmed targets
        # are strong enough for automatic rejection.
        # =================================================

        if (
            rules.get(
                "reject_multiple_high_confidence_targets",
                True,
            )
            and
            len(
                high_confidence_targets
            )
            >
            1
        ):

            routing_status = (
                "PRE_AI_REJECT"
            )

            routing_reason = (
                "MULTIPLE_HIGH_CONFIDENCE_TARGETS"
            )

            routing_stage = (
                "DETERMINISTIC_DIAGNOSIS_GATE"
            )


        # =================================================
        # CONTRADICTORY RULE EVIDENCE
        # =================================================

        elif contradiction:

            routing_status = (
                "AI_REVIEW"
            )

            routing_reason = (
                "CONTRADICTORY_ASSERTIONS"
            )

            routing_stage = (
                "DETERMINISTIC_TRIAGE"
            )


        # =================================================
        # MULTIPLE DETERMINISTIC CONFIRMED TARGETS
        #
        # Not strong enough for rejection unless the
        # targets were all high-confidence above.
        # =================================================

        elif len(
            confirmed_targets
        ) > 1:

            routing_status = (
                "AI_REVIEW"
            )

            routing_reason = (
                "MULTIPLE_DETERMINISTIC_"
                "TARGETS_UNCERTAIN"
            )

            routing_stage = (
                "DETERMINISTIC_TRIAGE"
            )


        # =================================================
        # MULTIPLE DISEASE MENTIONS IN RAW CASE TEXT
        #
        # IMPORTANT:
        #
        # mention != diagnosis
        #
        # Therefore:
        #
        # AI_REVIEW
        #
        # NOT PRE_AI_REJECT
        # =================================================

        elif has_multiple_scope_mentions:

            routing_status = (
                "AI_REVIEW"
            )

            routing_reason = (
                "MULTIPLE_SCOPE_DISEASE_MENTIONS"
            )

            routing_stage = (
                "RAW_TEXT_TRIAGE"
            )


        # =================================================
        # NO DETERMINISTIC CONFIRMED TARGET
        # =================================================

        elif len(
            confirmed_targets
        ) == 0:

            routing_status = (
                "AI_REVIEW"
            )

            routing_reason = (
                "NO_DETERMINISTIC_"
                "CONFIRMED_TARGET"
            )

            routing_stage = (
                "DETERMINISTIC_TRIAGE"
            )


        # =================================================
        # EXACTLY ONE DETERMINISTIC CONFIRMED TARGET
        # =================================================

        else:

            target = (
                confirmed_targets[
                    0
                ]
            )


            strong_single_target = (
                len(
                    high_confidence_targets
                )
                ==
                1
                and
                high_confidence_targets[
                    0
                ]
                ==
                target
            )


            if strong_single_target:

                routing_status = (
                    "SILVER_ACCEPT"
                )

                routing_reason = (
                    "SINGLE_HIGH_CONFIDENCE_TARGET"
                )

                routing_stage = (
                    "SILVER_TRIAGE"
                )

                silver_target = (
                    target
                )


            else:

                routing_status = (
                    "AI_REVIEW"
                )

                routing_reason = (
                    "WEAK_OR_AMBIGUOUS_"
                    "CONFIRMATION"
                )

                routing_stage = (
                    "DETERMINISTIC_TRIAGE"
                )


    # =====================================================
    # FINAL ELIGIBILITY
    # =====================================================

    pre_ai_eligible = (
        routing_status
        in {
            "SILVER_ACCEPT",
            "AI_REVIEW",
        }
    )


    # =====================================================
    # MASTER ROUTING ROW
    # =====================================================

    routing_rows.append({

        # -----------------------------------------
        # IDENTITY
        # -----------------------------------------

        "case_id":
            case_id,

        "article_id":
            normalize_id(
                case[
                    "article_id"
                ]
            ),

        # -----------------------------------------
        # STRUCTURAL
        # -----------------------------------------

        "has_raw_case_text":
            structural[
                "has_raw_case_text"
            ],

        "image_metadata_row_count":
            structural[
                "image_metadata_row_count"
            ],

        "image_count":
            structural[
                "image_count"
            ],

        "source_figure_count":
            structural[
                "source_figure_count"
            ],

        "failed_structural_rules":
            structural[
                "failed_rules"
            ],

        # -----------------------------------------
        # RAW-TEXT DISEASE INVENTORY
        # -----------------------------------------

        "raw_scope_diseases_all":
            raw_scope_diseases_all,

        "raw_scope_disease_names_all":
            disease_names(
                raw_scope_diseases_all
            ),

        "raw_scope_diseases":
            raw_scope_diseases,

        "raw_scope_disease_names":
            disease_names(
                raw_scope_diseases
            ),

        "raw_scope_disease_count":
            raw_scope_disease_count,

        "raw_scope_match_count":
            raw_count_info[
                "match_count"
            ],

        "has_multiple_scope_mentions":
            has_multiple_scope_mentions,

        "has_ambiguous_raw_alias":
            (
                case_id
                in
                ambiguous_cases
            ),

        # -----------------------------------------
        # PHYSICAL IMAGES
        # -----------------------------------------

        "physical_found_count":
            (
                physical[
                    "physical_found_count"
                ]
                if physical is not None
                else None
            ),

        "decodable_image_count":
            (
                physical[
                    "decodable_image_count"
                ]
                if physical is not None
                else None
            ),

        "missing_image_count":
            (
                physical[
                    "missing_image_count"
                ]
                if physical is not None
                else None
            ),

        "decode_failed_count":
            (
                physical[
                    "decode_failed_count"
                ]
                if physical is not None
                else None
            ),

        "size_mismatch_count":
            (
                physical[
                    "size_mismatch_count"
                ]
                if physical is not None
                else None
            ),

        "decodable_image_keys":
            (
                physical[
                    "decodable_image_keys"
                ]
                if physical is not None
                else []
            ),

        "decodable_files":
            (
                physical[
                    "decodable_files"
                ]
                if physical is not None
                else []
            ),

        "decodable_image_ids":
            (
                physical[
                    "decodable_image_ids"
                ]
                if physical is not None
                else []
            ),

        "decodable_source_image_ids":
            (
                physical[
                    "decodable_source_image_ids"
                ]
                if physical is not None
                else []
            ),

        # -----------------------------------------
        # CANDIDATE DISEASES
        # -----------------------------------------

        "candidate_diseases":
            candidate_map.get(
                case_id,
                [],
            ),

        "candidate_disease_names":
            disease_names(
                candidate_map.get(
                    case_id,
                    [],
                )
            ),

        "candidate_disease_count":
            len(
                candidate_map.get(
                    case_id,
                    [],
                )
            ),

        # -----------------------------------------
        # DETERMINISTIC DIAGNOSIS
        # -----------------------------------------

        "confirmed_targets":
            confirmed_targets,

        "confirmed_target_names":
            disease_names(
                confirmed_targets
            ),

        "confirmed_target_count":
            len(
                confirmed_targets
            ),

        "high_confidence_targets":
            high_confidence_targets,

        "high_confidence_target_names":
            disease_names(
                high_confidence_targets
            ),

        "high_confidence_target_count":
            len(
                high_confidence_targets
            ),

        "has_contradictory_assertions":
            contradiction,

        # -----------------------------------------
        # SILVER
        # -----------------------------------------

        "silver_target_concept_id":
            silver_target,

        "silver_target_disease":
            (
                disease_lookup.get(
                    silver_target
                )
                if silver_target is not None
                else None
            ),

        # -----------------------------------------
        # FINAL ROUTING
        # -----------------------------------------

        "routing_stage":
            routing_stage,

        "routing_status":
            routing_status,

        "routing_reason":
            routing_reason,

        "pre_ai_eligible":
            pre_ai_eligible,

        "rule_version":
            RULE_VERSION,
    })


routing = pd.DataFrame(
    routing_rows
)


# =========================================================
# FINAL ROUTING INTEGRITY
# =========================================================

if routing[
    "case_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate case_id in routing."
    )


if len(
    routing
) != len(
    all_candidate_case_ids
):

    raise RuntimeError(
        "Routing must contain exactly "
        "one row per candidate case."
    )


if set(
    routing[
        "case_id"
    ]
) != all_candidate_case_set:

    raise RuntimeError(
        "Routing population does not exactly "
        "match candidates.parquet."
    )


allowed_statuses = {
    "PRE_AI_REJECT",
    "SILVER_ACCEPT",
    "AI_REVIEW",
}


invalid_statuses = (
    set(
        routing[
            "routing_status"
        ]
    )
    -
    allowed_statuses
)


if invalid_statuses:

    raise RuntimeError(
        "Invalid routing statuses: "
        f"{invalid_statuses}"
    )


# =========================================================
# PRE-AI SURVIVOR VALIDATION
# =========================================================

pre_ai_survivors = routing[
    routing[
        "pre_ai_eligible"
    ]
    ==
    True
]


if (
    pre_ai_survivors[
        "has_raw_case_text"
    ]
    !=
    True
).any():

    raise RuntimeError(
        "Pre-AI eligible case lacks "
        "raw_case_text."
    )


if (
    pre_ai_survivors[
        "image_count"
    ]
    <=
    0
).any():

    raise RuntimeError(
        "Pre-AI eligible case has "
        "no source image."
    )


if (
    pre_ai_survivors[
        "decodable_image_count"
    ]
    .fillna(
        0
    )
    <=
    0
).any():

    raise RuntimeError(
        "Pre-AI eligible case has "
        "zero decodable images."
    )


# =========================================================
# SILVER VALIDATION
#
# SILVER must be substantially stricter than AI_REVIEW.
# =========================================================

silver_check = routing[
    routing[
        "routing_status"
    ]
    ==
    "SILVER_ACCEPT"
]


if (
    silver_check[
        "raw_scope_disease_count"
    ]
    >
    MAX_SCOPE_DISEASES_FOR_SILVER
).any():

    raise RuntimeError(
        "Silver case has multiple "
        "scope disease mentions."
    )


if (
    silver_check[
        "confirmed_target_count"
    ]
    !=
    1
).any():

    raise RuntimeError(
        "Silver case does not have "
        "exactly one deterministic "
        "confirmed target."
    )


if (
    silver_check[
        "high_confidence_target_count"
    ]
    !=
    1
).any():

    raise RuntimeError(
        "Silver case does not have "
        "exactly one high-confidence target."
    )


if (
    silver_check[
        "has_contradictory_assertions"
    ]
    ==
    True
).any():

    raise RuntimeError(
        "Contradictory case entered "
        "SILVER_ACCEPT."
    )


# =========================================================
# SAVE MASTER ROUTING
# =========================================================

routing.to_parquet(
    ROUTING_OUTPUT,
    index=False,
)


# =========================================================
# SPLIT FINAL POPULATIONS
# =========================================================

pre_ai_rejected = (
    routing[
        routing[
            "routing_status"
        ]
        ==
        "PRE_AI_REJECT"
    ]
    .copy()
)


pre_ai_eligible = (
    routing[
        routing[
            "pre_ai_eligible"
        ]
        ==
        True
    ]
    .copy()
)


silver = (
    routing[
        routing[
            "routing_status"
        ]
        ==
        "SILVER_ACCEPT"
    ]
    .copy()
)


ai_review = (
    routing[
        routing[
            "routing_status"
        ]
        ==
        "AI_REVIEW"
    ]
    .copy()
)


pre_ai_rejected.to_parquet(
    REJECT_OUTPUT,
    index=False,
)


pre_ai_eligible.to_parquet(
    ELIGIBLE_OUTPUT,
    index=False,
)


silver.to_parquet(
    SILVER_OUTPUT,
    index=False,
)


ai_review.to_parquet(
    AI_REVIEW_OUTPUT,
    index=False,
)


# =========================================================
# REPORT:
# PRE-AI REJECTION REASONS
# =========================================================

reject_reason_counts = (
    pre_ai_rejected[
        "routing_reason"
    ]
    .value_counts()
    .rename_axis(
        "routing_reason"
    )
    .reset_index(
        name="cases"
    )
)


reject_reason_counts.to_csv(
    REJECT_REASONS_OUTPUT,
    index=False,
)


# =========================================================
# REPORT:
# AI REVIEW REASONS
# =========================================================

ai_reason_counts = (
    ai_review[
        "routing_reason"
    ]
    .value_counts()
    .rename_axis(
        "routing_reason"
    )
    .reset_index(
        name="cases"
    )
)


ai_reason_counts.to_csv(
    AI_REASONS_OUTPUT,
    index=False,
)


# =========================================================
# REPORT:
# MULTIPLE SCOPE DISEASE MENTIONS
#
# IMPORTANT:
#
# These are NOT automatically rejected.
# =========================================================

multiple_scope_mentions = (
    routing[
        routing[
            "has_multiple_scope_mentions"
        ]
        ==
        True
    ][
        [
            "case_id",
            "article_id",
            "routing_status",
            "routing_reason",
            "raw_scope_disease_count",
            "raw_scope_diseases",
            "raw_scope_disease_names",
            "confirmed_targets",
            "confirmed_target_names",
            "high_confidence_targets",
            "high_confidence_target_names",
        ]
    ]
    .copy()
)


multiple_scope_mentions.to_csv(
    MULTI_SCOPE_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# REPORT:
# IMAGE FAILURES
# =========================================================

image_failures = (
    image_audit[
        image_audit[
            "decode_ok"
        ]
        !=
        True
    ]
    .copy()
)


image_failures.to_csv(
    IMAGE_FAILURES_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# IMAGE IDENTITY DIAGNOSTICS
# =========================================================

image_identity_diagnostics = {

    "raw_image_metadata_rows":
        int(
            len(
                images
            )
        ),

    "candidate_image_metadata_rows":
        int(
            len(
                candidate_image_rows
            )
        ),

    "rows_with_duplicate_image_id":
        int(
            duplicate_image_id_rows
        ),

    "rows_in_duplicate_case_file_groups":
        int(
            duplicate_case_file_rows
        ),

    "unique_candidate_case_file_images":
        int(
            len(
                physical_image_table
            )
        ),

    "physical_identity_definition":
        "case_id + file",

    "image_id_treated_as_globally_unique":
        False,
}


with IMAGE_IDENTITY_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        image_identity_diagnostics,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =========================================================
# SUMMARY
# =========================================================

status_counts = (
    routing[
        "routing_status"
    ]
    .value_counts()
)


reason_counts = (
    routing[
        "routing_reason"
    ]
    .value_counts()
)


stage_counts = (
    routing[
        "routing_stage"
    ]
    .value_counts()
)


summary = {

    "rule_version":
        RULE_VERSION,

    "candidate_cases":
        int(
            len(
                routing
            )
        ),

    "structural_survivors":
        int(
            len(
                structural_survivors
            )
        ),

    "pre_ai_rejected":
        int(
            len(
                pre_ai_rejected
            )
        ),

    "pre_ai_eligible":
        int(
            len(
                pre_ai_eligible
            )
        ),

    "silver_accept":
        int(
            len(
                silver
            )
        ),

    "ai_review":
        int(
            len(
                ai_review
            )
        ),

    "cases_with_multiple_scope_mentions":
        int(
            len(
                multiple_scope_mentions
            )
        ),

    "physical_files_scanned":
        int(
            physical_files_scanned
        ),

    "physical_images_checked":
        int(
            len(
                image_audit
            )
        ),

    "physical_images_decodable":
        int(
            image_audit[
                "decode_ok"
            ].sum()
        ),

    "physical_image_failures":
        int(
            len(
                image_failures
            )
        ),

    "rows_with_duplicate_image_id":
        int(
            duplicate_image_id_rows
        ),

    "rows_in_duplicate_case_file_groups":
        int(
            duplicate_case_file_rows
        ),

    "routing_statuses": {

        str(
            key
        ):
            int(
                value
            )

        for key, value
        in status_counts.items()
    },

    "routing_reasons": {

        str(
            key
        ):
            int(
                value
            )

        for key, value
        in reason_counts.items()
    },

    "routing_stages": {

        str(
            key
        ):
            int(
                value
            )

        for key, value
        in stage_counts.items()
    },
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
# FINAL CONSOLE SUMMARY
# =========================================================

print()
print(
    "=" * 72
)

print(
    "STEP 18 FINAL PRE-AI GATE v002"
)

print(
    "=" * 72
)


print(
    "Candidate cases:",
    len(
        routing
    ),
)


print(
    "Structural survivors:",
    len(
        structural_survivors
    ),
)


print()
print(
    "Routing statuses:"
)


print(
    status_counts.to_string()
)


print()
print(
    "Routing reasons:"
)


print(
    reason_counts.to_string()
)


print()
print(
    "Routing stages:"
)


print(
    stage_counts.to_string()
)


print()
print(
    "Pre-AI rejected:",
    len(
        pre_ai_rejected
    ),
)


print(
    "Pre-AI eligible:",
    len(
        pre_ai_eligible
    ),
)


print(
    "  Silver accept:",
    len(
        silver
    ),
)


print(
    "  AI review:",
    len(
        ai_review
    ),
)


print()
print(
    "Cases with multiple scope mentions:",
    len(
        multiple_scope_mentions
    ),
)


print()
print(
    "Physical images checked:",
    len(
        image_audit
    ),
)


print(
    "Decodable physical images:",
    int(
        image_audit[
            "decode_ok"
        ].sum()
    ),
)


print(
    "Physical image failures:",
    len(
        image_failures
    ),
)


print()
print(
    "Rows with duplicated image_id:",
    duplicate_image_id_rows
)


print(
    "Rows in duplicate case_id + file groups:",
    duplicate_case_file_rows
)


print()
print(
    "Saved:"
)


for output_path in [
    ROUTING_OUTPUT,
    REJECT_OUTPUT,
    ELIGIBLE_OUTPUT,
    SILVER_OUTPUT,
    AI_REVIEW_OUTPUT,
    IMAGE_AUDIT_OUTPUT,
    IMAGE_CASE_OUTPUT,
    SUMMARY_OUTPUT,
    REJECT_REASONS_OUTPUT,
    AI_REASONS_OUTPUT,
    MULTI_SCOPE_OUTPUT,
    IMAGE_FAILURES_OUTPUT,
    IMAGE_IDENTITY_OUTPUT,
]:

    print(
        " -",
        output_path
    )


print()
print(
    "FINAL STATUS: PASS"
)

"""
Candidate cases: 21646
Structural survivors: 7639

Routing statuses:
routing_status
PRE_AI_REJECT    14089
AI_REVIEW         6925
SILVER_ACCEPT      632

Routing reasons:
routing_reason
NO_SOURCE_IMAGE                      14007
NO_DETERMINISTIC_CONFIRMED_TARGET     4645
MULTIPLE_SCOPE_DISEASE_MENTIONS       2244
SINGLE_HIGH_CONFIDENCE_TARGET          632
MULTIPLE_HIGH_CONFIDENCE_TARGETS        81
CONTRADICTORY_ASSERTIONS                36
NO_DECODABLE_SOURCE_IMAGE                1

Routing stages:
routing_stage
STRUCTURAL_GATE                 14007
DETERMINISTIC_TRIAGE             4681
RAW_TEXT_TRIAGE                  2244
SILVER_TRIAGE                     632
DETERMINISTIC_DIAGNOSIS_GATE       81
PHYSICAL_IMAGE_GATE                 1

Pre-AI rejected: 14089
Pre-AI eligible: 7557
  Silver accept: 632
  AI review: 6925

Cases with multiple scope mentions: 6710

Physical images checked: 29055
Decodable physical images: 29054
Physical image failures: 1

Rows with duplicated image_id: 5012
Rows in duplicate case_id + file groups: 0

Saved:
 - data\pre_ai\routing.parquet
 - data\pre_ai\pre_ai_rejected.parquet
 - data\pre_ai\pre_ai_eligible.parquet
 - data\pre_ai\silver_accept.parquet
 - data\pre_ai\ai_review_queue.parquet
 - data\pre_ai\physical_image_audit.parquet
 - data\pre_ai\physical_image_case_summary.parquet
 - reports\pre_ai\step18_summary.json
 - reports\pre_ai\reject_reasons.csv
 - reports\pre_ai\ai_review_reasons.csv
 - reports\pre_ai\multiple_scope_mentions.csv
 - reports\pre_ai\image_failures.csv
 - reports\pre_ai\image_identity_diagnostics.json
"""