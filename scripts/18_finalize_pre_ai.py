from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import json

import pandas as pd
import yaml
from PIL import Image, features


# =========================================================
# PATHS
# =========================================================

CONFIG_PATH = Path(
    "configs/pre_ai.yaml"
)

VALIDATION_PATH = Path(
    "reports/candidates/candidate_validation.json"
)

KB_PATH = Path(
    "taxonomy/disease_knowledge_base.parquet"
)

SCOPE_PATH = Path(
    "reports/scope/disease_scope.parquet"
)

CASES_PATH = Path(
    "data/normalized/cases.parquet"
)

IMAGES_PATH = Path(
    "data/normalized/images.parquet"
)

CANDIDATES_PATH = Path(
    "data/candidates/candidates.parquet"
)

MATCHES_PATH = Path(
    "data/candidates/candidate_matches.parquet"
)

LABELS_PATH = Path(
    "data/screening/labels.parquet"
)


OUTPUT_DIR = Path(
    "data/pre_ai"
)

REPORT_DIR = Path(
    "reports/pre_ai"
)


ROUTING_OUTPUT = (
    OUTPUT_DIR / "routing.parquet"
)

PRE_AI_REJECTED_OUTPUT = (
    OUTPUT_DIR / "pre_ai_rejected.parquet"
)

PRE_AI_ELIGIBLE_OUTPUT = (
    OUTPUT_DIR / "pre_ai_eligible.parquet"
)

SILVER_OUTPUT = (
    OUTPUT_DIR / "silver_accept.parquet"
)

AI_QUEUE_OUTPUT = (
    OUTPUT_DIR / "ai_review_queue.parquet"
)

PHYSICAL_IMAGE_AUDIT_OUTPUT = (
    OUTPUT_DIR / "physical_image_audit.parquet"
)

PHYSICAL_IMAGE_CASE_SUMMARY_OUTPUT = (
    OUTPUT_DIR / "physical_image_case_summary.parquet"
)


SUMMARY_OUTPUT = (
    REPORT_DIR / "step18_summary.json"
)

REJECT_REASONS_OUTPUT = (
    REPORT_DIR / "reject_reasons.csv"
)

AI_REVIEW_REASONS_OUTPUT = (
    REPORT_DIR / "ai_review_reasons.csv"
)

MULTIPLE_SCOPE_OUTPUT = (
    REPORT_DIR / "multiple_scope_mentions.csv"
)

IMAGE_FAILURES_OUTPUT = (
    REPORT_DIR / "image_failures.csv"
)

IMAGE_IDENTITY_DIAGNOSTICS_OUTPUT = (
    REPORT_DIR / "image_identity_diagnostics.json"
)

RULED_OUT_REJECTS_OUTPUT = (
    REPORT_DIR / "all_candidate_targets_ruled_out.csv"
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

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    if hasattr(value, "tolist"):
        try:
            result = value.tolist()
            if isinstance(result, list):
                return result
        except Exception:
            pass

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    return [value]


def clean_string_list(values):
    output = []

    for value in values:
        value = normalize_id(value)

        if value is not None:
            output.append(value)

    return sorted(set(output))


def require_columns(df, required, name):
    missing = sorted(
        set(required) - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{name} missing required columns: {missing}"
        )


def bool_value(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def json_dump(path, payload):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# ARGUMENTS
# =========================================================

parser = ArgumentParser()

parser.add_argument(
    "--image-root",
    required=True,
    help=(
        "Root directory containing the extracted "
        "MultiCaRe image files."
    ),
)

args = parser.parse_args()

IMAGE_ROOT = Path(
    args.image_root
).expanduser().resolve()


if not IMAGE_ROOT.exists():
    raise RuntimeError(
        f"Image root does not exist: {IMAGE_ROOT}"
    )

if not IMAGE_ROOT.is_dir():
    raise RuntimeError(
        f"Image root is not a directory: {IMAGE_ROOT}"
    )


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CONFIG
# =========================================================

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    config = yaml.safe_load(f)


if not isinstance(
    config,
    dict,
):
    raise RuntimeError(
        "configs/pre_ai.yaml must contain a YAML mapping."
    )


VERSION = str(
    config.get(
        "version",
        "pre_ai_v003",
    )
)

rules = config.get(
    "rules",
    {},
)

if not isinstance(
    rules,
    dict,
):
    raise RuntimeError(
        "configs/pre_ai.yaml -> rules must be a mapping."
    )


MAX_SCOPE_DISEASES_FOR_SILVER = int(
    rules.get(
        "max_scope_diseases_for_silver",
        1,
    )
)

IGNORE_AMBIGUOUS_ALIASES = bool(
    rules.get(
        "ignore_ambiguous_aliases_for_scope_count",
        True,
    )
)

REJECT_MULTIPLE_HIGH_CONFIDENCE_TARGETS = bool(
    rules.get(
        "reject_multiple_high_confidence_targets",
        True,
    )
)

REJECT_ALL_RULED_OUT = bool(
    rules.get(
        "reject_all_candidate_targets_ruled_out",
        True,
    )
)


# =========================================================
# REQUIRE STEP 16 PASS
# =========================================================

with VALIDATION_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    validation = json.load(f)


if validation.get(
    "status"
) != "PASS":
    raise RuntimeError(
        "Step 16 validation did not PASS."
    )


# =========================================================
# PIL / WEBP SUPPORT
# =========================================================

if not features.check(
    "webp"
):
    raise RuntimeError(
        "Installed Pillow build does not support WebP."
    )


# =========================================================
# LOAD
# =========================================================

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
# REQUIRED COLUMNS
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

require_columns(
    images,
    [
        "case_id",
        "file",
        "image_id",
        "source_image_id",
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
        "disease",
        "alias",
        "field",
    ],
    "candidate_matches",
)

require_columns(
    labels,
    [
        "case_id",
        "concept_id",
        "disease",
        "diagnosis_status",
        "high_confidence_confirmed",
        "contradictory_assertions",
    ],
    "labels",
)


# =========================================================
# NORMALIZE IDS
# =========================================================

for df in [
    kb,
    scope,
    candidates,
    matches,
    labels,
]:
    if "concept_id" in df.columns:
        df[
            "concept_id"
        ] = df[
            "concept_id"
        ].map(
            normalize_id
        )


for df in [
    cases,
    images,
    candidates,
    matches,
    labels,
]:
    if "case_id" in df.columns:
        df[
            "case_id"
        ] = df[
            "case_id"
        ].map(
            normalize_id
        )


for df in [
    cases,
    candidates,
]:
    if "article_id" in df.columns:
        df[
            "article_id"
        ] = df[
            "article_id"
        ].map(
            normalize_id
        )


images[
    "file"
] = images[
    "file"
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


# =========================================================
# BASIC SOURCE INTEGRITY
# =========================================================

candidate_case_ids = set(
    candidates[
        "case_id"
    ]
    .dropna()
    .tolist()
)


if not candidate_case_ids:
    raise RuntimeError(
        "No candidate cases found."
    )


if candidates[
    "case_id"
].isna().any():
    raise RuntimeError(
        "candidates.parquet contains null case_id."
    )


case_duplicates = cases[
    "case_id"
].duplicated(
    keep=False
)


if case_duplicates.any():
    duplicate_cases = (
        cases.loc[
            case_duplicates,
            "case_id",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "cases.parquet has duplicate case_id values: "
        f"{duplicate_cases[:20]}"
    )


case_lookup = (
    cases.set_index(
        "case_id"
    )
)


missing_candidate_cases = (
    candidate_case_ids
    -
    set(
        case_lookup.index
    )
)


if missing_candidate_cases:
    raise RuntimeError(
        "Candidate cases missing from cases.parquet: "
        f"{sorted(missing_candidate_cases)[:20]}"
    )


# =========================================================
# DISEASE NAME LOOKUP
# =========================================================

disease_lookup = {}


for df in [
    kb,
    scope,
    candidates,
    labels,
]:
    if (
        "concept_id" in df.columns
        and
        "disease" in df.columns
    ):
        for row in df[
            [
                "concept_id",
                "disease",
            ]
        ].dropna(
            subset=[
                "concept_id",
            ]
        ).itertuples(
            index=False
        ):
            if row.concept_id not in disease_lookup:
                disease_lookup[
                    row.concept_id
                ] = str(
                    row.disease
                )


def disease_names(concept_ids):
    return [
        disease_lookup.get(
            concept_id,
            concept_id,
        )
        for concept_id
        in concept_ids
    ]


# =========================================================
# ONTOLOGY GRAPH
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

    if concept_id is None:
        continue

    parents = clean_string_list(
        as_list(
            row.parents
        )
    )

    parent_map[
        concept_id
    ] = set(
        parents
    )


@lru_cache(
    maxsize=None
)
def ancestors(concept_id):
    output = set()

    for parent in parent_map.get(
        concept_id,
        set(),
    ):
        if parent in output:
            continue

        output.add(
            parent
        )

        output.update(
            ancestors(
                parent
            )
        )

    return frozenset(
        output
    )


def most_specific(concept_ids):
    ids = set(
        clean_string_list(
            concept_ids
        )
    )

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
        ids - ancestor_ids
    )


# =========================================================
# CANDIDATE MAP
# =========================================================

candidate_case_concepts = (
    candidates[
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


candidate_map = (
    candidate_case_concepts
    .groupby(
        "case_id"
    )[
        "concept_id"
    ]
    .apply(
        lambda values:
            clean_string_list(
                values.tolist()
            )
    )
    .to_dict()
)


# =========================================================
# AMBIGUOUS ALIASES
#
# A raw alias matching >1 concept is not used when counting
# scope mentions for Silver triage.
# =========================================================

alias_concept_counts = (
    matches[
        [
            "alias",
            "concept_id",
        ]
    ]
    .dropna()
    .assign(
        alias_normalized=lambda x:
            x[
                "alias"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
    )
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
# RAW-TEXT SCOPE MENTION MAP
# =========================================================

scope_concept_ids = set(
    scope[
        "concept_id"
    ]
    .dropna()
    .tolist()
)


raw_scope_matches = matches[
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
            "case_id"
        ].isin(
            candidate_case_ids
        )
    )
    &
    (
        matches[
            "concept_id"
        ].isin(
            scope_concept_ids
        )
    )
].copy()


raw_scope_matches[
    "alias_normalized"
] = (
    raw_scope_matches[
        "alias"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


if IGNORE_AMBIGUOUS_ALIASES:
    raw_scope_matches = (
        raw_scope_matches[
            ~raw_scope_matches[
                "alias_normalized"
            ].isin(
                ambiguous_aliases
            )
        ]
        .copy()
    )


raw_scope_map = (
    raw_scope_matches
    .groupby(
        "case_id"
    )[
        "concept_id"
    ]
    .apply(
        lambda values:
            most_specific(
                values.tolist()
            )
    )
    .to_dict()
)


# =========================================================
# STEP 17 LABEL MAP
# =========================================================

if labels.duplicated(
    [
        "case_id",
        "concept_id",
    ]
).any():
    raise RuntimeError(
        "labels.parquet contains duplicate "
        "case_id + concept_id rows."
    )


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
# PHYSICAL IMAGE IDENTITY DIAGNOSTICS
# =========================================================

duplicated_image_id_rows = int(
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


# =========================================================
# CONSOLIDATE IMAGE METADATA TO PHYSICAL FINAL-IMAGE GRAIN
#
# Physical identity:
#     case_id + file
#
# image_id and source_image_id are provenance only.
# =========================================================

candidate_images = (
    images[
        images[
            "case_id"
        ].isin(
            candidate_case_ids
        )
    ]
    .copy()
)


candidate_images = (
    candidate_images[
        candidate_images[
            "file"
        ].notna()
    ]
    .copy()
)


physical_metadata_rows = []


for (
    case_id,
    file_name,
), group in candidate_images.groupby(
    [
        "case_id",
        "file",
    ],
    sort=False,
    dropna=False,
):
    image_ids = clean_string_list(
        group[
            "image_id"
        ].tolist()
    )

    source_image_ids = clean_string_list(
        group[
            "source_image_id"
        ].tolist()
    )

    physical_metadata_rows.append({
        "case_id":
            case_id,

        "file":
            file_name,

        "metadata_row_count":
            int(
                len(
                    group
                )
            ),

        "image_ids":
            image_ids,

        "source_image_ids":
            source_image_ids,
    })


physical_metadata = pd.DataFrame(
    physical_metadata_rows
)


# =========================================================
# INDEX PHYSICAL IMAGE FILES BY BASENAME
# =========================================================

print(
    "Indexing physical image files..."
)

basename_index = defaultdict(
    list
)


for path in IMAGE_ROOT.rglob(
    "*"
):
    if path.is_file():
        basename_index[
            path.name
        ].append(
            path
        )


# =========================================================
# VERIFY / DECODE EVERY PHYSICAL IMAGE
# =========================================================

print(
    "Checking physical images..."
)

physical_audit_rows = []


for row in physical_metadata.itertuples(
    index=False
):
    matches_for_file = basename_index.get(
        row.file,
        [],
    )

    physical_exists = (
        len(
            matches_for_file
        )
        >
        0
    )

    unique_path_match = (
        len(
            matches_for_file
        )
        ==
        1
    )

    resolved_path = None
    decode_ok = False
    decode_error = None
    width = None
    height = None
    image_format = None

    if not physical_exists:
        decode_error = (
            "FILE_NOT_FOUND"
        )

    elif not unique_path_match:
        decode_error = (
            "AMBIGUOUS_BASENAME_MULTIPLE_FILES"
        )

    else:
        path = matches_for_file[
            0
        ]

        resolved_path = str(
            path
        )

        try:
            with Image.open(
                path
            ) as image:
                image.verify()

            with Image.open(
                path
            ) as image:
                image.load()

                width = int(
                    image.width
                )

                height = int(
                    image.height
                )

                image_format = (
                    image.format
                )

            decode_ok = True

        except Exception as exc:
            decode_error = (
                f"{type(exc).__name__}: {exc}"
            )

    physical_audit_rows.append({
        "case_id":
            row.case_id,

        "file":
            row.file,

        "metadata_row_count":
            int(
                row.metadata_row_count
            ),

        "image_ids":
            row.image_ids,

        "source_image_ids":
            row.source_image_ids,

        "path_match_count":
            int(
                len(
                    matches_for_file
                )
            ),

        "physical_exists":
            bool(
                physical_exists
            ),

        "unique_path_match":
            bool(
                unique_path_match
            ),

        "resolved_path":
            resolved_path,

        "decode_ok":
            bool(
                decode_ok
            ),

        "decode_error":
            decode_error,

        "width":
            width,

        "height":
            height,

        "image_format":
            image_format,
    })


physical_image_audit = pd.DataFrame(
    physical_audit_rows
)


physical_image_audit.to_parquet(
    PHYSICAL_IMAGE_AUDIT_OUTPUT,
    index=False,
)


# =========================================================
# PHYSICAL IMAGE CASE SUMMARY
# =========================================================

physical_case_rows = []


for case_id in sorted(
    candidate_case_ids
):
    group = physical_image_audit[
        physical_image_audit[
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

    decodable_files = clean_string_list(
        decodable[
            "file"
        ].tolist()
    )

    decodable_image_ids = clean_string_list(
        [
            value
            for values
            in decodable[
                "image_ids"
            ].tolist()
            for value
            in as_list(
                values
            )
        ]
    )

    decodable_source_image_ids = clean_string_list(
        [
            value
            for values
            in decodable[
                "source_image_ids"
            ].tolist()
            for value
            in as_list(
                values
            )
        ]
    )

    physical_case_rows.append({
        "case_id":
            case_id,

        "source_image_count":
            int(
                len(
                    group
                )
            ),

        "decodable_image_count":
            int(
                len(
                    decodable
                )
            ),

        "decodable_files":
            decodable_files,

        "decodable_image_ids":
            decodable_image_ids,

        "decodable_source_image_ids":
            decodable_source_image_ids,
    })


physical_case_summary = pd.DataFrame(
    physical_case_rows
)


physical_image_case_summary = (
    physical_case_summary.set_index(
        "case_id"
    )
)


physical_case_summary.to_parquet(
    PHYSICAL_IMAGE_CASE_SUMMARY_OUTPUT,
    index=False,
)


# =========================================================
# ROUTING ALL CANDIDATE CASES
# =========================================================

print(
    "Routing candidate cases..."
)

route_rows = []


for case_id in sorted(
    candidate_case_ids
):
    case = case_lookup.loc[
        case_id
    ]

    article_id = normalize_id(
        case[
            "article_id"
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

    image_summary = (
        physical_image_case_summary.loc[
            case_id
        ]
    )

    source_image_count = int(
        image_summary[
            "source_image_count"
        ]
    )

    decodable_image_count = int(
        image_summary[
            "decodable_image_count"
        ]
    )

    decodable_files = as_list(
        image_summary[
            "decodable_files"
        ]
    )

    decodable_image_ids = as_list(
        image_summary[
            "decodable_image_ids"
        ]
    )

    decodable_source_image_ids = as_list(
        image_summary[
            "decodable_source_image_ids"
        ]
    )

    candidate_ids = (
        candidate_map.get(
            case_id,
            [],
        )
    )

    candidate_names = disease_names(
        candidate_ids
    )

    raw_scope_ids = (
        raw_scope_map.get(
            case_id,
            [],
        )
    )

    raw_scope_names = disease_names(
        raw_scope_ids
    )

    has_multiple_scope_mentions = (
        len(
            raw_scope_ids
        )
        >
        MAX_SCOPE_DISEASES_FOR_SILVER
    )

    # Defaults for structural rejects.
    confirmed_ids_raw = []
    confirmed_ids = []

    high_conf_ids_raw = []
    high_conf_ids = []

    contradiction = False

    candidate_label_count = 0
    all_candidates_have_labels = False

    ruled_out_candidate_ids = []
    ruled_out_candidate_names = []

    all_candidate_targets_ruled_out = False

    silver_target = None

    status = None
    reason = None
    stage = None

    # -----------------------------------------------------
    # STRUCTURAL HARD GATES
    # -----------------------------------------------------

    if (
        rules.get(
            "require_raw_case_text",
            True,
        )
        and
        not raw_case_text
    ):
        status = (
            "PRE_AI_REJECT"
        )

        reason = (
            "NO_RAW_CASE_TEXT"
        )

        stage = (
            "STRUCTURAL_GATE"
        )

    elif (
        rules.get(
            "require_source_image",
            True,
        )
        and
        source_image_count
        <=
        0
    ):
        status = (
            "PRE_AI_REJECT"
        )

        reason = (
            "NO_SOURCE_IMAGE"
        )

        stage = (
            "STRUCTURAL_GATE"
        )

    elif (
        rules.get(
            "require_decodable_image",
            True,
        )
        and
        decodable_image_count
        <=
        0
    ):
        status = (
            "PRE_AI_REJECT"
        )

        reason = (
            "NO_DECODABLE_SOURCE_IMAGE"
        )

        stage = (
            "PHYSICAL_IMAGE_GATE"
        )

    else:
        group = label_groups.get(
            case_id
        )

        if (
            group is None
            or
            group.empty
        ):
            raise RuntimeError(
                "Structurally surviving case has no "
                f"Step 17 labels: {case_id}"
            )

        # -------------------------------------------------
        # CONFIRMED TARGETS
        # -------------------------------------------------

        confirmed = group[
            group[
                "diagnosis_status"
            ]
            ==
            "confirmed"
        ]

        confirmed_ids_raw = clean_string_list(
            confirmed[
                "concept_id"
            ].tolist()
        )

        confirmed_ids = most_specific(
            confirmed_ids_raw
        )

        # -------------------------------------------------
        # HIGH-CONFIDENCE TARGETS
        # -------------------------------------------------

        high_conf = group[
            group[
                "high_confidence_confirmed"
            ].map(
                bool_value
            )
            ==
            True
        ]

        high_conf_ids_raw = clean_string_list(
            high_conf[
                "concept_id"
            ].tolist()
        )

        high_conf_ids = most_specific(
            high_conf_ids_raw
        )

        # -------------------------------------------------
        # CONTRADICTIONS
        # -------------------------------------------------

        contradiction = bool(
            group[
                "contradictory_assertions"
            ]
            .map(
                bool_value
            )
            .any()
        )

        # -------------------------------------------------
        # STRICT ALL-CANDIDATE RULED-OUT GATE
        #
        # Every candidate target must have a Step 17 label,
        # and every one must be exactly ruled_out.
        # -------------------------------------------------

        candidate_label_rows = (
            group[
                group[
                    "concept_id"
                ].isin(
                    candidate_ids
                )
            ]
            .copy()
        )

        candidate_label_count = int(
            len(
                candidate_label_rows
            )
        )

        candidate_labeled_ids = set(
            candidate_label_rows[
                "concept_id"
            ]
            .dropna()
            .tolist()
        )

        all_candidates_have_labels = (
            bool(
                candidate_ids
            )
            and
            candidate_labeled_ids
            ==
            set(
                candidate_ids
            )
        )

        ruled_out_candidate_ids = clean_string_list(
            candidate_label_rows.loc[
                candidate_label_rows[
                    "diagnosis_status"
                ]
                ==
                "ruled_out",
                "concept_id",
            ].tolist()
        )

        ruled_out_candidate_names = disease_names(
            ruled_out_candidate_ids
        )

        all_candidate_targets_ruled_out = (
            REJECT_ALL_RULED_OUT
            and
            bool(
                candidate_ids
            )
            and
            all_candidates_have_labels
            and
            not candidate_label_rows.empty
            and
            candidate_label_rows[
                "diagnosis_status"
            ].notna().all()
            and
            candidate_label_rows[
                "diagnosis_status"
            ].eq(
                "ruled_out"
            ).all()
            and
            len(
                confirmed_ids
            )
            ==
            0
            and
            not contradiction
        )

        # -------------------------------------------------
        # ROUTING PRIORITY
        # -------------------------------------------------

        if all_candidate_targets_ruled_out:
            status = (
                "PRE_AI_REJECT"
            )

            reason = (
                "ALL_CANDIDATE_TARGETS_RULED_OUT"
            )

            stage = (
                "DETERMINISTIC_DIAGNOSIS_GATE"
            )

        elif (
            REJECT_MULTIPLE_HIGH_CONFIDENCE_TARGETS
            and
            len(
                high_conf_ids
            )
            >
            1
        ):
            status = (
                "PRE_AI_REJECT"
            )

            reason = (
                "MULTIPLE_HIGH_CONFIDENCE_TARGETS"
            )

            stage = (
                "DETERMINISTIC_DIAGNOSIS_GATE"
            )

        elif contradiction:
            status = (
                "AI_REVIEW"
            )

            reason = (
                "CONTRADICTORY_ASSERTIONS"
            )

            stage = (
                "DETERMINISTIC_TRIAGE"
            )

        elif (
            len(
                confirmed_ids
            )
            >
            1
        ):
            status = (
                "AI_REVIEW"
            )

            reason = (
                "MULTIPLE_DETERMINISTIC_TARGETS_UNCERTAIN"
            )

            stage = (
                "DETERMINISTIC_TRIAGE"
            )

        elif has_multiple_scope_mentions:
            status = (
                "AI_REVIEW"
            )

            reason = (
                "MULTIPLE_SCOPE_DISEASE_MENTIONS"
            )

            stage = (
                "RAW_TEXT_TRIAGE"
            )

        elif (
            len(
                confirmed_ids
            )
            ==
            0
        ):
            status = (
                "AI_REVIEW"
            )

            reason = (
                "NO_DETERMINISTIC_CONFIRMED_TARGET"
            )

            stage = (
                "DETERMINISTIC_TRIAGE"
            )

        else:
            # Exactly one deterministic confirmed target.
            target = (
                confirmed_ids[
                    0
                ]
            )

            if (
                len(
                    high_conf_ids
                )
                ==
                1
                and
                high_conf_ids[
                    0
                ]
                ==
                target
            ):
                status = (
                    "SILVER_ACCEPT"
                )

                reason = (
                    "SINGLE_HIGH_CONFIDENCE_TARGET"
                )

                stage = (
                    "SILVER_TRIAGE"
                )

                silver_target = (
                    target
                )

            else:
                status = (
                    "AI_REVIEW"
                )

                reason = (
                    "WEAK_OR_AMBIGUOUS_CONFIRMATION"
                )

                stage = (
                    "DETERMINISTIC_TRIAGE"
                )

    pre_ai_eligible = (
        status
        in {
            "AI_REVIEW",
            "SILVER_ACCEPT",
        }
    )

    route_rows.append({
        "case_id":
            case_id,

        "article_id":
            article_id,

        "source_image_count":
            source_image_count,

        "decodable_image_count":
            decodable_image_count,

        "decodable_files":
            clean_string_list(
                decodable_files
            ),

        "decodable_image_ids":
            clean_string_list(
                decodable_image_ids
            ),

        "decodable_source_image_ids":
            clean_string_list(
                decodable_source_image_ids
            ),

        "candidate_diseases":
            candidate_ids,

        "candidate_disease_names":
            candidate_names,

        "candidate_disease_count":
            int(
                len(
                    candidate_ids
                )
            ),

        "raw_scope_diseases":
            raw_scope_ids,

        "raw_scope_disease_names":
            raw_scope_names,

        "raw_scope_disease_count":
            int(
                len(
                    raw_scope_ids
                )
            ),

        "has_multiple_scope_mentions":
            bool(
                has_multiple_scope_mentions
            ),

        "confirmed_targets_raw":
            confirmed_ids_raw,

        "confirmed_targets":
            confirmed_ids,

        "confirmed_target_count":
            int(
                len(
                    confirmed_ids
                )
            ),

        "high_confidence_targets_raw":
            high_conf_ids_raw,

        "high_confidence_targets":
            high_conf_ids,

        "high_confidence_target_count":
            int(
                len(
                    high_conf_ids
                )
            ),

        "has_contradictory_assertions":
            bool(
                contradiction
            ),

        "candidate_label_count":
            int(
                candidate_label_count
            ),

        "all_candidates_have_labels":
            bool(
                all_candidates_have_labels
            ),

        "ruled_out_candidate_ids":
            ruled_out_candidate_ids,

        "ruled_out_candidate_names":
            ruled_out_candidate_names,

        "ruled_out_candidate_count":
            int(
                len(
                    ruled_out_candidate_ids
                )
            ),

        "all_candidate_targets_ruled_out":
            bool(
                all_candidate_targets_ruled_out
            ),

        "silver_target_concept_id":
            silver_target,

        "silver_target_disease":
            (
                disease_lookup.get(
                    silver_target
                )
                if silver_target
                else None
            ),

        "routing_status":
            status,

        "routing_reason":
            reason,

        "routing_stage":
            stage,

        "pre_ai_eligible":
            bool(
                pre_ai_eligible
            ),

        "pre_ai_version":
            VERSION,
    })


routing = pd.DataFrame(
    route_rows
)


# =========================================================
# ROUTING INTEGRITY
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
    candidate_case_ids
):
    raise RuntimeError(
        "Routing must contain exactly one row "
        "per candidate case."
    )


if set(
    routing[
        "case_id"
    ]
) != candidate_case_ids:
    raise RuntimeError(
        "Routing case set differs from candidates."
    )


valid_statuses = {
    "PRE_AI_REJECT",
    "AI_REVIEW",
    "SILVER_ACCEPT",
}


if not set(
    routing[
        "routing_status"
    ].dropna().unique()
).issubset(
    valid_statuses
):
    raise RuntimeError(
        "Unexpected routing_status generated."
    )


# =========================================================
# RULED-OUT HARD-GATE VALIDATION
# =========================================================

ruled_out_rejects = routing[
    routing[
        "routing_reason"
    ]
    ==
    "ALL_CANDIDATE_TARGETS_RULED_OUT"
].copy()


if not ruled_out_rejects.empty:
    if not (
        ruled_out_rejects[
            "all_candidate_targets_ruled_out"
        ]
        ==
        True
    ).all():
        raise RuntimeError(
            "Invalid ALL_CANDIDATE_TARGETS_RULED_OUT reject."
        )

    if (
        ruled_out_rejects[
            "confirmed_target_count"
        ]
        !=
        0
    ).any():
        raise RuntimeError(
            "Ruled-out hard reject still has "
            "a confirmed target."
        )

    if (
        ruled_out_rejects[
            "has_contradictory_assertions"
        ]
        ==
        True
    ).any():
        raise RuntimeError(
            "Contradictory case was hard-rejected "
            "by ruled-out gate."
        )

    if not (
        ruled_out_rejects[
            "all_candidates_have_labels"
        ]
        ==
        True
    ).all():
        raise RuntimeError(
            "Ruled-out hard reject has unlabeled candidates."
        )


# =========================================================
# SPLIT ROUTING OUTPUTS
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


ai_queue = (
    routing[
        routing[
            "routing_status"
        ]
        ==
        "AI_REVIEW"
    ]
    .copy()
)


if len(
    pre_ai_eligible
) != (
    len(
        silver
    )
    +
    len(
        ai_queue
    )
):
    raise RuntimeError(
        "pre_ai_eligible != silver + ai_review."
    )


# =========================================================
# SAVE PARQUET OUTPUTS
# =========================================================

routing.to_parquet(
    ROUTING_OUTPUT,
    index=False,
)

pre_ai_rejected.to_parquet(
    PRE_AI_REJECTED_OUTPUT,
    index=False,
)

pre_ai_eligible.to_parquet(
    PRE_AI_ELIGIBLE_OUTPUT,
    index=False,
)

silver.to_parquet(
    SILVER_OUTPUT,
    index=False,
)

ai_queue.to_parquet(
    AI_QUEUE_OUTPUT,
    index=False,
)


# =========================================================
# REPORTS
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
        name="case_count"
    )
)


reject_reason_counts.to_csv(
    REJECT_REASONS_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


ai_reason_counts = (
    ai_queue[
        "routing_reason"
    ]
    .value_counts()
    .rename_axis(
        "routing_reason"
    )
    .reset_index(
        name="case_count"
    )
)


ai_reason_counts.to_csv(
    AI_REVIEW_REASONS_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


multiple_scope_report = (
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
            "raw_scope_disease_count",
            "raw_scope_diseases",
            "raw_scope_disease_names",
            "routing_status",
            "routing_reason",
        ]
    ]
    .copy()
)


multiple_scope_report.to_csv(
    MULTIPLE_SCOPE_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


image_failures = (
    physical_image_audit[
        physical_image_audit[
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


ruled_out_reject_report = (
    ruled_out_rejects[
        [
            "case_id",
            "article_id",
            "candidate_disease_count",
            "candidate_diseases",
            "candidate_disease_names",
            "ruled_out_candidate_ids",
            "ruled_out_candidate_names",
            "ruled_out_candidate_count",
        ]
    ]
    .copy()
)


ruled_out_reject_report.to_csv(
    RULED_OUT_REJECTS_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# IMAGE IDENTITY DIAGNOSTICS
# =========================================================

image_identity_diagnostics = {
    "physical_identity":
        "case_id + file",

    "image_id_role":
        "provenance_only_not_unique_primary_key",

    "source_image_id_role":
        "provenance_only_not_unique_primary_key",

    "normalized_image_rows":
        int(
            len(
                images
            )
        ),

    "candidate_image_rows":
        int(
            len(
                candidate_images
            )
        ),

    "physical_case_file_rows_checked":
        int(
            len(
                physical_image_audit
            )
        ),

    "decodable_physical_images":
        int(
            physical_image_audit[
                "decode_ok"
            ].sum()
        )
        if not physical_image_audit.empty
        else 0,

    "physical_image_failures":
        int(
            len(
                image_failures
            )
        ),

    "rows_with_duplicated_image_id":
        int(
            duplicated_image_id_rows
        ),

    "rows_in_duplicate_case_id_file_groups":
        int(
            duplicate_case_file_rows
        ),
}


json_dump(
    IMAGE_IDENTITY_DIAGNOSTICS_OUTPUT,
    image_identity_diagnostics,
)


# =========================================================
# SUMMARY
# =========================================================

routing_status_counts = (
    routing[
        "routing_status"
    ]
    .value_counts()
)


routing_reason_counts = (
    routing[
        "routing_reason"
    ]
    .value_counts()
)


routing_stage_counts = (
    routing[
        "routing_stage"
    ]
    .value_counts()
)


structural_survivors = int(
    (
        routing[
            "source_image_count"
        ]
        >
        0
    ).sum()
)


summary = {
    "version":
        VERSION,

    "candidate_cases":
        int(
            len(
                routing
            )
        ),

    "structural_survivors_with_source_image":
        structural_survivors,

    "routing_statuses": {
        str(key):
            int(
                value
            )

        for key, value
        in routing_status_counts.items()
    },

    "routing_reasons": {
        str(key):
            int(
                value
            )

        for key, value
        in routing_reason_counts.items()
    },

    "routing_stages": {
        str(key):
            int(
                value
            )

        for key, value
        in routing_stage_counts.items()
    },

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
                ai_queue
            )
        ),

    "all_candidate_targets_ruled_out_rejects":
        int(
            len(
                ruled_out_rejects
            )
        ),

    "cases_with_multiple_scope_mentions":
        int(
            routing[
                "has_multiple_scope_mentions"
            ].sum()
        ),

    "physical_images_checked":
        int(
            len(
                physical_image_audit
            )
        ),

    "decodable_physical_images":
        int(
            physical_image_audit[
                "decode_ok"
            ].sum()
        )
        if not physical_image_audit.empty
        else 0,

    "physical_image_failures":
        int(
            len(
                image_failures
            )
        ),

    "rows_with_duplicated_image_id":
        int(
            duplicated_image_id_rows
        ),

    "rows_in_duplicate_case_id_file_groups":
        int(
            duplicate_case_file_rows
        ),
}


json_dump(
    SUMMARY_OUTPUT,
    summary,
)


# =========================================================
# PRINT
# =========================================================

print()
print(
    "=" * 76
)
print(
    "STEP 18 FINAL PRE-AI GATE v003"
)
print(
    "=" * 76
)

print(
    "Candidate cases:",
    len(
        routing
    ),
)

print(
    "Structural survivors with source images:",
    structural_survivors,
)

print()
print(
    "Routing statuses:"
)
print(
    routing_status_counts.to_string()
)

print()
print(
    "Routing reasons:"
)
print(
    routing_reason_counts.to_string()
)

print()
print(
    "Routing stages:"
)
print(
    routing_stage_counts.to_string()
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
        ai_queue
    ),
)

print()
print(
    "ALL_CANDIDATE_TARGETS_RULED_OUT rejects:",
    len(
        ruled_out_rejects
    ),
)

print(
    "Cases with multiple scope mentions:",
    int(
        routing[
            "has_multiple_scope_mentions"
        ].sum()
    ),
)

print()
print(
    "Physical images checked:",
    len(
        physical_image_audit
    ),
)

print(
    "Decodable physical images:",
    int(
        physical_image_audit[
            "decode_ok"
        ].sum()
    )
    if not physical_image_audit.empty
    else 0,
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
    duplicated_image_id_rows,
)

print(
    "Rows in duplicate case_id + file groups:",
    duplicate_case_file_rows,
)

print()
print(
    "Saved:"
)

for path in [
    ROUTING_OUTPUT,
    PRE_AI_REJECTED_OUTPUT,
    PRE_AI_ELIGIBLE_OUTPUT,
    SILVER_OUTPUT,
    AI_QUEUE_OUTPUT,
    PHYSICAL_IMAGE_AUDIT_OUTPUT,
    PHYSICAL_IMAGE_CASE_SUMMARY_OUTPUT,
    SUMMARY_OUTPUT,
    REJECT_REASONS_OUTPUT,
    AI_REVIEW_REASONS_OUTPUT,
    MULTIPLE_SCOPE_OUTPUT,
    IMAGE_FAILURES_OUTPUT,
    IMAGE_IDENTITY_DIAGNOSTICS_OUTPUT,
    RULED_OUT_REJECTS_OUTPUT,
]:
    print(
        " -",
        path,
    )

print()
print(
    "FINAL STATUS: PASS"
)
