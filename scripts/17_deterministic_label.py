from pathlib import Path
import json
import re
import sys

import pandas as pd
import yaml


# =========================================================
# PATHS
# =========================================================

CONFIG_PATH = Path(
    "configs/labeling.yaml"
)

VALIDATION_PATH = Path(
    "reports/candidates/"
    "candidate_validation.json"
)

ALIAS_DIAGNOSTICS_PATH = Path(
    "reports/candidates/"
    "alias_diagnostics.csv"
)

CANDIDATES_PATH = Path(
    "data/candidates/"
    "candidates.parquet"
)

MATCHES_PATH = Path(
    "data/candidates/"
    "candidate_matches.parquet"
)

CASES_PATH = Path(
    "data/normalized/"
    "cases.parquet"
)

IMAGES_PATH = Path(
    "data/normalized/"
    "images.parquet"
)

OUTPUT = Path(
    "data/screening"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


HARD_GATE_OUTPUT = (
    OUTPUT /
    "hard_gate_audit.parquet"
)

HARD_EXCLUDED_OUTPUT = (
    OUTPUT /
    "hard_excluded.parquet"
)

ELIGIBLE_CASES_OUTPUT = (
    OUTPUT /
    "eligible_cases.parquet"
)

MENTIONS_OUTPUT = (
    OUTPUT /
    "mention_assertions.parquet"
)

LABELS_OUTPUT = (
    OUTPUT /
    "labels.parquet"
)


# =========================================================
# CONFIG
# =========================================================

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(f)


RULE_VERSION = config[
    "version"
]


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
        "Step 16 validation did not PASS. "
        "Do not run deterministic labeling."
    )


print(
    "Step 16 validation: PASS"
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


def evidence_span(
    text,
    mention_start,
    mention_end,
):

    """
    Return a conservative sentence-ish span
    surrounding a disease mention.

    This is not used as medical ground truth.
    It only gives a local literal evidence span.
    """

    if not text:
        return (
            mention_start,
            mention_end,
        )


    left = 0

    for separator in [
        ".",
        "!",
        "?",
        "\n",
    ]:

        position = text.rfind(
            separator,
            0,
            mention_start,
        )

        if position >= 0:

            left = max(
                left,
                position + 1,
            )


    right_candidates = []

    for separator in [
        ".",
        "!",
        "?",
        "\n",
    ]:

        position = text.find(
            separator,
            mention_end,
        )

        if position >= 0:

            right_candidates.append(
                position + 1
            )


    right = (
        min(
            right_candidates
        )
        if right_candidates
        else len(text)
    )


    # Remove leading/trailing whitespace
    # while preserving exact source offsets.
    while (
        left < right
        and text[left].isspace()
    ):

        left += 1


    while (
        right > left
        and text[
            right - 1
        ].isspace()
    ):

        right -= 1


    return (
        left,
        right,
    )


def local_clause(
    sentence,
    alias_start,
    alias_end,
):

    """
    Restrict context around the mention so
    language describing a neighbouring disease
    is less likely to leak into the assertion.
    """

    before = (
        sentence[
            :alias_start
        ]
        .lower()
    )

    after = (
        sentence[
            alias_end:
        ]
        .lower()
    )


    before = re.split(
        r"[;:]|\b(?:but|whereas|however)\b",
        before,
    )[-1][-120:]


    after = re.split(
        r"[;:]|\b(?:but|whereas|however)\b",
        after,
    )[0][:120]


    return (
        before,
        after,
    )


def classify_mention(
    sentence,
    alias_start,
    alias_end,
):

    """
    Conservative rule-based assertion classifier.

    IMPORTANT:
    failure to match a rule means UNCLEAR,
    not negative.
    """

    before, after = local_clause(
        sentence,
        alias_start,
        alias_end,
    )


    local = (
        before
        + " <disease> "
        + after
    )


    # -----------------------------------------------------
    # HISTORY / OTHER EXPERIENCER
    # -----------------------------------------------------

    if re.search(
        r"\b(?:family history|"
        r"mother|father|sister|brother)"
        r"\b",
        before,
    ):

        return (
            "historical",
            "FAMILY_HISTORY",
        )


    if re.search(
        r"\b(?:history of|"
        r"previous|prior|past|"
        r"previously treated for|"
        r"resolved)\b",
        before,
    ):

        return (
            "historical",
            "PAST_HISTORY",
        )


    # -----------------------------------------------------
    # EXPLICIT NEGATION / EXCLUSION
    # -----------------------------------------------------

    if (
        re.search(
            r"\b(?:no evidence of|"
            r"negative for|"
            r"ruled out|"
            r"excluded|"
            r"absence of|"
            r"without evidence of)"
            r"\s*(?:\w+\s+){0,4}$",
            before,
        )
        or
        re.match(
            r"\s*(?:was |is |"
            r"has been |had been )?"
            r"(?:ruled out|"
            r"excluded|"
            r"negative|"
            r"not confirmed)\b",
            after,
        )
    ):

        return (
            "ruled_out",
            "EXPLICIT_NEGATION",
        )


    # -----------------------------------------------------
    # EXPLICIT CONFIRMATION
    #
    # Keep this deliberately narrow.
    # -----------------------------------------------------

    if (
        re.search(
            r"\b(?:confirmed|"
            r"diagnosed with|"
            r"diagnosis of|"
            r"diagnosis was|"
            r"positive for)"
            r"\s+(?:\w+\s+){0,4}$",
            before,
        )
        or
        re.match(
            r"\s*(?:\w+\s+){0,3}"
            r"(?:was |is |"
            r"has been |had been )?"
            r"(?:confirmed|"
            r"diagnosed|"
            r"established)\b",
            after,
        )
    ):

        return (
            "confirmed",
            "EXPLICIT_CONFIRMATION",
        )


    # -----------------------------------------------------
    # PROBABLE
    # -----------------------------------------------------

    if re.search(
        r"\b(?:probable|"
        r"likely|"
        r"presumed|"
        r"consistent with(?: a diagnosis of)?)"
        r"\b",
        local,
    ):

        return (
            "probable",
            "PROBABLE_LANGUAGE",
        )


    # -----------------------------------------------------
    # SUSPECTED
    # -----------------------------------------------------

    if re.search(
        r"\b(?:suspect\w*|"
        r"possible|"
        r"potential|"
        r"may represent|"
        r"might represent|"
        r"could represent|"
        r"cannot exclude)\b",
        local,
    ):

        return (
            "suspected",
            "SUSPECTED_LANGUAGE",
        )


    # -----------------------------------------------------
    # DIFFERENTIAL
    # -----------------------------------------------------

    if re.search(
        r"\b(?:differential"
        r"(?: diagnosis)?|"
        r"versus|"
        r"vs\.?|"
        r"considered|"
        r"to rule out|"
        r"r/o)\b",
        local,
    ):

        return (
            "differential",
            "DIFFERENTIAL_LANGUAGE",
        )


    return (
        "unclear",
        "NO_EXPLICIT_ASSERTION",
    )


# =========================================================
# LOAD DATA
# =========================================================

print(
    "Loading candidate data..."
)


candidates = pd.read_parquet(
    CANDIDATES_PATH
)

matches = pd.read_parquet(
    MATCHES_PATH
)

cases = pd.read_parquet(
    CASES_PATH
)

images = pd.read_parquet(
    IMAGES_PATH
)


# =========================================================
# NORMALIZE IDS
# =========================================================

for df in [
    candidates,
    matches,
    cases,
    images,
]:

    if (
        "case_id"
        in df.columns
    ):

        df[
            "case_id"
        ] = df[
            "case_id"
        ].map(
            normalize_id
        )


for df in [
    candidates,
    matches,
]:

    df[
        "concept_id"
    ] = df[
        "concept_id"
    ].map(
        normalize_id
    )


# =========================================================
# AMBIGUOUS ALIASES
# =========================================================

ambiguous_aliases = set()


if ALIAS_DIAGNOSTICS_PATH.exists():

    alias_diag = pd.read_csv(
        ALIAS_DIAGNOSTICS_PATH
    )


    mask = (
        alias_diag[
            "ambiguous_alias"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )


    ambiguous_aliases = set(
        alias_diag.loc[
            mask,
            "alias",
        ]
        .astype(str)
        .str.lower()
    )


print(
    "Ambiguous aliases:",
    len(
        ambiguous_aliases
    ),
)


# =========================================================
# CASE LOOKUP
# =========================================================

case_lookup = (
    cases
    .drop_duplicates(
        "case_id"
    )
    .set_index(
        "case_id"
    )
)


candidate_case_ids = set(
    candidates[
        "case_id"
    ]
)


# =========================================================
# ACTUAL IMAGE COUNTS
# =========================================================

image_counts = (
    images[
        images[
            "case_id"
        ].isin(
            candidate_case_ids
        )
    ]
    .groupby(
        "case_id"
    )[
        "image_id"
    ]
    .nunique()
)


# =========================================================
# HARD GATES
# =========================================================

print(
    "Applying minimal hard gates..."
)


gate_rows = []


for case_id in sorted(
    candidate_case_ids
):

    if (
        case_id
        not in case_lookup.index
    ):

        # This should be impossible after Step 16.
        raise RuntimeError(
            f"Validated candidate case "
            f"missing from cases.parquet: "
            f"{case_id}"
        )


    case = case_lookup.loc[
        case_id
    ]


    raw_text = (
        case[
            "raw_case_text"
        ]
    )


    if not isinstance(
        raw_text,
        str,
    ):

        raw_text = ""


    image_count = int(
        image_counts.get(
            case_id,
            0,
        )
    )


    has_image = (
        image_count > 0
    )


    has_raw_text = bool(
        raw_text.strip()
    )


    failed_rules = []


    if not has_image:

        failed_rules.append(
            "NO_SOURCE_IMAGE"
        )


    if not has_raw_text:

        failed_rules.append(
            "NO_RAW_CASE_TEXT"
        )


    gate_rows.append({
        "case_id":
            case_id,

        "article_id":
            normalize_id(
                case[
                    "article_id"
                ]
            ),

        "source_image_count":
            image_count,

        "has_source_image":
            has_image,

        "has_raw_case_text":
            has_raw_text,

        "failed_hard_rules":
            failed_rules,

        "hard_gate_status":
            (
                "HARD_EXCLUDE"
                if failed_rules
                else "ELIGIBLE"
            ),

        "rule_version":
            RULE_VERSION,
    })


gate_df = pd.DataFrame(
    gate_rows
)


gate_df.to_parquet(
    HARD_GATE_OUTPUT,
    index=False,
)


hard_excluded = (
    gate_df[
        gate_df[
            "hard_gate_status"
        ]
        ==
        "HARD_EXCLUDE"
    ]
    .copy()
)


hard_excluded.to_parquet(
    HARD_EXCLUDED_OUTPUT,
    index=False,
)


eligible_cases = (
    gate_df[
        gate_df[
            "hard_gate_status"
        ]
        ==
        "ELIGIBLE"
    ]
    .copy()
)


eligible_cases.to_parquet(
    ELIGIBLE_CASES_OUTPUT,
    index=False,
)


eligible_case_ids = set(
    eligible_cases[
        "case_id"
    ]
)


print(
    "Candidate cases:",
    len(
        candidate_case_ids
    ),
)

print(
    "Hard excluded:",
    len(
        hard_excluded
    ),
)

print(
    "Eligible:",
    len(
        eligible_case_ids
    ),
)


# =========================================================
# RAW-TEXT MATCHES ONLY
# =========================================================

raw_matches = matches[
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
            eligible_case_ids
        )
    )
].copy()


raw_matches = (
    raw_matches
    .sort_values(
        [
            "case_id",
            "concept_id",
            "start",
            "end",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# CLASSIFY EACH MENTION
# =========================================================

print(
    "Classifying raw-text "
    "disease mentions..."
)


mention_rows = []


for match in raw_matches.itertuples(
    index=False
):

    text = case_lookup.loc[
        match.case_id,
        "raw_case_text",
    ]


    mention_start = int(
        match.start
    )

    mention_end = int(
        match.end
    )


    evidence_start, evidence_end = (
        evidence_span(
            text,
            mention_start,
            mention_end,
        )
    )


    evidence_text = text[
        evidence_start:
        evidence_end
    ]


    relative_start = (
        mention_start
        -
        evidence_start
    )

    relative_end = (
        mention_end
        -
        evidence_start
    )


    status, reason = (
        classify_mention(
            evidence_text,
            relative_start,
            relative_end,
        )
    )


    alias = str(
        match.alias
    ).lower()


    ambiguous = (
        alias
        in ambiguous_aliases
    )


    mention_rows.append({
        "case_id":
            match.case_id,

        "article_id":
            match.article_id,

        "concept_id":
            match.concept_id,

        "disease":
            match.disease,

        "alias":
            match.alias,

        "matched_text":
            match.matched_text,

        "mention_start":
            mention_start,

        "mention_end":
            mention_end,

        "evidence_text":
            evidence_text,

        "evidence_start":
            evidence_start,

        "evidence_end":
            evidence_end,

        "mention_status":
            status,

        "assertion_reason":
            reason,

        "ambiguous_alias":
            ambiguous,

        "auto_confirm_eligible":
            (
                status
                ==
                "confirmed"
                and
                not ambiguous
            ),

        "rule_version":
            RULE_VERSION,
    })


mention_columns = [
    "case_id",
    "article_id",
    "concept_id",
    "disease",
    "alias",
    "matched_text",
    "mention_start",
    "mention_end",
    "evidence_text",
    "evidence_start",
    "evidence_end",
    "mention_status",
    "assertion_reason",
    "ambiguous_alias",
    "auto_confirm_eligible",
    "rule_version",
]


mention_assertions = pd.DataFrame(
    mention_rows,
    columns=mention_columns,
)


mention_assertions.to_parquet(
    MENTIONS_OUTPUT,
    index=False,
)


# =========================================================
# AGGREGATE TO CASE × DISEASE
# =========================================================

print(
    "Aggregating case-disease labels..."
)


eligible_candidates = candidates[
    candidates[
        "case_id"
    ].isin(
        eligible_case_ids
    )
].copy()


assertion_groups = {
    key: group.copy()
    for key, group in (
        mention_assertions.groupby(
            [
                "case_id",
                "concept_id",
            ],
            sort=False,
        )
    )
}


label_rows = []


for candidate in (
    eligible_candidates.itertuples(
        index=False
    )
):

    key = (
        candidate.case_id,
        candidate.concept_id,
    )


    group = assertion_groups.get(
        key
    )


    # -----------------------------------------------------
    # METADATA-ONLY / NO RAW TEXT DISEASE MENTION
    # -----------------------------------------------------

    if (
        group is None
        or len(
            group
        ) == 0
    ):

        label_rows.append({
            "case_id":
                candidate.case_id,

            "article_id":
                candidate.article_id,

            "concept_id":
                candidate.concept_id,

            "disease":
                candidate.disease,

            "diagnosis_status":
                "unclear",

            "diagnosis_evidence":
                "",

            "evidence_start":
                None,

            "evidence_end":
                None,

            "mention_count":
                0,

            "confirmed_mention_count":
                0,

            "ruled_out_mention_count":
                0,

            "probable_mention_count":
                0,

            "suspected_mention_count":
                0,

            "differential_mention_count":
                0,

            "historical_mention_count":
                0,

            "unclear_mention_count":
                0,

            "trajectory":
                "",

            "contradictory_assertions":
                False,

            "has_unambiguous_confirmation":
                False,

            "high_confidence_confirmed":
                False,

            "label_reason":
                "NO_RAW_TEXT_MENTION",

            "candidate_reason":
                candidate.candidate_reason,

            "source_image_count":
                int(
                    candidate.image_count
                ),

            "rule_version":
                RULE_VERSION,
        })

        continue


    group = (
        group
        .sort_values(
            "mention_start"
        )
        .reset_index(
            drop=True
        )
    )


    statuses = group[
        "mention_status"
    ].tolist()


    counts = {
        status:
            statuses.count(
                status
            )
        for status in [
            "confirmed",
            "ruled_out",
            "probable",
            "suspected",
            "differential",
            "historical",
            "unclear",
        ]
    }


    contradiction = (
        counts[
            "confirmed"
        ]
        > 0
        and
        counts[
            "ruled_out"
        ]
        > 0
    )


    meaningful = group[
        group[
            "mention_status"
        ]
        !=
        "unclear"
    ]


    if len(
        meaningful
    ):

        chosen = meaningful.iloc[
            -1
        ]

    else:

        chosen = group.iloc[
            -1
        ]


    has_unambiguous_confirmation = bool(
        (
            (
                group[
                    "mention_status"
                ]
                ==
                "confirmed"
            )
            &
            (
                ~group[
                    "ambiguous_alias"
                ]
            )
        ).any()
    )


    # -----------------------------------------------------
    # Conservative case-disease aggregation
    # -----------------------------------------------------

    if contradiction:

        final_status = (
            "unclear"
        )

        label_reason = (
            "CONTRADICTORY_ASSERTIONS"
        )

        high_confidence = False


    else:

        final_status = (
            chosen[
                "mention_status"
            ]
        )


        if (
            final_status
            ==
            "confirmed"
        ):

            if (
                has_unambiguous_confirmation
            ):

                label_reason = (
                    "EXPLICIT_CONFIRMATION"
                )

                high_confidence = True

            else:

                label_reason = (
                    "AMBIGUOUS_ALIAS_ONLY_"
                    "CONFIRMATION"
                )

                high_confidence = False

        else:

            label_reason = (
                "NO_HIGH_CONFIDENCE_"
                "CONFIRMATION"
            )

            high_confidence = False


    trajectory = ">".join(
        statuses
    )


    label_rows.append({
        "case_id":
            candidate.case_id,

        "article_id":
            candidate.article_id,

        "concept_id":
            candidate.concept_id,

        "disease":
            candidate.disease,

        "diagnosis_status":
            final_status,

        "diagnosis_evidence":
            chosen[
                "evidence_text"
            ],

        "evidence_start":
            int(
                chosen[
                    "evidence_start"
                ]
            ),

        "evidence_end":
            int(
                chosen[
                    "evidence_end"
                ]
            ),

        "mention_count":
            len(
                group
            ),

        "confirmed_mention_count":
            counts[
                "confirmed"
            ],

        "ruled_out_mention_count":
            counts[
                "ruled_out"
            ],

        "probable_mention_count":
            counts[
                "probable"
            ],

        "suspected_mention_count":
            counts[
                "suspected"
            ],

        "differential_mention_count":
            counts[
                "differential"
            ],

        "historical_mention_count":
            counts[
                "historical"
            ],

        "unclear_mention_count":
            counts[
                "unclear"
            ],

        "trajectory":
            trajectory,

        "contradictory_assertions":
            contradiction,

        "has_unambiguous_confirmation":
            has_unambiguous_confirmation,

        "high_confidence_confirmed":
            high_confidence,

        "label_reason":
            label_reason,

        "candidate_reason":
            candidate.candidate_reason,

        "source_image_count":
            int(
                candidate.image_count
            ),

        "rule_version":
            RULE_VERSION,
    })


labels = pd.DataFrame(
    label_rows
)


# =========================================================
# LABEL INTEGRITY
# =========================================================

if labels.duplicated(
    [
        "case_id",
        "concept_id",
    ]
).any():

    raise RuntimeError(
        "Duplicate case-disease "
        "labels generated."
    )


# Evidence must still be literal source text.
bad_evidence = []


for row in labels.itertuples(
    index=False
):

    if (
        row.evidence_start
        is None
        or pd.isna(
            row.evidence_start
        )
    ):

        continue


    text = case_lookup.loc[
        row.case_id,
        "raw_case_text",
    ]


    actual = text[
        int(
            row.evidence_start
        ):
        int(
            row.evidence_end
        )
    ]


    if (
        actual
        !=
        row.diagnosis_evidence
    ):

        bad_evidence.append(
            (
                row.case_id,
                row.concept_id,
            )
        )


if bad_evidence:

    raise RuntimeError(
        "Generated diagnosis evidence "
        "offsets are invalid."
    )


labels.to_parquet(
    LABELS_OUTPUT,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

reason_counts = (
    labels[
        "label_reason"
    ]
    .value_counts()
)


status_counts = (
    labels[
        "diagnosis_status"
    ]
    .value_counts()
)


high_conf_case_count = (
    labels.loc[
        labels[
            "high_confidence_confirmed"
        ],
        "case_id",
    ]
    .nunique()
)


print()
print(
    "=" * 70
)

print(
    "STEP 17 DETERMINISTIC LABELING"
)

print(
    "=" * 70
)

print(
    "Candidate cases:",
    len(
        candidate_case_ids
    ),
)

print(
    "Hard excluded cases:",
    len(
        hard_excluded
    ),
)

print(
    "Eligible cases:",
    len(
        eligible_case_ids
    ),
)

print(
    "Raw-text mention assertions:",
    len(
        mention_assertions
    ),
)

print(
    "Case-disease labels:",
    len(
        labels
    ),
)

print(
    "Cases with >=1 high-confidence "
    "confirmed hypothesis:",
    high_conf_case_count,
)

print()

print(
    "Diagnosis statuses:"
)

print(
    status_counts.to_string()
)

print()

print(
    "Label reasons:"
)

print(
    reason_counts.to_string()
)

print()

print(
    "Saved:"
)

print(
    " -",
    HARD_GATE_OUTPUT,
)

print(
    " -",
    HARD_EXCLUDED_OUTPUT,
)

print(
    " -",
    ELIGIBLE_CASES_OUTPUT,
)

print(
    " -",
    MENTIONS_OUTPUT,
)

print(
    " -",
    LABELS_OUTPUT,
)

print()
print(
    "FINAL STATUS: PASS"
)