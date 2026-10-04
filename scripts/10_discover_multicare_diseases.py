# Discover diseases thực sự xuất hiện trong MultiCaRe
# sử dụng data/normalized/cases.parquet
# kqua: reports/scope/disease_catalog.parquet

from collections import (
    Counter,
    defaultdict,
)

from pathlib import Path
import ahocorasick
import pandas as pd
from tqdm import tqdm


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


MIN_ALIAS_LENGTH = 4


def values_to_text(value):

    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple),
    ):

        return " ".join(
            str(x)
            for x in value
            if x is not None
        )

    # PyArrow/Pandas sometimes
    # converts lists to array-like values.
    if hasattr(
        value,
        "tolist",
    ):

        try:
            x = value.tolist()

            if isinstance(x, list):
                return " ".join(
                    str(v)
                    for v in x
                )
        except Exception:
            pass

    return str(value)


alias_map = defaultdict(
    set
)


for row in KB.itertuples():

    # Do not use the infectious
    # root itself as a diagnosis.
    if getattr(
        row,
        "is_scope_root",
        False,
    ):
        continue

    names = [
        row.disease,
        *row.aliases,
    ]

    for alias in names:

        alias = (
            str(alias)
            .strip()
            .lower()
        )

        if (
            len(alias)
            >= MIN_ALIAS_LENGTH
        ):

            alias_map[
                alias
            ].add(
                row.concept_id
            )


automaton = (
    ahocorasick.Automaton()
)


for alias, concept_ids in (
    alias_map.items()
):

    automaton.add_word(
        alias,
        (
            alias,
            tuple(
                concept_ids
            ),
        ),
    )


automaton.make_automaton()


def find_concepts(text):

    if not isinstance(
        text,
        str,
    ):
        return []

    text = text.lower()

    found = []

    for end_index, payload in (
        automaton.iter(text)
    ):

        alias, concept_ids = (
            payload
        )

        start = (
            end_index
            - len(alias)
            + 1
        )

        stop = end_index + 1

        # Whole-word-ish boundaries.
        if (
            start > 0
            and text[
                start - 1
            ].isalnum()
        ):
            continue

        if (
            stop < len(text)
            and text[
                stop
            ].isalnum()
        ):
            continue

        found.extend(
            concept_ids
        )

    return found


article_text = {}


for row in ARTICLES.itertuples():

    text = " ".join([
        values_to_text(
            getattr(
                row,
                "title",
                "",
            )
        ),

        values_to_text(
            getattr(
                row,
                "keywords",
                "",
            )
        ),

        values_to_text(
            getattr(
                row,
                "mesh_terms",
                "",
            )
        ),
    ])

    article_text[
        str(row.article_id)
    ] = text


raw_mentions = Counter()
candidate_cases = Counter()

text_case_counts = Counter()
metadata_case_counts = Counter()


for row in tqdm(
    CASES.itertuples(),
    total=len(CASES),
):

    text_matches = find_concepts(
        row.raw_case_text
    )

    metadata_matches = (
        find_concepts(
            article_text.get(
                str(
                    row.article_id
                ),
                "",
            )
        )
    )

    raw_mentions.update(
        text_matches
        + metadata_matches
    )

    text_unique = set(
        text_matches
    )

    metadata_unique = set(
        metadata_matches
    )

    all_unique = (
        text_unique
        | metadata_unique
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


catalog = KB.copy()


catalog[
    "raw_mentions"
] = (
    catalog["concept_id"]
    .map(raw_mentions)
    .fillna(0)
    .astype(int)
)


catalog[
    "candidate_cases"
] = (
    catalog["concept_id"]
    .map(candidate_cases)
    .fillna(0)
    .astype(int)
)


catalog[
    "text_candidate_cases"
] = (
    catalog["concept_id"]
    .map(text_case_counts)
    .fillna(0)
    .astype(int)
)


catalog[
    "metadata_candidate_cases"
] = (
    catalog["concept_id"]
    .map(metadata_case_counts)
    .fillna(0)
    .astype(int)
)


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