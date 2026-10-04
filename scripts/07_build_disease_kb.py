from pathlib import Path
from functools import lru_cache
import re
import pandas as pd


SOURCE = Path(
    "taxonomy/doid.obo"
)

OUTPUT = Path(
    "taxonomy/disease_kb_base.parquet"
)

INFECTIOUS_ROOT = (
    "DOID:0050117"
)


def parse_term(block):

    id_match = re.search(
        r"(?m)^id: (DOID:\d+)$",
        block,
    )

    name_match = re.search(
        r"(?m)^name: (.+)$",
        block,
    )

    if not id_match or not name_match:
        return None

    if re.search(
        r"(?m)^is_obsolete: true$",
        block,
    ):
        return None

    concept_id = id_match.group(1)

    name = (
        name_match.group(1)
        .strip()
    )

    synonyms = re.findall(
        r'(?m)^synonym: "([^"]+)"',
        block,
    )

    parents = re.findall(
        r"(?m)^is_a: "
        r"(DOID:\d+)",
        block,
    )

    alt_ids = re.findall(
        r"(?m)^alt_id: "
        r"(DOID:\d+)",
        block,
    )

    aliases = sorted(
        set(
            [name]
            + synonyms
        )
    )

    return {
        "concept_id": concept_id,
        "disease": name,
        "aliases": aliases,
        "parents": parents,
        "alt_ids": alt_ids,
    }


text = SOURCE.read_text(
    encoding="utf-8",
    errors="replace",
)

blocks = re.split(
    r"(?m)^\[Term\]\s*$",
    text,
)[1:]


terms = {}

for block in blocks:

    row = parse_term(block)

    if row:
        terms[
            row["concept_id"]
        ] = row


@lru_cache(maxsize=None)
def is_infectious(concept_id):

    if concept_id == INFECTIOUS_ROOT:
        return True

    term = terms.get(
        concept_id
    )

    if not term:
        return False

    return any(
        is_infectious(parent)
        for parent
        in term["parents"]
    )


rows = []

for concept_id, term in terms.items():

    rows.append({
        **term,

        "is_infectious":
            is_infectious(
                concept_id
            ),

        "infectious_source":
            "Human Disease Ontology",

        "infectious_root":
            INFECTIOUS_ROOT,
    })


df = pd.DataFrame(rows)


# Do not treat the root itself
# as a trainable disease class.
df["is_scope_root"] = (
    df["concept_id"]
    == INFECTIOUS_ROOT
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_parquet(
    OUTPUT,
    index=False,
)


print("All concepts:", len(df))

print(
    "Infectious concepts:",
    int(
        df[
            "is_infectious"
        ].sum()
    ),
)

print(
    df[
        df["is_infectious"]
    ][
        [
            "concept_id",
            "disease",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)