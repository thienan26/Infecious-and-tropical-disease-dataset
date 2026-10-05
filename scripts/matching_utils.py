from collections import defaultdict
import ahocorasick


MIN_ALIAS_LENGTH = 4


def as_list(value):

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple, set),
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


def build_matcher(
    kb,
    allowed_concepts=None,
    min_alias_length=MIN_ALIAS_LENGTH,
):

    allowed = (
        set(allowed_concepts)
        if allowed_concepts is not None
        else None
    )

    alias_map = defaultdict(
        set
    )


    for row in kb.itertuples():

        concept_id = str(
            row.concept_id
        )


        if (
            allowed is not None
            and concept_id not in allowed
        ):
            continue


        # Never treat ontology root
        # itself as a diagnosis.
        if getattr(
            row,
            "is_scope_root",
            False,
        ):
            continue


        names = [
            row.disease,
            *as_list(
                getattr(
                    row,
                    "aliases",
                    [],
                )
            ),
        ]


        for name in names:

            if name is None:
                continue

            alias = (
                str(name)
                .strip()
                .lower()
            )


            if len(
                alias
            ) < min_alias_length:

                continue


            alias_map[
                alias
            ].add(
                concept_id
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
                    sorted(
                        concept_ids
                    )
                ),
            ),
        )


    automaton.make_automaton()

    return automaton


def find_matches(
    automaton,
    text,
    field,
    field_item_index=None,
):

    if not isinstance(
        text,
        str,
    ):

        return []


    if not text:

        return []


    lowered = text.lower()

    matches = []


    for end_index, payload in (
        automaton.iter(
            lowered
        )
    ):

        alias, concept_ids = (
            payload
        )


        start = (
            end_index
            - len(alias)
            + 1
        )

        end = end_index + 1


        # Whole-word-ish boundary.
        if (
            start > 0
            and lowered[
                start - 1
            ].isalnum()
        ):
            continue


        if (
            end < len(lowered)
            and lowered[
                end
            ].isalnum()
        ):
            continue


        matched_text = text[
            start:end
        ]


        for concept_id in concept_ids:

            matches.append({
                "concept_id":
                    concept_id,

                "field":
                    field,

                "field_item_index":
                    field_item_index,

                "alias":
                    alias,

                "matched_text":
                    matched_text,

                "start":
                    start,

                "end":
                    end,
            })


    return matches