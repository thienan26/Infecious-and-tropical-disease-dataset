import sys
import pandas as pd


scope = pd.read_parquet(
    "reports/scope/"
    "disease_scope.parquet"
)


errors = []


# 1
if scope[
    "concept_id"
].duplicated().any():

    errors.append(
        "Duplicate concept_id"
    )


# 2
if not (
    scope[
        "candidate_cases"
    ]
    > 0
).all():

    errors.append(
        "Disease with zero "
        "candidate_cases"
    )


# 3
if not (
    scope[
        "is_infectious"
    ]
    == True
).all():

    errors.append(
        "Non-infectious disease "
        "present in scope"
    )


# 4
allowed = {
    "tropical",
    "unknown",
    "not_tropical",
}

invalid = (
    set(
        scope[
            "tropical_status"
        ]
        .dropna()
        .unique()
    )
    - allowed
)

if invalid:

    errors.append(
        f"Invalid tropical status: "
        f"{invalid}"
    )


# 5
if scope[
    "disease"
].isna().any():

    errors.append(
        "Missing disease name"
    )


print(
    "=" * 55
)

print(
    "DISEASE SCOPE VALIDATION"
)

print(
    "=" * 55
)


if errors:

    for error in errors:

        print(
            "[FAIL]",
            error
        )

    print()
    print(
        "FINAL STATUS: FAIL"
    )

    sys.exit(1)


print(
    "[PASS] unique concept IDs"
)

print(
    "[PASS] all diseases observed"
)

print(
    "[PASS] all diseases infectious"
)

print(
    "[PASS] tropical statuses valid"
)

print(
    "[PASS] disease names present"
)

print()

print(
    "Disease concepts:",
    len(scope)
)

print(
    "Tropical:",
    int(
        (
            scope[
                "tropical_status"
            ]
            == "tropical"
        ).sum()
    )
)

print(
    "Unknown tropical:",
    int(
        (
            scope[
                "tropical_status"
            ]
            == "unknown"
        ).sum()
    )
)

print()

print(
    "FINAL STATUS: PASS"
)