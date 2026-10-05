"""
Mục đích cốt lõi của Bước 15 là tạo ra một tập hợp các "ứng viên" (candidate) tiềm năng nhất. Hệ thống sẽ mang danh sách bệnh của bạn đi quét qua toàn bộ văn bản gốc, tìm xem bất cứ ca bệnh nào có nhắc đến các căn bệnh này (kể cả từ viết tắt, bí danh) và gom chúng lại.
Lưu ý quan trọng tại sao Bước 15 chưa phải là "Lọc":
Ở bước này, chiến lược là "thà bắt nhầm còn hơn bỏ sót" (ưu tiên recall).

Nếu một ca bệnh hoàn toàn không có ảnh (0 images), nó vẫn được giữ lại.

Nếu tên bệnh chỉ vô tình nằm trên Tiêu đề bài báo mà không hề được nhắc đến trong hồ sơ khám (metadata-only), nó vẫn được giữ lại.

"""


from collections import defaultdict
from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm

from matching_utils import (
    as_list,
    build_matcher,
    find_matches,
)


# =========================================================
# PATHS
# =========================================================

SCOPE_PATH = Path(
    "reports/scope/"
    "disease_scope.parquet"
)

KB_PATH = Path(
    "taxonomy/"
    "disease_knowledge_base.parquet"
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


OUTPUT_ROOT = Path(
    "data/candidates"
)

REPORT_ROOT = Path(
    "reports/candidates"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


MATCHES_OUTPUT = (
    OUTPUT_ROOT /
    "candidate_matches.parquet"
)

CANDIDATES_OUTPUT = (
    OUTPUT_ROOT /
    "candidates.parquet"
)

INVENTORY_OUTPUT = (
    OUTPUT_ROOT /
    "candidate_case_inventory.parquet"
)

COUNTS_OUTPUT = (
    REPORT_ROOT /
    "candidate_counts_by_disease.csv"
)

SUMMARY_OUTPUT = (
    REPORT_ROOT /
    "candidate_summary.json"
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


def json_list(value):

    return json.dumps(
        list(value),
        ensure_ascii=False,
    )


# =========================================================
# LOAD DATA
# =========================================================

print(
    "Loading disease scope..."
)

scope = pd.read_parquet(
    SCOPE_PATH
)


print(
    "Loading disease KB..."
)

kb = pd.read_parquet(
    KB_PATH
)


print(
    "Loading cases..."
)

cases = pd.read_parquet(
    CASES_PATH
)


print(
    "Loading articles..."
)

articles = pd.read_parquet(
    ARTICLES_PATH
)


print(
    "Loading images..."
)

images = pd.read_parquet(
    IMAGES_PATH
)


# =========================================================
# BASIC SCHEMA CHECK
# =========================================================

required_scope = {
    "concept_id",
    "disease",
}

required_cases = {
    "case_id",
    "article_id",
    "raw_case_text",
}

required_articles = {
    "article_id",
    "title",
    "keywords",
    "mesh_terms",
}

required_images = {
    "image_id",
    "case_id",
}


for name, df, required in [
    (
        "scope",
        scope,
        required_scope,
    ),
    (
        "cases",
        cases,
        required_cases,
    ),
    (
        "articles",
        articles,
        required_articles,
    ),
    (
        "images",
        images,
        required_images,
    ),
]:

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{name} missing "
            f"columns: {missing}"
        )


# =========================================================
# NORMALIZE IDS
# =========================================================

scope[
    "concept_id"
] = scope[
    "concept_id"
].map(
    normalize_id
)


cases[
    "case_id"
] = cases[
    "case_id"
].map(
    normalize_id
)


cases[
    "article_id"
] = cases[
    "article_id"
].map(
    normalize_id
)


articles[
    "article_id"
] = articles[
    "article_id"
].map(
    normalize_id
)


images[
    "case_id"
] = images[
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


# =========================================================
# SCOPE
# =========================================================

scope_ids = set(
    scope[
        "concept_id"
    ]
)


if None in scope_ids:

    raise RuntimeError(
        "Null concept ID in scope."
    )


disease_lookup = dict(
    zip(
        scope[
            "concept_id"
        ],
        scope[
            "disease"
        ],
    )
)


print(
    "Diseases in scope:",
    len(
        scope_ids
    ),
)


# =========================================================
# BUILD MATCHER
# =========================================================

print(
    "Building disease matcher..."
)


matcher = build_matcher(
    kb,
    allowed_concepts=
        scope_ids,
)


# =========================================================
# ARTICLE LOOKUP
# =========================================================

article_lookup = {}


for row in (
    articles.itertuples(
        index=False
    )
):

    article_lookup[
        str(
            row.article_id
        )
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
# IMAGE SUMMARY
# =========================================================

print(
    "Building image summary..."
)


image_summary = {}


for case_id, group in (
    images.groupby(
        "case_id",
        sort=False,
    )
):

    image_ids = sorted(
        set(
            str(x)
            for x in group[
                "image_id"
            ].dropna()
        )
    )

    source_image_ids = []

    if (
        "source_image_id"
        in group.columns
    ):

        source_image_ids = sorted(
            set(
                str(x)
                for x in group[
                    "source_image_id"
                ].dropna()
            )
        )


    image_summary[
        str(case_id)
    ] = {
        "image_count":
            len(
                image_ids
            ),

        "image_ids":
            image_ids,

        "source_image_count":
            len(
                source_image_ids
            ),

        "source_image_ids":
            source_image_ids,
    }


# =========================================================
# MATCH ALL CASES
# =========================================================

print(
    "Matching cases..."
)


match_rows = []


for case in tqdm(
    cases.itertuples(
        index=False
    ),
    total=len(
        cases
    ),
):

    case_id = str(
        case.case_id
    )

    article_id = (
        str(
            case.article_id
        )
        if case.article_id
        is not None
        else None
    )


    raw_text = (
        case.raw_case_text
        if isinstance(
            case.raw_case_text,
            str,
        )
        else ""
    )


    article = (
        article_lookup.get(
            article_id,
            {},
        )
    )


    # -----------------------------------------------------
    # RAW CASE TEXT
    # -----------------------------------------------------

    matches = find_matches(
        matcher,
        raw_text,
        field="raw_case_text",
    )


    for match in matches:

        match[
            "case_id"
        ] = case_id

        match[
            "article_id"
        ] = article_id

        match_rows.append(
            match
        )


    # -----------------------------------------------------
    # ARTICLE TITLE
    # -----------------------------------------------------

    title = article.get(
        "title",
        "",
    )


    matches = find_matches(
        matcher,
        title,
        field="article_title",
    )


    for match in matches:

        match[
            "case_id"
        ] = case_id

        match[
            "article_id"
        ] = article_id

        match_rows.append(
            match
        )


    # -----------------------------------------------------
    # KEYWORDS
    # -----------------------------------------------------

    for index, keyword in enumerate(
        article.get(
            "keywords",
            [],
        )
    ):

        if keyword is None:
            continue


        matches = find_matches(
            matcher,
            str(
                keyword
            ),
            field="article_keyword",
            field_item_index=index,
        )


        for match in matches:

            match[
                "case_id"
            ] = case_id

            match[
                "article_id"
            ] = article_id

            match_rows.append(
                match
            )


    # -----------------------------------------------------
    # MeSH
    # -----------------------------------------------------

    for index, mesh in enumerate(
        article.get(
            "mesh_terms",
            [],
        )
    ):

        if mesh is None:
            continue


        matches = find_matches(
            matcher,
            str(
                mesh
            ),
            field="article_mesh",
            field_item_index=index,
        )


        for match in matches:

            match[
                "case_id"
            ] = case_id

            match[
                "article_id"
            ] = article_id

            match_rows.append(
                match
            )


# =========================================================
# MATCH TABLE
# =========================================================

matches_df = pd.DataFrame(
    match_rows
)


if matches_df.empty:

    raise RuntimeError(
        "No candidate disease "
        "matches found."
    )


matches_df[
    "disease"
] = matches_df[
    "concept_id"
].map(
    disease_lookup
)


if matches_df[
    "disease"
].isna().any():

    raise RuntimeError(
        "Matched concept outside "
        "disease scope."
    )


# Stable ordering.
matches_df = (
    matches_df
    .sort_values(
        [
            "case_id",
            "concept_id",
            "field",
            "field_item_index",
            "start",
            "end",
            "alias",
        ],
        na_position="first",
    )
    .reset_index(
        drop=True
    )
)


matches_df.to_parquet(
    MATCHES_OUTPUT,
    index=False,
)


# =========================================================
# AGGREGATE TO CASE × DISEASE
# =========================================================

print(
    "Aggregating candidate hypotheses..."
)


candidate_rows = []


for (
    case_id,
    concept_id,
), group in matches_df.groupby(
    [
        "case_id",
        "concept_id",
    ],
    sort=False,
):


    fields = set(
        group[
            "field"
        ]
    )


    aliases = sorted(
        set(
            group[
                "alias"
            ]
        )
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

        candidate_reason = (
            "RAW_TEXT_AND_METADATA"
        )

    elif has_raw:

        candidate_reason = (
            "RAW_TEXT"
        )

    else:

        candidate_reason = (
            "METADATA_ONLY"
        )


    article_ids = (
        group[
            "article_id"
        ]
        .dropna()
        .unique()
    )


    article_id = (
        str(
            article_ids[0]
        )
        if len(
            article_ids
        )
        else None
    )


    image_info = (
        image_summary.get(
            str(
                case_id
            ),
            {
                "image_count":
                    0,

                "image_ids":
                    [],

                "source_image_count":
                    0,

                "source_image_ids":
                    [],
            },
        )
    )


    candidate_rows.append({
        "case_id":
            str(
                case_id
            ),

        "article_id":
            article_id,

        "concept_id":
            str(
                concept_id
            ),

        "disease":
            disease_lookup[
                concept_id
            ],

        "matched_aliases":
            aliases,

        "match_count":
            len(
                group
            ),

        "raw_text_match_count":
            int(
                (
                    group[
                        "field"
                    ]
                    ==
                    "raw_case_text"
                ).sum()
            ),

        "metadata_match_count":
            int(
                (
                    group[
                        "field"
                    ]
                    !=
                    "raw_case_text"
                ).sum()
            ),

        "has_raw_text_match":
            has_raw,

        "has_title_match":
            has_title,

        "has_keyword_match":
            has_keyword,

        "has_mesh_match":
            has_mesh,

        "candidate_reason":
            candidate_reason,

        "image_count":
            image_info[
                "image_count"
            ],

        "image_ids":
            image_info[
                "image_ids"
            ],

        "source_image_count":
            image_info[
                "source_image_count"
            ],

        "source_image_ids":
            image_info[
                "source_image_ids"
            ],
    })


candidates = pd.DataFrame(
    candidate_rows
)


# =========================================================
# CANDIDATE KEY VALIDATION
# =========================================================

duplicate_key = (
    candidates.duplicated(
        subset=[
            "case_id",
            "concept_id",
        ]
    )
)


if duplicate_key.any():

    raise RuntimeError(
        "Duplicate "
        "(case_id, concept_id) "
        "candidate rows."
    )


if not set(
    candidates[
        "concept_id"
    ]
).issubset(
    scope_ids
):

    raise RuntimeError(
        "Candidate concept "
        "outside scope."
    )


candidates = (
    candidates
    .sort_values(
        [
            "disease",
            "case_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


candidates.to_parquet(
    CANDIDATES_OUTPUT,
    index=False,
)


# =========================================================
# BUILD WIDE AUDIT INVENTORY
# =========================================================

print(
    "Building candidate audit inventory..."
)


case_columns = [
    "case_id",
    "article_id",
    "raw_case_text",
]


for optional in [
    "raw_text_sha256",
    "raw_text_chars",
]:

    if optional in cases.columns:

        case_columns.append(
            optional
        )


case_info = cases[
    case_columns
].copy()


article_info = articles[
    [
        "article_id",
        "title",
        "keywords",
        "mesh_terms",
    ]
].copy()


inventory = (
    candidates
    .merge(
        case_info,
        on=[
            "case_id",
            "article_id",
        ],
        how="left",
        validate="many_to_one",
    )
    .merge(
        article_info,
        on="article_id",
        how="left",
        validate="many_to_one",
    )
)


if inventory[
    "raw_case_text"
].isna().any():

    raise RuntimeError(
        "Candidate inventory lost "
        "raw case text during join."
    )


inventory.to_parquet(
    INVENTORY_OUTPUT,
    index=False,
)


# =========================================================
# DISEASE COUNTS REPORT
# =========================================================

counts = (
    candidates.groupby(
        [
            "concept_id",
            "disease",
        ],
        as_index=False,
    )
    .agg(
        candidate_cases=(
            "case_id",
            "nunique",
        ),

        raw_text_candidate_cases=(
            "has_raw_text_match",
            "sum",
        ),

        candidate_cases_with_images=(
            "image_count",
            lambda x:
                int(
                    (
                        x > 0
                    ).sum()
                ),
        ),
    )
)


metadata_only_counts = (
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


counts[
    "metadata_only_cases"
] = (
    counts[
        "concept_id"
    ]
    .map(
        metadata_only_counts
    )
    .fillna(0)
    .astype(int)
)


counts = counts.sort_values(
    "candidate_cases",
    ascending=False,
)


counts.to_csv(
    COUNTS_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# SUMMARY
# =========================================================

summary = {
    "scope_diseases":
        int(
            len(
                scope_ids
            )
        ),

    "matched_diseases":
        int(
            candidates[
                "concept_id"
            ].nunique()
        ),

    "normalized_cases":
        int(
            len(
                cases
            )
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
                matches_df
            )
        ),

    "raw_text_hypotheses":
        int(
            candidates[
                "has_raw_text_match"
            ].sum()
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

    "candidate_hypotheses_with_images":
        int(
            (
                candidates[
                    "image_count"
                ]
                > 0
            ).sum()
        ),

    "unique_candidate_cases_with_images":
        int(
            candidates.loc[
                candidates[
                    "image_count"
                ]
                > 0,
                "case_id",
            ].nunique()
        ),
}


with SUMMARY_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print(
    "=" * 65
)

print(
    "CANDIDATE CASE BUILD COMPLETE"
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
print(
    "Saved:"
)

print(
    " -",
    MATCHES_OUTPUT,
)

print(
    " -",
    CANDIDATES_OUTPUT,
)

print(
    " -",
    INVENTORY_OUTPUT,
)

print(
    " -",
    COUNTS_OUTPUT,
)

print(
    " -",
    SUMMARY_OUTPUT,
)

print()
print(
    "FINAL STATUS: PASS"
)

"""
=================================================================
CANDIDATE CASE BUILD COMPLETE
=================================================================
scope_diseases: 390
matched_diseases: 386
normalized_cases: 98641
candidate_cases: 21646
case_disease_hypotheses: 39803
match_occurrences: 78859
raw_text_hypotheses: 33466
metadata_only_hypotheses: 6337
candidate_hypotheses_with_images: 14052
unique_candidate_cases_with_images: 7639

Saved:
 - data\candidates\candidate_matches.parquet
 - data\candidates\candidates.parquet
 - data\candidates\candidate_case_inventory.parquet
 - reports\candidates\candidate_counts_by_disease.csv
 - reports\candidates\candidate_summary.json

 candidate_matches.parquet (Bảng Tọa độ gốc): Đánh dấu chính xác vị trí từ khóa. Ví dụ: Ca bệnh 001 xuất hiện chữ "malaria" (sốt rét) ở ký tự số 421 đến 428. Bảng này giống như một tấm bản đồ, giúp AI dán nhãn ở bước sau biết chính xác phải đọc khúc nào để xác minh bệnh, thay vì phải tự bơi trong biển chữ.

candidates.parquet (Bảng Giả thuyết): Gom nhóm theo "Ca bệnh × Căn bệnh". Một bệnh nhân có thể có nhiều giả thuyết (Ví dụ: Ca 001 - Giả thuyết Sốt rét; Ca 001 - Giả thuyết Sốt xuất huyết). Đây sẽ là bảng dữ liệu chính để xử lý tiếp.

candidate_case_inventory.parquet (Bảng Hồ sơ kiểm toán): Bảng này gộp toàn bộ nội dung chữ, siêu dữ liệu bài báo (tiêu đề, từ khóa) và số lượng ảnh. Nó chủ yếu dùng để bạn (con người) đọc, kiểm tra và dễ hình dung.
"""