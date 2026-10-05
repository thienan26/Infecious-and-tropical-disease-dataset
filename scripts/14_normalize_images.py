# case_images.parquet + captions_and_labels.csv
"""
case PMC123_1
│
├── image 10001
│   └── source_image = PMC123_Fig2
│
└── image 10002
    └── source_image = PMC123_Fig2 """

from pathlib import Path
import pandas as pd

# ============================================================
# Helpers for Flattening Nested Parquet
# ============================================================
def to_python_list(value):
    if value is None: return []
    if isinstance(value, dict): return [value]
    if isinstance(value, (list, tuple)): return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, dict): return [converted]
        if isinstance(converted, list): return converted
    if hasattr(value, "as_py"):
        converted = value.as_py()
        if isinstance(converted, dict): return [converted]
        if isinstance(converted, list): return converted
    return [value]

def to_dict(value):
    if value is None: return {}
    if isinstance(value, dict): return value
    if hasattr(value, "as_py"):
        converted = value.as_py()
        if isinstance(converted, dict): return converted
    return {}

def flatten_case_images(raw_df):
    rows = []
    for outer in raw_df.itertuples(index=False):
        article_id = getattr(outer, "article_id", None)
        nested_cases = getattr(outer, "case_images", None)
        
        for case_item in to_python_list(nested_cases):
            if case_item is None: continue
            case_record = to_dict(case_item)
            case_id = case_record.get("case_id")
            if not case_id: continue
            
            for image_item in to_python_list(case_record.get("case_image_list")):
                if image_item is None: continue
                img_rec = to_dict(image_item).copy()
                
                # Trong raw data gọi là image_id, nhưng ở bảng chuẩn hóa 
                # ta dùng source_image_id để phân biệt với image_id cuối cùng
                img_id = img_rec.get("image_id")
                img_file = img_rec.get("file")
                
                if not img_id and img_file:
                    img_id = f"{case_id}_{img_file}"
                    
                rows.append({
                    "article_id": article_id,
                    "case_id": case_id,
                    "source_image_id": img_id,
                    "source_caption": img_rec.get("caption"),
                    "text_references": img_rec.get("text_references")
                })
                
    flat = pd.DataFrame(rows)
    # Chuẩn hóa kiểu chuỗi cho ID
    for col in ["article_id", "case_id", "source_image_id"]:
        if col in flat.columns:
            flat[col] = flat[col].astype("string").str.strip()
            
    return flat.drop_duplicates(subset=["article_id", "case_id", "source_image_id"]).reset_index(drop=True)

# ============================================================
# Main Execution
# ============================================================
def main():
    RAW = Path("data/raw/multicare")
    NORMALIZED = Path("data/normalized")
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    
    print("Loading normalized cases...")
    cases_df = pd.read_parquet(NORMALIZED / "cases.parquet")
    valid_cases = set(cases_df["case_id"].unique())
    
    print("Loading case_images.parquet...")
    raw_case_images = pd.read_parquet(RAW / "case_images.parquet")
    
    print("Flattening nested image metadata...")
    source_images = flatten_case_images(raw_case_images)
    
    print("Loading captions_and_labels.csv...")
    labels_df = pd.read_csv(RAW / "captions_and_labels.csv")
    
    # Chuẩn hóa bảng labels
    # Đổi 'file_id' thành 'image_id' (đây sẽ là ID của ảnh con cuối cùng)
    labels_df = labels_df.rename(columns={"file_id": "image_id"})
    labels_df["image_id"] = labels_df["image_id"].astype("string").str.strip()
    labels_df["main_image"] = labels_df["main_image"].astype("string").str.strip()
    
    print("Merging images...")
    # Nối theo main_image của file CSV và source_image_id của file Parquet
    images = labels_df.merge(
        source_images,
        left_on="main_image",
        right_on="source_image_id",
        how="left"
    )
    
    # ============================================================
    # Validation (Giữ case tồn tại, lọc case lỗi)
    # ============================================================
    images["case_valid"] = images["case_id"].isin(valid_cases)
    
    rejected = images[~images["case_valid"] | images["case_id"].isna()].copy()
    valid_images = images[images["case_valid"] & images["case_id"].notna()].copy()
    
    if len(rejected) > 0:
        print(f"Warning: {len(rejected)} images rejected. Saving to image_normalization_rejects.parquet")
        rejected["reason"] = "CASE_NOT_FOUND_IN_NORMALIZED_CASES"
        rejected.to_parquet(NORMALIZED / "image_normalization_rejects.parquet", index=False)
        
    print("\n=================================================================")
    print("IMAGE NORMALIZATION COMPLETE")
    print("=================================================================")
    print(f"Final images: {len(valid_images)}")
    print(f"Cases with images: {valid_images['case_id'].nunique()}")
    print(f"Unique source images: {valid_images['source_image_id'].nunique()}")
    
    mean_images = valid_images.groupby("case_id").size().mean()
    max_images = valid_images.groupby("case_id").size().max()
    print(f"Mean images per case: {mean_images:.2f}")
    print(f"Maximum images in one case: {max_images}")
    
    duplicate_ids = valid_images.duplicated(subset=['image_id']).sum()
    print(f"Duplicate final image IDs: {duplicate_ids}")
    
    # Save final
    valid_images.drop(columns=["case_valid"], inplace=True)
    valid_images.to_parquet(NORMALIZED / "images.parquet", index=False)
    
    print(f"Saved: {NORMALIZED / 'images.parquet'}")
    if duplicate_ids == 0:
        print("FINAL STATUS: PASS")
    else:
        print("FINAL STATUS: FAIL (Duplicates found)")

if __name__ == "__main__":
    main()

    """
    IMAGE NORMALIZATION COMPLETE
=============================
Final images: 142125
Cases with images: 35605
Unique source images: 79454
Mean images per case: 3.99
Maximum images in one case: 47
Duplicate final image IDs: 2871
Saved: data\normalized\images.parquet
    """