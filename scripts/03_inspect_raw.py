from pathlib import Path
import pyarrow.parquet as pq


ROOT = Path(
    "data/raw/multicare"
)


for filename in [
    "metadata.parquet",
    "cases.parquet",
    "case_images.parquet",
]:

    path = ROOT / filename

    print()
    print("=" * 60)
    print(filename)
    print("=" * 60)

    table = pq.read_table(
        path
    )

    print("Rows:", table.num_rows)
    print("Columns:")

    for field in table.schema:
        print(
            " -",
            field.name,
            ":",
            field.type,
        )

    print()
    print("FIRST ROW:")
    print(
        table.slice(0, 1)
        .to_pylist()[0]
    )