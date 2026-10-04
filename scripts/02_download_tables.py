from pathlib import Path
import hashlib
import json
import requests


RECORD_ID = "20416562"

API_URL = (
    f"https://zenodo.org/api/records/"
    f"{RECORD_ID}"
)

OUTPUT = Path("data/raw/multicare")

REQUIRED = {
    "metadata.parquet",
    "cases.parquet",
    "case_images.parquet",
    "captions_and_labels.csv",
    "data_dictionary.csv",
}


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    record = requests.get(
        API_URL,
        timeout=60,
    ).json()

    available = {
        item["key"]: item
        for item in record["files"]
    }

    missing = REQUIRED - set(available)

    if missing:
        raise RuntimeError(
            f"Missing files: {missing}"
        )

    lock = {
        "record_id": RECORD_ID,
        "files": {},
    }

    for filename in sorted(REQUIRED):

        info = available[filename]

        url = info["links"]["self"]

        destination = OUTPUT / filename

        print(
            f"Downloading {filename}..."
        )

        with requests.get(
            url,
            stream=True,
            timeout=120,
        ) as response:

            response.raise_for_status()

            with destination.open("wb") as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        lock["files"][filename] = {
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(
                destination
            ),
        }

        print(
            "Saved:",
            destination
        )

    with open(
        OUTPUT / "source_lock.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            lock,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()