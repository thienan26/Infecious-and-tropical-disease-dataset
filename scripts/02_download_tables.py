from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests
import yaml

CONFIG_PATH = Path("configs/source.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

RECORD_ID = str(CONFIG["zenodo"]["record_id"])
API_URL = CONFIG["zenodo"]["api_url"]
OUTPUT = Path(CONFIG["paths"]["raw_root"])
REQUIRED = set(CONFIG["required_files"])


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

    response = requests.get(
        API_URL,
        timeout=60,
    )
    response.raise_for_status()
    record = response.json()

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
        "dataset": CONFIG["dataset"]["name"],
        "provider": CONFIG["dataset"]["provider"],
        "record_id": RECORD_ID,
        "api_url": API_URL,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "zenodo": {
            "id": record.get("id"),
            "doi": record.get("doi"),
            "conceptdoi": record.get("conceptdoi"),
            "created": record.get("created"),
            "updated": record.get("updated"),
            "version": record.get("metadata", {}).get("version"),
        },
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
        ) as file_response:

            file_response.raise_for_status()

            with destination.open("wb") as f:
                for chunk in file_response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        lock["files"][filename] = {
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "source_url": url,
            "source_checksum": info.get("checksum"),
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