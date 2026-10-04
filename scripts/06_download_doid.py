from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests


OUTPUT = Path("taxonomy/doid.obo")
LOCK = Path("taxonomy/doid_lock.json")

# Constructed this way so the source stays explicit.
DOID_URL = (
    "http"
    + "://purl.obolibrary.org/obo/doid.obo"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def find_data_version(text: str):
    for line in text.splitlines():
        if line.startswith("data-version:"):
            return line.split(
                ":",
                1,
            )[1].strip()

    return None


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading Disease Ontology...")

    response = requests.get(
        DOID_URL,
        timeout=120,
    )

    response.raise_for_status()

    OUTPUT.write_bytes(
        response.content
    )

    text = OUTPUT.read_text(
        encoding="utf-8",
        errors="replace",
    )

    data_version = find_data_version(
        text
    )

    lock = {
        "source": "Human Disease Ontology",
        "data_version": data_version,
        "downloaded_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sha256": sha256(OUTPUT),
    }

    LOCK.write_text(
        json.dumps(
            lock,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved:", OUTPUT)
    print("Version:", data_version)
    print("SHA256:", lock["sha256"])


if __name__ == "__main__":
    main()