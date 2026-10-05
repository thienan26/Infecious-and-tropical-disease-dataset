from pathlib import Path

import requests
import yaml


with Path(
    "configs/source.yaml"
).open(
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(f)


URL = config[
    "zenodo"
][
    "api_url"
]

response = requests.get(URL, timeout=60)
response.raise_for_status()

record = response.json()

print("Title:")
print(record["metadata"]["title"])
print()

print("Files:")
for item in record["files"]:
    print(
        f"{item['key']:40} "
        f"{item['size'] / 1024 / 1024:.2f} MB"
    )