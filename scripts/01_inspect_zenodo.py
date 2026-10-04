import requests

URL = "https://zenodo.org/api/records/20416562"

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