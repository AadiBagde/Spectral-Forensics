
from pathlib import Path
import csv
import hashlib
import io

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "raw" / "genimage"
MANIFEST = ROOT / "datasets" / "manifests" / "image_manifest.csv"

DATASET_ID = "nebula/GenImage-arrow"

QUOTAS = {
    "train": 2000,  # per class (4,000 total)
    "test": 1000,   # per class (2,000 total)
}


def get_image(value):
    if isinstance(value, Image.Image):
        return value.copy()

    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).copy()

    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).copy()
        if value.get("path"):
            return Image.open(value["path"]).copy()

    raise ValueError(f"Unsupported image format: {type(value)}")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_split(split, quota, writer, seen_hashes):
    config = f"adm-{split}"

    print(f"\nOpening {config} in streaming mode...")
    dataset = load_dataset(
        DATASET_ID,
        config,
        split=split,
        streaming=True,
    )

    dataset = dataset.shuffle(seed=42, buffer_size=2000)
    counts = {0: 0, 1: 0}

    for row in tqdm(dataset, desc=split):
        label = int(row["label"])

        if label not in counts:
            continue

        if counts[label] >= quota:
            if all(counts[k] >= quota for k in counts):
                break
            continue

        try:
            image = get_image(row["image"]).convert("RGB")

            # Deduplicate by decoded pixel content.
            pixel_hash = hashlib.sha256(
                image.tobytes()
            ).hexdigest()

            if pixel_hash in seen_hashes:
                continue

            seen_hashes.add(pixel_hash)

            class_name = "real" if label == 0 else "ai"
            folder = OUT / split / class_name
            folder.mkdir(parents=True, exist_ok=True)

            filename = f"adm_{counts[label]:05d}.png"
            path = folder / filename

            image.save(path, format="PNG", optimize=True)

            writer.writerow({
                "image_path": path.relative_to(ROOT).as_posix(),
                "label": class_name,
                "generator": (
                    "real_source" if label == 0
                    else str(row.get("generator", "ADM"))
                ),
                "source_dataset": DATASET_ID,
                "source_config": config,
                "split": split,
                "width": image.width,
                "height": image.height,
                "sha256": sha256(path),
            })

            counts[label] += 1

        except Exception as exc:
            print(f"\nSkipping one record: {exc}")

        if all(counts[k] >= quota for k in counts):
            break

    print(f"{split} counts: {counts}")

    if any(counts[k] < quota for k in counts):
        print("WARNING: requested quota was not reached.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_path", "label", "generator",
        "source_dataset", "source_config", "split",
        "width", "height", "sha256",
    ]

    # Prevent an accidental rerun from mixing old and new manifests.
    if any(OUT.iterdir()):
        raise FileExistsError(
            f"{OUT} is not empty. Move or remove the old dataset "
            "before running this fresh-export script again."
        )

    seen_hashes = set()

    with MANIFEST.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        export_split("train", QUOTAS["train"], writer, seen_hashes)
        export_split("test", QUOTAS["test"], writer, seen_hashes)

    print(f"\nManifest saved at: {MANIFEST}")
    print(f"Dataset exported to: {OUT}")


if __name__ == "__main__":
    main()