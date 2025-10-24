from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import pandas as pd


@dataclass(frozen=True)
class SampleRecord:             # typed container for the fields
    filename: str
    path: str
    split: str
    index: str
    tumor: str
    plane: str
    sequence: str
    label: int


CLASS_ORDER: Sequence[str] = ("glioma", "meningioma", "no_tumor", "pituitary") # defines the class names 
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_ORDER)} # maps names to label integers


def parse_filename(filename: str) -> Dict[str, str]:    
    """
    Parse filenames like `brisc2025_train_00452_gl_co_t1.jpg` and return
    dictionary with split/index/tumor/plane/sequence information.
    """
    parts = filename.split("_")
    if len(parts) < 6:
        raise ValueError(f"Cannot parse filename structure: {filename}")

    split = parts[1]
    index = parts[2]
    tumor_code = parts[3]
    plane = parts[4]
    sequence = parts[5].split(".")[0]

    tumor_map = {
        "gl": "glioma",
        "me": "meningioma",
        "pi": "pituitary",
        "no": "no_tumor",
    }

    tumor_name = tumor_map.get(tumor_code)
    if tumor_name is None:
        raise KeyError(f"Unknown tumor code '{tumor_code}' in file {filename}")

    return dict(
        split=split,
        index=index,
        tumor=tumor_name,
        plane=plane,
        sequence=sequence,
    )


def iter_samples(dataset_root: Path) -> Iterable[SampleRecord]:
    classification_root = dataset_root / "classification_task"
    if not classification_root.exists():
        raise FileNotFoundError(f"Missing classification_task directory at {classification_root}")

    for split_dir in ("train", "test"):
        split_path = classification_root / split_dir
        if not split_path.exists():
            continue

        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_idx = CLASS_TO_INDEX.get(class_name)
            if class_idx is None:
                continue

            for image_path in class_dir.rglob("*.jpg"):
                filename = image_path.name
                parsed = parse_filename(filename)

                rel_path = image_path.relative_to(dataset_root).as_posix()
                yield SampleRecord(
                    filename=filename,
                    path=rel_path,
                    split=parsed["split"],
                    index=parsed["index"],
                    tumor=parsed["tumor"],
                    plane=parsed["plane"],
                    sequence=parsed["sequence"],
                    label=class_idx,
                )


def build_manifest(dataset_root: str | Path, output_csv: str | Path) -> Path:
    dataset_root = Path(dataset_root).resolve()
    records = list(iter_samples(dataset_root))
    if not records:
        raise RuntimeError(f"No samples found under {dataset_root}")

    df = pd.DataFrame(
        [r.__dict__ for r in records],
        columns=[
            "filename",
            "path",
            "split",
            "index",
            "tumor",
            "plane",
            "sequence",
            "label",
        ],
    )

    df = df.sort_values(["split", "tumor", "index"]).reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return output_path


if __name__ == "__main__":
    dataset_dir = Path("data/brisc2025")
    manifest_path = Path("data/manifests/classification_manifest.csv")
    written_path = build_manifest(dataset_root=dataset_dir, output_csv=manifest_path)
    print(f"Classification manifest written to {written_path}")
