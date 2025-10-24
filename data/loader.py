## funzioni di caricamento del dataset + preprocessing
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import pandas as pd
import os 
from torchvision.transforms import transforms 
import glob
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from sklearn.model_selection import train_test_split

BASE_DIR = "data/brisc2025/"

def transform_filenames_to_paths(filenames):
    """
    Transform filenames like:
      'brisc2025_train_00452_gl_co_t1.jpg'
    into:
      'train/glioma/brisc2025_train_00452_gl_co_t1.jpg'
    keeping the full filename intact.
    """
    class_map = {
        'gl': 'glioma',
        'pi': 'pituitary',
        'me': 'meningioma',
        'no': 'no_tumor'
    }

    transformed = []
    for name in filenames:
        parts = name.split('_')
        if len(parts) < 5:
            continue  # skip malformed names

        split = parts[1]        # e.g., 'train'
        class_code = parts[3]   # e.g., 'gl'
        class_name = class_map.get(class_code, 'unknown')

        path = os.path.join(split, class_name, name)
        transformed.append(path)

    return transformed

def check_image_shapes(folder_path, target_shape=(3, 512, 512)):
    """
    Checks all .jpg images in a folder (and subfolders) for shape mismatches.

    Args:
        folder_path (str): Path to the dataset directory.
        target_shape (tuple): Expected (C, H, W) shape.

    Returns:
        list: List of (filepath, actual_shape) for mismatched images.
    """
    transform = transforms.ToTensor()
    mismatched = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = transform(img)
                    if tensor.shape != target_shape:
                        mismatched.append((path, tuple(tensor.shape)))
                except Exception as e:
                    print(f"⚠️ Error reading {path}: {e}")

    if mismatched:
        print(f"\nFound {len(mismatched)} images with shape different from {target_shape}:")
        for path, shape in mismatched[:10]:  # show first 10 only
            print(f" - {path}: {shape}")
        if len(mismatched) > 10:
            print(f"...and {len(mismatched)-10} more.")
    else:
        print(f"✅ All images match {target_shape}.")

    return mismatched

def show_filenames_by_filters(df, plane=None, tumor=None, split=None):
    """
    Filter and display filenames from df_meta based on plane, tumor, and split.

    Args:
        df (pd.DataFrame): The metadata dataframe.
        plane (str, optional): Plane filter ('ax', 'co', 'sa').
        tumor (str, optional): Tumor type filter ('glioma', 'meningioma', 'pituitary', 'no_tumor').
        split (str, optional): Data split ('train' or 'test').

    Returns:
        pd.Series: The list of matching filenames.
    """
    filtered = df.copy()

    if plane:
        filtered = filtered[filtered['Plane'] == plane]
    if tumor:
        filtered = filtered[filtered['Tumor'] == tumor]
    if split:
        filtered = filtered[filtered['Split'] == split]

    if filtered.empty:
        print("⚠️ No images found for the given filters.")
        return None

    print(f"✅ Found {len(filtered)} image(s) matching your filters:")
    #display(filtered[['Filename', 'Split', 'Tumor', 'Plane']])

    return filtered['Filename']

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        """
        image_paths: list of image file paths
        labels: list or tensor of integer class labels
        transform: torchvision transforms to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
    
def get_df(dir, splits, classes):
    meta = []
    for sp in splits:
        for cls in classes:
            files = glob.glob(os.path.join(BASE_DIR, "classification_task", sp, cls, "*.jpg"))
            for f in files:
                parts = os.path.basename(f).split("_")
                meta.append({
                    "Filename": os.path.basename(f),
                    "Split": parts[1],
                    "Index": parts[2],
                    "Tumor": parts[3],
                    "Plane": parts[4],
                    "Sequence": parts[5].split(".")[0]
                })
    df_meta = pd.DataFrame(meta)
    tumor_map = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": "no_tumor"
    }

    # Apply mapping to the DataFrame
    df_meta["Tumor"] = df_meta["Tumor"].map(tumor_map)
    return df_meta

def get_dataset(train_paths,
                eval_paths,
                test_paths,
                train_labels,
                eval_labels,
                test_labels,
                train_transform=None,
                eval_transform=None):

    imagenet_stats = dict(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.05,
                    hue=0.02)],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats),
        ])

    if eval_transform is None:
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats),
        ])

    train_set = ImagePathDataset(train_paths, train_labels, train_transform)
    eval_set = ImagePathDataset(eval_paths, eval_labels, eval_transform)
    test_set = ImagePathDataset(test_paths, test_labels, eval_transform)
    
    return train_set, eval_set, test_set


def _ensure_list(value: Union[pd.Series, List[str]]) -> List[str]:
    if isinstance(value, pd.Series):
        return value.tolist()
    return list(value)


def _resolve_sample_paths(paths: List[str], base_dir: Optional[Union[str, os.PathLike]]) -> List[str]:
    if base_dir is None:
        return [str(Path(p)) for p in paths]

    root = Path(base_dir)
    resolved = []
    for p in paths:
        path_obj = Path(p)
        if path_obj.is_absolute():
            resolved.append(str(path_obj))
        else:
            resolved.append(str(root / path_obj))
    return resolved


def load_manifest_dataframe(manifest_path: Union[str, os.PathLike],
                            path_column: str = "path",
                            label_column: str = "label",
                            split_column: str = "split") -> pd.DataFrame:
    """
    Load a CSV manifest describing samples and ensure required columns exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required_columns = {path_column, label_column, split_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing required columns: {missing}")
    return df


def _append_stratification_column(
    df: pd.DataFrame,
    label_column: str,
    stratify_columns: Optional[Sequence[str]],
) -> pd.DataFrame:
    """
    Ensure the manifest DataFrame carries a `_strata` column for stratified splits.
    """
    df = df.copy()

    if stratify_columns:
        missing = [col for col in stratify_columns if col not in df.columns]
        if missing:
            combined = df[label_column].astype(str)
        else:
            combined = df[stratify_columns[0]].astype(str)
            for col in stratify_columns[1:]:
                combined = combined + "_" + df[col].astype(str)
    else:
        combined = df[label_column].astype(str)

    df["_strata"] = combined
    return df


def split_manifest_dataframe(
    manifest_path: Union[str, os.PathLike],
    train_split_value: str = "train",
    test_split_value: str = "test",
    val_fraction: float = 0.2,
    stratify_columns: Optional[Sequence[str]] = ("tumor", "plane"),
    random_state: int = 42,
    path_column: str = "path",
    label_column: str = "label",
    split_column: str = "split",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a manifest CSV and return train/val/test DataFrames with optional stratification.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in the interval [0.0, 1.0).")

    df = load_manifest_dataframe(
        manifest_path,
        path_column=path_column,
        label_column=label_column,
        split_column=split_column,
    )
    df = _append_stratification_column(df, label_column, stratify_columns)

    train_df = df[df[split_column] == train_split_value].copy()
    test_df = df[df[split_column] == test_split_value].copy()

    if val_fraction and val_fraction > 0.0 and len(train_df) > 0:
        stratify_arg = train_df["_strata"]
        # If stratification cannot be honoured (e.g., only one member), fall back gracefully.
        try:
            train_subset, val_subset = train_test_split(
                train_df,
                test_size=val_fraction,
                stratify=stratify_arg,
                random_state=random_state,
            )
        except ValueError:
            train_subset, val_subset = train_test_split(
                train_df,
                test_size=val_fraction,
                stratify=None,
                random_state=random_state,
            )
    else:
        train_subset = train_df
        val_subset = train_df.iloc[0:0].copy()

    return (
        train_subset.reset_index(drop=True),
        val_subset.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_datasets_from_manifest(
    manifest_path: Union[str, os.PathLike],
    base_dir: Optional[Union[str, os.PathLike]] = None,
    train_split_value: str = "train",
    test_split_value: str = "test",
    val_fraction: float = 0.2,
    stratify_columns: Optional[Sequence[str]] = ("tumor", "plane"),
    random_state: int = 42,
    train_transform=None,
    eval_transform=None,
    path_column: str = "path",
    label_column: str = "label",
    split_column: str = "split",
) -> Tuple[ImagePathDataset, ImagePathDataset, ImagePathDataset]:
    """
    Build ImagePathDataset objects (train/val/test) from a manifest CSV.
    """
    df = load_manifest_dataframe(
        manifest_path,
        path_column=path_column,
        label_column=label_column,
        split_column=split_column,
    )
    df = _append_stratification_column(df, label_column, stratify_columns)
    strata_key = "_strata"

    train_df = df[df[split_column] == train_split_value]
    test_df = df[df[split_column] == test_split_value]

    if val_fraction and val_fraction > 0.0 and len(train_df) > 0:
        train_subset, val_subset = train_test_split(
            train_df,
            test_size=val_fraction,
            stratify=train_df[strata_key],
            random_state=random_state,
        )
    else:
        train_subset = train_df
        val_subset = train_df.iloc[0:0]

    def _subset_to_lists(subset: pd.DataFrame) -> Tuple[List[str], List[int]]:
        subset_paths = _ensure_list(subset[path_column])
        resolved_paths = _resolve_sample_paths(subset_paths, base_dir)
        subset_labels = [int(label) for label in subset[label_column]]
        return resolved_paths, subset_labels

    train_paths, train_labels = _subset_to_lists(train_subset)
    val_paths, val_labels = _subset_to_lists(val_subset)
    test_paths, test_labels = _subset_to_lists(test_df)

    train_set, val_set, test_set = get_dataset(
        train_paths=train_paths,
        eval_paths=val_paths,
        test_paths=test_paths,
        train_labels=train_labels,
        eval_labels=val_labels,
        test_labels=test_labels,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    return train_set, val_set, test_set
