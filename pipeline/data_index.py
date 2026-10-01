"""One index per dataset: every image used, its label, grouping unit and official partition.

The index is the single source of truth for extraction (row order = feature order) and for
splitting. Rows are sorted by path, so the order does not depend on the filesystem.

Columns: path (relative to BASE_DIR), label, class_name, group, partition, plus dataset extras.
  group      unit that must not be split across train/val/test ('' when none is available)
  partition  official partition (train/val/test) or '' when the dataset has none

Usage: python pipeline/data_index.py [dataset ...]
"""
import io
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, DATASET_DIR, INDEX_DIR, atomic_write_bytes, sha256_strings

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _rel(p):
    return os.path.relpath(p, BASE_DIR)


def _walk_images(root):
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(IMG_EXT):
                yield r, f


def index_lung():
    # LC25000 lung subset (adenocarcinoma vs squamous), "Train and Validation Set" folder only.
    # group = source image, from SIFT + RANSAC matching (pipeline/lc25000_groups.py).
    root = os.path.join(DATASET_DIR, "TGCA/lung_colon_image_set/Train and Validation Set")
    src = pd.read_csv(os.path.join(INDEX_DIR, "lc25000_source_groups.csv"))
    src = src[src.folder == "Train and Validation Set"]
    source = dict(zip(zip(src.class_name, src.path.map(os.path.basename)), src.group))
    classes = {"lung_aca": 0, "lung_scc": 1}
    rows = []
    for name, label in classes.items():
        for _, f in _walk_images(os.path.join(root, name)):
            rows.append(dict(path=_rel(os.path.join(root, name, f)), label=label, class_name=name,
                             group=source[(name, f)], partition=""))
    return rows


def index_breakhis():
    # Filename: SOB_<B|M>_<type>-<year>-<slide id>-<mag>-<seq>. The slide id carries letter suffixes
    # (22549AB, 22549CD) and one number can appear under two tumour types (13412 DC and LC), so the
    # conservative grouping unit is the numeric patient number.
    root = os.path.join(DATASET_DIR, "BreakHis - Breast Cancer Histopathological Database/"
                        "dataset_cancer_v1/dataset_cancer_v1/classificacao_binaria")
    rows = []
    for r, f in _walk_images(root):
        low = r.lower()
        if "benign" in low:
            label, cname = 0, "benign"
        elif "malignant" in low:
            label, cname = 1, "malignant"
        else:
            continue
        parts = os.path.splitext(f)[0].split("-")
        slide = "-".join(parts[:3])
        patient = re.match(r"\d+", parts[2]).group(0)
        rows.append(dict(path=_rel(os.path.join(r, f)), label=label, class_name=cname, group=patient,
                         slide=slide, tumour_type=parts[0].split("_")[-1], magnification=parts[3], partition=""))
    return rows


def _index_class_folders(root, group_fn, partition=""):
    rows = []
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for label, cname in enumerate(classes):
        for r, f in _walk_images(os.path.join(root, cname)):
            rows.append(dict(path=_rel(os.path.join(r, f)), label=label, class_name=cname,
                             group=group_fn(f), partition=partition))
    return rows


def index_nct():
    # No patient/slide identifiers in the public release: group is empty (patch-level split, declared).
    return _index_class_folders(os.path.join(DATASET_DIR, "NCT-CRC-HE-100K"), lambda f: "")


def index_nct_val7k():
    # CRC-VAL-HE-7K: independent patients, the official external test set for NCT-CRC-HE-100K.
    root = os.path.join(DATASET_DIR, "CRC-VAL-HE-7K")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found: download CRC-VAL-HE-7K (Zenodo 1214456) first")
    return _index_class_folders(root, lambda f: "", partition="test")


def index_hubmap():
    # Tiles named <source image id>_<x>_<y>.tiff; class = organ.
    return _index_class_folders(os.path.join(DATASET_DIR, "HUBMAP_TILED_TIFF"), lambda f: f.split("_")[0])


def index_bach():
    # Tiles named <image id>_<x>_<y>.tiff (n001, b001, is001, iv001); no patient ids are public.
    # Folders 0..3 = normal < benign < in situ < invasive.
    rows = _index_class_folders(os.path.join(DATASET_DIR, "BACH/TILED_TIFF"), lambda f: f.split("_")[0])
    names = {"0": "normal", "1": "benign", "2": "in_situ", "3": "invasive"}
    for row in rows:
        row["class_name"] = names[row["class_name"]]
    return rows


def index_panda():
    # Tiles named <slide id>_<x>_<y>.jpg in a folder per slide ISUP grade (0-5); every slide is in one folder.
    root = os.path.join(DATASET_DIR, "PANDA_TILED_TIFF")
    rows = []
    for grade in sorted(os.listdir(root)):
        for f in os.listdir(os.path.join(root, grade)):
            if not f.lower().endswith(IMG_EXT):
                continue
            slide, x, y = os.path.splitext(f)[0].split("_")
            rows.append(dict(path=_rel(os.path.join(root, grade, f)), label=int(grade), class_name=f"isup{grade}",
                             group=slide, x=int(x), y=int(y), partition=""))
    return rows


def index_bracs():
    # Official RoI partition (train/val/test folders); group = patient id from BRACS.xlsx.
    # Folder order 0_N < 1_PB < 2_UDH < 3_FEA < 4_ADH < 5_DCIS < 6_IC.
    root = os.path.join(DATASET_DIR, "BRACS/histoimage.na.icar.cnr.it/BRACS_RoI/latest_version")
    wsi = pd.read_excel(os.path.join(DATASET_DIR, "BRACS/BRACS.xlsx"), sheet_name="WSI_Information")
    patient = dict(zip(wsi["WSI Filename"], wsi["Patient Id"]))
    three = {"N": "benign", "PB": "benign", "UDH": "benign", "FEA": "atypical", "ADH": "atypical",
             "DCIS": "malignant", "IC": "malignant"}
    rows = []
    for part in ["train", "val", "test"]:
        for r in _index_class_folders(os.path.join(root, part), lambda f: "", partition=part):
            f = os.path.basename(r["path"])
            w = "_".join(f.split("_")[:2])
            r["wsi"] = w
            r["group"] = str(patient[w])
            r["label_3class"] = ["benign", "atypical", "malignant"].index(three[r["class_name"].split("_", 1)[1]])
            rows.append(r)
    return rows


def index_sicap():
    # Official partition from sicapv2_labels/{Train,Val,Test}.xlsx (slide-disjoint, verified).
    # G4C (cribriform) is always co-labelled with G4 and is collapsed into G4: NC < G3 < G4 < G5.
    rows = []
    names = ["NC", "G3", "G4", "G5"]
    for part, fname in [("train", "Train.xlsx"), ("val", "Val.xlsx"), ("test", "Test.xlsx")]:
        df = pd.read_excel(os.path.join(DATASET_DIR, "SICAPv2/sicapv2_labels", fname))
        vals = np.stack([df.NC, df.G3, df.G4 + df.G4C, df.G5], axis=1)
        if ((vals > 0).sum(axis=1) != 1).any():
            raise ValueError(f"SICAPv2 {fname}: rows without exactly one grade")
        for name, label, g4c in zip(df.image_name, vals.argmax(axis=1), df.G4C):
            path = os.path.join(DATASET_DIR, "SICAPv2/sicapv2_patches", name)
            rows.append(dict(path=_rel(path), label=int(label), class_name=names[label],
                             group=name.split("_")[0], cribriform=int(g4c), partition=part))
    return rows


DATASETS = {
    "lung": index_lung, "breakhis": index_breakhis, "nct": index_nct, "nct_val7k": index_nct_val7k,
    "hubmap": index_hubmap, "bach": index_bach, "panda": index_panda, "bracs": index_bracs, "sicap": index_sicap,
}


def build_index(name):
    df = pd.DataFrame(DATASETS[name]()).sort_values("path", kind="stable").reset_index(drop=True)
    if df.path.duplicated().any():
        raise ValueError(f"{name}: duplicate paths in index")
    missing = [p for p in df.path if not os.path.isfile(os.path.join(BASE_DIR, p))]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} indexed files missing, e.g. {missing[:3]}")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    atomic_write_bytes(os.path.join(INDEX_DIR, f"{name}.csv"), buf.getvalue().encode())
    return df


def load_index(name):
    path = os.path.join(INDEX_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return build_index(name)
    return pd.read_csv(path, keep_default_na=False, dtype={"group": str, "partition": str})


def index_hash(df):
    return sha256_strings(df.path.tolist())


if __name__ == "__main__":
    for name in sys.argv[1:] or [n for n in DATASETS if n != "nct_val7k"]:
        df = build_index(name)
        parts = df.partition.replace("", "-").value_counts().to_dict()
        print(f"{name:10s} {len(df):7d} images | classes {df.label.nunique()} | "
              f"groups {df.group.replace('', np.nan).nunique()} | partitions {parts} | hash {index_hash(df)[:12]}")
