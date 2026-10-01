"""Source-image grouping for the LC25000 lung images.

LC25000 was made by augmenting 250 original images per class (rotations and flips, Borkowski et al. 2019)
to 5,000 per class, so derivatives of one original must share a group. A perceptual-hash grouping misses most of them.

Method, independent of the models being benchmarked:
  1. SIFT keypoints on every image (grayscale, 384 px) and on its horizontal mirror (with rotation, a mirror
     covers every flip); descriptors are RootSIFT.
  2. Candidates: every descriptor of an image (both orientations) votes for the image holding its nearest
     descriptor (Lowe ratio test); image pairs with enough votes are candidates. The ratio test rejects
     descriptors with several identical copies, so each image's nearest neighbours by a small grayscale
     thumbnail, compared under all eight rotations by 90 degrees and flips, are added as candidates.
  3. Verification: RANSAC similarity transform (rotation + scale + translation) between the two keypoint
     sets; a pair with >= MIN_INLIERS inliers comes from the same original.
  4. Groups = connected components of the verified pairs, per class. Classes are never merged.
The "Test Set" folder is included so its overlap with "Train and Validation Set" can be reported.

Output: indexes/lc25000_source_groups.csv (path, class_name, folder, group), and
verification/lc25000_groups.json (counts, component sizes, inlier distribution, test-folder overlap).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE_DIR, CODE_DIR, atomic_write_json  # noqa: E402

ROOT = os.path.join(BASE_DIR, "dataset/TGCA/lung_colon_image_set")
FOLDERS = ["Train and Validation Set", "Test Set"]
CLASSES = ["lung_aca", "lung_scc"]
SIZE, N_FEATURES = 384, 300
RATIO, MIN_VOTES, MAX_CANDIDATES = 0.8, 6, 30
MIN_INLIERS = 15
THUMB, K_THUMB = 32, 20


def sift(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    thumb = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    det = cv2.SIFT_create(nfeatures=N_FEATURES)
    out = []
    for im in (img, img[:, ::-1].copy()):
        kp, d = det.detectAndCompute(im, None)
        if d is None:
            d, kp = np.zeros((0, 128), np.float32), []
        d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))  # RootSIFT
        out.append((np.array([k.pt for k in kp], np.float32).reshape(-1, 2), d.astype(np.float32)))
    return out, thumb


def unit(x):
    x = x.reshape(len(x), -1)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def thumb_candidates(thumbs, device, k=K_THUMB, chunk=512):
    """Pairs (a, b), a < b: b among the k most similar images to a (Pearson correlation of thumbnails, maximum
    over the eight rotations by 90 degrees and flips of a)."""
    n = len(thumbs)
    base = torch.from_numpy(unit(thumbs)).to(device)
    dih = np.stack([np.stack([np.rot90(t, r)[:, ::-1] if f else np.rot90(t, r) for r in range(4) for f in (0, 1)])
                    for t in thumbs])  # n, 8, T, T
    pairs = set()
    for a in range(0, n, chunk):
        q = torch.from_numpy(unit(dih[a:a + chunk].reshape(-1, THUMB, THUMB))).to(device)
        s = (q @ base.T).reshape(-1, 8, n).amax(1)
        s[torch.arange(len(s)), torch.arange(a, a + len(s))] = -2.0
        nb = s.topk(min(k, n - 1), dim=1).indices.cpu().numpy()
        for i, row in enumerate(nb):
            pairs.update((min(a + i, b), max(a + i, b)) for b in row)
    return pairs


def nearest_two(q, db, owner_q, owner_db, device, qchunk=4096, dchunk=262144):
    """For each query descriptor: the two nearest db descriptors belonging to another image (L2 on unit vectors)."""
    best_s = torch.full((len(q), 2), -2.0, device=device)
    best_i = torch.full((len(q), 2), -1, dtype=torch.long, device=device)
    dbt = torch.from_numpy(db).to(device, torch.float16)
    own_db = torch.from_numpy(owner_db).to(device)
    for a in range(0, len(q), qchunk):
        qa = torch.from_numpy(q[a:a + qchunk]).to(device, torch.float16)
        own_q = torch.from_numpy(owner_q[a:a + qchunk]).to(device)
        s_run, i_run = best_s[a:a + qchunk], best_i[a:a + qchunk]
        for b in range(0, len(db), dchunk):
            s = (qa @ dbt[b:b + dchunk].T).float()
            s[own_q[:, None] == own_db[None, b:b + dchunk]] = -2.0
            s2, i2 = s.topk(2, dim=1)
            cat_s, cat_i = torch.cat([s_run, s2], 1), torch.cat([i_run, i2 + b], 1)
            top = cat_s.topk(2, dim=1)
            s_run, i_run = top.values, cat_i.gather(1, top.indices)
        best_s[a:a + qchunk], best_i[a:a + qchunk] = s_run, i_run
    # cosine -> L2 distance for unit vectors
    dist = torch.sqrt(torch.clamp(2 - 2 * best_s, min=0))
    return dist.cpu().numpy(), best_i.cpu().numpy()


def verify(args):
    (pa, da), variants_b = args
    best = 0
    for pb, db in variants_b:  # original and mirrored orientation of image b
        if len(da) < 3 or len(db) < 3:
            continue
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < RATIO * x[1].distance]
        if len(good) < 3:
            continue
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                             maxIters=2000, confidence=0.999)
        if inl is not None:
            best = max(best, int(inl.sum()))
    return best


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, report = [], {"method": __doc__.split("Method")[1].split("Output")[0].strip(),
                        "params": dict(size=SIZE, n_features=N_FEATURES, ratio=RATIO, min_votes=MIN_VOTES,
                                       max_candidates=MAX_CANDIDATES, min_inliers=MIN_INLIERS,
                                       thumb_size=THUMB, thumb_neighbours=K_THUMB,
                                       opencv=cv2.__version__), "classes": {}}
    for cls in CLASSES:
        t0 = time.time()
        paths, folders = [], []
        for folder in FOLDERS:
            d = os.path.join(ROOT, folder, cls)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpeg", ".jpg", ".png")):
                    paths.append(os.path.join(d, f)); folders.append(folder)
        n = len(paths)
        with Pool(16) as pool:
            res = pool.map(sift, paths, chunksize=32)
        feats, thumbs = [r[0] for r in res], np.stack([r[1] for r in res])
        print(f"{cls}: {n} images, SIFT {time.time() - t0:.0f}s", flush=True)

        # database = original orientation; queries = both orientations
        db = np.concatenate([f[0][1] for f in feats])
        owner_db = np.concatenate([np.full(len(f[0][1]), i) for i, f in enumerate(feats)])
        q = np.concatenate([f[v][1] for f in feats for v in (0, 1)])
        owner_q = np.concatenate([np.full(len(f[v][1]), i) for i, f in enumerate(feats) for v in (0, 1)])
        dist, idx = nearest_two(q, db, owner_q, owner_db, device)
        ok = dist[:, 0] < RATIO * dist[:, 1]
        votes = pd.DataFrame({"a": owner_q[ok], "b": owner_db[idx[ok, 0]]})
        votes = votes.assign(a2=np.minimum(votes.a, votes.b), b2=np.maximum(votes.a, votes.b))
        counts = votes.groupby(["a2", "b2"]).size().rename("v").reset_index()
        counts = counts[counts.v >= MIN_VOTES].sort_values("v", ascending=False)
        counts = pd.concat([counts.groupby("a2").head(MAX_CANDIDATES),
                            counts.groupby("b2").head(MAX_CANDIDATES)]).drop_duplicates()
        sift_pairs = {tuple(p) for p in counts[["a2", "b2"]].values.tolist()}
        thumb_pairs = thumb_candidates(thumbs, device)
        pairs = np.array(sorted(sift_pairs | thumb_pairs), dtype=np.int64).reshape(-1, 2)
        print(f"{cls}: {len(pairs)} candidate pairs ({len(sift_pairs)} keypoint votes, {len(thumb_pairs)} thumbnails, "
              f"{len(thumb_pairs - sift_pairs)} new), {time.time() - t0:.0f}s", flush=True)

        with Pool(16) as pool:
            inliers = np.array(pool.map(verify, [(feats[a][0], feats[b]) for a, b in pairs], chunksize=64))
        edges = pairs[inliers >= MIN_INLIERS]
        comp = components(n, edges)
        _, gid = np.unique(comp, return_inverse=True)
        sizes = np.bincount(gid)
        folder_arr = np.array(folders)
        test_groups = set(gid[folder_arr == "Test Set"])
        train_groups = set(gid[folder_arr != "Test Set"])
        report["classes"][cls] = {
            "images": n, "images_per_folder": {f: int((folder_arr == f).sum()) for f in FOLDERS},
            "candidate_pairs": int(len(pairs)), "candidate_pairs_keypoints": len(sift_pairs),
            "candidate_pairs_thumbnail_only": len(thumb_pairs - sift_pairs), "verified_pairs": int(len(edges)), "groups": int(len(sizes)),
            "group_size_quantiles": {str(p): float(np.percentile(sizes, p)) for p in (0, 5, 25, 50, 75, 95, 100)},
            "singletons": int((sizes == 1).sum()),
            "inlier_histogram": {f"{lo}-{hi - 1}": int(((inliers >= lo) & (inliers < hi)).sum())
                                 for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100), (100, 10**6)]},
            "test_folder_groups": int(len(test_groups)),
            "test_folder_groups_also_in_train_val": int(len(test_groups & train_groups)),
            "test_folder_images_in_shared_groups": int(np.isin(gid[folder_arr == "Test Set"], list(train_groups)).sum()),
            "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps({cls: report["classes"][cls]}, indent=1), flush=True)
        rel = [os.path.relpath(p, BASE_DIR) for p in paths]
        rows += [dict(path=r, class_name=cls, folder=f, group=f"{cls}_src{g:04d}") for r, f, g in zip(rel, folders, gid)]

    out = os.path.join(CODE_DIR, "indexes", "lc25000_source_groups.csv")
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    atomic_write_json(os.path.join(CODE_DIR, "verification", "lc25000_groups.json"), report)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
