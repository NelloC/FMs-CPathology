"""pipeline/report.py on small synthetic result trees (no real results, no GPU)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import report

METRICS = report.STATS_METRICS  # order of the last axis of boot_<dataset>.npz
SEEDS = [42, 43, 44]


def _seed_json(root, model, dataset, seed, test):
    d = os.path.join(root, model, dataset)
    os.makedirs(d, exist_ok=True)
    json.dump({"model": model, "dataset": dataset, "seed": seed, "test": test}, open(os.path.join(d, f"seed{seed}.json"), "w"))


def _test_metrics(rng, base):
    return {k: float(np.clip(base + rng.normal(0, 0.01), 0, 1)) for k in ["acc", "bal_acc", "f1_weighted", "f1_macro", "auc", "qwk"]}


def build_tree(tmp_path, with_stats=True, stale=False):
    """probe: vit_in1k, phikon, uni2h on lung (3 seeds, random) and sicap (official: identical seeds);
    hoptimus0 only on lung (a model missing a dataset); no BRACS, no PANDA, no ABMIL, no external."""
    res = tmp_path / "results"
    probe = res / "probe"
    rng = np.random.default_rng(0)
    base = {"vit_in1k": 0.80, "phikon": 0.90, "uni2h": 0.97, "hoptimus0": 0.95}
    truth = {}
    for m, b in base.items():
        for s in SEEDS:
            t = _test_metrics(rng, b)
            _seed_json(probe, m, "lung", s, t)
            truth[(m, "lung", s)] = t
    sicap = {m: _test_metrics(rng, base[m] - 0.1) for m in ["vit_in1k", "phikon", "uni2h"]}
    sicap["phikon"]["qwk"] = sicap["uni2h"]["qwk"] - 0.001  # tie with the best
    for m, t in sicap.items():
        for s in SEEDS:
            _seed_json(probe, m, "sicap", s, t)
            truth[(m, "sicap", s)] = t
    if not with_stats:
        return res, truth
    st = res / "stats"
    st.mkdir(parents=True)
    rows, pw = [], []
    models = ["vit_in1k", "phikon", "uni2h", "hoptimus0"]
    for m in models:
        v = np.array([truth[(m, "lung", s)]["auc"] for s in SEEDS])
        mean = v.mean() + (0.01 if stale and m == "phikon" else 0)
        rows.append(dict(dataset="lung", model=m, metric="auc", primary=True, mean=mean, sd=v.std(ddof=1),
                         ci_low=v.mean() - 0.02, ci_high=min(1, v.mean() + 0.02), n_seeds=3))
    for m in ["vit_in1k", "phikon", "uni2h"]:  # stats.py's (too narrow) seed-averaged CI for the official partition
        q = sicap[m]["qwk"]
        rows.append(dict(dataset="sicap", model=m, metric="qwk", primary=True, mean=q, sd=0.0, ci_low=q - 0.001,
                         ci_high=q + 0.001, n_seeds=3))
    pd.DataFrame(rows).to_csv(st / "summary.csv", index=False)
    # lung: uni2h significantly better than every other model
    for a, b in [("vit_in1k", "phikon"), ("vit_in1k", "uni2h"), ("vit_in1k", "hoptimus0"), ("phikon", "uni2h"),
                 ("phikon", "hoptimus0"), ("uni2h", "hoptimus0")]:
        pw.append(dict(dataset="lung", metric="auc", model_a=a, model_b=b, diff=0.0, ci_low=0, ci_high=0, p=0.001, p_holm=0.004))
    # sicap in pairwise.csv says 'all significant' (seed-averaged); the single-seed recomputation must say phikon ~ uni2h
    for a, b in [("vit_in1k", "phikon"), ("vit_in1k", "uni2h"), ("phikon", "uni2h")]:
        pw.append(dict(dataset="sicap", metric="qwk", model_a=a, model_b=b, diff=0.0, ci_low=0, ci_high=0, p=0.001, p_holm=0.003))
    pd.DataFrame(pw).to_csv(st / "pairwise.csv", index=False)
    # boot_sicap.npz: [B+1, seeds, models, metrics], identical point estimates over seeds
    B, ms = 400, ["phikon", "vit_in1k", "uni2h"]
    boot = np.empty((B + 1, len(SEEDS), len(ms), len(METRICS)))
    r = np.random.default_rng(1)
    shared = r.normal(0, 0.03, (B, len(SEEDS)))
    for mi, m in enumerate(ms):
        for ki, k in enumerate(METRICS):
            boot[0, :, mi, ki] = sicap[m][k]
            boot[1:, :, mi, ki] = sicap[m][k] + shared + r.normal(0, 0.002 if m != "vit_in1k" else 0.01, (B, len(SEEDS)))
    np.savez(st / "boot_sicap.npz", boot=boot, models=np.array(ms), seeds=np.array(SEEDS), n_boot=B,
             metrics=np.array(METRICS), input_hash="0" * 16)
    ranks = []
    for d, mm in [("lung", models), ("sicap", ["vit_in1k", "phikon", "uni2h"])]:
        for i, m in enumerate(mm):
            p = np.zeros(len(models))
            p[i] = 1.0
            ranks.append(dict(dataset=d, model=m, metric=report.primary(d), mean_rank=i + 1.0,
                              **{f"p_rank{j + 1}": p[j] for j in range(len(models))}))
    pd.DataFrame(ranks).to_csv(st / "ranks.csv", index=False)
    json.dump({"all": {"datasets": ["lung", "sicap"], "chi2": 6.0, "p": 0.0498, "nemenyi_cd": 1.91,
                       "mean_rank": {"vit_in1k": 3.0, "phikon": 2.0, "uni2h": 1.0}}}, open(st / "friedman.json", "w"))
    return res, truth


def run(res, tmp_path, **kw):
    out = tmp_path / "generated"
    rep = report.Report(str(res), str(out), **kw)
    rep.run()
    return rep, out, json.load(open(out / "summary.json"))


def cell_row(tex, model):
    """The table row of one model (display name at the start of the line)."""
    name = report.MODEL_NAMES[model]
    rows = [l for l in tex.splitlines() if l.startswith(name + " &")]
    assert len(rows) == 1, (model, rows)
    return rows[0].rstrip(" \\").split(" & ")[1:]


def test_missing_everything_does_not_fail(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    rep, out, summ = run(res, tmp_path)
    assert (out / "summary.json").exists() and (out / "macros.tex").exists()
    text = "\n".join(summ["skipped"])
    for what in ["results_categorical", "results_ordinal", "ABMIL", "precision", "sensitivity", "external"]:
        assert what in text


def test_main_tables_numbers_equal_source_json(tmp_path):
    res, truth = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    tex = open(out / "tables" / "results_categorical.tex").read()
    for m in ["vit_in1k", "phikon", "uni2h", "hoptimus0"]:
        v = np.array([truth[(m, "lung", s)]["auc"] for s in SEEDS])
        (cell,) = cell_row(tex, m)
        assert f"{v.mean():.4f}$\\pm${v.std(ddof=1):.4f}" in cell
    # summary.json records the same numbers with their source files
    cells = [c for c in summ["tables"]["results_categorical"]["cells"] if c["quantity"] == "auc mean"]
    for c in cells:
        v = np.mean([truth[(c["row"], "lung", s)]["auc"] for s in SEEDS])
        assert c["value"] == pytest.approx(v, abs=1e-12)
        assert len(c["sources"]) == 3 and all(s.startswith("results/probe/") for s in c["sources"])
    # CI from stats summary.csv for seed-split datasets
    s = pd.read_csv(res / "stats" / "summary.csv")
    row = s[(s.dataset == "lung") & (s.model == "phikon")].iloc[0]
    assert f"[{row.ci_low:.4f}, {row.ci_high:.4f}]" in cell_row(tex, "phikon")[0]


def test_official_partition_single_value_and_single_seed_ci(tmp_path):
    res, truth = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    tex = open(out / "tables" / "results_ordinal.tex").read()
    assert "SICAPv2$^{\\dagger}$" in tex
    boot = np.load(res / "stats" / "boot_sicap.npz")
    ms = list(boot["models"])
    for m in ["vit_in1k", "phikon", "uni2h"]:
        (cell,) = cell_row(tex, m)
        q = truth[(m, "sicap", 42)]["qwk"]
        assert "\\pm" not in cell  # one result, no SD
        assert f"{q:.4f}" in cell
        lo, hi = np.percentile(boot["boot"][1:, 0, ms.index(m), METRICS.index("qwk")], [2.5, 97.5])
        assert f"[{lo:.4f}, {hi:.4f}]" in cell
    # --official-ci stats: stats.py's (seed-averaged) CI from summary.csv
    rep2, out2, _ = run(res, tmp_path / "b", official_ci="stats")
    tex2 = open(out2 / "tables" / "results_ordinal.tex").read()
    q = truth[("uni2h", "sicap", 42)]["qwk"]
    assert f"[{q - 0.001:.4f}, {q + 0.001:.4f}]" in cell_row(tex2, "uni2h")[0]


def test_missing_model_and_dataset(tmp_path):
    res, _ = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    tex = open(out / "tables" / "results_ordinal.tex").read()
    # hoptimus0 has no SICAPv2 result and no other ordinal result: no row at all; BRACS/PANDA absent: no column
    assert report.MODEL_NAMES["hoptimus0"] not in tex
    assert "BRACS" not in tex and "PANDA" not in tex
    assert any("bracs" in s for s in summ["skipped"]) and any("panda" in s for s in summ["skipped"])
    # the fixed display order is kept
    cat = open(out / "tables" / "results_categorical.tex").read()
    pos = [cat.index(report.MODEL_NAMES[m] + " &") for m in ["vit_in1k", "phikon", "uni2h", "hoptimus0"]]
    assert pos == sorted(pos)


def test_significance_marking(tmp_path):
    res, _ = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    cat = open(out / "tables" / "results_categorical.tex").read()
    # lung: uni2h significantly better than all -> bold, nobody underlined
    assert "\\textbf{" in cell_row(cat, "uni2h")[0]
    assert all("\\underline" not in cell_row(cat, m)[0] and "\\textbf" not in cell_row(cat, m)[0]
               for m in ["vit_in1k", "phikon", "hoptimus0"])
    # sicap (single-seed recomputation): phikon not distinguishable from uni2h -> both underlined, vit_in1k unmarked
    ordi = open(out / "tables" / "results_ordinal.tex").read()
    assert "\\underline{" in cell_row(ordi, "uni2h")[0] and "\\underline{" in cell_row(ordi, "phikon")[0]
    assert "\\underline" not in cell_row(ordi, "vit_in1k")[0] and "\\textbf" not in ordi


def test_marks_rule():
    rep = report.Report.__new__(report.Report)
    rep.alpha, rep.warnings = 0.05, []
    means = {"a": 0.9, "b": 0.8, "c": 0.7}
    p = {frozenset("ab"): 0.01, frozenset("ac"): 0.001, frozenset("bc"): 0.5}
    assert report.Report.marks(rep, means, p, "t")[0] == {"a": "bold", "b": "", "c": ""}
    p[frozenset("ab")] = 0.2
    assert report.Report.marks(rep, means, p, "t")[0] == {"a": "underline", "b": "underline", "c": ""}
    del p[frozenset("ac")]  # missing test: conservative (not significant) and reported
    assert report.Report.marks(rep, means, p, "t")[0]["c"] == "underline" and rep.warnings


def test_holm_matches_hand_computation():
    assert np.allclose(report.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


def test_stale_stats_warning(tmp_path):
    res, _ = build_tree(tmp_path, stale=True)
    rep, out, summ = run(res, tmp_path)
    assert any("stale" in w and "phikon" in w for w in summ["warnings"])


def test_no_stats_still_writes_tables(tmp_path):
    res, truth = build_tree(tmp_path, with_stats=False)
    rep, out, summ = run(res, tmp_path)
    tex = open(out / "tables" / "results_categorical.tex").read()
    v = np.array([truth[("uni2h", "lung", s)]["auc"] for s in SEEDS])
    assert f"{v.mean():.4f}$\\pm${v.std(ddof=1):.4f}" in cell_row(tex, "uni2h")[0]
    assert any("no bootstrap CI" in s for s in summ["skipped"])


def test_figures_and_macros(tmp_path):
    res, _ = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    assert (out / "figures" / "cd_all.pdf").exists() and (out / "figures" / "rank_stability.pdf").exists()
    assert summ["figures"]["cd_all"]["data"]["cd"] == 1.91
    macros = open(out / "macros.tex").read()
    assert "\\RptFriedmanAllChisq" in macros and "{6.00}" in macros
    assert any("categorical" in s and "friedman" in s.lower() for s in summ["skipped"])


def test_precision_same_device_reference(tmp_path):
    res, _ = build_tree(tmp_path)
    rng = np.random.default_rng(3)
    d = res / "precision" / "phikon" / "lung"
    d.mkdir(parents=True)
    rows, y = np.arange(40), rng.integers(0, 2, 40)
    f32 = rng.normal(size=(40, 8))
    p32 = rng.random((40, 2))
    flip = p32.copy()
    flip[:4] = flip[:4, ::-1]  # INT8 changes 4 of 40 predictions w.r.t. FP32 CPU
    for p in report.PRECISIONS:
        feats = f32 + (0.3 * rng.normal(size=f32.shape) if p == "INT8_CPU" else 0)
        probs = flip if p == "INT8_CPU" else p32
        np.savez(d / f"{p}.npz", rows=rows, targets=y, probs=probs, feats=feats)
        j = {"n_images": 40, "metrics": {"acc": 0.9 if p != "INT8_CPU" else 0.85},
             "timing": {"ms_per_image_mean": 2.0 if p.endswith("GPU") else 50.0, "ms_per_image_sd": 0.1,
                        "images_per_s_mean": 500.0 if p.endswith("GPU") else 20.0, "images_per_s_sd": 1.0, "cpu_threads": 4}}
        if p != "FP32_GPU":
            j["vs_fp32_gpu"] = {"prediction_agreement": 0.5, "cosine_mean": 0.5, "cosine_min": 0.1}
        json.dump(j, open(d / f"{p}.json", "w"))
    rep, out, summ = run(res, tmp_path)
    tex = open(out / "tables" / "precision_summary.tex").read()
    int8 = [l for l in tex.splitlines() if "INT8 CPU" in l][0]
    assert "0.9000 (0.9000)" in int8  # agreement 36/40 vs FP32 CPU, not the json's vs-GPU 0.5
    fp16 = [l for l in tex.splitlines() if "FP16 GPU" in l][0]
    assert "0.5000 (0.5000)" in fp16  # GPU precisions: json's comparison with FP32 GPU
    timing = open(out / "tables" / "precision_timing.tex").read()
    assert "50.00$\\pm$0.10" in [l for l in timing.splitlines() if "INT8 CPU" in l][0]
    assert (out / "figures" / "precision_latency.pdf").exists()
    rec = [c for c in summ["tables"]["precision_summary"]["cells"] if c["col"] == "INT8_CPU" and c["quantity"] == "mean agreement"][0]
    assert rec["value"] == pytest.approx(0.9) and any(s.endswith("FP32_CPU.npz") for s in rec["sources"])


def test_sensitivity_and_abmil(tmp_path):
    res, truth = build_tree(tmp_path)
    for s in SEEDS:  # CLS probe for phikon on lung (2 seeds only -> flagged)
        if s != 44:
            _seed_json(res / "probe_cls", "phikon", "lung", s, {"auc": 0.5 + s / 1000})
        _seed_json(res / "abmil", "phikon", "panda", s, {"qwk": 0.8})
        _seed_json(res / "probe", "phikon", "panda", s, {"qwk": 0.6, "auc": 0.9})
    rep, out, summ = run(res, tmp_path)
    tex = open(out / "tables" / "embedding_sensitivity.tex").read()
    assert f"{np.mean([0.542, 0.543]):.4f}" in tex and "$^{n=2}$" in tex
    assert any("2 seeds" in w for w in summ["warnings"])
    ab = open(out / "tables" / "abmil_panda.tex").read()
    assert "0.8000" in ab and "0.6000" in ab and "Delta" not in ab
    assert not any("ABMIL: FAILED" in w for w in summ["skipped"])


def test_identical_seeds_on_seed_split_dataset_warns(tmp_path):
    res, _ = build_tree(tmp_path, with_stats=False)
    for s in SEEDS:
        _seed_json(res / "probe", "phikon", "lung", s, {"auc": 0.9, "qwk": 0.5})
    rep, out, summ = run(res, tmp_path)
    assert any("identical" in w and "phikon" in w for w in summ["warnings"])


def test_benchmark_table_layout_numbers_and_marks(tmp_path):
    """Table III: Dataset | Model | Accuracy | F1-Score | AUC | QWK, 3 decimals, mean +- SD from
    the seed json; QWK only for ordinal datasets; official partition single value; marks equal results_table's."""
    res, truth = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    tex = open(out / "tables" / "performance_benchmark.tex").read()
    assert "Dataset & Model & Accuracy & F1-Score & AUC & QWK" in tex
    rows = [l.rstrip(" \\").split(" & ") for l in tex.splitlines() if " & " in l and not l.startswith("Dataset")]
    lung_rows = {r[1]: r for r in rows[:4]}  # the 4 lung models come first (display order)
    for m in ["vit_in1k", "phikon", "uni2h", "hoptimus0"]:
        r = lung_rows[report.MODEL_NAMES[m]]
        for k, cell in zip(["acc", "f1_weighted", "auc"], r[2:5]):
            v = np.array([truth[(m, "lung", s)][k] for s in SEEDS])
            assert f"{v.mean():.3f}$\\pm${report.fmt_sd(v.std(ddof=1), 3)}" in cell, (m, k, cell)
        assert r[5] == "--"  # no QWK for a nominal dataset
    sicap = [r for r in rows if "SICAP" in r[0]]
    assert sicap and "$\\pm$" not in " ".join(sicap[0][2:])  # official partition: single value
    assert sicap[0][5] != "--"
    # the primary-metric marks equal those of the main results table
    cat = open(out / "tables" / "results_categorical.tex").read()
    for m in ["vit_in1k", "phikon", "uni2h", "hoptimus0"]:
        mark_main = [k for k in ("underline", "textbf") if k in cell_row(cat, m)[0]]
        mark_bench = [k for k in ("underline", "textbf") if k in lung_rows[report.MODEL_NAMES[m]][4]]
        assert mark_main == mark_bench, m


def _qual_tree(tmp_path, rows_b=None):
    """Two models, 40 BACH test tiles (10 per class, 2 tiles per source group), tiny PNG images."""
    from PIL import Image
    res = tmp_path / "results"
    data = tmp_path / "data"
    rng = np.random.default_rng(1)
    n = 40
    y = np.repeat(np.arange(4), 10)
    groups = np.array([f"g{i // 2}" for i in range(n)])
    paths = np.array([f"img/t{i}.png" for i in range(n)])
    (data / "img").mkdir(parents=True)
    for p in paths:
        Image.fromarray((rng.random((8, 8, 3)) * 255).astype("uint8")).save(data / p)
    for m, rows in [("uni2h", np.arange(n)), ("ctranspath", np.arange(n) if rows_b is None else rows_b)]:
        d = res / "probe" / m / "bach"
        d.mkdir(parents=True)
        probs = rng.random((n, 4))
        np.savez(d / "seed42.npz", rows=rows, paths=paths, groups=groups, targets=y, probs=probs)
    return res, data


def test_qualitative_same_patches_stratified_and_true_predictions(tmp_path):
    res, data = _qual_tree(tmp_path)
    rep = report.Report(str(res), str(tmp_path / "gen"))
    rep.data_root = str(data)
    report.setup_matplotlib()
    rep.qualitative()
    d = rep.figures["qualitative"]["data"]
    pats = d["patches"]
    assert len(pats) == 16
    assert sorted(p["true"] for p in pats).count("Invasive") == 4  # 4 per class
    for c in ["Normal", "Benign", "In situ", "Invasive"]:
        gs = [p["group"] for p in pats if p["true"] == c]
        assert len(gs) == len(set(gs)) == 4  # different source groups
    z = {m: np.load(res / "probe" / m / "bach" / "seed42.npz") for m in ["uni2h", "ctranspath"]}
    names = report.CLASS_NAMES["bach"]
    for p in pats:  # both models on the SAME patch, predictions exactly the saved argmax
        for m in ["uni2h", "ctranspath"]:
            assert p[m] == names[int(z[m]["probs"][p["row"]].argmax())]
        assert p["true"] == names[int(z["uni2h"]["targets"][p["row"]])]
    # deterministic: same draw on a rerun
    rep2 = report.Report(str(res), str(tmp_path / "gen2"))
    rep2.data_root = str(data)
    rep2.qualitative()
    assert rep2.figures["qualitative"]["data"]["patches"] == pats


def test_qualitative_refuses_different_test_items(tmp_path):
    res, data = _qual_tree(tmp_path, rows_b=np.arange(40)[::-1].copy())
    rep = report.Report(str(res), str(tmp_path / "gen"))
    rep.data_root = str(data)
    rep.qualitative()
    assert "qualitative" not in rep.figures and any("different test items" in s for s in rep.skipped)


def test_accuracy_barchart_values_are_seed_mean_and_sd(tmp_path):
    res, truth = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    assert (out / "figures" / "accuracy_barchart.pdf").exists()
    vals = summ["figures"]["accuracy_barchart"]["data"]["values"]
    for m in ["vit_in1k", "phikon", "uni2h", "hoptimus0"]:
        v = np.array([truth[(m, "lung", s)]["acc"] for s in SEEDS])
        assert vals[f"{m}/lung"]["mean"] == pytest.approx(v.mean(), abs=1e-12)
        assert vals[f"{m}/lung"]["sd"] == pytest.approx(v.std(ddof=1), abs=1e-12)
    assert not any(k.endswith("/sicap") for k in vals)  # categorical datasets only


def test_qwk_heatmap_values(tmp_path):
    res, truth = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    assert (out / "figures" / "qwk_heatmap.pdf").exists()
    vals = summ["figures"]["qwk_heatmap"]["data"]["values"]
    assert set(k.split("/")[1] for k in vals) == {"sicap"}  # ordinal datasets only (the tree has sicap)
    for m in ["vit_in1k", "phikon", "uni2h"]:
        v = vals[f"{m}/sicap"]
        assert v["official"] and v["mean"] == pytest.approx(truth[(m, "sicap", SEEDS[0])]["qwk"], abs=1e-12)


def test_text_macros_equal_source_values(tmp_path):
    res, truth = build_tree(tmp_path)
    rep, out, summ = run(res, tmp_path)
    means = {m: np.mean([truth[(m, "lung", s)]["auc"] for s in SEEDS]) for m in ["vit_in1k", "phikon", "uni2h", "hoptimus0"]}
    mac = summ["macros"]
    assert mac["RptLungAUCMin"]["value"] == pytest.approx(min(means.values()), abs=1e-12)
    assert mac["RptLungAUCMax"]["value"] == pytest.approx(max(means.values()), abs=1e-12)
    assert mac["RptLungAUCMin"]["text"] == f"{min(means.values()):.3f}"
    tex = open(out / "macros.tex").read()
    assert "\\RptLungAUCMin" in tex and "\\RptLungAUCMax" in tex
