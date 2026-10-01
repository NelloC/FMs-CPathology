"""Manuscript tables (LaTeX) and figures (PDF) generated directly from the result files, so that no number
in the paper is transcribed by hand.

Reads (whatever is present; everything missing is skipped and reported):
  results/probe/<model>/<dataset>/seed<k>.json        per-seed test metrics (mean +- SD over seeds)
  results/stats/{summary,pairwise,ranks}.csv, friedman.json, boot_<dataset>.npz
                                                       95% cluster-bootstrap CIs, Holm-corrected paired tests,
                                                       rank stability, Friedman / Nemenyi
  results/abmil/<model>/panda/seed<k>.json + results/stats_abmil/   slide-level ABMIL on PANDA
  results/precision/<model>/<dataset>/<precision>.{json,npz}       precision / latency
  results/probe_cls, results/probe_patch_mean                       embedding sensitivity
  results/external/<model>/<dataset>/seed<k>.json + results/stats_external/   external test
  verification/models.json                                          official embedding of each model

Official-partition datasets (BRACS, SICAPv2): the probe is deterministic, so every seed gives the same
predictions and the table reports one value with its CI, no SD; stats.py bootstraps a single seed. With
--official-ci single_seed (default) their CI and Holm-corrected paired tests are recomputed here from that seed's
replicates in boot_<dataset>.npz (same rules as stats.py); --official-ci stats uses stats.py's values.

Writes (to --out, default generated/):
  tables/*.tex    tabular bodies to \\input inside a table environment (caption and label stay in main.tex)
  figures/*.pdf   vector figures at final size (one column = 3.5 in, two columns = 7.16 in)
  macros.tex      \\newcommand's for numbers quoted in the text (Friedman statistics, ...)
  summary.json    every number used, with its source file(s), plus the list of skipped items and warnings

Usage: python pipeline/report.py [--results results] [--out generated]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ORDER = ["vit_in1k", "ctranspath", "phikon", "conch", "uni2h", "virchow2", "hoptimus0", "gigapath"]
MODEL_NAMES = {"vit_in1k": "ViT-B/16 (ImageNet)", "ctranspath": "CTransPath", "phikon": "Phikon", "conch": "CONCH",
               "uni2h": "UNI2-h", "virchow2": "Virchow2", "hoptimus0": "H-optimus-0", "gigapath": "Prov-GigaPath"}
# Okabe-Ito (colour-blind safe); the pale yellow is left out (too little contrast on white), one colour per model
MODEL_COLORS = {"vit_in1k": "#999999", "ctranspath": "#E69F00", "phikon": "#56B4E9", "conch": "#009E73",
                "uni2h": "#0072B2", "virchow2": "#D55E00", "hoptimus0": "#CC79A7", "gigapath": "#000000"}
MODEL_MARKERS = {"vit_in1k": "o", "ctranspath": "s", "phikon": "^", "conch": "D", "uni2h": "v", "virchow2": "P",
                 "hoptimus0": "X", "gigapath": "*"}
EXTRA_COLORS = ["#882255", "#44AA99", "#117733", "#332288", "#AA4499", "#DDCC77"]

DATASET_ORDER = ["lung", "breakhis", "nct", "hubmap", "bach", "panda", "bracs", "sicap"]
DATASET_NAMES = {"lung": "LC25000", "breakhis": "BreaKHis", "nct": "NCT-CRC", "hubmap": "HuBMAP", "bach": "BACH",
                 "panda": "PANDA", "bracs": "BRACS", "sicap": "SICAPv2", "nct_val7k": "CRC-VAL-HE-7K"}
CATEGORICAL = ["lung", "breakhis", "nct", "hubmap", "bach"]
ORDINAL = ["panda", "bracs", "sicap"]
OFFICIAL = {"bracs", "sicap"}  # official train/val/test partitions: one result, seeds identical
METRIC_NAMES = {"auc": "AUC", "qwk": "QWK", "acc": "Acc.", "bal_acc": "Bal. acc.", "f1_weighted": "F1 (w)",
                "f1_macro": "F1 (macro)"}
STATS_METRICS = ["acc", "bal_acc", "f1_weighted", "auc", "qwk"]  # order of the last axis of boot_<dataset>.npz
PRECISIONS = ["FP32_GPU", "FP16_GPU", "BF16_GPU", "FP32_CPU", "INT8_CPU"]
PRECISION_NAMES = {"FP32_GPU": "FP32", "FP16_GPU": "FP16", "BF16_GPU": "BF16", "FP32_CPU": "FP32", "INT8_CPU": "INT8"}
PRECISION_MARKERS = {"FP32_GPU": "o", "FP16_GPU": "s", "BF16_GPU": "^", "FP32_CPU": "o", "INT8_CPU": "D"}
SECONDARY = ["acc", "bal_acc", "f1_weighted"]
CLASS_NAMES = {"bach": ["Normal", "Benign", "In situ", "Invasive"]}  # label order of indexes/bach.csv
EMBEDDINGS = [("official", "probe"), ("cls", "probe_cls"), ("patch_mean", "probe_patch_mean")]
EMBEDDING_NAMES = {"official": "Official", "cls": "CLS", "patch_mean": "Patch mean"}

COL_W, TEXT_W = 3.5, 7.16  # IEEE two-column widths (in)


def primary(dataset):
    return "qwk" if dataset in ORDINAL else "auc"


MODELS_ONLY = None      # --models: restrict every table/figure to these models
STATS_SUFFIX = ""       # --stats-suffix: read results/stats<suffix>, stats_abmil<suffix>, stats_external<suffix>


def order_by(names, order):
    """Fixed display order for known names, unknown names after them (sorted); --models filter applied."""
    names = [n for n in dict.fromkeys(names) if MODELS_ONLY is None or order is not MODEL_ORDER or n in MODELS_ONLY]
    return [n for n in order if n in names] + sorted(n for n in names if n not in order)


def tex_escape(s):
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def model_name(m):
    return MODEL_NAMES.get(m, tex_escape(m))


def model_label(m):  # plain text (figures)
    return MODEL_NAMES.get(m, m)


def dataset_name(d):
    return DATASET_NAMES.get(d, tex_escape(d))


def model_color(m):
    if m in MODEL_COLORS:
        return MODEL_COLORS[m]
    return EXTRA_COLORS[int(hashlib.md5(m.encode()).hexdigest(), 16) % len(EXTRA_COLORS)]


def fmt(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


def fmt_sd(x, d=4):
    """An SD that is positive but rounds to zero is shown as '<0.0001' rather than a misleading '0.0000'."""
    if x is not None and math.isfinite(x) and 0 < x < 0.5 * 10 ** -d:
        return f"\\textless{{}}{10 ** -d:.{d}f}"
    return fmt(x, d)


def holm(p):
    """Holm step-down adjustment (identical to stats.holm)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def seed_of(path):
    return int(re.search(r"seed(\d+)\.json$", path).group(1))


class Stats:
    """One stats.py output directory (results/stats, results/stats_abmil, ...), loaded lazily."""

    def __init__(self, report, path):
        self.r, self.path = report, path
        self._cache = {}

    def _csv(self, name):
        if name not in self._cache:
            f = os.path.join(self.path, name)
            self._cache[name] = pd.read_csv(f) if os.path.exists(f) else None
            if self._cache[name] is None:
                self.r.skip(f"{self.r.rel(f)} not found")
        return self._cache[name]

    def friedman(self):
        f = os.path.join(self.path, "friedman.json")
        if not os.path.exists(f):
            self.r.skip(f"{self.r.rel(f)} not found")
            return {}, f
        return json.load(open(f)), f

    def ranks(self):
        return self._csv("ranks.csv"), os.path.join(self.path, "ranks.csv")

    def boot(self, dataset):
        key = f"boot_{dataset}"
        if key not in self._cache:
            f = os.path.join(self.path, f"{key}.npz")
            self._cache[key] = None
            if os.path.exists(f):
                c = np.load(f, allow_pickle=False)
                self._cache[key] = dict(boot=c["boot"], models=[str(m) for m in c["models"]],
                                        seeds=[int(s) for s in c["seeds"]], metrics=[str(m) for m in c["metrics"]],
                                        input_hash=str(c["input_hash"]), path=f)
        return self._cache[key]

    def check_fresh(self, pred_root, dataset):
        """Warn when the probe predictions changed after stats.py ran (same hash as stats.input_hash)."""
        b = self.boot(dataset)
        if b is None:
            return
        # stats.input_hash hashes the file paths as stats.py saw them: absolute (default --probe-dir) or relative to
        # the code directory (e.g. --probe-dir results/abmil); accept either spelling
        blobs = {}
        for m in b["models"]:
            for f in sorted(glob.glob(os.path.join(pred_root, m, dataset, "seed*.npz"))):
                blobs[f] = open(f, "rb").read()
        ok = False
        for spelling in (lambda f: f, lambda f: os.path.relpath(f, self.r.base)):
            h = hashlib.sha256()
            for f, data in blobs.items():
                h.update(spelling(f).encode())
                h.update(data)
            ok |= h.hexdigest()[:16] == b["input_hash"].split("-")[0]  # stats.py appends tags ("-official1", "-allclasses")
        if not ok:
            self.r.warn(f"{self.r.rel(b['path'])}: prediction files changed since stats.py ran (input hash mismatch) "
                        "- rerun stats.py")
        present = {os.path.basename(os.path.dirname(os.path.dirname(f)))
                   for f in glob.glob(os.path.join(pred_root, "*", dataset, "seed*.npz"))}
        if MODELS_ONLY is not None:
            present &= MODELS_ONLY
        if present - set(b["models"]):
            self.r.warn(f"{self.r.rel(b['path'])}: models with predictions but not in stats: "
                        f"{sorted(present - set(b['models']))} - rerun stats.py")

    def single_seed(self, dataset):
        """(boot array of seed 0 [B+1, models, metrics], info) for an official-partition dataset, after checking that
        every seed has the same point estimates (identical predictions)."""
        b = self.boot(dataset)
        if b is None:
            return None
        pt = b["boot"][0]  # [seeds, models, metrics]
        if not np.allclose(pt, pt[:1], rtol=0, atol=1e-12, equal_nan=True):
            self.r.warn(f"{dataset}: seeds of the official partition do not give identical results in "
                        f"{self.r.rel(b['path'])}; using stats.py's seed-averaged values")
            return None
        return b

    def ci(self, dataset, model, metric, single_seed):
        """(lo, hi, stats_mean, source) or None."""
        if single_seed:
            b = self.single_seed(dataset)
            if b is not None and model in b["models"]:
                reps = b["boot"][1:, 0, b["models"].index(model), b["metrics"].index(metric)]
                lo, hi = np.nanpercentile(reps, [2.5, 97.5])
                return float(lo), float(hi), float(b["boot"][0, 0, b["models"].index(model), b["metrics"].index(metric)]), \
                    b["path"] + f" [boot[1:, seed {b['seeds'][0]}]]"
        s = self._csv("summary.csv")
        if s is None:
            return None
        row = s[(s.dataset == dataset) & (s.model == model) & (s.metric == metric)]
        if row.empty:
            return None
        row = row.iloc[0]
        return float(row.ci_low), float(row.ci_high), float(row["mean"]), os.path.join(self.path, "summary.csv")

    def pvalues(self, dataset, metric, single_seed):
        """({frozenset(a, b): Holm-adjusted p}, source)."""
        if single_seed:
            b = self.single_seed(dataset)
            if b is not None:
                reps = b["boot"][1:, 0, :, b["metrics"].index(metric)]
                pairs, ps = [], []
                ms = b["models"]
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        d = reps[:, i] - reps[:, j]
                        d = d[np.isfinite(d)]
                        pairs.append(frozenset((ms[i], ms[j])))
                        ps.append(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan)
                return dict(zip(pairs, holm(ps))), b["path"] + f" [boot[1:, seed {b['seeds'][0]}], Holm]"
        p = self._csv("pairwise.csv")
        if p is None:
            return {}, None
        p = p[(p.dataset == dataset) & (p.metric == metric)]
        return {frozenset((r.model_a, r.model_b)): float(r.p_holm) for r in p.itertuples()}, \
            os.path.join(self.path, "pairwise.csv")


class Report:
    def __init__(self, results, out, alpha=0.05, official_ci="single_seed", verification=None):
        self.results, self.out, self.alpha, self.official_ci = os.path.abspath(results), os.path.abspath(out), alpha, official_ci
        self.base = os.path.dirname(self.results)
        self.verification = verification or os.path.join(self.base, "verification")
        self.skipped, self.warnings = [], []
        self.tables, self.figures, self.macros = {}, {}, {}
        self.stats = Stats(self, os.path.join(self.results, "stats" + STATS_SUFFIX))
        os.makedirs(os.path.join(self.out, "tables"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "figures"), exist_ok=True)

    # ------------------------------------------------------------------ bookkeeping
    def rel(self, path):
        return os.path.relpath(path, self.base) if os.path.isabs(path) else path

    def skip(self, msg):
        if msg not in self.skipped:
            self.skipped.append(msg)

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def record(self, table, row, col, quantity, value, text, sources):
        self.tables.setdefault(table, {"file": None, "cells": []})["cells"].append(dict(
            row=row, col=col, quantity=quantity, value=None if value is None or not np.isfinite(value) else float(value),
            text=text, sources=sorted({self.rel(s) for s in sources if s})))

    def write_table(self, name, tabular, sources, note="", resize=True):
        path = os.path.join(self.out, "tables", f"{name}.tex")
        head = ["% Generated by pipeline/report.py - do not edit by hand. Every number comes from the files below "
                "(per-cell sources in summary.json):"]
        head += [f"%   {p}" for p in self.source_patterns(sources)]
        head += [f"% {line}" for line in note.strip().splitlines()] if note else []
        # typeset at \footnotesize; shrunk to the line width only when wider (never enlarged)
        body = ("{\\setlength{\\tabcolsep}{3pt}\\footnotesize%\n\\setbox0=\\hbox{%\n" + tabular +
                "}%\n\\centerline{\\ifdim\\wd0>\\linewidth\\resizebox{\\linewidth}{!}{\\copy0}\\else\\box0\\fi}}\n")
        with open(path, "w") as f:
            f.write("\n".join(head) + "\n" + body)
        self.tables.setdefault(name, {"cells": []})["file"] = self.rel(path)
        print(f"[table]  {self.rel(path)}")

    def source_patterns(self, sources):
        """Compact globs of the source files (model and dataset directories and seed numbers collapsed)."""
        out = set()
        for s in sources:
            if not s:
                continue
            s = re.sub(r" \[.*\]$", "", self.rel(s))
            parts = s.split("/")
            if len(parts) == 5 and parts[0] == "results":
                parts[2], parts[3] = "*", "*"
            out.add(re.sub(r"seed\d+", "seed*", "/".join(parts)))
        return sorted(out)

    def save_figure(self, fig, name, data, sources):
        import matplotlib.pyplot as plt
        path = os.path.join(self.out, "figures", f"{name}.pdf")
        fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
        self.figures[name] = {"file": self.rel(path), "data": data, "sources": sorted({self.rel(s) for s in sources})}
        print(f"[figure] {self.rel(path)}")

    # ------------------------------------------------------------------ loading
    def discover(self, root, datasets=None):
        """(models, datasets) that have at least one seed json under root, in display order."""
        found = glob.glob(os.path.join(root, "*", "*", "seed*.json"))
        ms = order_by([f.split(os.sep)[-3] for f in found], MODEL_ORDER)
        ds = order_by([f.split(os.sep)[-2] for f in found], DATASET_ORDER)
        if datasets is not None:
            ds = [d for d in datasets if d in ds]
        return ms, ds

    def runs(self, root, model, dataset):
        out = {}
        for f in sorted(glob.glob(os.path.join(root, model, dataset, "seed*.json")), key=seed_of):
            try:
                out[seed_of(f)] = (json.load(open(f)), f)
            except (OSError, ValueError) as e:
                self.warn(f"unreadable {self.rel(f)}: {e!r}")
        return out

    def seed_summary(self, root, model, dataset, metric):
        """Mean / SD over seeds of one test metric, from the per-seed json files."""
        runs = self.runs(root, model, dataset)
        if not runs:
            return None
        vals = np.array([r["test"].get(metric, np.nan) for r, _ in runs.values()], float)
        tag = f"{self.rel(os.path.join(root, model, dataset))} {metric}"
        identical = len(vals) > 1 and np.allclose(vals, vals[0], rtol=0, atol=1e-12, equal_nan=True)
        official = dataset in OFFICIAL
        if official and len(vals) > 1 and not identical:
            self.warn(f"{tag}: official partition but seeds differ ({vals.min():.4f}-{vals.max():.4f}); reported as mean +- SD")
            official = False
        if not official and identical and dataset not in OFFICIAL:
            self.warn(f"{tag}: all {len(vals)} seeds give identical results on a seed-split dataset")
        if np.isnan(vals).any():
            self.warn(f"{tag}: NaN in seeds {[s for s, v in zip(runs, vals) if np.isnan(v)]}")
        mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        sd = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan")
        return dict(mean=float(vals[0]) if official else mean, sd=None if official else sd, n=len(vals),
                    seeds=list(runs), values=vals.tolist(), official=official, sources=[f for _, f in runs.values()])

    # ------------------------------------------------------------------ significance
    def marks(self, means, pvals, context):
        """Best model (highest mean) in bold when it is significantly better (Holm p < alpha) than every other model;
        otherwise the best and every model not significantly different from it are underlined."""
        valid = {m: v for m, v in means.items() if v is not None and np.isfinite(v)}
        if len(valid) < 2:
            return {m: "" for m in means}, None
        best = max(valid, key=lambda m: valid[m])
        tied = []
        for m in valid:
            if m == best:
                continue
            p = pvals.get(frozenset((best, m)))
            if p is None or not np.isfinite(p):
                self.warn(f"{context}: no paired test for {best} vs {m}; treated as not significant")
                tied.append(m)
            elif p >= self.alpha:
                tied.append(m)
        out = {m: "" for m in means}
        if tied:
            for m in [best] + tied:
                out[m] = "underline"
        else:
            out[best] = "bold"
        return out, best

    @staticmethod
    def apply_mark(text, mark):
        return f"\\textbf{{{text}}}" if mark == "bold" else f"\\underline{{{text}}}" if mark == "underline" else text

    # ------------------------------------------------------------------ 1. main results
    def value_cell(self, s, ci, mark, digits=4):
        """(latex, parts) for 'mean +- SD' (or the single value of an official partition) over '[lo, hi]'."""
        top = fmt(s["mean"], digits) if s["official"] or s["sd"] is None else f"{fmt(s['mean'], digits)}$\\pm${fmt_sd(s['sd'], digits)}"
        top = self.apply_mark(top, mark)
        if ci is None:
            return top
        return f"\\makecell{{{top}\\\\{{\\scriptsize[{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]}}}}"

    def gather(self, root, stats, dataset, models, table):
        """Per-model seed summaries + CI + significance marks for the primary metric of one dataset."""
        metric = primary(dataset)
        res = {}
        for m in models:
            s = self.seed_summary(root, m, dataset, metric)
            if s is None:
                self.skip(f"{table}: {m}/{dataset} has no results in {self.rel(root)}")
                continue
            single = s["official"] and self.official_ci == "single_seed"
            ci = stats.ci(dataset, m, metric, single)
            if ci is None:
                self.skip(f"{table}: no bootstrap CI for {m}/{dataset} in {self.rel(stats.path)} (rerun stats.py)")
            elif abs(ci[2] - s["mean"]) > 1e-9:
                self.warn(f"{table}: {m}/{dataset} {metric}: seed-json mean {s['mean']:.6f} != stats mean {ci[2]:.6f} "
                          f"({self.rel(ci[3])}); stats.py is stale - rerun it")
            s["ci"] = ci
            res[m] = s
        if not res:
            return res, None, None
        single = any(s["official"] for s in res.values()) and self.official_ci == "single_seed"
        pv, psrc = stats.pvalues(dataset, metric, single)
        mk, best = self.marks({m: s["mean"] for m, s in res.items()}, pv, f"{table}/{dataset}")
        for m, s in res.items():
            s["mark"] = mk[m]
            s["p_vs_best"] = None if m == best else pv.get(frozenset((best, m)))
            s["p_source"] = psrc
        high = [m for m, s in res.items() if np.isfinite(s["mean"]) and s["mean"] >= 0.999]
        if len(high) >= 2:
            self.warn(f"{dataset}: ceiling effect - {len(high)} models with {metric} >= 0.999 ({', '.join(high)})")
        return res, metric, best

    def results_table(self, task, datasets):
        name = f"results_{task}"
        root = os.path.join(self.results, "probe")
        models, present = self.discover(root, datasets)
        for d in datasets:
            if d not in present:
                self.skip(f"{name}: dataset {d} has no probe results")
        if not present:
            self.skip(f"{name}: no datasets - table not written")
            return
        cols, sources = {}, set()
        for d in present:
            self.stats.check_fresh(root, d)
            res, metric, best = self.gather(root, self.stats, d, models, name)
            cols[d] = (res, metric)
        lines = ["\\begin{tabular}{l" + "c" * len(present) + "}", "\\toprule",
                 "Model & " + " & ".join(f"{dataset_name(d)}" + ("$^{\\dagger}$" if d in OFFICIAL else "") for d in present) + " \\\\",
                 " & " + " & ".join(METRIC_NAMES[primary(d)] for d in present) + " \\\\", "\\midrule"]
        for m in models:
            cells = []
            for d in present:
                res, metric = cols[d]
                s = res.get(m)
                if s is None:
                    cells.append("--")
                    continue
                ci = s["ci"][:2] if s["ci"] else None
                cells.append(self.value_cell(s, ci, s["mark"]))
                src = s["sources"] + ([s["ci"][3]] if s["ci"] else []) + ([s["p_source"]] if s["p_source"] else [])
                sources.update(src)
                self.record(name, m, d, f"{metric} mean" if not s["official"] else f"{metric} (official partition)",
                            s["mean"], fmt(s["mean"]), s["sources"])
                if s["sd"] is not None:
                    self.record(name, m, d, f"{metric} sd", s["sd"], fmt(s["sd"]), s["sources"])
                if s["ci"]:
                    self.record(name, m, d, f"{metric} ci_low", s["ci"][0], fmt(s["ci"][0]), [s["ci"][3]])
                    self.record(name, m, d, f"{metric} ci_high", s["ci"][1], fmt(s["ci"][1]), [s["ci"][3]])
                self.record(name, m, d, "mark", None, s["mark"] or "none", [s["p_source"]])
                if s["p_vs_best"] is not None:
                    self.record(name, m, d, "p_holm vs best", s["p_vs_best"], "", [s["p_source"]])
            if any(c != "--" for c in cells):
                lines.append(f"{model_name(m)} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        n_seeds = sorted({s["n"] for res, _ in cols.values() for s in res.values()})
        note = (f"Suggested caption: {METRIC_NAMES[primary(present[0])]} on the test set, mean $\\pm$ SD over {'/'.join(map(str, n_seeds))} "
                "seeds; below, 95% cluster-bootstrap CI of the seed-averaged metric. "
                "Bold: best model, significantly better than every other model (paired bootstrap, Holm-corrected, p < "
                f"{self.alpha}); underlined: best model and the models not significantly different from it. "
                "$^\\dagger$ official partition: one result (seeds identical), no SD.")
        self.write_table(name, "\n".join(lines) + "\n", sources, note)

    def secondary_table(self, task, datasets):
        name = f"secondary_{task}"
        root = os.path.join(self.results, "probe")
        models, present = self.discover(root, datasets)
        if not present:
            self.skip(f"{name}: no datasets - table not written")
            return
        sources = set()
        lines = ["\\begin{tabular}{ll" + "c" * len(SECONDARY) + "}", "\\toprule",
                 "Dataset & Model & " + " & ".join(METRIC_NAMES[k] for k in SECONDARY) + " \\\\", "\\midrule"]
        for di, d in enumerate(present):
            rows = []
            for m in models:
                ss = {k: self.seed_summary(root, m, d, k) for k in SECONDARY}
                if any(v is None for v in ss.values()):
                    continue
                cells = []
                for k, s in ss.items():
                    cells.append(fmt(s["mean"]) if s["official"] else f"{fmt(s['mean'])}$\\pm${fmt_sd(s['sd'])}")
                    self.record(name, m, d, f"{k} mean", s["mean"], fmt(s["mean"]), s["sources"])
                    if not s["official"]:
                        self.record(name, m, d, f"{k} sd", s["sd"], fmt(s["sd"]), s["sources"])
                    sources.update(s["sources"])
                rows.append((m, cells))
            for i, (m, cells) in enumerate(rows):
                first = f"\\multirow{{{len(rows)}}}{{*}}{{{dataset_name(d)}{'$^{\\dagger}$' if d in OFFICIAL else ''}}}" if i == 0 else ""
                lines.append(f"{first} & {model_name(m)} & " + " & ".join(cells) + " \\\\")
            if rows and di < len(present) - 1:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        self.write_table(name, "\n".join(lines) + "\n", sources,
                         "Suggested caption: accuracy, balanced accuracy and weighted F1 (mean $\\pm$ SD over seeds; "
                         "$^\\dagger$ official partition, one result).")

    def benchmark_table(self, digits=3):
        """Table III of the manuscript (Dataset | Model | Accuracy | F1-Score |
        AUC | QWK): mean +- SD over seeds (one value for official partitions). The primary metric (AUC categorical,
        QWK ordinal) carries the significance marks of results_table; QWK is shown for ordinal datasets only
        (it is not meaningful for nominal classes). CIs are in results_categorical / results_ordinal."""
        name = "performance_benchmark"
        root = os.path.join(self.results, "probe")
        models, present = self.discover(root, DATASET_ORDER)
        if not present:
            self.skip(f"{name}: no datasets - table not written")
            return
        cols = ["acc", "f1_weighted", "auc", "qwk"]
        sources = set()
        lines = ["\\begin{tabular}{llcccc}", "\\toprule",
                 "Dataset & Model & Accuracy & F1-Score & AUC & QWK \\\\", "\\midrule"]
        for di, d in enumerate(present):
            self.stats.check_fresh(root, d)
            res, pmetric, best = self.gather(root, self.stats, d, models, name)
            rows = []
            for m in models:
                if m not in res:
                    continue
                cells = []
                for k in cols:
                    if k == "qwk" and d not in ORDINAL:
                        cells.append("--")
                        continue
                    s = res[m] if k == pmetric else self.seed_summary(root, m, d, k)
                    if s is None:
                        cells.append("--")
                        continue
                    text = fmt(s["mean"], digits) if s["official"] or s["sd"] is None else \
                        f"{fmt(s['mean'], digits)}$\\pm${fmt_sd(s['sd'], digits)}"
                    if k == pmetric:
                        text = self.apply_mark(text, res[m]["mark"])
                        self.record(name, m, d, "mark", None, res[m]["mark"] or "none", [res[m]["p_source"]])
                    cells.append(text)
                    self.record(name, m, d, f"{k} mean", s["mean"], fmt(s["mean"], digits), s["sources"])
                    if not s["official"] and s["sd"] is not None:
                        self.record(name, m, d, f"{k} sd", s["sd"], fmt_sd(s["sd"], digits), s["sources"])
                    sources.update(s["sources"])
                    if k == pmetric and res[m]["p_source"]:
                        sources.add(res[m]["p_source"])
                rows.append((m, cells))
            for i, (m, cells) in enumerate(rows):
                first = (f"\\multirow{{{len(rows)}}}{{*}}{{{dataset_name(d)}{'$^{\\dagger}$' if d in OFFICIAL else ''}}}"
                         if i == 0 else "")
                lines.append(f"{first} & {model_name(m)} & " + " & ".join(cells) + " \\\\")
            if rows and di < len(present) - 1:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        n_seeds = sorted({s["n"] for d in present for s in self.gather(root, self.stats, d, models, name)[0].values()
                          if not s["official"]})
        self.write_table(name, "\n".join(lines) + "\n", sources,
                         f"Suggested caption: test-set performance, mean $\\pm$ SD over {'/'.join(map(str, n_seeds))} seeds "
                         "($^\\dagger$ official partition: one result). Primary metric (AUC categorical, QWK ordinal): bold = "
                         "best model, significantly better than every other (paired cluster bootstrap, Holm, p < "
                         f"{self.alpha}); underlined = best model and models not significantly different from it. "
                         "F1: weighted. QWK only for ordinal datasets.")

    # ------------------------------------------------------------------ qualitative figure (Fig. 6)
    def qualitative(self, dataset="bach", models=("uni2h", "ctranspath"), seed=42, per_class=4, sample_seed=0):
        """Fig. 6: the SAME randomly drawn test patches for every model (no selection by outcome). Stratified draw:
        per_class patches per class, each from a different source group, rng(sample_seed) over the seed-`seed` test
        set. Each tile shows the true class and every model's prediction (green correct, red wrong)."""
        import matplotlib.pyplot as plt
        from PIL import Image
        name = "qualitative"
        root = os.path.join(self.results, "probe")
        files = [os.path.join(root, m, dataset, f"seed{seed}.npz") for m in models]
        if not all(os.path.exists(f) for f in files):
            self.skip(f"{name}: missing {[self.rel(f) for f in files if not os.path.exists(f)]}")
            return
        z = [np.load(f) for f in files]
        for zi in z[1:]:
            if not (np.array_equal(zi["rows"], z[0]["rows"]) and np.array_equal(zi["targets"], z[0]["targets"])):
                self.skip(f"{name}: models were evaluated on different test items")
                return
        y, paths, groups = z[0]["targets"], z[0]["paths"].astype(str), z[0]["groups"].astype(str)
        preds = [zi["probs"].argmax(1) for zi in z]
        classes = CLASS_NAMES.get(dataset)
        if classes is None:
            self.skip(f"{name}: no class names for {dataset}")
            return
        rng = np.random.default_rng(sample_seed)
        picked = []
        for c in range(len(classes)):
            cand = np.flatnonzero(y == c)
            order = rng.permutation(cand)
            seen = set()
            for i in order:  # one tile per source group
                if groups[i] not in seen:
                    seen.add(groups[i])
                    picked.append(int(i))
                if len(seen) == per_class:
                    break
        data_root = getattr(self, "data_root", None) or os.environ.get("VFM_BASE_DIR") or os.path.dirname(os.path.dirname(self.base))
        ncol = 2 * per_class
        nrow = int(math.ceil(len(picked) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_W, nrow * 1.45), squeeze=False)
        rec = []
        for ax, i in zip(axes.flat, picked):
            with Image.open(os.path.join(data_root, paths[i])) as im:
                ax.imshow(np.asarray(im.convert("RGB")))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
            lines = [(f"True: {classes[y[i]]}", "#222222")]
            for m, pr in zip(models, preds):
                ok = pr[i] == y[i]
                lines.append((f"{model_label(m)}: {classes[pr[i]]}", "#1a7f37" if ok else "#c62828"))
            for k, (t, col) in enumerate(lines):
                ax.text(0.5, -0.06 - 0.17 * k, t, transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
                        color=col, fontweight="bold" if k else "normal")
            rec.append({"row": int(z[0]["rows"][i]), "path": paths[i], "group": groups[i], "true": classes[y[i]],
                        **{m: classes[pr[i]] for m, pr in zip(models, preds)}})
        for ax in list(axes.flat)[len(picked):]:
            ax.axis("off")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.14, wspace=0.08, hspace=0.75)
        acc_sample = {m: float(np.mean([pr[i] == y[i] for i in picked])) for m, pr in zip(models, preds)}
        acc_full = {m: float(np.mean(pr == y)) for m, pr in zip(models, preds)}
        for tag, (m, pr) in zip("AB", zip(models, preds)):  # caption numbers: model A = models[0], B = models[1]
            n_ok = int(sum(pr[i] == y[i] for i in picked))
            self.macros[f"Qual{tag}Correct"] = (str(n_ok), n_ok, files[models.index(m)])
            self.macros[f"Qual{tag}Acc"] = (fmt(acc_full[m], 3), acc_full[m], files[models.index(m)])
        self.macros["QualN"] = (str(len(picked)), len(picked), files[0])
        self.macros["QualNTest"] = (f"{len(y):,}".replace(",", "{,}"), len(y), files[0])
        self.save_figure(fig, name, {"dataset": dataset, "seed": seed, "sample_seed": sample_seed, "per_class": per_class,
                                     "patches": rec, "accuracy_on_sample": acc_sample, "accuracy_full_test": acc_full,
                                     "n_test": int(len(y)),
                                     "suggested_caption": f"{DATASET_NAMES[dataset]} test patches (seed {seed} split) drawn at "
                                     f"random, {per_class} per class from different source images, the same for both "
                                     "models; true class and each model's prediction (green correct, red wrong)."},
                         files)

    # ------------------------------------------------------------------ accuracy bar chart (categorical)
    def accuracy_barchart(self, metric="acc"):
        """Test accuracy per categorical dataset and model: bar = mean over seeds, error bar = SD over seeds (ddof=1),
        values from the per-seed json files (the same numbers as Table III). y axis from 0 (no truncation)."""
        import matplotlib.pyplot as plt
        name = "accuracy_barchart"
        root = os.path.join(self.results, "probe")
        models, present = self.discover(root, CATEGORICAL)
        if not present or not models:
            self.skip(f"{name}: no categorical results")
            return
        fig, ax = plt.subplots(figsize=(COL_W, 2.35))
        width = 0.8 / len(models)
        data, sources = {}, set()
        for j, m in enumerate(models):
            xs, ys, es = [], [], []
            for k, d in enumerate(present):
                s = self.seed_summary(root, m, d, metric)
                if s is None:
                    continue
                xs.append(k - 0.4 + width * (j + 0.5))
                ys.append(s["mean"])
                es.append(s["sd"] if s["sd"] is not None and np.isfinite(s["sd"]) else 0.0)
                data[f"{m}/{d}"] = {"mean": s["mean"], "sd": s["sd"], "n": s["n"]}
                sources.update(s["sources"])
            ax.bar(xs, ys, width=width, color=model_color(m), edgecolor="white", linewidth=0.3, label=model_label(m),
                   yerr=es, error_kw={"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6, "ecolor": "#222222"})
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([dataset_name(d) for d in present])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Test accuracy")
        ax.yaxis.grid(True, linewidth=0.3, color="#dddddd")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        h, lab = ax.get_legend_handles_labels()
        ncol = 3
        nrow = int(math.ceil(len(h) / ncol))
        order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < len(h)]  # read row by row
        ax.legend([h[i] for i in order], [lab[i] for i in order], ncol=ncol, fontsize=6, frameon=False,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8)
        fig.tight_layout(pad=0.2)
        self.save_figure(fig, name, {"metric": metric, "values": data,
                                     "suggested_caption": "Test accuracy per categorical dataset: bars, mean over five "
                                     "seeds; error bars, SD over seeds."}, sources)

    # ------------------------------------------------------------------ QWK heatmap (ordinal)
    def qwk_heatmap(self):
        """QWK per ordinal dataset (rows) and model (columns): mean over seeds, or the single official-partition
        result; the same numbers as the QWK column of Table III. Colour scale fixed to [0, 1]."""
        import matplotlib.pyplot as plt
        name = "qwk_heatmap"
        root = os.path.join(self.results, "probe")
        models, present = self.discover(root, ORDINAL)
        if not present or not models:
            self.skip(f"{name}: no ordinal results")
            return
        M = np.full((len(present), len(models)), np.nan)
        data, sources = {}, set()
        for i, d in enumerate(present):
            for j, m in enumerate(models):
                s = self.seed_summary(root, m, d, "qwk")
                if s is None:
                    continue
                M[i, j] = s["mean"]
                data[f"{m}/{d}"] = {"mean": s["mean"], "sd": s["sd"], "n": s["n"], "official": s["official"]}
                sources.update(s["sources"])
        fig, ax = plt.subplots(figsize=(COL_W, 0.42 * len(present) + 0.75))
        im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                            color="white" if M[i, j] > 0.6 else "#222222")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([model_label(m).replace(" (", "\n(") for m in models], fontsize=6.5)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([dataset_name(d) + ("$^{\\dagger}$" if d in OFFICIAL else "") for d in present], fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("QWK", fontsize=7)
        cb.ax.tick_params(labelsize=6, length=2)
        fig.tight_layout(pad=0.2)
        self.save_figure(fig, name, {"values": data, "suggested_caption": "QWK on the ordinal datasets for each model "
                                     "(mean over five seeds for PANDA; single official-partition result for BRACS and "
                                     "SICAPv2)."}, sources)

    # ------------------------------------------------------------------ numbers quoted in the Results text
    def text_macros(self):
        """Macros for numbers quoted in the Results text (\\Rpt...). Every value is recomputed from the per-seed json
        files (or the stats summary for the external test); ranges are min/max over the models shown."""
        root = os.path.join(self.results, "probe")
        models, _ = self.discover(root, DATASET_ORDER)
        if not models:
            self.skip("text macros: no probe results")
            return

        def mean(r, m, d, k):
            s = self.seed_summary(r, m, d, k)
            return None if s is None else (s["mean"], s["sources"])

        def put(name, vals, digits=3):
            vals = [v for v in vals if v is not None]
            if not vals:
                self.skip(f"text macros: no values for {name}")
                return
            src = sorted({f for _, fs in vals for f in fs})
            lo, hi = min(v for v, _ in vals), max(v for v, _ in vals)
            self.macros[name + "Min"] = (fmt(lo, digits), lo, src)
            self.macros[name + "Max"] = (fmt(hi, digits), hi, src)

        path_models = [m for m in models if m != "vit_in1k"]
        # ceiling datasets: lowest AUC of any model
        put("CeilAUC", [mean(root, m, d, "auc") for m in models for d in ("lung", "nct", "hubmap")])
        for d, tag in (("breakhis", "Breakhis"), ("bach", "Bach"), ("lung", "Lung")):
            put(f"{tag}AUC", [mean(root, m, d, "auc") for m in models])
            put(f"{tag}ImageAUC", [mean(os.path.join(self.results, "probe_imagesplit"), m, d, "auc") for m in models])
        # mean accuracy over the categorical datasets
        def mean_acc(m):
            v = [mean(root, m, d, "acc") for d in CATEGORICAL]
            return None if any(x is None for x in v) else (float(np.mean([x[0] for x in v])), [f for x in v for f in x[1]])
        put("VitMeanAcc", [mean_acc("vit_in1k")] if "vit_in1k" in models else [])
        put("PathMeanAcc", [mean_acc(m) for m in path_models])
        put("VitBreakhisAcc", [mean(root, "vit_in1k", "breakhis", "acc")] if "vit_in1k" in models else [])
        # majority-class share of the BreaKHis test sets (mean over seeds)
        fs = sorted(glob.glob(os.path.join(root, models[0], "breakhis", "seed*.npz")))
        if fs:
            shares = [float(np.bincount(np.load(f)["targets"]).max() / len(np.load(f)["targets"])) for f in fs]
            put("BreakhisMajority", [(float(np.mean(shares)), fs)])
        # external test (NCT -> CRC-VAL-HE-7K) and the internal NCT test, accuracy
        ext = os.path.join(self.results, "external")
        put("ExtAcc", [mean(ext, m, "nct_val7k", "acc") for m in models])
        put("NctAcc", [mean(root, m, "nct", "acc") for m in models])
        # PANDA slide level (ABMIL) vs tile level, and BRACS at fixed resolution (0.5 um/px tiles)
        from scipy.stats import spearmanr
        abm = os.path.join(self.results, "abmil")
        slide = {m: mean(abm, m, "panda", "qwk") for m in models}
        tile = {m: mean(root, m, "panda", "qwk") for m in models}
        both = [m for m in models if slide[m] and tile[m]]
        put("AbmilQWK", [slide[m] for m in both])
        if len(both) > 2:
            rho = float(spearmanr([tile[m][0] for m in both], [slide[m][0] for m in both])[0])
            self.macros["AbmilRankRho"] = (fmt(rho, 2), rho, sorted({f for m in both for f in slide[m][1] + tile[m][1]}))
        bt = os.path.join(self.results, "probe_bracs_tiles")
        tiles = {m: mean(bt, m, "bracs", "qwk") for m in models}
        whole = {m: mean(root, m, "bracs", "qwk") for m in models}
        both = [m for m in models if tiles[m] and whole[m]]
        put("BracsTileQWKPath", [tiles[m] for m in both if m != "vit_in1k"])
        put("BracsWholeQWKPath", [whole[m] for m in both if m != "vit_in1k"])
        if "vit_in1k" in both:
            put("BracsTileQWKVit", [tiles["vit_in1k"]])
        if len(both) > 2:
            rho = float(spearmanr([whole[m][0] for m in both], [tiles[m][0] for m in both])[0])
            self.macros["BracsTileRankRho"] = (fmt(rho, 2), rho, sorted({f for m in both for f in tiles[m][1] + whole[m][1]}))
        # speed-ups from the precision runs: per model, ms/image averaged over the categorical datasets (as in the
        # precision summary table), then FP32 / reduced precision on the same device
        prec = os.path.join(self.results, "precision")
        def ms(m, p):
            fs = [os.path.join(prec, m, d, f"{p}.json") for d in CATEGORICAL]
            if not all(os.path.exists(f) for f in fs):
                return None
            return float(np.mean([json.load(open(f))["timing"]["ms_per_image_mean"] for f in fs])), fs
        half, int8 = [], []
        for m in models:
            g32, c32, c8 = ms(m, "FP32_GPU"), ms(m, "FP32_CPU"), ms(m, "INT8_CPU")
            for p in ("FP16_GPU", "BF16_GPU"):
                h = ms(m, p)
                if g32 and h:
                    half.append((g32[0] / h[0], g32[1] + h[1]))
            if c32 and c8:
                int8.append((c32[0] / c8[0], c32[1] + c8[1]))
        put("GpuHalfSpeedup", half, digits=1)
        put("CpuIntSpeedup", int8, digits=1)
        tpath = os.path.join(self.base, "indexes", "tissue_panda.csv")
        if os.path.exists(tpath):
            t = pd.read_csv(tpath)
            frac = float((t.tissue_frac < 0.10).mean())
            self.macros["PandaExcludedPct"] = (f"{100 * frac:.0f}", frac, [tpath])

    # ------------------------------------------------------------------ overview figure (Fig. 1)
    def overview(self):
        """Schematic of the benchmark: five stages left to right. Content = facts stated in the Methods
        (no numbers or results), names as in Tables I and II."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        stages = [
            ("Datasets", ["Categorical:", "LC25000, BreaKHis,", "NCT-CRC, HuBMAP, BACH", "", "Ordinal:",
                          "PANDA, BRACS, SICAPv2", "", "Patient-, slide- or", "source-image-disjoint", "splits"]),
            ("Frozen embeddings", ["UNI2-h", "Virchow2", "CONCH", "Phikon", "CTransPath", "ViT-B/16 (ImageNet)", "",
                                   "each with its own", "preprocessing and", "embedding"]),
            ("Downstream models", ["Linear probe", "(logistic regression)", "on every dataset", "", "ABMIL",
                                   "(slide level, PANDA)"]),
            ("Evaluation", ["Accuracy, F1, AUC;", "QWK (ordinal)", "", "5 seeds, bootstrap CIs,", "paired tests,",
                            "Friedman / Nemenyi", "", "External test:", "CRC-VAL-HE-7K"]),
            ("Efficiency", ["Single host", "", "GPU: FP32, FP16, BF16", "CPU: FP32, INT8", "", "Agreement with FP32,",
                            "time per image"]),
        ]
        tints = ["#E8F1FA", "#FDF1E0", "#E6F4EF", "#F3EAF3", "#EFEFEF"]
        edges = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#666666"]
        W, H = TEXT_W, 2.0
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_axis_off()
        n, gap, left = len(stages), 0.22, 0.04
        bw = (W - 2 * left - gap * (n - 1)) / n
        top, bottom = H - 0.05, 0.34
        xs = [left + i * (bw + gap) for i in range(n)]
        for i, ((title, lines), x) in enumerate(zip(stages, xs)):
            ax.add_patch(FancyBboxPatch((x, bottom), bw, top - bottom, boxstyle="round,pad=0,rounding_size=0.06",
                                        facecolor=tints[i], edgecolor=edges[i], linewidth=0.8))
            ax.text(x + bw / 2, top - 0.08, f"{i + 1}  {title}", ha="center", va="top", fontsize=7.5,
                    fontweight="bold", color="#222222")
            ax.text(x + bw / 2, top - 0.33, "\n".join(lines), ha="center", va="top", fontsize=6.6,
                    color="#222222", linespacing=1.25)
            if i < n - 1 and i != 3:  # evaluation -> efficiency are parallel analyses, not a sequence
                ax.add_patch(FancyArrowPatch((x + bw + 0.02, (top + bottom) / 2), (x + bw + gap - 0.02, (top + bottom) / 2),
                                             arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color="#444444"))
        # efficiency uses the same frozen encoders: connector from stage 2 to stage 5 below the boxes
        y = bottom - 0.16
        x2, x5 = xs[1] + bw / 2, xs[4] + bw / 2
        ax.plot([x2, x2, x5], [bottom, y, y], color="#444444", linewidth=0.9)
        ax.add_patch(FancyArrowPatch((x5, y), (x5, bottom - 0.01), arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
                                     color="#444444"))
        ax.text((x2 + x5) / 2, y - 0.05, "same frozen encoders", ha="center", va="top", fontsize=6.6, color="#444444")
        self.save_figure(fig, "overview", {"stages": [t for t, _ in stages]}, [])

    # ------------------------------------------------------------------ additional models (supplement)
    def additional_models(self, extra=("hoptimus0", "gigapath"), digits=3):
        """Primary metric of models evaluated in addition to the main set (same protocol), one row per model;
        read directly from the per-seed json files, independently of the --models filter."""
        name = "additional_models"
        root = os.path.join(self.results, "probe")
        present = [d for d in DATASET_ORDER if any(os.path.isdir(os.path.join(root, m, d)) for m in extra)]
        rows, sources = [], set()
        for m in extra:
            cells = []
            for d in present:
                s = self.seed_summary(root, m, d, primary(d))
                if s is None:
                    cells.append("--")
                    continue
                cells.append(fmt(s["mean"], digits) if s["official"] or s["sd"] is None
                             else f"{fmt(s['mean'], digits)}$\\pm${fmt_sd(s['sd'], digits)}")
                self.record(name, m, d, f"{primary(d)} mean", s["mean"], fmt(s["mean"], digits), s["sources"])
                sources.update(s["sources"])
            if any(c != "--" for c in cells):
                rows.append(f"{MODEL_NAMES.get(m, m)} & " + " & ".join(cells) + " \\\\")
        if not rows:
            self.skip(f"{name}: no results for {extra}")
            return
        lines = ["\\begin{tabular}{l" + "c" * len(present) + "}", "\\toprule",
                 "Model & " + " & ".join(dataset_name(d) + ("$^{\\dagger}$" if d in OFFICIAL else "") for d in present) + " \\\\",
                 " & " + " & ".join(METRIC_NAMES[primary(d)] for d in present) + " \\\\", "\\midrule"] + rows + \
                ["\\bottomrule", "\\end{tabular}"]
        self.write_table(name, "\n".join(lines) + "\n", sources,
                         "Suggested caption: primary metric (AUC categorical, QWK ordinal), mean $\\pm$ SD over seeds.")

    # ------------------------------------------------------------------ 2. critical difference
    def cd_diagrams(self):
        fried, fsrc = self.stats.friedman()
        if not fried:
            self.skip("critical-difference diagrams: friedman.json empty or missing")
        for group in ["categorical", "ordinal", "all"]:
            g = fried.get(group)
            if not g:
                if fried:
                    self.skip(f"CD diagram '{group}': not in friedman.json (needs >= 2 datasets with every model)")
                continue
            key = group.capitalize()
            self.macros[f"Friedman{key}Chisq"] = (fmt(g["chi2"], 2), g["chi2"], fsrc)
            self.macros[f"Friedman{key}P"] = (self.fmt_p(g["p"]), g["p"], fsrc)
            self.macros[f"Friedman{key}N"] = (str(len(g["datasets"])), len(g["datasets"]), fsrc)
            self.macros[f"Friedman{key}K"] = (str(len(g["mean_rank"])), len(g["mean_rank"]), fsrc)
            if g.get("nemenyi_cd") is None:
                self.skip(f"CD diagram '{group}': no Nemenyi CD (too many models for the q table)")
                continue
            self.macros[f"Friedman{key}CD"] = (fmt(g["nemenyi_cd"], 2), g["nemenyi_cd"], fsrc)
            if len(g["datasets"]) < 3:
                self.warn(f"Friedman '{group}': only {len(g['datasets'])} datasets ({', '.join(g['datasets'])}); "
                          f"CD = {g['nemenyi_cd']:.2f} spans most of the rank range - low power")
            fig = self.draw_cd(g["mean_rank"], g["nemenyi_cd"])
            self.save_figure(fig, f"cd_{group}", dict(mean_rank=g["mean_rank"], cd=g["nemenyi_cd"], chi2=g["chi2"], p=g["p"],
                                                      datasets=g["datasets"]), [fsrc])

    @staticmethod
    def fmt_p(p):
        if p < 1e-4:
            m, e = f"{p:.1e}".split("e")
            return f"{m}\\times 10^{{{int(e)}}}"
        return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"

    def draw_cd(self, mean_rank, cd):
        import matplotlib.pyplot as plt
        models = sorted(mean_rank, key=lambda m: mean_rank[m])
        r = [mean_rank[m] for m in models]
        k = len(models)
        hi = max(k, math.ceil(1 + cd))
        # cliques: maximal runs of models whose mean ranks differ by <= CD
        cl = []
        for i in range(k):
            j = max(j for j in range(i, k) if r[j] - r[i] <= cd)
            if j > i:
                cl.append((i, j))
        cl = [c for c in cl if not any(o != c and o[0] <= c[0] and c[1] <= o[1] for o in cl)]
        n_left = math.ceil(k / 2)
        rows = max(n_left, k - n_left)
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        fp = FontProperties(family=plt.rcParams["font.family"], size=7)

        def width(t):  # rendered width in inches of a (possibly multi-line) 7-pt label
            return max(TextPath((0, 0), line, prop=fp).get_extents().width for line in t.split("\n")) / 72

        labels = []
        for i, m in enumerate(models):  # labels wider than 1.15 in wrap before the rank / the parenthesis
            t = f"{model_label(m)} ({r[i]:.2f})"
            if width(t) > 1.15 and " (" in model_label(m):
                t = model_label(m).replace(" (", "\n(", 1) + f" ({r[i]:.2f})"
            labels.append(t)
        two_lines = any("\n" in t for t in labels)
        y_axis, dy_cl, dy_lab = 0.50, 0.06, (0.20 if two_lines else 0.15)
        y_lab0 = y_axis + 0.10 + len(cl) * dy_cl + 0.06
        h = y_lab0 + (rows - 1) * dy_lab + (0.16 if two_lines else 0.08)
        ml = max(width(labels[i]) for i in range(n_left)) + 0.20
        mr = max(width(labels[i]) for i in range(n_left, k)) + 0.20
        xl, xr = max(0.75, ml), COL_W - max(0.75, mr)

        def x(v):
            return xl + (v - 1) / (hi - 1) * (xr - xl)

        fig = plt.figure(figsize=(COL_W, h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, COL_W)
        ax.set_ylim(h, 0)
        ax.set_axis_off()
        ink = "#222222"
        ax.plot([x(1), x(hi)], [y_axis, y_axis], color=ink, lw=0.8)
        for t in range(1, hi + 1):
            ax.plot([x(t), x(t)], [y_axis - 0.05, y_axis], color=ink, lw=0.8)
            ax.text(x(t), y_axis - 0.07, str(t), ha="center", va="bottom", fontsize=7)
        for t in np.arange(1.5, hi, 1.0):
            ax.plot([x(t), x(t)], [y_axis - 0.025, y_axis], color=ink, lw=0.6)
        yc = 0.20
        ax.plot([x(1), x(1 + cd)], [yc, yc], color=ink, lw=1.0)
        for v in (1, 1 + cd):
            ax.plot([x(v), x(v)], [yc - 0.03, yc + 0.03], color=ink, lw=1.0)
        ax.text((x(1) + x(1 + cd)) / 2, yc - 0.035, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
        for i, (a, b) in enumerate(cl):
            yy = y_axis + 0.10 + i * dy_cl
            ax.plot([x(r[a]) - 0.03, x(r[b]) + 0.03], [yy, yy], color=ink, lw=1.8, solid_capstyle="round")
        for i, m in enumerate(models):
            left = i < n_left
            row = i if left else k - 1 - i
            yy = y_lab0 + row * dy_lab
            xe = xl - 0.08 if left else xr + 0.08
            ax.plot([x(r[i]), x(r[i]), xe], [y_axis, yy, yy], color=ink, lw=0.6)
            ax.plot([x(r[i])], [y_axis], marker=MODEL_MARKERS.get(m, "o"), color=model_color(m), ms=4.5, mec="white", mew=0.4,
                    zorder=5)
            ax.text(xe + (-0.04 if left else 0.04), yy, labels[i], ha="right" if left else "left",
                    va="center", fontsize=7, linespacing=1.0)
        return fig

    # ------------------------------------------------------------------ 3. rank stability
    def rank_stability(self):
        ranks, src = self.stats.ranks()
        if ranks is None or ranks.empty:
            self.skip("rank stability: ranks.csv missing or empty")
            return
        datasets = order_by(ranks.dataset.unique(), DATASET_ORDER)
        models = order_by(ranks.model.unique(), MODEL_ORDER)
        info = {}
        for rw in ranks.itertuples():
            pr = np.array([getattr(rw, f"p_rank{j}", np.nan) for j in range(1, len(models) + 1)], float)
            pr = np.nan_to_num(pr)
            cum = np.cumsum(pr)
            lo = int(np.argmax(cum >= 0.025 - 1e-12)) + 1
            hi = int(np.argmax(cum >= 0.975 - 1e-12)) + 1
            info[(rw.model, rw.dataset)] = dict(mean_rank=float(rw.mean_rank), p_rank1=float(rw.p_rank1), lo=lo, hi=hi,
                                                 p=pr.tolist())
        name = "rank_stability"
        lines = ["\\begin{tabular}{l" + "c" * len(datasets) + "}", "\\toprule",
                 "Model & " + " & ".join(dataset_name(d) for d in datasets) + " \\\\", "\\midrule"]
        for m in models:
            cells = []
            for d in datasets:
                v = info.get((m, d))
                if v is None:
                    cells.append("--")
                    continue
                cells.append(f"{v['mean_rank']:.2f} ({100 * v['p_rank1']:.0f})")
                self.record(name, m, d, "mean rank", v["mean_rank"], f"{v['mean_rank']:.2f}", [src])
                self.record(name, m, d, "P(rank 1) %", 100 * v["p_rank1"], f"{100 * v['p_rank1']:.0f}", [src])
            lines.append(f"{model_name(m)} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        self.write_table(name, "\n".join(lines) + "\n", [src],
                         "Suggested caption: rank stability on the primary metric: mean rank over seeds x bootstrap "
                         "resamples (percentage of resamples in which the model ranks first).")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(TEXT_W, 2.3), layout="constrained")
        w = 0.8 / max(len(models), 1)
        for mi, m in enumerate(models):
            xs, ys, lo, hi = [], [], [], []
            for di, d in enumerate(datasets):
                v = info.get((m, d))
                if v is None:
                    continue
                xs.append(di - 0.4 + w * (mi + 0.5))
                ys.append(v["mean_rank"])
                lo.append(v["mean_rank"] - v["lo"])
                hi.append(v["hi"] - v["mean_rank"])
            ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt=MODEL_MARKERS.get(m, "o"),
                        color=model_color(m), ms=4.5, lw=0.9, capsize=0, label=model_label(m), mec="white", mew=0.3)
        ax.set_xticks(range(len(datasets)), [DATASET_NAMES.get(d, d) for d in datasets])
        ax.set_ylim(len(models) + 0.5, 0.5)
        ax.set_yticks(range(1, len(models) + 1))
        ax.set_ylabel("Rank (1 = best)")
        for di in range(1, len(datasets)):
            ax.axvline(di - 0.5, color="#dddddd", lw=0.5, zorder=0)
        ax.grid(axis="y", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.2,
                  columnspacing=1.0)
        self.save_figure(fig, "rank_stability", {f"{m}/{d}": v for (m, d), v in info.items()}, [src])

    # ------------------------------------------------------------------ 4. ABMIL vs tile-level probe on PANDA
    def abmil(self):
        aroot = os.path.join(self.results, "abmil")
        models, ds = self.discover(aroot, ["panda"])
        if not ds:
            self.skip("ABMIL: no results/abmil/<model>/panda results")
            return
        astats = Stats(self, os.path.join(self.results, "stats_abmil" + STATS_SUFFIX))
        astats.check_fresh(aroot, "panda")
        proot = os.path.join(self.results, "probe")
        pmodels, _ = self.discover(proot, ["panda"])
        models = order_by(set(models) | set(pmodels), MODEL_ORDER)
        tile, _, _ = self.gather(proot, self.stats, "panda", models, "abmil_panda")
        slide, _, _ = self.gather(aroot, astats, "panda", models, "abmil_panda")
        name = "abmil_panda"
        sources = set()
        lines = ["\\begin{tabular}{lcc}", "\\toprule",
                 "Model & Tile-level probe & Slide-level ABMIL \\\\", "\\midrule"]
        data = {}
        for m in models:
            t, s = tile.get(m), slide.get(m)
            if t is None and s is None:
                continue
            cells = []
            for lvl, x in (("tile", t), ("slide", s)):
                if x is None:
                    cells.append("--")
                    continue
                cells.append(self.value_cell(x, x["ci"][:2] if x["ci"] else None, x["mark"]))
                self.record(name, m, lvl, "qwk mean", x["mean"], fmt(x["mean"]), x["sources"])
                self.record(name, m, lvl, "qwk sd", x["sd"], fmt(x["sd"]), x["sources"])
                if x["ci"]:
                    self.record(name, m, lvl, "qwk ci_low", x["ci"][0], fmt(x["ci"][0]), [x["ci"][3]])
                    self.record(name, m, lvl, "qwk ci_high", x["ci"][1], fmt(x["ci"][1]), [x["ci"][3]])
                    sources.add(x["ci"][3])
                self.record(name, m, lvl, "mark", None, x["mark"] or "none", [x["p_source"]])
                sources.update(x["sources"] + [x["p_source"]])
            data[m] = dict(tile=None if t is None else dict(mean=t["mean"], sd=t["sd"], ci=t["ci"][:2] if t["ci"] else None),
                           slide=None if s is None else dict(mean=s["mean"], sd=s["sd"], ci=s["ci"][:2] if s["ci"] else None))
            lines.append(f"{model_name(m)} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        self.write_table(name, "\n".join(lines) + "\n", {x for x in sources if x},
                         "Suggested caption: PANDA ISUP grading, QWK (mean $\\pm$ SD over seeds; below, 95% cluster-bootstrap CI). "
                         "Tile-level: linear probe on tiles (tile labels = slide grade); slide-level: gated ABMIL over the "
                         "tile embeddings of each slide (same slide-disjoint splits). Marks as in the main results table, "
                         "within each column.")
        import matplotlib.pyplot as plt
        ms = [m for m in models if m in data]
        fig, ax = plt.subplots(figsize=(COL_W, 0.9 + 0.22 * len(ms)), layout="constrained")
        for i, m in enumerate(ms):
            pts = [(lvl, data[m][lvl]) for lvl in ("tile", "slide") if data[m][lvl]]
            if len(pts) == 2:
                ax.plot([pts[0][1]["mean"], pts[1][1]["mean"]], [i, i], color=model_color(m), lw=1.0, alpha=0.6)
            for lvl, v in pts:
                err = None if v["ci"] is None else [[v["mean"] - v["ci"][0]], [v["ci"][1] - v["mean"]]]
                ax.errorbar([v["mean"]], [i], xerr=err, fmt="o", ms=4.5, color=model_color(m), lw=0.9, capsize=1.5,
                            mfc=model_color(m) if lvl == "slide" else "white", mec=model_color(m), mew=1.0)
        ax.set_yticks(range(len(ms)), [model_label(m) for m in ms])
        ax.set_ylim(len(ms) - 0.5, -0.5)
        ax.set_xlabel("QWK (PANDA test set)")
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#444444", mfc="white", ms=4.5, label="Tile-level probe"),
                           Line2D([], [], marker="o", ls="", color="#444444", ms=4.5, label="Slide-level ABMIL")],
                  loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        self.save_figure(fig, "abmil_panda", data, {x for x in sources if x})

    # ------------------------------------------------------------------ 5. precision / latency
    def precision(self):
        root = os.path.join(self.results, "precision")
        found = glob.glob(os.path.join(root, "*", "*", "*.json"))
        if not found:
            self.skip("precision: no results/precision/<model>/<dataset>/<precision>.json")
            return
        models = order_by([f.split(os.sep)[-3] for f in found], MODEL_ORDER)
        datasets = order_by([f.split(os.sep)[-2] for f in found], DATASET_ORDER)
        R = {}
        for m in models:
            for d in datasets:
                for p in PRECISIONS:
                    f = os.path.join(root, m, d, f"{p}.json")
                    if not os.path.exists(f):
                        if os.path.isdir(os.path.join(root, m, d)):
                            self.skip(f"precision: {m}/{d}/{p} missing")
                        continue
                    j = json.load(open(f))
                    v = dict(acc=j["metrics"]["acc"], ms=j["timing"]["ms_per_image_mean"], ms_sd=j["timing"]["ms_per_image_sd"],
                             ips=j["timing"]["images_per_s_mean"], ips_sd=j["timing"]["images_per_s_sd"], n=j["n_images"],
                             threads=j["timing"].get("cpu_threads"), sources=[f], ref=None, agree=None, cos=None, cos_min=None)
                    if p == "FP32_GPU":
                        v["ref"] = "--"
                    elif p in ("FP16_GPU", "BF16_GPU", "FP32_CPU"):
                        c = j.get("vs_fp32_gpu", {})
                        v.update(ref="FP32 GPU", agree=c.get("prediction_agreement"), cos=c.get("cosine_mean"),
                                 cos_min=c.get("cosine_min"))
                    else:  # INT8 on the CPU: compared with FP32 on the CPU (same device), from the saved predictions
                        a, b = os.path.join(root, m, d, "INT8_CPU.npz"), os.path.join(root, m, d, "FP32_CPU.npz")
                        c = j.get("vs_fp32_gpu", {})
                        v["int8_vs_fp32_gpu"] = dict(agree=c.get("prediction_agreement"), cos=c.get("cosine_mean"))
                        if os.path.exists(a) and os.path.exists(b):
                            A, B = np.load(a), np.load(b)
                            if np.array_equal(A["rows"], B["rows"]):
                                fa, fb = A["feats"].astype(np.float64), B["feats"].astype(np.float64)
                                cos = (fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1))
                                v.update(ref="FP32 CPU", agree=float((A["probs"].argmax(1) == B["probs"].argmax(1)).mean()),
                                         cos=float(cos.mean()), cos_min=float(cos.min()))
                                v["sources"] += [a, b]
                            else:
                                self.warn(f"precision {m}/{d}: INT8_CPU and FP32_CPU images differ; INT8 compared with FP32 GPU")
                        if v["ref"] is None:
                            v.update(ref="FP32 GPU", agree=c.get("prediction_agreement"), cos=c.get("cosine_mean"),
                                     cos_min=c.get("cosine_min"))
                    R[(m, d, p)] = v
        # suspicious values
        for (m, d, p), v in R.items():
            if v["agree"] is not None and v["agree"] < 0.95:
                self.warn(f"precision {m}/{d}/{p}: agreement with {v['ref']} only {v['agree']:.3f}")
            if v["cos"] is not None and v["cos"] < 0.99:
                self.warn(f"precision {m}/{d}/{p}: mean feature cosine with {v['ref']} {v['cos']:.4f}")
        for m in models:
            for d in datasets:
                g, c = R.get((m, d, "FP32_GPU")), R.get((m, d, "INT8_CPU"))
                f32c = R.get((m, d, "FP32_CPU"))
                if c and f32c and c["ms"] > f32c["ms"]:
                    self.warn(f"precision {m}/{d}: INT8 CPU slower than FP32 CPU ({c['ms']:.1f} vs {f32c['ms']:.1f} ms/img)")
                for p in ("FP16_GPU", "BF16_GPU"):
                    h = R.get((m, d, p))
                    if g and h and h["ms"] > g["ms"]:
                        self.warn(f"precision {m}/{d}: {p} slower than FP32 GPU ({h['ms']:.2f} vs {g['ms']:.2f} ms/img)")
        # per-dataset inference time (mean +- SD over the timed repetitions): model x precision rows, dataset columns
        name = "precision_timing"
        lines = ["\\begin{tabular}{ll" + "c" * len(datasets) + "}", "\\toprule",
                 "Model & Prec. & " + " & ".join(DATASET_NAMES.get(d, d) for d in datasets) + " \\\\", "\\midrule"]
        srcs = set()
        for mi, m in enumerate(models):
            ps = [p for p in PRECISIONS if any((m, d, p) in R for d in datasets)]
            for i, p in enumerate(ps):
                cells = []
                for d in datasets:
                    v = R.get((m, d, p))
                    if v is None:
                        cells.append("--")
                        continue
                    srcs.update(v["sources"])
                    self.record(name, m, f"{d}/{p}", "ms_per_image_mean", v["ms"], f"{v['ms']:.2f}", v["sources"])
                    self.record(name, m, f"{d}/{p}", "ms_per_image_sd", v["ms_sd"], f"{v['ms_sd']:.2f}", v["sources"])
                    cells.append(f"{v['ms']:.2f}$\\pm${v['ms_sd']:.2f}")
                dev = "GPU" if p.endswith("GPU") else "CPU"
                first = f"\\multirow{{{len(ps)}}}{{*}}{{{model_name(m)}}}" if i == 0 else ""
                lines.append(f"{first} & {PRECISION_NAMES[p]} {dev} & " + " & ".join(cells) + " \\\\")
            if mi < len(models) - 1:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        thr = sorted({R[k]["threads"] for k in R if k[2].endswith("CPU") and R[k]["threads"]})
        self.write_table(name, "\n".join(lines) + "\n", srcs,
                         "Suggested caption: backbone inference time per image (ms), mean $\\pm$ SD over the timed "
                         f"repetitions (batch 32); CPU threads: {'/'.join(map(str, thr)) or 'n/a'}.")
        # summary over datasets: model x precision
        name = "precision_summary"
        lines = ["\\begin{tabular}{llcccc}", "\\toprule",
                 "Model & Prec. & Acc. & Agree (min) & ms/img & img/s \\\\", "\\midrule"]
        srcs, fig_data = set(), {}
        for mi, m in enumerate(models):
            full = [d for d in datasets if all((m, d, p) in R for p in PRECISIONS)]
            if not full:
                self.skip(f"precision_summary: {m} has no dataset with all precisions")
                continue
            if len(full) < len(datasets):
                self.skip(f"precision_summary: {m} averaged over {full} only (others incomplete)")
            for i, p in enumerate(PRECISIONS):
                vs = [R[(m, d, p)] for d in full]
                acc = float(np.mean([v["acc"] for v in vs]))
                ms = float(np.mean([v["ms"] for v in vs]))
                ips = float(np.mean([v["ips"] for v in vs]))
                ag = [v["agree"] for v in vs if v["agree"] is not None]
                agm, agmin = (float(np.mean(ag)), float(np.min(ag))) if ag else (None, None)
                src = [s for v in vs for s in v["sources"]]
                srcs.update(src)
                fig_data[f"{m}/{p}"] = dict(acc=acc, ms=ms, ips=ips, agree_mean=agm, agree_min=agmin, datasets=full)
                for q, val, txt in (("mean acc over datasets", acc, fmt(acc)), ("mean agreement", agm, fmt(agm)),
                                    ("min agreement", agmin, fmt(agmin)), ("mean ms/img", ms, f"{ms:.2f}"),
                                    ("mean img/s", ips, f"{ips:.1f}")):
                    self.record(name, m, p, q, val, txt, src)
                dev = "GPU" if p.endswith("GPU") else "CPU"
                first = f"\\multirow{{5}}{{*}}{{{model_name(m)}}}" if i == 0 else ""
                agtxt = "--" if agm is None else f"{fmt(agm)} ({fmt(agmin)})"
                lines.append(f"{first} & {PRECISION_NAMES[p]} {dev} & {fmt(acc)} & {agtxt} & {ms:.2f} & {ips:.1f} \\\\")
            if mi < len(models) - 1:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        if fig_data:
            self.write_table(name, "\n".join(lines) + "\n", srcs,
                             "Suggested caption: precision and latency averaged over the categorical datasets (accuracy, "
                             "agreement with the same-device FP32 reference: mean (minimum), backbone ms/image, images/s).")
            self.precision_figure(models, fig_data, srcs)

    def precision_figure(self, models, data, sources):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.55), layout="constrained")
        for ax, dev, precs in ((axes[0], "GPU", ["FP32_GPU", "FP16_GPU", "BF16_GPU"]), (axes[1], "CPU", ["FP32_CPU", "INT8_CPU"])):
            for m in models:
                pts = [(p, data[f"{m}/{p}"]) for p in precs if f"{m}/{p}" in data]
                if not pts:
                    continue
                ax.plot([v["ms"] for _, v in pts], [v["acc"] for _, v in pts], color=model_color(m), lw=0.7, alpha=0.7)
                for p, v in pts:
                    ax.plot(v["ms"], v["acc"], marker=PRECISION_MARKERS[p], color=model_color(m), ms=4, ls="",
                            mfc=model_color(m) if p.startswith("FP32") else "white", mew=0.9)
            ax.set_xscale("log")
            xs = [data[f"{m}/{p}"]["ms"] for m in models for p in precs if f"{m}/{p}" in data]
            if xs:
                lo_, hi_ = min(xs) / 1.4, max(xs) * 1.4
                ax.set_xlim(lo_, hi_)
                ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if lo_ <= t <= hi_]
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(f"ms / image, {dev} (log)")
            ax.grid(color="#eeeeee", lw=0.5)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Accuracy (mean over datasets)")
        lo = min(v["acc"] for v in data.values())
        hi = max(v["acc"] for v in data.values())
        for ax in axes:
            ax.set_ylim(lo - 0.02 * (hi - lo + 1e-3) - 0.01, hi + 0.02 * (hi - lo + 1e-3) + 0.01)
        axes[1].tick_params(labelleft=False)
        ms = [m for m in models if any(k.startswith(m + "/") for k in data)]
        handles = [Line2D([], [], color=model_color(m), marker="o", ms=3.5, lw=0.8, label=model_label(m)) for m in ms]
        handles += [Line2D([], [], color="#444444", marker=PRECISION_MARKERS[p], ls="", ms=3.5,
                           mfc="#444444" if p.startswith("FP32") else "white", label=PRECISION_NAMES[p])
                    for p in ["FP32_GPU", "FP16_GPU", "BF16_GPU", "INT8_CPU"]]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, handletextpad=0.3,
                   columnspacing=0.8)
        self.save_figure(fig, "precision_latency", data, sources)

    # ------------------------------------------------------------------ 6. embedding sensitivity
    def official_embedding(self):
        f = os.path.join(self.verification, "models.json")
        try:
            return {m: v["meta"]["official"] for m, v in json.load(open(f))["models"].items()}, f
        except (OSError, ValueError, KeyError):
            self.skip(f"sensitivity: {self.rel(f)} not readable - official embedding type not shown")
            return {}, None

    def sensitivity(self):
        roots = {k: os.path.join(self.results, d) for k, d in EMBEDDINGS}
        alt = {k: self.discover(r) for k, r in roots.items() if k != "official"}
        models = order_by([m for ms, _ in alt.values() for m in ms], MODEL_ORDER)
        for k, r in roots.items():
            if k != "official" and not alt[k][0]:
                self.skip(f"sensitivity: no {self.rel(r)} results")
        if not models:
            self.skip("sensitivity: no alternative-embedding results - table not written")
            return
        off, osrc = self.official_embedding()
        name = "embedding_sensitivity"
        lines = ["\\begin{tabular}{llc" + "c" * (len(EMBEDDINGS)) + "}", "\\toprule",
                 "Model & Dataset & Metric & " + " & ".join(EMBEDDING_NAMES[k] for k, _ in EMBEDDINGS) + " \\\\", "\\midrule"]
        srcs = {osrc} if osrc else set()
        missing = {}
        for mi, m in enumerate(models):
            rows = []
            for d in DATASET_ORDER + sorted(set(alt["cls"][1] + alt["patch_mean"][1]) - set(DATASET_ORDER)):
                metric = primary(d)
                ss = {k: self.seed_summary(roots[k], m, d, metric) for k, _ in EMBEDDINGS}
                for k in ss:
                    if ss[k] is None and (k == "official" or os.path.isdir(os.path.join(roots[k], m))):
                        missing.setdefault((m, k), []).append(d)
                if all(ss[k] is None for k in ss if k != "official"):
                    continue
                n_ref = ss["official"]["n"] if ss["official"] else None
                cells = []
                for k, _ in EMBEDDINGS:
                    s = ss[k]
                    if s is None:
                        cells.append("--")
                        continue
                    t = fmt(s["mean"]) if s["official"] else f"{fmt(s['mean'])}$\\pm${fmt_sd(s['sd'])}"
                    if n_ref is not None and s["n"] != n_ref:
                        t += f"$^{{n={s['n']}}}$"
                        self.warn(f"sensitivity: {m}/{d} {k} has {s['n']} seeds vs {n_ref} for the official embedding (run incomplete?)")
                    cells.append(t)
                    srcs.update(s["sources"])
                    self.record(name, m, d, f"{k} {metric} mean", s["mean"], fmt(s["mean"]), s["sources"])
                    if not s["official"]:
                        self.record(name, m, d, f"{k} {metric} sd", s["sd"], fmt(s["sd"]), s["sources"])
                # official embedding == CLS: the two probes must agree exactly
                if off.get(m, "").startswith("cls") and ss["official"] and ss["cls"] and ss["official"]["n"] == ss["cls"]["n"] \
                        and not np.allclose(ss["official"]["values"], ss["cls"]["values"], atol=1e-6, equal_nan=True):
                    self.warn(f"sensitivity: {m}/{d}: official embedding is CLS but official and cls probes differ "
                              f"({ss['official']['mean']:.4f} vs {ss['cls']['mean']:.4f})")
                rows.append((d, metric, cells))
            label = model_name(m) + (f" \\\\ {{\\scriptsize official: {tex_escape(off[m])}}}" if m in off else "")
            for i, (d, metric, cells) in enumerate(rows):
                first = f"\\multirow{{{len(rows)}}}{{*}}{{\\makecell[l]{{{label}}}}}" if i == 0 else ""
                lines.append(f"{first} & {dataset_name(d)} & {METRIC_NAMES[metric]} & " + " & ".join(cells) + " \\\\")
            if rows and mi < len(models) - 1:
                lines.append("\\midrule")
        for (m, k), ds in missing.items():
            self.skip(f"sensitivity: {m} {k}: no results for {', '.join(ds)}")
        for k, _ in EMBEDDINGS[1:]:
            absent = [m for m in models if not os.path.isdir(os.path.join(roots[k], m))]
            if absent:
                self.skip(f"sensitivity: {k}: no results for {', '.join(absent)}")
        lines += ["\\bottomrule", "\\end{tabular}"]
        self.write_table(name, "\n".join(lines) + "\n", srcs,
                         "Suggested caption: embedding sensitivity: primary metric of the linear probe on the official "
                         "embedding, the class token (CLS) and the mean of the patch tokens (mean $\\pm$ SD over seeds; one "
                         "value for official partitions). $^{n=k}$: only k seeds available.")

    # ------------------------------------------------------------------ 7. external test
    def external(self):
        root = os.path.join(self.results, "external")
        models, datasets = self.discover(root)
        if not datasets:
            self.skip("external test: no results/external/<model>/<dataset>/seed*.json")
            return
        estats = Stats(self, os.path.join(self.results, "stats_external" + STATS_SUFFIX))
        proot = os.path.join(self.results, "probe")
        name = "external"
        srcs = set()
        for ed in datasets:
            estats.check_fresh(root, ed)
        ext = {ed: self.gather(root, estats, ed, models, name)[0] for ed in datasets}
        internal, _, _ = self.gather(proot, self.stats, "nct", models, name)
        ms = order_by(set(models), MODEL_ORDER)
        cols = [("nct", "internal")] + [(ed, "external") for ed in datasets]
        lines = ["\\begin{tabular}{l" + "cc" * len(cols) + "}", "\\toprule",
                 "Model & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{dataset_name(d)} ({kind})}}" for d, kind in cols) + " \\\\",
                 " & " + " & ".join("AUC & Acc." for _ in cols) + " \\\\", "\\midrule"]
        for m in ms:
            cells = []
            for d, kind in cols:
                root_ = proot if kind == "internal" else root
                s = internal.get(m) if kind == "internal" else ext[d].get(m)
                if s is None:
                    cells += ["--", "--"]
                    continue
                cells.append(self.value_cell(s, s["ci"][:2] if s["ci"] else None, s["mark"]))
                acc = self.seed_summary(root_, m, d, "acc")
                cells.append(f"{fmt(acc['mean'])}$\\pm${fmt_sd(acc['sd'])}" if acc and not acc["official"] else fmt(acc["mean"]) if acc else "--")
                self.record(name, m, f"{d}/{kind}", "auc mean", s["mean"], fmt(s["mean"]), s["sources"])
                if s["sd"] is not None:
                    self.record(name, m, f"{d}/{kind}", "auc sd", s["sd"], fmt(s["sd"]), s["sources"])
                if s["ci"]:
                    self.record(name, m, f"{d}/{kind}", "auc ci_low", s["ci"][0], fmt(s["ci"][0]), [s["ci"][3]])
                    self.record(name, m, f"{d}/{kind}", "auc ci_high", s["ci"][1], fmt(s["ci"][1]), [s["ci"][3]])
                    srcs.add(s["ci"][3])
                if acc:
                    self.record(name, m, f"{d}/{kind}", "acc mean", acc["mean"], fmt(acc["mean"]), acc["sources"])
                    srcs.update(acc["sources"])
                srcs.update(s["sources"])
            lines.append(f"{model_name(m)} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        self.write_table(name, "\n".join(lines) + "\n", srcs,
                         "Suggested caption: external test: probes trained on NCT-CRC-HE-100K evaluated on the internal "
                         "NCT-CRC test split and on the external set (mean $\\pm$ SD over seeds; below, 95% bootstrap CI).")

    # ------------------------------------------------------------------ output
    def write_macros(self):
        path = os.path.join(self.out, "macros.tex")
        lines = ["% Generated by pipeline/report.py - do not edit by hand."]
        for k, (txt, _, _) in sorted(self.macros.items()):
            cmd = "\\Rpt" + re.sub(r"[^A-Za-z]", "", k)
            lines.append(f"\\providecommand{{{cmd}}}{{}}\\renewcommand{{{cmd}}}{{\\ensuremath{{{txt}}}}}")
        open(path, "w").write("\n".join(lines) + "\n")
        print(f"[macros] {self.rel(path)}")

    def write_summary(self):
        path = os.path.join(self.out, "summary.json")
        out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), results_dir=self.results, alpha=self.alpha,
                   official_ci=self.official_ci, skipped=self.skipped, warnings=self.warnings, tables=self.tables,
                   figures=self.figures,
                   macros={("Rpt" + re.sub(r"[^A-Za-z]", "", k)): dict(text=t, value=v, source=[self.rel(x) for x in s] if isinstance(s, (list, tuple)) else self.rel(s))
                           for k, (t, v, s) in self.macros.items()})

        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, float)):
                return float(o) if np.isfinite(o) else None
            if isinstance(o, np.integer):
                return int(o)
            return o
        json.dump(clean(out), open(path, "w"), indent=1)
        print(f"[json]   {self.rel(path)}")

    def run(self):
        setup_matplotlib()
        steps = [("benchmark table (Table III)", self.benchmark_table),
                 ("main results (categorical)", lambda: self.results_table("categorical", CATEGORICAL)),
                 ("main results (ordinal)", lambda: self.results_table("ordinal", ORDINAL)),
                 ("secondary (categorical)", lambda: self.secondary_table("categorical", CATEGORICAL)),
                 ("secondary (ordinal)", lambda: self.secondary_table("ordinal", ORDINAL)),
                 ("overview figure", self.overview),
                 ("qualitative figure", self.qualitative),
                 ("accuracy bar chart", self.accuracy_barchart),
                 ("QWK heatmap", self.qwk_heatmap),
                 ("text macros", self.text_macros),
                 ("additional models", self.additional_models),
                 ("critical difference", self.cd_diagrams), ("rank stability", self.rank_stability),
                 ("ABMIL", self.abmil), ("precision", self.precision), ("embedding sensitivity", self.sensitivity),
                 ("external test", self.external)]
        for label, fn in steps:
            try:
                fn()
            except Exception as e:  # one broken input must not stop the other outputs
                import traceback
                traceback.print_exc()
                self.skip(f"{label}: FAILED with {e!r}")
        self.write_macros()
        self.write_summary()
        print(f"\n{len(self.skipped)} skipped:" + "".join(f"\n  - {s}" for s in self.skipped))
        print(f"{len(self.warnings)} warnings:" + "".join(f"\n  - {w}" for w in self.warnings))


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "axes.titlesize": 7.5, "lines.linewidth": 1.0, "savefig.dpi": 300, "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222", "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#222222",
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(CODE_DIR, "results"))
    ap.add_argument("--out", default=os.path.join(CODE_DIR, "generated"))
    ap.add_argument("--verification", default=None, help="directory with models.json (default: <results>/../verification)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--official-ci", choices=["single_seed", "stats"], default="single_seed")
    ap.add_argument("--models", nargs="+", default=None, help="only these models (default: all found on disk)")
    ap.add_argument("--stats-suffix", default="", help="read results/stats<suffix> etc. (statistics computed for another model set)")
    args = ap.parse_args(argv)
    global MODELS_ONLY, STATS_SUFFIX
    MODELS_ONLY, STATS_SUFFIX = (set(args.models) if args.models else None), args.stats_suffix
    rep = Report(args.results, args.out, args.alpha, args.official_ci, args.verification)
    rep.run()
    return rep


if __name__ == "__main__":
    main()
