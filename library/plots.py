"""v3 figures built only from a results JSON (no model is rerun). See README.md."""
import os
import re

import matplotlib.pyplot as plt

from library import evaluation as ev
from library.runner import OUT_DIR, load_results, summarise

BAND_LABEL = "mean +- SD over seeds (descriptive)"
# Fixed categorical order (colour follows the entity, never its rank); markers are the secondary encoding.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#5f5e5a"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
CONDITION_ORDER = ["clean", "polluted", "mean", "median", "most_frequent", "std_constant", "knn", "iterative",
                   "mean_indicator", "native", "duplicated", "deduplicated", "naive_split"]
METRICS = {
    "test_nrmse": "test nRMSE (RMSE / std(y clean))",
    "gap": "generalisation gap (test - train nRMSE)",
    "fit_time": "fit time per fold [s]",
}


def _style(key, order):
    i = order.index(key) if key in order else len(order)
    return {"color": PALETTE[i % len(PALETTE)], "marker": MARKERS[i % len(MARKERS)],
            "linestyle": "-" if i < len(PALETTE) else "--"}


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def _draw(ax, table, key_col, keys, order, x_axis):
    for key in keys:
        part = table[table[key_col] == key].set_index("x").reindex(x_axis)
        if part["mean"].isna().all():
            continue
        st = _style(key, order)
        n_null = int(part["n_null"].fillna(0).sum())
        label = f"{key} ({n_null} null folds)" if n_null else key
        ax.plot(x_axis, part["mean"], label=label, linewidth=2, markersize=5, **st)
        ax.fill_between(x_axis, part["mean"] - part["sd"], part["mean"] + part["sd"], color=st["color"],
                        alpha=0.15, linewidth=0)


def _finish(fig, ax, title, x_label, y_label, n_seeds, path):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, color="#d9d8d3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, title=f"band: {BAND_LABEL}, n = {n_seeds}", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_experiment(source, out_dir=OUT_DIR):
    """
    Write all figures of one experiment from its results JSON (path or exp_id) into <out_dir>/<exp_id>/:
    per condition <condition>_<metric>.pdf for test_nrmse, gap and fit_time (one line per model), and per model
    model_<model>_test_nrmse.pdf comparing conditions. Returns the list of written paths.
    """
    payload = load_results(source, out_dir) if isinstance(source, str) else source
    x_axis, x_label = payload["x_axis"], payload["x_label"]
    model_order = list(ev.MODELS)
    n_seeds = len(payload["seeds"])
    directory = os.path.join(out_dir, _slug(payload["exp_id"]))
    os.makedirs(directory, exist_ok=True)
    paths = []
    for metric, y_label in METRICS.items():
        table = summarise(payload, metric)
        for condition in payload["conditions"]:
            sub = table[table["condition"] == condition]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            _draw(ax, sub, "model", [m for m in payload["models"] if m in set(sub["model"])], model_order, x_axis)
            paths.append(_finish(fig, ax, f"{payload['title']} - {condition}", x_label, y_label, n_seeds,
                                 os.path.join(directory, f"{_slug(condition)}_{metric}.pdf")))
    table = summarise(payload, "test_nrmse")
    for model in payload["models"]:
        sub = table[table["model"] == model]
        if sub["condition"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        _draw(ax, sub, "condition", [c for c in payload["conditions"] if c in set(sub["condition"])],
              CONDITION_ORDER, x_axis)
        paths.append(_finish(fig, ax, f"{payload['title']} - {model} across conditions", x_label,
                             METRICS["test_nrmse"], n_seeds,
                             os.path.join(directory, f"model_{_slug(model)}_test_nrmse.pdf")))
    return paths
