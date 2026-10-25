"""v3 experiment runner and results JSON (see README.md)."""
import hashlib
import inspect
import json
import math
import os
import platform
import warnings

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from library.v3 import evaluation as ev
from library.v3.data import realised_meta

OUT_DIR = "Plot/v3"
DEFAULT_NOISE_RATIO = 0.2


def resolve_condition(name):
    """
    Condition vocabulary:
      'clean'            unpolluted data, imputer None (reference)
      'polluted'         polluted data, imputer None (complete-but-degraded data, e.g. distinctness)
      an imputer name    polluted data imputed per fold ('mean', 'median', 'most_frequent', 'std_constant',
                         'knn', 'iterative', 'mean_indicator', 'native'; 'native' runs only NaN-capable models)
      'duplicated', 'deduplicated', 'naive_split'
                         duplicate-records conditions (pollute must return groups)
    """
    if name == "clean":
        return {"polluted": False, "imputer": None, "kind": "cv"}
    if name == "polluted":
        return {"polluted": True, "imputer": None, "kind": "cv"}
    if name in ev.DUPLICATE_CONDITIONS:
        return {"polluted": True, "imputer": None, "kind": "duplicates"}
    if name in ev.IMPUTERS and name is not None:
        return {"polluted": True, "imputer": name, "kind": "cv"}
    raise ValueError(f"unknown condition {name!r}")


def derive_seeds(seed):
    """Independent sub-seeds for data generation, pollution and splits/models from one repetition seed."""
    data, pollute, split = np.random.SeedSequence(int(seed)).generate_state(3)
    return {"data": int(data), "pollute": int(pollute), "split": int(split)}


def _data_hash(X, y):
    h = hashlib.sha1(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes())
    h.update(np.ascontiguousarray(np.asarray(y, dtype=float)).tobytes())
    return h.hexdigest()[:12]


def takes_x(make_data):
    """True if make_data accepts (seed, x): two or more positional parameters, or a parameter named 'x'."""
    params = inspect.signature(make_data).parameters.values()
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    return len(positional) >= 2 or varargs or any(p.name == "x" for p in params)


def _polluted(make_data, pollute, seed, x, condition):
    """Data and pollution for one (seed, x, condition): identical across conditions and models (paired)."""
    seeds = derive_seeds(seed)
    X, y, y_scale = make_data(seeds["data"], x) if takes_x(make_data) else make_data(seeds["data"])
    meta, groups = {}, None
    if resolve_condition(condition)["polluted"]:
        # Legacy polluters seed the global numpy RNG; keep the caller's state untouched (n_jobs=1 runs in-process).
        state = np.random.get_state()
        try:
            out = pollute(X.copy(), np.array(y, copy=True), x, seeds["pollute"])
        finally:
            np.random.set_state(state)
        if len(out) == 4:
            X, y, meta, groups = out
        else:
            X, y, meta = out
    meta = {**dict(meta or {}), "realised": realised_meta(X), "seeds": seeds}
    return X, np.asarray(y), y_scale, meta, groups


def _task(make_data, pollute, seed, x, condition, model, n_splits, test_size):
    with threadpool_limits(limits=1):
        spec = resolve_condition(condition)
        X, y, y_scale, meta, groups = _polluted(make_data, pollute, seed, x, condition)
        split_seed = derive_seeds(seed)["split"]
        if spec["kind"] == "duplicates":
            if groups is None:
                raise ValueError(f"condition {condition!r} needs pollute to return groups")
            folds = ev.evaluate_duplicates(X, y, groups, model, condition, split_seed, y_scale, n_splits, test_size)
        else:
            folds = ev.evaluate(X, y, model, spec["imputer"], split_seed, y_scale, n_splits, test_size)
    base = {"seed": seed, "x": x, "condition": condition, "model": model, "imputer": spec["imputer"],
            "data_hash": _data_hash(X, y), "y_scale": float(y_scale)}
    return [{**base, **f, "meta": meta} for f in folds]


def _skip(model, condition):
    return resolve_condition(condition)["imputer"] == "native" and model not in ev.NATIVE_NAN_MODELS


def run_experiment(exp_id, title, x_label, x_axis, make_data, pollute, models, conditions, seeds, n_jobs=1,
                   n_splits=ev.N_SPLITS, test_size=ev.TEST_SIZE, config=None, out_dir=OUT_DIR):
    """
    Run every (seed, x, condition, model) task in parallel and write <out_dir>/results/<exp_id>.json.

    make_data(seed) or make_data(seed, x) -> (X, y, y_scale); pollute(X, y, x, seed) -> (X, y, meta) or (X, y, meta, groups), called
    on copies. If make_data takes x the data (and the 'clean' condition) are generated and fitted per x, for
    sample-size or shape axes. Otherwise 'clean' is fitted once per (seed, model) and reused for every x (records
    carry reused_from_x), which requires pollute to keep the number of rows; a mismatch raises. Returns the results payload (also written to disk; see results_path / load_results).
    """
    for c in conditions:
        resolve_condition(c)
    # Unpolluted conditions do not depend on x when make_data ignores x: fit them once per (seed, model).
    per_x = takes_x(make_data)

    def reused(c):
        return not per_x and not resolve_condition(c)["polluted"]

    tasks = [(s, x, c, m) for s in seeds for x in x_axis for c in conditions for m in models
             if not _skip(m, c) and (not reused(c) or x == x_axis[0])]
    done = Parallel(n_jobs=n_jobs)(
        delayed(_task)(make_data, pollute, s, x, c, m, n_splits, test_size) for s, x, c, m in tasks)
    done = {task[:1] + task[2:] if reused(task[2]) else task: recs for task, recs in zip(tasks, done)}
    chunks = []
    clean_rows, other_rows = {}, {}
    for s, x, c, m in ((s, x, c, m) for s in seeds for x in x_axis for c in conditions for m in models):
        if _skip(m, c):
            continue
        recs = done[(s, c, m) if reused(c) else (s, x, c, m)]
        origin = x_axis[0] if reused(c) and x != x_axis[0] else None
        chunks.append([{**r, "x": x, "reused_from_x": origin} for r in recs])
        rows = recs[0]["meta"]["realised"]["n_rows"]
        if reused(c):
            clean_rows[s] = rows
        elif resolve_condition(c)["kind"] == "cv" and resolve_condition(c)["polluted"]:
            other_rows.setdefault(s, {})[x] = rows
    for s, by_x in other_rows.items():
        bad = sorted(x for x, rows in by_x.items() if s in clean_rows and rows != clean_rows[s])
        if bad:
            raise ValueError(
                f"pollute changed the number of rows for x={bad} (seed {s}) but 'clean' was fitted once on the "
                f"unpolluted data; define make_data(seed, x) to generate the data per x, or drop 'clean'")
    payload = {
        "version": "v3",
        "exp_id": exp_id,
        "title": title,
        "x_label": x_label,
        "x_axis": list(x_axis),
        "models": list(models),
        "conditions": list(conditions),
        "seeds": list(seeds),
        "config": {
            "n_splits": n_splits,
            "test_size": test_size,
            "noise_ratio": DEFAULT_NOISE_RATIO,
            "gp": {"length_scale_bounds": list(ev.GP_LENGTH_SCALE_BOUNDS), "n_restarts_optimizer": ev.GP_RESTARTS},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
            **(config or {}),
        },
        "records": [r for chunk in chunks for r in chunk],
    }
    payload = to_strict(payload)
    path = results_path(exp_id, out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, allow_nan=False, indent=1)
    return payload


def to_strict(obj):
    """Recursively convert numpy types to Python and non-finite floats to None (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): to_strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_strict(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def results_path(exp_id, out_dir=OUT_DIR):
    return os.path.join(out_dir, "results", f"{exp_id}.json")


def _reject_constant(name):
    raise ValueError(f"non-strict JSON constant {name}")


def load_results(path_or_exp_id, out_dir=OUT_DIR):
    """Load a results payload from a path or an exp_id (strict JSON: NaN/Infinity are rejected)."""
    path = path_or_exp_id if path_or_exp_id.endswith(".json") else results_path(path_or_exp_id, out_dir)
    with open(path) as f:
        return json.load(f, parse_constant=_reject_constant)


def records_frame(payload):
    """Per-fold records as a DataFrame (meta column kept as dicts); adds gap = test_nrmse - train_nrmse."""
    df = pd.DataFrame(payload["records"])
    df["gap"] = df["test_nrmse"] - df["train_nrmse"]
    return df


def summarise(payload, metric="test_nrmse"):
    """
    Folds averaged within each seed (null metrics skipped and counted in n_null, with a warning), then mean and SD (ddof=1; 0 for one seed) over seeds per
    (condition, model, x). The SD band is descriptive, not a confidence interval.
    """
    df = records_frame(payload)
    per_seed = df.groupby(["condition", "model", "x", "seed"], sort=False)[metric].mean().reset_index()
    out = per_seed.groupby(["condition", "model", "x"], sort=False)[metric].agg(
        mean="mean", sd=lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0, n_seeds="count").reset_index()
    nulls = df[metric].isna().groupby([df["condition"], df["model"], df["x"]], sort=False).sum()
    out["n_null"] = nulls.reindex(pd.MultiIndex.from_frame(out[["condition", "model", "x"]])).to_numpy().astype(int)
    if out["n_null"].any():
        bad = out[out["n_null"] > 0]
        warnings.warn(f"{int(bad['n_null'].sum())} null {metric} fold values in {len(bad)} group(s) were skipped "
                      f"in the mean (first: {bad.iloc[0][['condition', 'model', 'x']].to_dict()}); see n_null",
                      RuntimeWarning, stacklevel=2)
    return out
