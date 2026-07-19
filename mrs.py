import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


OBJ_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"]


def _safe_iqr(data):
    """IQR per column, floored so later divisions never hit zero."""
    iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
    return np.maximum(iqr, 1e-12)


def density_filter(pf, threshold, n_neighbors=10):
    """Keep the densest `threshold` fraction of PF points (IQR-normalized NN)."""
    iqr = _safe_iqr(pf)
    pf_norm = pf / iqr
    k = min(n_neighbors + 1, pf_norm.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k).fit(pf_norm)
    dist_nbrs, _ = nbrs.kneighbors(pf_norm)
    sparse_score = dist_nbrs[:, 1:].mean(axis=1)
    cutoff = np.quantile(sparse_score, threshold)
    return pf[sparse_score <= cutoff]


def pf_to_csv(pf, output_dir, run_type, sub_id):
    obj_cols = [f"o{i}" for i in range(pf.shape[1])]
    df = pd.DataFrame(pf, columns=obj_cols).astype(int)
    pf_path = Path(output_dir) / f"{run_type}_{sub_id}_pf.csv"
    df.to_csv(pf_path, index=False)
    print(f"Saved {len(df)} points → {pf_path}")
    return df, pf_path


def get_mrs_params():
    return {"k": 100, "epsilon": 0.10}


def get_pf_info(df):
    data = df.values.astype(float)
    return {
        "data": data,
        "obj_names": list(df.columns),
        "n": data.shape[0],
        "d": data.shape[1],
        "iqr": _safe_iqr(data),
        "nadir": data.min(axis=0),
        "ideal": data.max(axis=0),
    }


def get_query_point(data, aspi, iqr):
    dists = np.abs((data - aspi) / iqr).sum(axis=1)
    query_idx = int(np.argmin(dists))
    return query_idx, data[query_idx]


def compute_mrs_at_query(data, k, d, iqr, query_idx):
    poly = PolynomialFeatures(degree=2, include_bias=True)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(data / iqr)
    _, nn_idx = nn.kneighbors((data[query_idx] / iqr).reshape(1, -1))
    nbrs = data[nn_idx[0, 1:]] - data[query_idx]

    betas_raw = np.zeros((d, d))
    betas = np.zeros((d, d))
    for j in range(d):
        input_cols = [c for c in range(d) if c != j]
        X_poly = poly.fit_transform(nbrs[:, input_cols])
        coef, _, _, _ = np.linalg.lstsq(X_poly, nbrs[:, j], rcond=None)
        for pos, i in enumerate(input_cols):
            betas_raw[i, j] = coef[pos + 1]
            betas[i, j] = coef[pos + 1] * iqr[i] / iqr[j]

    return betas_raw, betas


def save_mrs_results(run_type, seed, betas_raw, betas, output_dir, obj_names):
    """Save raw and IQR-scaled MRS matrices. Raises if either path exists."""
    output_dir = Path(output_dir)
    raw_path = output_dir / f"{run_type}_{seed}_betas_raw.csv"
    scaled_path = output_dir / f"{run_type}_{seed}_betas.csv"
    for path in (raw_path, scaled_path):
        if path.exists():
            raise ValueError(f"File {path} already exists")

    pd.DataFrame(betas_raw, columns=obj_names, index=obj_names).to_csv(raw_path)
    pd.DataFrame(betas, columns=obj_names, index=obj_names).to_csv(scaled_path)
    return raw_path, scaled_path


def plot_bar_chart(betas_raw, obj_names, query_pt, d, output_dir, run_type, seed,
                   obj_colors=None):
    obj_colors = obj_colors or OBJ_COLORS
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, d, figsize=(4 * d, 5), facecolor="#FAFAF8", squeeze=False)
    fig.suptitle(
        f"Tradeoff rates at the query point\n"
        + "  ".join(f"{n}={v:.1f}" for n, v in zip(obj_names, query_pt))
        + "\n(change in oⱼ per 1 unit improvement in oᵢ)",
        fontsize=10, fontweight="bold", y=1.03,
    )
    for i, ax in enumerate(axes[0]):
        others = [j for j in range(d) if j != i]
        vals = [betas_raw[i, j] for j in others]
        bars = ax.bar(
            np.arange(len(others)), vals,
            color=[obj_colors[j] for j in others],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val - 0.02,
                f"{val:.3f}", ha="center", va="top",
                fontsize=9, fontweight="bold", color="white",
            )
        ax.axhline(0, color="red", lw=1.2, alpha=0.8)
        ax.set_xticks(np.arange(len(others)))
        ax.set_xticklabels([obj_names[j] for j in others], fontsize=11)
        ax.set_title(f"Improving {obj_names[i]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("change in oⱼ" if i == 0 else "", fontsize=9)
        ax.set_ylim(-1.5, 0.2)
        ax.set_facecolor("#F7F7F5")
        ax.grid(True, axis="y", lw=0.3, color="#DDDDDD")
        ax.set_xlim(-0.5, len(others) - 0.5)

    plt.tight_layout()
    out_path = output_dir / f"{run_type}_{seed}_mrs_bar_chart.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
