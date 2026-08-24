import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OBJ_COLORS = {
    "PTS": "#4C78A8",   # blue
    "AST": "#F58518",   # orange
    "REB": "#54A24B",   # green
    "STL": "#E45756",   # red
    "BLK": "#B279A2",   # purple
    "3P": "#FF9DA6",   # pink
    "G": "#9D755D",   # brown
    "FG%": "#2CB1BC",  # teal
}


def _safe_iqr(data):
    """IQR per column, floored so later divisions never hit zero."""
    iqr = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
    return np.maximum(iqr, 1e-12)


def density_filter(pf, threshold, n_neighbors=10):
    """Keep the densest `threshold` fraction of PF points (IQR-normalized NN)."""
    is_dataframe = isinstance(pf, pd.DataFrame)
    values = pf.to_numpy(dtype=float) if is_dataframe else np.asarray(pf, dtype=float)
    
    iqr = _safe_iqr(values)
    pf_norm = values / iqr
    k = min(n_neighbors + 1, pf_norm.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k).fit(pf_norm)
    dist_nbrs, _ = nbrs.kneighbors(pf_norm)
    sparse_score = dist_nbrs[:, 1:].mean(axis=1)
    cutoff = np.quantile(sparse_score, threshold)
    kept_idx = np.flatnonzero(sparse_score <= cutoff)
    filtered_pf = pf.iloc[kept_idx].reset_index(drop=True) if is_dataframe else values[kept_idx]
    return filtered_pf, kept_idx


def pf_to_csv(pf, output_dir, obj_names):
    df = pd.DataFrame(pf, columns=obj_names)
    pf_path = Path(output_dir) / "pf.csv"
    df.to_csv(pf_path, index=False)
    print(f"Saved {len(df)} points → {pf_path}")
    return df, pf_path


def get_mrs_params():
    return {"k": 100, "epsilon": 0.30}


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


def save_mrs_results(betas_raw, betas, output_dir, obj_names):
    """Save raw and IQR-scaled MRS matrices. Raises if either path exists."""
    output_dir = Path(output_dir)
    raw_path = output_dir / "betas_raw.csv"
    scaled_path = output_dir / "betas.csv"
    for path in (raw_path, scaled_path):
        if path.exists():
            raise ValueError(f"File {path} already exists")

    pd.DataFrame(betas_raw, columns=obj_names, index=obj_names).to_csv(raw_path)
    pd.DataFrame(betas, columns=obj_names, index=obj_names).to_csv(scaled_path)
    return raw_path, scaled_path


def plot_bar_chart(
    betas, obj_names, query_pt, d, output_dir, obj_colors=None
):
    """Bar chart of IQR-scaled MRS (Δ%IQR oⱼ per 1% IQR improvement in oᵢ)."""
    obj_colors = obj_colors or OBJ_COLORS
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, d, figsize=(4 * d, 5), facecolor="#FAFAF8", squeeze=False)
    fig.suptitle(
        f"Tradeoff rates at the query point (IQR-scaled)\n"
        + "  ".join(f"{n}={v:.1f}" for n, v in zip(obj_names, query_pt))
        + "\n(% of IQR(oⱼ) change per 1% IQR improvement in oᵢ)",
        fontsize=10, fontweight="bold", y=1.03,
    )
    for i, ax in enumerate(axes[0]):
        others = [j for j in range(d) if j != i]
        vals = [betas[i, j] for j in others]
        bars = ax.bar(
            np.arange(len(others)), vals,
            color=[obj_colors[obj_names[j]] for j in others],
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
        ax.set_ylabel("% IQR(oⱼ) per 1% IQR(oᵢ)" if i == 0 else "", fontsize=9)
        ax.set_ylim(-1.5, 0.2)
        ax.set_facecolor("#F7F7F5")
        ax.grid(True, axis="y", lw=0.3, color="#DDDDDD")
        ax.set_xlim(-0.5, len(others) - 0.5)

    out_path = output_dir / "mrs_bar_chart.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def compute_validity_radii(data, k, d, iqr, query_idx, epsilon):
    """Validity radius (raw + % of IQR) for each directed pair i→j at the query point.

    Radius is the step size in objective i where the quadratic prediction error
    for objective j reaches `epsilon` (relative to the linear term).
    """
    poly = PolynomialFeatures(degree=2, include_bias=True)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(data / iqr)
    _, nn_idx = nn.kneighbors((data[query_idx] / iqr).reshape(1, -1))
    nbrs = data[nn_idx[0, 1:]] - data[query_idx]

    radii_pct = np.full((d, d), np.nan)
    radii_raw = np.full((d, d), np.nan)
    for j in range(d):
        input_cols = [c for c in range(d) if c != j]
        feat_names = [f"x{p}" for p in range(len(input_cols))]
        X_poly = poly.fit_transform(nbrs[:, input_cols])
        all_names = list(poly.get_feature_names_out(feat_names))
        coef, _, _, _ = np.linalg.lstsq(X_poly, nbrs[:, j], rcond=None)

        for pos, i in enumerate(input_cols):
            b = coef[pos + 1]
            c_ii = coef[all_names.index(f"x{pos}^2")]
            if abs(b) < 1e-8 or abs(c_ii) < 1e-10:
                radii_pct[i, j] = np.inf
                radii_raw[i, j] = np.inf
            else:
                dr = epsilon * abs(b) / abs(c_ii)
                radii_raw[i, j] = dr
                radii_pct[i, j] = dr / iqr[i] * 100

    return radii_pct, radii_raw


def plot_validity_bar_chart(
    radii_pct, radii_raw, obj_names, d, epsilon, output_dir,
    obj_colors=None, query_label="query",
):
    """Bar chart of validity radii (% of IQR); bar labels show raw units."""
    obj_colors = obj_colors or OBJ_COLORS
    output_dir = Path(output_dir)

    pct_vals, raw_vals, colors, labels = [], [], [], []
    for i in range(d):
        for j in range(d):
            if i == j or np.isnan(radii_pct[i, j]):
                continue
            pct_vals.append(min(radii_pct[i, j], 200))
            raw_vals.append(min(radii_raw[i, j], 999))
            colors.append(obj_colors[obj_names[i]])
            labels.append(f"{obj_names[i]}→{obj_names[j]}")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#FAFAF8")
    bars = ax.bar(
        np.arange(len(pct_vals)), pct_vals, color=colors,
        alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    for bar, raw, pct in zip(bars, raw_vals, pct_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, min(pct, 195) + 1.5,
            f"{raw:.1f}" if raw < 999 else ">999",
            ha="center", va="bottom", fontsize=7.5,
            color="#333333", rotation=90,
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Validity radius (% of IQR of reference objective)", fontsize=10)
    ax.set_title(
        f"Validity range — {query_label} point  (ε={epsilon:.0%})\n"
        "Bar height = step size where prediction error = ε  |  "
        "Number = raw units  |  Colour = reference objective",
        fontsize=10, fontweight="bold",
    )
    ax.set_ylim(0, 220)
    ax.set_facecolor("#F7F7F5")
    ax.grid(True, axis="y", lw=0.3, color="#DDDDDD")
    handles = [
        Patch(color=obj_colors[name], alpha=0.85, label=name)
        for name in obj_names
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left", ncol=3)
    fig.tight_layout()

    out_path = output_dir / "validity_bar_chart.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
