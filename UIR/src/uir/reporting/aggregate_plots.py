from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _display_label(label: str) -> str:
    replacements = {
        "default": "базовый",
        "peak020": "порог пика 0,20",
        "corner060": "порог угла 0,60",
        "roi501 default": "ROI 501, базовый",
        "roi501 peak020": "ROI 501, порог пика 0,20",
        "roi650 default": "ROI 650, базовый",
        "roi650 peak020": "ROI 650, порог пика 0,20",
        "resample": "с ресэмплингом",
        "--resample": "с ресэмплингом",
    }
    return replacements.get(label, label)


def _display_degradation(value: str) -> str:
    replacements = {
        "awgn": "AWGN",
        "blur_awgn": "размытие+AWGN",
    }
    return replacements.get(value, value)


def _annotate_bars(ax: plt.Axes, bars, *, fmt: str = "{:.0f}", dy: float = 3.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_real_pair_threshold_comparison(rows: list[dict[str, object]], out_path: Path) -> None:
    plot_rows = list(rows)
    if not plot_rows:
        return

    raw_labels = [str(row["label"]) for row in plot_rows]
    labels = [_display_label(label) for label in raw_labels]
    matches = [int(row.get("match_count") or 0) for row in plot_rows]
    p95 = [float(row["match_residual_l2_p95"]) for row in plot_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ["#4c78a8", "#59a14f", "#e15759"]

    bars = axes[0].bar(labels, matches, color=colors[: len(labels)])
    axes[0].set_title("Проверка порогов на реальной паре: совпадения")
    axes[0].set_ylabel("Количество совпадений")
    axes[0].grid(True, axis="y", alpha=0.25)
    _annotate_bars(axes[0], bars, fmt="{:.0f}")

    bars = axes[1].bar(labels, p95, color=colors[: len(labels)])
    axes[1].set_title("Проверка порогов на реальной паре: P95 невязки")
    axes[1].set_ylabel("P95 невязки, воксели")
    axes[1].grid(True, axis="y", alpha=0.25)
    _annotate_bars(axes[1], bars, fmt="{:.2f}")

    fig.suptitle("Базовая настройка, порог пика 0,20 и порог угла 0,60", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_real2_roi_comparison(rows: list[dict[str, object]], out_path: Path) -> None:
    def sort_key(row: dict[str, object]) -> tuple[int, str]:
        label = str(row["label"])
        digits = "".join(ch for ch in label if ch.isdigit())
        roi = int(digits[:3]) if digits else 0
        return roi, label

    plot_rows = sorted(rows, key=sort_key)
    if not plot_rows:
        return

    raw_labels = [str(row["label"]) for row in plot_rows]
    labels = [_display_label(label) for label in raw_labels]
    matches = [int(row.get("match_count") or 0) for row in plot_rows]
    fractions = [float(row["model_consistent_match_fraction"]) * 100.0 for row in plot_rows]
    p95 = [float(row["match_residual_l2_p95"]) for row in plot_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#4c78a8" if "default" in label else "#59a14f" for label in raw_labels]

    bars = axes[0].bar(labels, matches, color=colors[: len(labels)])
    axes[0].set_title("Сопоставленные ключевые точки")
    axes[0].set_ylabel("Количество совпадений")
    axes[0].grid(True, axis="y", alpha=0.25)
    _annotate_bars(axes[0], bars, fmt="{:.0f}")

    bars = axes[1].bar(labels, fractions, color=colors[: len(labels)])
    axes[1].set_title("Доля согласованных с моделью совпадений")
    axes[1].set_ylabel("Согласованные совпадения, %")
    axes[1].set_ylim(max(0.0, min(fractions) - 2.0), 100.0)
    axes[1].grid(True, axis="y", alpha=0.25)
    _annotate_bars(axes[1], bars, fmt="{:.2f}")

    bars = axes[2].bar(labels, p95, color=colors[: len(labels)])
    axes[2].set_title("P95 невязки")
    axes[2].set_ylabel("P95 невязки, воксели")
    axes[2].grid(True, axis="y", alpha=0.25)
    _annotate_bars(axes[2], bars, fmt="{:.2f}")

    fig.suptitle("Вторая реальная пара: ROI 501 и ROI 650", fontsize=12)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_synthetic_roi_ladder(rows: list[dict[str, object]], out_path: Path) -> None:
    roi_sizes = sorted({int(row["roi_size"]) for row in rows})
    if not roi_sizes:
        return

    success_counts: list[int] = []
    fail_counts: list[int] = []
    median_matches: list[float] = []
    for roi_size in roi_sizes:
        group = [row for row in rows if int(row["roi_size"]) == roi_size]
        successes = [row for row in group if bool(row.get("registration_succeeded"))]
        success_counts.append(len(successes))
        fail_counts.append(len(group) - len(successes))
        matches = [int(row.get("match_count") or 0) for row in successes]
        median_matches.append(float(np.median(matches)) if matches else 0.0)

    labels = [str(roi_size) for roi_size in roi_sizes]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    success_bars = axes[0].bar(x, success_counts, label="успешно", color="#59a14f")
    fail_bars = axes[0].bar(x, fail_counts, bottom=success_counts, label="ошибка", color="#e15759")
    axes[0].set_title("Синтетические прогоны по размеру ROI")
    axes[0].set_xlabel("Размер ребра ROI")
    axes[0].set_ylabel("Количество прогонов")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    totals = [success + fail for success, fail in zip(success_counts, fail_counts)]
    axes[0].set_ylim(0, max(totals) * 1.14)
    for bar, success, fail in zip(success_bars, success_counts, fail_counts):
        if success > 0:
            axes[0].annotate(
                f"{success}",
                xy=(bar.get_x() + bar.get_width() / 2, success / 2),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        if fail > 0:
            fail_y = success + fail / 2
            text = f"ошибок={fail}"
            if fail >= 3:
                axes[0].annotate(
                    text,
                    xy=(bar.get_x() + bar.get_width() / 2, fail_y),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            else:
                axes[0].annotate(
                    text,
                    xy=(bar.get_x() + bar.get_width() / 2, success + fail),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                )

    bars = axes[1].bar(labels, median_matches, color="#4c78a8")
    axes[1].set_title("Медианное число совпадений в успешных прогонах")
    axes[1].set_xlabel("Размер ребра ROI")
    axes[1].set_ylabel("Медиана совпадений")
    axes[1].grid(True, axis="y", alpha=0.25)
    _annotate_bars(axes[1], bars, fmt="{:.0f}")

    fig.suptitle("Синтетический эксперимент: лестница ROI", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def plot_real_pair_roi_summary(rows: list[dict[str, object]], out_path: Path) -> None:
    plot_rows = sorted(rows, key=lambda row: int(row["roi_size"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    error_points = [
        (int(row["roi_size"]), float(row["translation_l2_error_voxels"]))
        for row in plot_rows
        if row.get("translation_l2_error_voxels") not in (None, "")
    ]
    if error_points:
        xs, ys = zip(*error_points)
        axes[0].plot(xs, ys, marker="o", linewidth=1.8)
    axes[0].set_title("Ошибка смещения ROI относительно полного объема")
    axes[0].set_xlabel("Размер ребра ROI")
    axes[0].set_ylabel("L2-ошибка смещения, воксели")
    axes[0].grid(True, alpha=0.3)

    match_points = [(int(row["roi_size"]), int(row.get("match_count") or 0)) for row in plot_rows]
    if match_points:
        xs, ys = zip(*match_points)
        axes[1].plot(xs, ys, marker="o", linewidth=1.8, color="#3b5b92")
    axes[1].set_title("Сопоставленные точки по размеру ROI")
    axes[1].set_xlabel("Размер ребра ROI")
    axes[1].set_ylabel("Количество совпадений")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_resolution_pair_summary(rows: list[dict[str, object]], out_path: Path, *, crop_size: int | None = None) -> None:
    if crop_size is None:
        crop_sizes = [int(row["crop_size"]) for row in rows if row.get("crop_size") not in (None, "")]
        if not crop_sizes:
            return
        crop_size = max(crop_sizes)

    subset = [row for row in rows if int(row.get("crop_size") or 0) == crop_size]
    if not subset:
        return

    modes = ["resample"]
    colors = {"resample": "#1f77b4"}
    labels = {"resample": "с ресэмплингом"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Регистрация кропов с физическим разрешением и ресэмплингом, crop={crop_size}", fontsize=13)

    for mode in modes:
        mode_rows = sorted([row for row in subset if row.get("mode") == mode], key=lambda row: int(row["ratio"]))
        if not mode_rows:
            continue

        ratios = [int(row["ratio"]) for row in mode_rows]
        success = [1 if row.get("registration_succeeded") else 0 for row in mode_rows]
        matches = [int(row.get("match_count") or 0) for row in mode_rows]

        axes[0, 0].plot(ratios, success, marker="o", linewidth=1.8, color=colors[mode], label=labels[mode])
        axes[0, 1].plot(ratios, matches, marker="o", linewidth=1.8, color=colors[mode], label=labels[mode])

        err_vox = [
            (int(row["ratio"]), float(row["translation_l2_error_voxels"]))
            for row in mode_rows
            if row.get("translation_l2_error_voxels") not in (None, "")
        ]
        if err_vox:
            xs, ys = zip(*err_vox)
            axes[1, 0].plot(xs, ys, marker="o", linewidth=1.8, color=colors[mode], label=labels[mode])

        err_phys = [
            (int(row["ratio"]), float(row["translation_l2_error_physical"]))
            for row in mode_rows
            if row.get("translation_l2_error_physical") not in (None, "")
        ]
        if err_phys:
            xs, ys = zip(*err_phys)
            axes[1, 1].plot(xs, ys, marker="o", linewidth=1.8, color=colors[mode], label=labels[mode])

    axes[0, 0].set_title("Успешность регистрации")
    axes[0, 0].set_xlabel("Коэффициент разрешения")
    axes[0, 0].set_ylabel("Успех")
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_yticklabels(["ошибка", "успех"])

    axes[0, 1].set_title("Сопоставленные ключевые точки")
    axes[0, 1].set_xlabel("Коэффициент разрешения")
    axes[0, 1].set_ylabel("Количество совпадений")

    axes[1, 0].set_title("Ошибка смещения в низкоразрешенных вокселях")
    axes[1, 0].set_xlabel("Коэффициент разрешения")
    axes[1, 0].set_ylabel("L2-ошибка")

    axes[1, 1].set_title("Ошибка смещения в физических единицах")
    axes[1, 1].set_xlabel("Коэффициент разрешения")
    axes[1, 1].set_ylabel("L2-ошибка")

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xticks(sorted({int(row["ratio"]) for row in subset}))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_noise_sweep(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    roi_size: int,
    metric: str = "translation_l2_error_voxels",
    ylabel: str = "L2-ошибка смещения, воксели",
) -> None:
    subset = [r for r in rows if int(r["roi_size"]) == roi_size]
    if not subset:
        return

    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in subset:
        groups[(str(row["transform_id"]), float(row["blur_sigma_xy"]))].append(row)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"ROI {roi_size}: влияние уровня шума", fontsize=12)

    for i, (transform_id, blur_sigma) in enumerate(sorted(groups)):
        group_rows = sorted(groups[(transform_id, blur_sigma)], key=lambda r: float(r["awgn_variance"]))
        xs = [float(r["awgn_variance"]) for r in group_rows]
        label = f"{transform_id}, размытие={blur_sigma:g}"
        color = colors[i % len(colors)]

        err_points = [(float(r["awgn_variance"]), float(r[metric])) for r in group_rows if r.get(metric) not in (None, "")]
        if err_points:
            xs_err, ys_err = zip(*err_points)
            axes[0].plot(xs_err, ys_err, marker="o", linewidth=1.8, markersize=5, label=label, color=color)

        match_points = [(float(r["awgn_variance"]), int(r["match_count"])) for r in group_rows if r.get("match_count") not in (None, "")]
        if match_points:
            xs_mc, ys_mc = zip(*match_points)
            axes[1].plot(xs_mc, ys_mc, marker="o", linewidth=1.8, markersize=5, label=label, color=color)

    axes[0].set_title(f"{ylabel}: зависимость от шума")
    axes[0].set_xlabel("Дисперсия AWGN")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    handles, _ = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=8)

    axes[1].set_title("Количество совпадений: зависимость от шума")
    axes[1].set_xlabel("Дисперсия AWGN")
    axes[1].set_ylabel("Количество совпадений")
    axes[1].grid(True, alpha=0.3)
    handles, _ = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scale_sweep(rows: list[dict[str, object]], out_path: Path) -> None:
    subset = [r for r in rows if r.get("isotropic_scale") not in (None, "")]
    if not subset:
        return

    groups: dict[tuple[int, str, float, float], list[dict[str, object]]] = defaultdict(list)
    for row in subset:
        groups[
            (
                int(row["roi_size"]),
                str(row["degradation"]),
                float(row["blur_sigma_xy"]),
                float(row["awgn_variance"]),
            )
        ].append(row)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Синтетический эксперимент: изотропное масштабирование", fontsize=12)

    for i, key in enumerate(sorted(groups)):
        roi_size, degradation, blur_sigma, variance = key
        group_rows = sorted(groups[key], key=lambda r: float(r["isotropic_scale"]))
        xs = [float(r["isotropic_scale"]) for r in group_rows]
        label = f"ROI={roi_size}, {_display_degradation(degradation)}, размытие={blur_sigma:g}, дисперсия={variance:g}"
        color = colors[i % len(colors)]

        err_points = [
            (float(r["isotropic_scale"]), float(r["translation_l2_error_voxels"]))
            for r in group_rows
            if r.get("translation_l2_error_voxels") not in (None, "")
        ]
        if err_points:
            xs_err, ys_err = zip(*err_points)
            axes[0].plot(xs_err, ys_err, marker="o", linewidth=1.8, markersize=5, label=label, color=color)

        match_points = [
            (float(r["isotropic_scale"]), int(r["match_count"]))
            for r in group_rows
            if r.get("match_count") not in (None, "")
        ]
        if match_points:
            xs_mc, ys_mc = zip(*match_points)
            axes[1].plot(xs_mc, ys_mc, marker="o", linewidth=1.8, markersize=5, label=label, color=color)

    axes[0].set_title("Ошибка смещения: зависимость от масштаба")
    axes[0].set_xlabel("Изотропный масштаб в воксельном пространстве")
    axes[0].set_ylabel("L2-ошибка смещения, воксели")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    handles, _ = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=8)

    axes[1].set_title("Количество совпадений: зависимость от масштаба")
    axes[1].set_xlabel("Изотропный масштаб в воксельном пространстве")
    axes[1].set_ylabel("Количество совпадений")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    handles, _ = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_heatmap(rows: list[dict[str, object]], out_path: Path) -> None:
    cell_data: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        v = row.get("translation_l2_error_voxels")
        if v not in (None, ""):
            cell_data[(str(row["transform_id"]), int(row["roi_size"]))].append(float(v))

    transform_ids = sorted({k[0] for k in cell_data})
    roi_sizes = sorted({k[1] for k in cell_data})
    if not transform_ids or not roi_sizes:
        return

    matrix = np.full((len(transform_ids), len(roi_sizes)), np.nan)
    for i, tid in enumerate(transform_ids):
        for j, rsz in enumerate(roi_sizes):
            vals = cell_data.get((tid, rsz), [])
            if vals:
                matrix[i, j] = float(np.median(vals))

    fig, ax = plt.subplots(figsize=(max(6, len(roi_sizes) * 1.8), max(4, len(transform_ids) * 1.0)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(roi_sizes)))
    ax.set_xticklabels([str(s) for s in roi_sizes])
    ax.set_yticks(range(len(transform_ids)))
    ax.set_yticklabels(transform_ids)
    ax.set_xlabel("Размер ROI, воксели")
    ax.set_ylabel("Преобразование")
    ax.set_title("Медианная L2-ошибка смещения, воксели: преобразование × размер ROI")

    vmax = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    for i in range(len(transform_ids)):
        for j in range(len(roi_sizes)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Медианная L2-ошибка смещения, воксели")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def plot_reliability_curve(rows: list[dict[str, object]], out_path: Path) -> None:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        v = row.get("translation_l2_error_voxels")
        if v not in (None, ""):
            groups[int(row["roi_size"])].append(float(v))

    if not groups:
        return

    all_errors = [v for vals in groups.values() for v in vals]
    thresholds = np.linspace(0.0, float(np.max(all_errors)), 400)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, roi_size in enumerate(sorted(groups)):
        errors = np.array(sorted(groups[roi_size]))
        fracs = np.array([(errors <= t).mean() for t in thresholds])
        ax.plot(thresholds, fracs, linewidth=1.8, color=colors[i % len(colors)], label=f"ROI={roi_size}")

    ax.set_title("Кривая надежности: доля прогонов в пределах порога ошибки")
    ax.set_xlabel("Порог L2-ошибки смещения, воксели")
    ax.set_ylabel("Доля прогонов ≤ порога")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def plot_noise_accuracy(rows: list[dict[str, object]], out_path: Path) -> None:
    points: list[tuple[float, float]] = []
    for row in rows:
        expected = row.get("expected_noise_std")
        observed = row.get("noise_std")
        if expected not in (None, "") and observed not in (None, ""):
            try:
                points.append((float(expected), float(observed)))
            except (ValueError, TypeError):
                pass

    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo = min(xs + ys) * 0.95
    hi = max(xs + ys) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=20, alpha=0.6, color="#3b5b92")
    ax.plot([lo, hi], [lo, hi], "--", color="#aaaaaa", linewidth=1.2, label="идеал (y=x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Точность шума: ожидаемое и наблюдаемое СКО")
    ax.set_xlabel("Ожидаемое СКО шума")
    ax.set_ylabel("Наблюдаемое СКО шума")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
