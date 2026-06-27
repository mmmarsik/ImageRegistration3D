from __future__ import annotations

import argparse
from pathlib import Path

from uir.analysis.synthetic_runs import (
    collect_synthetic_rows,
    render_synthetic_summary_lines,
    write_synthetic_outputs,
)
from uir.reporting.aggregate_plots import (
    plot_error_heatmap,
    plot_noise_accuracy,
    plot_noise_sweep,
    plot_reliability_curve,
    plot_scale_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate normalized synthetic run summaries into compact reports."
    )
    parser.add_argument("runs_root", type=Path, help="UIR/runs directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_synthetic_rows(args.runs_root)
    csv_path, json_path = write_synthetic_outputs(args.runs_root, rows)
    out_dir = args.runs_root / "summary"

    heatmap_plot = out_dir / "synthetic_error_heatmap.png"
    reliability_plot = out_dir / "synthetic_reliability_curve.png"
    noise_accuracy_plot = out_dir / "synthetic_noise_accuracy.png"
    scale_sweep_plot = out_dir / "synthetic_scale_sweep.png"

    plot_error_heatmap(rows, heatmap_plot)
    plot_reliability_curve(rows, reliability_plot)
    plot_noise_accuracy(rows, noise_accuracy_plot)
    plot_scale_sweep(rows, scale_sweep_plot)

    roi_sizes = sorted({int(r["roi_size"]) for r in rows})
    sweep_metrics = [
        ("translation_l2_error_voxels", "L2-ошибка смещения, воксели", "translation_l2"),
        ("linear_rms_error", "RMS-ошибка линейной части", "linear_rms"),
        ("max_abs_transform_element_error", "Максимальная абсолютная ошибка элемента матрицы", "max_abs"),
    ]
    for roi_size in roi_sizes:
        for metric, ylabel, metric_slug in sweep_metrics:
            noise_sweep_plot = out_dir / f"synthetic_noise_sweep_roi{roi_size}_{metric_slug}.png"
            plot_noise_sweep(rows, noise_sweep_plot, roi_size=roi_size, metric=metric, ylabel=ylabel)
            print(f"Noise sweep roi={roi_size} {metric_slug}: {noise_sweep_plot}")

    for line in render_synthetic_summary_lines(rows):
        print(line)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Heatmap: {heatmap_plot}")
    print(f"Reliability: {reliability_plot}")
    print(f"Noise accuracy: {noise_accuracy_plot}")
    print(f"Scale sweep: {scale_sweep_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
