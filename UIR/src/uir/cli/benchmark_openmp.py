from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

plt = None


CSV_COLUMNS = [
    "variant",
    "mode",
    "threads",
    "repeat",
    "openmp_enabled",
    "openmp_version",
    "openmp_max_threads",
    "skip_save",
    "voxel_count",
    "load_seconds",
    "artifact_seconds",
    "transform_seconds",
    "save_seconds",
    "total_seconds",
    "throughput_mvox_per_s",
    "timing_json",
]


def project_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def configure_matplotlib_cache(runs_root: Path) -> None:
    global plt
    if plt is not None:
        return

    mpl_config_dir = runs_root / ".matplotlib"
    cache_dir = runs_root / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    plt = pyplot


def default_threads() -> list[int]:
    cpu_count = os.cpu_count() or 4
    values = [1, 2, 4, 6, 8, cpu_count]
    return sorted({value for value in values if 1 <= value <= cpu_count})


def parse_args() -> argparse.Namespace:
    uir_dir = project_dir()
    parser = argparse.ArgumentParser(
        description="Benchmark uir_affine synthetic volume generation with and without OpenMP."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=uir_dir / "runs" / "openmp_benchmark",
        help="Directory for benchmark CSV, JSON, plots, and optional generated stacks.",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=uir_dir / "build" / "openmp_benchmark",
        help="Directory for OpenMP and sequential CMake builds.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=default_threads(),
        help="OMP_NUM_THREADS values to test for the OpenMP build.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats for transform-only runs.")
    parser.add_argument(
        "--full-repeats",
        type=int,
        default=0,
        help="Repeats for full pipeline runs that write the transformed PNG stack.",
    )
    parser.add_argument("--build-jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing benchmark builds.")
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append new runs to an existing openmp_benchmark.json before regenerating summaries and plots.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def configure_and_build(build_dir: Path, *, enable_openmp: bool, jobs: int) -> Path:
    uir_dir = project_dir()
    run_command(
        [
            "cmake",
            "-S",
            str(uir_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DUIR_ENABLE_OPENMP={'ON' if enable_openmp else 'OFF'}",
        ]
    )
    run_command(["cmake", "--build", str(build_dir), "-j", str(jobs)])
    return build_dir / "uir_affine"


def run_benchmark_case(
    exe: Path,
    *,
    runs_root: Path,
    variant: str,
    mode: str,
    threads: int,
    repeat: int,
) -> dict[str, object]:
    case_dir = runs_root / "raw" / mode / variant / f"threads_{threads}" / f"repeat_{repeat}"
    timing_json = case_dir / "timing.json"
    artifact_dir = case_dir / "artifacts"
    output_dir = case_dir / "transformed_png_stack"
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe),
        str(output_dir),
        str(artifact_dir),
        "--timing-json",
        str(timing_json),
    ]
    if mode == "transform_only":
        cmd.append("--skip-save")
    elif mode != "full_pipeline":
        raise ValueError(f"Unknown benchmark mode: {mode}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    run_command(cmd, env=env)

    timing = json.loads(timing_json.read_text(encoding="utf-8"))
    voxel_count = int(timing["voxel_count"])
    transform_seconds = float(timing["transform_seconds"])
    throughput = (voxel_count / 1_000_000.0) / transform_seconds if transform_seconds > 0 else 0.0

    return {
        "variant": variant,
        "mode": mode,
        "threads": threads,
        "repeat": repeat,
        "openmp_enabled": bool(timing["openmp_enabled"]),
        "openmp_version": "" if timing.get("openmp_version") is None else timing["openmp_version"],
        "openmp_max_threads": int(timing["openmp_max_threads"]),
        "skip_save": bool(timing["skip_save"]),
        "voxel_count": voxel_count,
        "load_seconds": float(timing["load_seconds"]),
        "artifact_seconds": float(timing["artifact_seconds"]),
        "transform_seconds": transform_seconds,
        "save_seconds": float(timing["save_seconds"]),
        "total_seconds": float(timing["total_seconds"]),
        "throughput_mvox_per_s": throughput,
        "timing_json": str(timing_json),
    }


def write_rows(rows: list[dict[str, object]], summary_dir: Path) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)

    csv_path = summary_dir / "openmp_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    json_path = summary_dir / "openmp_benchmark.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def grouped_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["variant"]), str(row["mode"]), int(row["threads"]))
        groups.setdefault(key, []).append(row)

    stats_rows: list[dict[str, object]] = []
    for (variant, mode, threads), group in sorted(groups.items()):
        transform_values = [float(row["transform_seconds"]) for row in group]
        total_values = [float(row["total_seconds"]) for row in group]
        load_values = [float(row["load_seconds"]) for row in group]
        save_values = [float(row["save_seconds"]) for row in group]
        throughput_values = [float(row["throughput_mvox_per_s"]) for row in group]
        stats_rows.append(
            {
                "variant": variant,
                "mode": mode,
                "threads": threads,
                "repeats": len(group),
                "transform_mean": mean(transform_values),
                "transform_stdev": stdev(transform_values) if len(transform_values) > 1 else 0.0,
                "total_mean": mean(total_values),
                "total_stdev": stdev(total_values) if len(total_values) > 1 else 0.0,
                "load_mean": mean(load_values),
                "save_mean": mean(save_values),
                "throughput_mean": mean(throughput_values),
            }
        )
    return stats_rows


def write_stats(stats_rows: list[dict[str, object]], summary_dir: Path) -> Path:
    path = summary_dir / "openmp_benchmark_stats.csv"
    columns = [
        "variant",
        "mode",
        "threads",
        "repeats",
        "transform_mean",
        "transform_stdev",
        "total_mean",
        "total_stdev",
        "load_mean",
        "save_mean",
        "throughput_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)
    return path


def stats_lookup(stats_rows: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, object]]:
    return {
        (str(row["variant"]), str(row["mode"]), int(row["threads"])): row
        for row in stats_rows
    }


def plot_transform_time(stats_rows: list[dict[str, object]], out_path: Path) -> None:
    assert plt is not None
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"openmp": "#1f77b4", "no_openmp": "#7a7a7a"}

    for variant in ["no_openmp", "openmp"]:
        subset = [
            row
            for row in stats_rows
            if row["mode"] == "transform_only" and row["variant"] == variant
        ]
        if not subset:
            continue
        subset.sort(key=lambda row: int(row["threads"]))
        xs = [int(row["threads"]) for row in subset]
        ys = [float(row["transform_mean"]) for row in subset]
        yerr = [float(row["transform_stdev"]) for row in subset]
        label = "OpenMP отключен" if variant == "no_openmp" else "OpenMP"
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=label, color=colors[variant])

    ax.set_title("Время аффинного ресэмплинга")
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Время преобразования, с; меньше лучше")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_speedup(stats_rows: list[dict[str, object]], out_path: Path) -> None:
    assert plt is not None
    lookup = stats_lookup(stats_rows)
    baseline = lookup.get(("no_openmp", "transform_only", 1))
    if baseline is None:
        baseline = lookup.get(("openmp", "transform_only", 1))
    if baseline is None:
        return

    baseline_time = float(baseline["transform_mean"])
    subset = [
        row
        for row in stats_rows
        if row["mode"] == "transform_only" and row["variant"] == "openmp"
    ]
    subset.sort(key=lambda row: int(row["threads"]))
    if not subset:
        return

    xs = [int(row["threads"]) for row in subset]
    speedups = [baseline_time / float(row["transform_mean"]) for row in subset]
    efficiencies = [speedup / threads for speedup, threads in zip(speedups, xs)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(xs, speedups, marker="o", linewidth=1.8, label="Измерено")
    axes[0].plot(xs, xs, linestyle="--", color="#999999", label="Идеал")
    axes[0].set_title("Ускорение относительно версии без OpenMP")
    axes[0].set_xlabel("OMP_NUM_THREADS")
    axes[0].set_ylabel("Ускорение")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(xs, efficiencies, marker="o", linewidth=1.8, color="#2ca02c")
    axes[1].set_title("Параллельная эффективность")
    axes[1].set_xlabel("OMP_NUM_THREADS")
    axes[1].set_ylabel("Ускорение / число потоков")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_throughput(stats_rows: list[dict[str, object]], out_path: Path) -> None:
    assert plt is not None
    subset = [
        row
        for row in stats_rows
        if row["mode"] == "transform_only" and row["variant"] == "openmp"
    ]
    subset.sort(key=lambda row: int(row["threads"]))
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [int(row["threads"]) for row in subset],
        [float(row["throughput_mean"]) for row in subset],
        marker="o",
        linewidth=1.8,
        color="#9467bd",
    )
    ax.set_title("Производительность аффинного ресэмплинга")
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Млн вокселей / с")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_component_times(stats_rows: list[dict[str, object]], out_path: Path) -> None:
    assert plt is not None
    subset = [
        row
        for row in stats_rows
        if row["mode"] == "full_pipeline" and row["variant"] == "openmp"
    ]
    if not subset:
        subset = [
            row
            for row in stats_rows
            if row["mode"] == "transform_only" and row["variant"] == "openmp"
        ]
    subset.sort(key=lambda row: int(row["threads"]))
    if not subset:
        return

    xs = [str(int(row["threads"])) for row in subset]
    load = [float(row["load_mean"]) for row in subset]
    transform = [float(row["transform_mean"]) for row in subset]
    save = [float(row["save_mean"]) for row in subset]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xs, load, label="Загрузка PNG-стека", color="#7f7f7f")
    ax.bar(xs, transform, bottom=load, label="Аффинный ресэмплинг", color="#1f77b4")
    bottom = [a + b for a, b in zip(load, transform)]
    ax.bar(xs, save, bottom=bottom, label="Сохранение PNG-стека", color="#ff7f0e")
    mode_label = "полный пайплайн" if subset[0]["mode"] == "full_pipeline" else "только преобразование"
    ax.set_title(f"Компоненты времени ({mode_label})")
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Секунды")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_environment_report(summary_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = summary_dir / "openmp_benchmark_environment.json"
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "threads_tested": sorted({int(row["threads"]) for row in rows if row["variant"] == "openmp"}),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    if args.repeats < 0:
        raise SystemExit("--repeats must be >= 0")
    if args.full_repeats < 0:
        raise SystemExit("--full-repeats must be >= 0")
    if any(thread < 1 for thread in args.threads):
        raise SystemExit("--threads values must be positive")

    runs_root = args.runs_root
    summary_dir = runs_root / "summary"
    plots_dir = summary_dir / "plots"
    build_root = args.build_root
    configure_matplotlib_cache(runs_root)

    if not args.skip_build:
        no_openmp_exe = configure_and_build(build_root / "no_openmp", enable_openmp=False, jobs=args.build_jobs)
        openmp_exe = configure_and_build(build_root / "openmp", enable_openmp=True, jobs=args.build_jobs)
    else:
        no_openmp_exe = build_root / "no_openmp" / "uir_affine"
        openmp_exe = build_root / "openmp" / "uir_affine"

    rows: list[dict[str, object]] = []
    existing_json = summary_dir / "openmp_benchmark.json"
    if args.append_existing and existing_json.exists():
        rows.extend(json.loads(existing_json.read_text(encoding="utf-8")))

    for repeat in range(1, args.repeats + 1):
        rows.append(
            run_benchmark_case(
                no_openmp_exe,
                runs_root=runs_root,
                variant="no_openmp",
                mode="transform_only",
                threads=1,
                repeat=repeat,
            )
        )

    for threads in sorted(set(args.threads)):
        for repeat in range(1, args.repeats + 1):
            rows.append(
                run_benchmark_case(
                    openmp_exe,
                    runs_root=runs_root,
                    variant="openmp",
                    mode="transform_only",
                    threads=threads,
                    repeat=repeat,
                )
            )

    for repeat in range(1, args.full_repeats + 1):
        rows.append(
            run_benchmark_case(
                no_openmp_exe,
                runs_root=runs_root,
                variant="no_openmp",
                mode="full_pipeline",
                threads=1,
                repeat=repeat,
            )
        )
        for threads in sorted(set(args.threads)):
            rows.append(
                run_benchmark_case(
                    openmp_exe,
                    runs_root=runs_root,
                    variant="openmp",
                    mode="full_pipeline",
                    threads=threads,
                    repeat=repeat,
                )
            )

    csv_path, json_path = write_rows(rows, summary_dir)
    stats_rows = grouped_stats(rows)
    stats_csv_path = write_stats(stats_rows, summary_dir)
    env_path = write_environment_report(summary_dir, rows)

    plots_dir.mkdir(parents=True, exist_ok=True)
    transform_time_plot = plots_dir / "openmp_transform_time.png"
    speedup_plot = plots_dir / "openmp_speedup_efficiency.png"
    throughput_plot = plots_dir / "openmp_throughput.png"
    component_plot = plots_dir / "openmp_component_times.png"

    plot_transform_time(stats_rows, transform_time_plot)
    plot_speedup(stats_rows, speedup_plot)
    plot_throughput(stats_rows, throughput_plot)
    plot_component_times(stats_rows, component_plot)

    print(f"Rows: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Stats: {stats_csv_path}")
    print(f"Environment: {env_path}")
    print(f"Transform time plot: {transform_time_plot}")
    print(f"Speedup plot: {speedup_plot}")
    print(f"Throughput plot: {throughput_plot}")
    print(f"Component plot: {component_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
