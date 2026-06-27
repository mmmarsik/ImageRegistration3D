#include "synthetic_affine.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
using Clock = std::chrono::steady_clock;

struct ProgramOptions
{
    fs::path input_stack_dir = uir::default_input_stack_dir();
    fs::path out_dir = uir::default_output_stack_dir();
    fs::path artifact_dir = out_dir.parent_path();
    fs::path timing_json_path;
    bool skip_save = false;
};

struct RunTimings
{
    double load_seconds = 0.0;
    double artifact_seconds = 0.0;
    double transform_seconds = 0.0;
    double save_seconds = 0.0;
    double total_seconds = 0.0;
};

enum class InfoCommand
{
    None,
    TransformTag,
    TransformDescription,
    ParallelInfo,
};

double seconds_since(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

InfoCommand parse_info_command(int argc, char **argv)
{
    if (argc != 2)
    {
        return InfoCommand::None;
    }

    const std::string mode = argv[1];
    if (mode == "--transform-tag")
    {
        return InfoCommand::TransformTag;
    }
    if (mode == "--transform-description")
    {
        return InfoCommand::TransformDescription;
    }
    if (mode == "--parallel-info")
    {
        return InfoCommand::ParallelInfo;
    }
    return InfoCommand::None;
}

void print_parallel_info(std::ostream &out)
{
#ifdef _OPENMP
    out << "openmp_enabled=1\n";
    out << "openmp_version=" << _OPENMP << "\n";
    out << "openmp_max_threads=" << uir::openmp_max_threads() << "\n";
#else
    out << "openmp_enabled=0\n";
    out << "openmp_max_threads=1\n";
#endif
}

bool handle_info_command(InfoCommand command)
{
    switch (command)
    {
    case InfoCommand::TransformTag:
        std::cout << uir::experiment_transform_tag() << "\n";
        return true;
    case InfoCommand::TransformDescription:
        std::cout << uir::experiment_transform_description();
        return true;
    case InfoCommand::ParallelInfo:
        print_parallel_info(std::cout);
        return true;
    case InfoCommand::None:
        return false;
    }
    return false;
}

ProgramOptions parse_program_options(int argc, char **argv)
{
    ProgramOptions options;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];

        if (arg == "--skip-save")
        {
            options.skip_save = true;
        }
        else if (arg == "--input-stack")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("--input-stack requires a directory");
            }
            options.input_stack_dir = argv[++i];
        }
        else if (arg == "--timing-json")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("--timing-json requires a path");
            }
            options.timing_json_path = argv[++i];
        }
        else if (arg.rfind("--", 0) == 0)
        {
            throw std::runtime_error("Unknown option: " + arg);
        }
        else
        {
            positional.push_back(arg);
        }
    }

    if (positional.size() > 2)
    {
        throw std::runtime_error(
            "Usage: uir_affine [out_dir [artifact_dir]] [--input-stack dir] [--skip-save] [--timing-json path]");
    }

    if (!positional.empty())
    {
        options.out_dir = positional[0];
        options.artifact_dir = options.out_dir.parent_path();
    }
    if (positional.size() >= 2)
    {
        options.artifact_dir = positional[1];
    }

    return options;
}

void ensure_output_dirs(const ProgramOptions &options)
{
    if (!options.skip_save)
    {
        fs::create_directories(options.out_dir);
    }
    fs::create_directories(options.artifact_dir);
}

void print_run_header(const ProgramOptions &options, const cv::Matx44d &transform)
{
    std::cout << "T_full = \n"
              << cv::Mat(transform) << "\n\n";
    std::cout << "T_full_inv = \n"
              << cv::Mat(transform.inv()) << "\n\n";
    std::cout << "Input stack = " << options.input_stack_dir << "\n";
    std::cout << "Output dir = " << options.out_dir << "\n";
    std::cout << "Artifact dir = " << options.artifact_dir << "\n";
    std::cout << "Skip save = " << (options.skip_save ? "yes" : "no") << "\n";
#ifdef _OPENMP
    std::cout << "OpenMP = enabled, max threads = " << uir::openmp_max_threads() << "\n";
#else
    std::cout << "OpenMP = disabled\n";
#endif
}

void print_timings(const RunTimings &timings)
{
    std::cout << "Timings: load=" << timings.load_seconds
              << "s artifacts=" << timings.artifact_seconds
              << "s transform=" << timings.transform_seconds
              << "s save=" << timings.save_seconds
              << "s total=" << timings.total_seconds << "s\n";
}

void write_timing_json(const fs::path &path,
                       const ProgramOptions &options,
                       const RunTimings &timings,
                       size_t voxel_count,
                       size_t output_size)
{
    if (!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path());
    }

    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open timing JSON: " + path.string());
    }

    out << std::fixed << std::setprecision(9);
    out << "{\n";
#ifdef _OPENMP
    out << "  \"openmp_enabled\": true,\n";
    out << "  \"openmp_version\": " << _OPENMP << ",\n";
#else
    out << "  \"openmp_enabled\": false,\n";
    out << "  \"openmp_version\": null,\n";
#endif
    out << "  \"openmp_max_threads\": " << uir::openmp_max_threads() << ",\n";
    out << "  \"skip_save\": " << (options.skip_save ? "true" : "false") << ",\n";
    out << "  \"voxel_count\": " << voxel_count << ",\n";
    out << "  \"output_size\": " << output_size << ",\n";
    out << "  \"load_seconds\": " << timings.load_seconds << ",\n";
    out << "  \"artifact_seconds\": " << timings.artifact_seconds << ",\n";
    out << "  \"transform_seconds\": " << timings.transform_seconds << ",\n";
    out << "  \"save_seconds\": " << timings.save_seconds << ",\n";
    out << "  \"total_seconds\": " << timings.total_seconds << "\n";
    out << "}\n";
}
} // namespace

int main(int argc, char **argv)
{
    if (handle_info_command(parse_info_command(argc, argv)))
    {
        return 0;
    }

    const Clock::time_point total_start = Clock::now();
    RunTimings timings;
    ProgramOptions options = parse_program_options(argc, argv);
    ensure_output_dirs(options);

    const Clock::time_point load_start = Clock::now();
    uir::Image3D img(options.input_stack_dir);
    const Clock::time_point load_end = Clock::now();
    timings.load_seconds = seconds_since(load_start, load_end);

    const cv::Matx44d transform = uir::build_experiment_transform(img.shape());
    print_run_header(options, transform);

    const Clock::time_point artifact_start = Clock::now();
    uir::write_transform_artifacts(transform, options.artifact_dir, img.shape());
    const Clock::time_point artifact_end = Clock::now();
    timings.artifact_seconds = seconds_since(artifact_start, artifact_end);

    const Clock::time_point transform_start = Clock::now();
    std::vector<uint16_t> out = img.apply_affine_transform(transform);
    const Clock::time_point transform_end = Clock::now();
    timings.transform_seconds = seconds_since(transform_start, transform_end);

    if (!options.skip_save)
    {
        const Clock::time_point save_start = Clock::now();
        img.save_png_stack(out, options.out_dir);
        const Clock::time_point save_end = Clock::now();
        timings.save_seconds = seconds_since(save_start, save_end);
    }

    const Clock::time_point total_end = Clock::now();
    timings.total_seconds = seconds_since(total_start, total_end);

    if (!options.timing_json_path.empty())
    {
        write_timing_json(options.timing_json_path, options, timings, img.data().size(), out.size());
    }

    std::cout << "Done. out size = " << out.size() << "\n";
    print_timings(timings);
    return 0;
}
