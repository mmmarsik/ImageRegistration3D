#pragma once

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace uir
{
namespace fs = std::filesystem;

constexpr int kVolumeSizeX = 500;
constexpr int kVolumeSizeY = 500;
constexpr int kVolumeSizeZ = 500;

struct VolumeShape
{
    int x;
    int y;
    int z;
};

fs::path default_input_stack_dir();
fs::path default_output_stack_dir();

int openmp_max_threads();

std::string experiment_transform_tag();
std::string experiment_transform_description();
std::string experiment_transform_description(VolumeShape shape);
cv::Matx44d build_experiment_transform();
cv::Matx44d build_experiment_transform(VolumeShape shape);
void write_transform_artifacts(const cv::Matx44d &transform, const fs::path &artifact_dir, VolumeShape shape);

class Image3D
{
public:
    explicit Image3D(const fs::path &images_stack_path = default_input_stack_dir());

    int X() const;
    int Y() const;
    int Z() const;
    VolumeShape shape() const;

    uint16_t &at(int x, int y, int z);
    const uint16_t &at(int x, int y, int z) const;
    const std::vector<uint16_t> &data() const;

    std::vector<uint16_t> apply_affine_transform(const cv::Matx44d &transform) const;
    void save_png_stack(const std::vector<uint16_t> &vol, const fs::path &out_dir) const;

private:
    int X_;
    int Y_;
    int Z_;
    std::vector<uint16_t> volume_;
    fs::path images_stack_path_;

    std::size_t idx(int x, int y, int z) const;
    uint16_t get0(int x, int y, int z) const;
    uint16_t trilinear_u16(double x, double y, double z) const;
};

} // namespace uir
