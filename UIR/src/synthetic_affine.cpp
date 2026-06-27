#include "synthetic_affine.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef UIR_PROJECT_DIR
#define UIR_PROJECT_DIR "."
#endif

namespace uir
{
namespace
{
struct ExperimentTransformSpec
{
    double rot_x_deg;
    double rot_y_deg;
    double rot_z_deg;
    double scale_x;
    double scale_y;
    double scale_z;
    double sh_xy;
    double sh_xz;
    double sh_yx;
    double sh_yz;
    double sh_zx;
    double sh_zy;
    double tx;
    double ty;
    double tz;
};

double env_or_default(const char *name, double default_value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return default_value;
    }

    char *end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || (end != nullptr && *end != '\0'))
    {
        throw std::runtime_error(std::string("Invalid floating-point value for ") + name + ": " + raw);
    }
    return value;
}

const ExperimentTransformSpec &experiment_transform_spec()
{
    static const ExperimentTransformSpec spec{
        env_or_default("UIR_ROT_X_DEG", 4.0),
        env_or_default("UIR_ROT_Y_DEG", -6.0),
        env_or_default("UIR_ROT_Z_DEG", 12.0),
        env_or_default("UIR_SCALE_X", 1.06),
        env_or_default("UIR_SCALE_Y", 0.95),
        env_or_default("UIR_SCALE_Z", 1.03),
        env_or_default("UIR_SH_XY", 0.0025),
        env_or_default("UIR_SH_XZ", 0.0),
        env_or_default("UIR_SH_YX", 0.0),
        env_or_default("UIR_SH_YZ", -0.0015),
        env_or_default("UIR_SH_ZX", 0.0),
        env_or_default("UIR_SH_ZY", 0.0),
        env_or_default("UIR_TX", 8.0),
        env_or_default("UIR_TY", -6.0),
        env_or_default("UIR_TZ", 5.0)};
    return spec;
}

std::string format_slug_value(double value)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << value;
    std::string s = out.str();

    while (!s.empty() && s.back() == '0')
    {
        s.pop_back();
    }
    if (!s.empty() && s.back() == '.')
    {
        s.pop_back();
    }
    if (s.empty())
    {
        s = "0";
    }

    std::replace(s.begin(), s.end(), '-', 'm');
    std::replace(s.begin(), s.end(), '.', 'p');
    return s;
}

int extract_trailing_number(const fs::path &path)
{
    const std::string stem = path.stem().string();
    const size_t pos = stem.find_last_not_of("0123456789");

    if (pos == std::string::npos || pos + 1 >= stem.size())
    {
        throw std::runtime_error("Cannot extract numeric suffix from filename: " + path.filename().string());
    }

    return std::stoi(stem.substr(pos + 1));
}

cv::Mat read_gray_u16_png(const fs::path &path)
{
    cv::Mat slice = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (slice.empty())
    {
        throw std::runtime_error("Failed to read image: " + path.string());
    }

    cv::Mat gray;
    if (slice.channels() == 3)
    {
        cv::cvtColor(slice, gray, cv::COLOR_BGR2GRAY);
    }
    else if (slice.channels() == 1)
    {
        gray = slice;
    }
    else
    {
        throw std::runtime_error("Unsupported number of channels in input image: " + path.string());
    }

    if (gray.type() == CV_8UC1)
    {
        cv::Mat gray16;
        gray.convertTo(gray16, CV_16UC1);
        return gray16;
    }

    if (gray.type() != CV_16UC1)
    {
        throw std::runtime_error("Unsupported grayscale type in image: " + path.string());
    }

    return gray;
}

cv::Matx44d make_translation(double tx, double ty, double tz)
{
    cv::Matx44d T = cv::Matx44d::eye();
    T(0, 3) = tx;
    T(1, 3) = ty;
    T(2, 3) = tz;
    return T;
}

cv::Matx44d make_rotation_x_deg(double angle_deg)
{
    const double a = angle_deg * CV_PI / 180.0;
    const double c = std::cos(a);
    const double s = std::sin(a);

    cv::Matx44d R = cv::Matx44d::eye();
    R(1, 1) = c;
    R(1, 2) = -s;
    R(2, 1) = s;
    R(2, 2) = c;
    return R;
}

cv::Matx44d make_rotation_y_deg(double angle_deg)
{
    const double a = angle_deg * CV_PI / 180.0;
    const double c = std::cos(a);
    const double s = std::sin(a);

    cv::Matx44d R = cv::Matx44d::eye();
    R(0, 0) = c;
    R(0, 2) = s;
    R(2, 0) = -s;
    R(2, 2) = c;
    return R;
}

cv::Matx44d make_rotation_z_deg(double angle_deg)
{
    const double a = angle_deg * CV_PI / 180.0;
    const double c = std::cos(a);
    const double s = std::sin(a);

    cv::Matx44d R = cv::Matx44d::eye();
    R(0, 0) = c;
    R(0, 1) = -s;
    R(1, 0) = s;
    R(1, 1) = c;
    return R;
}

cv::Matx44d make_scale(double sx, double sy, double sz)
{
    cv::Matx44d S = cv::Matx44d::eye();
    S(0, 0) = sx;
    S(1, 1) = sy;
    S(2, 2) = sz;
    return S;
}

cv::Matx44d make_shear(double sh_xy, double sh_xz, double sh_yx,
                       double sh_yz, double sh_zx, double sh_zy)
{
    cv::Matx44d H = cv::Matx44d::eye();

    H(0, 1) = sh_xy;
    H(0, 2) = sh_xz;
    H(1, 0) = sh_yx;
    H(1, 2) = sh_yz;
    H(2, 0) = sh_zx;
    H(2, 1) = sh_zy;

    return H;
}

void write_mat44_csv(const cv::Matx44d &mat, const fs::path &path)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out << std::fixed << std::setprecision(12);

    for (int r = 0; r < 4; r++)
    {
        for (int c = 0; c < 4; c++)
        {
            if (c > 0)
            {
                out << ",";
            }
            out << mat(r, c);
        }
        out << "\n";
    }
}

void write_mat34_csv(const cv::Matx44d &mat, const fs::path &path)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out << std::fixed << std::setprecision(12);

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 4; c++)
        {
            if (c > 0)
            {
                out << ",";
            }
            out << mat(r, c);
        }
        out << "\n";
    }
}

void write_text_file(const fs::path &path, const std::string &contents)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }
    out << contents;
}
} // namespace

fs::path default_input_stack_dir()
{
    return fs::path(UIR_PROJECT_DIR) / "resources" / "bhi_2_2.32um_voi";
}

fs::path default_output_stack_dir()
{
    return fs::path(UIR_PROJECT_DIR) / "resources" / "identity_copy";
}

int openmp_max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

std::string experiment_transform_tag()
{
    const ExperimentTransformSpec &spec = experiment_transform_spec();

    std::ostringstream out;
    out << "centered_affine"
        << "_rz" << format_slug_value(spec.rot_z_deg)
        << "_ry" << format_slug_value(spec.rot_y_deg)
        << "_rx" << format_slug_value(spec.rot_x_deg)
        << "_sx" << format_slug_value(spec.scale_x)
        << "_sy" << format_slug_value(spec.scale_y)
        << "_sz" << format_slug_value(spec.scale_z)
        << "_shxy" << format_slug_value(spec.sh_xy)
        << "_shyz" << format_slug_value(spec.sh_yz)
        << "_tx" << format_slug_value(spec.tx)
        << "_ty" << format_slug_value(spec.ty)
        << "_tz" << format_slug_value(spec.tz);
    return out.str();
}

std::string experiment_transform_description()
{
    return experiment_transform_description({kVolumeSizeX, kVolumeSizeY, kVolumeSizeZ});
}

std::string experiment_transform_description(VolumeShape shape)
{
    const ExperimentTransformSpec &spec = experiment_transform_spec();
    const double cx = 0.5 * (shape.x - 1);
    const double cy = 0.5 * (shape.y - 1);
    const double cz = 0.5 * (shape.z - 1);

    std::ostringstream out;
    out << "transform_tag=" << experiment_transform_tag() << "\n";
    out << "formula=translation * from_center * shear * scale * rotation * to_center\n";
    out << "rotation_order=Rz * Ry * Rx\n";
    out << "volume_size=(" << shape.x << ", " << shape.y << ", " << shape.z << ")\n";
    out << "center=(" << cx << ", " << cy << ", " << cz << ")\n";
    out << "rotation_deg=(rx=" << spec.rot_x_deg
        << ", ry=" << spec.rot_y_deg
        << ", rz=" << spec.rot_z_deg << ")\n";
    out << "scale=(sx=" << spec.scale_x
        << ", sy=" << spec.scale_y
        << ", sz=" << spec.scale_z << ")\n";
    out << "shear=(sh_xy=" << spec.sh_xy
        << ", sh_xz=" << spec.sh_xz
        << ", sh_yx=" << spec.sh_yx
        << ", sh_yz=" << spec.sh_yz
        << ", sh_zx=" << spec.sh_zx
        << ", sh_zy=" << spec.sh_zy << ")\n";
    out << "translation=(tx=" << spec.tx
        << ", ty=" << spec.ty
        << ", tz=" << spec.tz << ")\n";
    return out.str();
}

cv::Matx44d build_experiment_transform()
{
    return build_experiment_transform({kVolumeSizeX, kVolumeSizeY, kVolumeSizeZ});
}

cv::Matx44d build_experiment_transform(VolumeShape shape)
{
    const ExperimentTransformSpec &spec = experiment_transform_spec();
    const double cx = 0.5 * (shape.x - 1);
    const double cy = 0.5 * (shape.y - 1);
    const double cz = 0.5 * (shape.z - 1);

    const cv::Matx44d to_center = make_translation(-cx, -cy, -cz);
    const cv::Matx44d from_center = make_translation(cx, cy, cz);

    const cv::Matx44d rotation =
        make_rotation_z_deg(spec.rot_z_deg) *
        make_rotation_y_deg(spec.rot_y_deg) *
        make_rotation_x_deg(spec.rot_x_deg);

    const cv::Matx44d scale = make_scale(spec.scale_x, spec.scale_y, spec.scale_z);
    const cv::Matx44d shear = make_shear(
        spec.sh_xy, spec.sh_xz, spec.sh_yx, spec.sh_yz, spec.sh_zx, spec.sh_zy);
    const cv::Matx44d translation = make_translation(spec.tx, spec.ty, spec.tz);
    return translation * from_center * shear * scale * rotation * to_center;
}

void write_transform_artifacts(const cv::Matx44d &transform, const fs::path &artifact_dir, VolumeShape shape)
{
    write_mat44_csv(transform, artifact_dir / "T_full_4x4.csv");
    write_mat44_csv(transform.inv(), artifact_dir / "T_full_inv_4x4.csv");
    write_mat34_csv(transform, artifact_dir / "T_full_3x4.csv");
    write_mat34_csv(transform.inv(), artifact_dir / "expected_reg_transform.csv");
    write_text_file(artifact_dir / "transform_info.txt", experiment_transform_description(shape));
}

Image3D::Image3D(const fs::path &images_stack_path)
    : X_(0), Y_(0), Z_(0), images_stack_path_(images_stack_path)
{
    std::vector<fs::path> slices;

    for (const auto &slice : fs::directory_iterator(images_stack_path_))
    {
        if (slice.is_regular_file())
        {
            slices.push_back(slice.path());
        }
    }

    std::sort(slices.begin(), slices.end(),
              [](const fs::path &a, const fs::path &b)
              {
                  return extract_trailing_number(a) < extract_trailing_number(b);
              });

    if (slices.empty())
    {
        throw std::runtime_error("No regular files found in input stack: " + images_stack_path_.string());
    }

    Z_ = static_cast<int>(slices.size());

    for (int z = 0; z < static_cast<int>(slices.size()); z++)
    {
        cv::Mat gray = read_gray_u16_png(slices[z]);
        if (z == 0)
        {
            X_ = gray.cols;
            Y_ = gray.rows;
            volume_.resize(size_t(X_) * size_t(Y_) * size_t(Z_));
        }
        else if (gray.rows != Y_ || gray.cols != X_)
        {
            throw std::runtime_error("Slice shape mismatch for image: " + slices[z].string());
        }

        for (int y = 0; y < Y_; y++)
        {
            for (int x = 0; x < X_; x++)
            {
                at(x, y, z) = gray.at<uint16_t>(y, x);
            }
        }
    }
}

int Image3D::X() const
{
    return X_;
}

int Image3D::Y() const
{
    return Y_;
}

int Image3D::Z() const
{
    return Z_;
}

VolumeShape Image3D::shape() const
{
    return {X_, Y_, Z_};
}

uint16_t &Image3D::at(int x, int y, int z)
{
    return volume_[idx(x, y, z)];
}

const uint16_t &Image3D::at(int x, int y, int z) const
{
    return volume_[idx(x, y, z)];
}

const std::vector<uint16_t> &Image3D::data() const
{
    return volume_;
}

std::vector<uint16_t> Image3D::apply_affine_transform(const cv::Matx44d &transform) const
{
    std::vector<uint16_t> out(volume_.size());
    const cv::Matx44d invT = transform.inv();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int z = 0; z < Z_; z++)
    {
        for (int y = 0; y < Y_; y++)
        {
            for (int x = 0; x < X_; x++)
            {
                cv::Vec4d p_dst((double)x, (double)y, (double)z, 1.0);
                cv::Vec4d p_src = invT * p_dst;

                uint16_t val = trilinear_u16(p_src[0], p_src[1], p_src[2]);
                out[idx(x, y, z)] = val;
            }
        }
    }

    return out;
}

void Image3D::save_png_stack(const std::vector<uint16_t> &vol, const fs::path &out_dir) const
{
    fs::create_directories(out_dir);

    for (int z = 0; z < Z_; z++)
    {
        cv::Mat slice(Y_, X_, CV_16UC1);

        for (int y = 0; y < Y_; y++)
        {
            for (int x = 0; x < X_; x++)
            {
                slice.at<uint16_t>(y, x) = vol[idx(x, y, z)];
            }
        }

        char filename[64];
        std::snprintf(filename, sizeof(filename), "slice_%04d.png", z);

        fs::path path = out_dir / filename;
        bool ok = cv::imwrite(path.string(), slice);
        assert(ok);
    }
}

std::size_t Image3D::idx(int x, int y, int z) const
{
    return (size_t(z) * size_t(Y_) + size_t(y)) * size_t(X_) + size_t(x);
}

uint16_t Image3D::get0(int x, int y, int z) const
{
    if (x < 0 || y < 0 || z < 0 || x >= X_ || y >= Y_ || z >= Z_)
        return 0;
    return at(x, y, z);
}

uint16_t Image3D::trilinear_u16(double x, double y, double z) const
{
    if (x < 0.0 || y < 0.0 || z < 0.0 ||
        x > (double)(X_ - 1) || y > (double)(Y_ - 1) || z > (double)(Z_ - 1))
        return 0;

    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int z0 = (int)std::floor(z);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    double u = x - (double)x0;
    double v = y - (double)y0;
    double w = z - (double)z0;

    double c000 = (double)get0(x0, y0, z0);
    double c100 = (double)get0(x1, y0, z0);
    double c010 = (double)get0(x0, y1, z0);
    double c110 = (double)get0(x1, y1, z0);
    double c001 = (double)get0(x0, y0, z1);
    double c101 = (double)get0(x1, y0, z1);
    double c011 = (double)get0(x0, y1, z1);
    double c111 = (double)get0(x1, y1, z1);

    double c00 = (1.0 - u) * c000 + u * c100;
    double c10 = (1.0 - u) * c010 + u * c110;
    double c01 = (1.0 - u) * c001 + u * c101;
    double c11 = (1.0 - u) * c011 + u * c111;

    double c0 = (1.0 - v) * c00 + v * c10;
    double c1 = (1.0 - v) * c01 + v * c11;

    double val = (1.0 - w) * c0 + w * c1;

    long long iv = llround(val);
    if (iv < 0)
        iv = 0;
    if (iv > 65535)
        iv = 65535;
    return (uint16_t)iv;
}

} // namespace uir
