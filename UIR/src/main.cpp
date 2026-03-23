#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

#ifndef UIR_PROJECT_DIR
#define UIR_PROJECT_DIR "."
#endif

cv::Matx44d make_translation(double tx, double ty, double tz);
cv::Matx44d make_rotation_x_deg(double angle_deg);
cv::Matx44d make_rotation_y_deg(double angle_deg);
cv::Matx44d make_rotation_z_deg(double angle_deg);
cv::Matx44d make_scale(double sx, double sy, double sz);
cv::Matx44d make_shear(double sh_xy, double sh_xz, double sh_yx,
                       double sh_yz, double sh_zx, double sh_zy);

namespace
{
constexpr int kVolumeSizeX = 500;
constexpr int kVolumeSizeY = 500;
constexpr int kVolumeSizeZ = 500;

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

const ExperimentTransformSpec &experiment_transform_spec()
{
    static const ExperimentTransformSpec spec{
        19.0, -22.0, 68.0,
        1.5, 0.88, 1.33,
        0.01, 0.0, 0.0, -0.01, 0.0, 0.0,
        12.0, -9.0, 7.0};
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
    const ExperimentTransformSpec &spec = experiment_transform_spec();
    const double cx = 0.5 * (kVolumeSizeX - 1);
    const double cy = 0.5 * (kVolumeSizeY - 1);
    const double cz = 0.5 * (kVolumeSizeZ - 1);

    std::ostringstream out;
    out << "transform_tag=" << experiment_transform_tag() << "\n";
    out << "formula=translation * from_center * shear * scale * rotation * to_center\n";
    out << "rotation_order=Rz * Ry * Rx\n";
    out << "volume_size=(" << kVolumeSizeX << ", " << kVolumeSizeY << ", " << kVolumeSizeZ << ")\n";
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

fs::path default_input_stack_dir()
{
    return fs::path(UIR_PROJECT_DIR) / "resources" / "bhi_2_2.32um_voi";
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

cv::Matx44d build_experiment_transform()
{
    const ExperimentTransformSpec &spec = experiment_transform_spec();
    const double cx = 0.5 * (kVolumeSizeX - 1);
    const double cy = 0.5 * (kVolumeSizeY - 1);
    const double cz = 0.5 * (kVolumeSizeZ - 1);

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
} // namespace

class Image3D
{
public:
    explicit Image3D(const fs::path &images_stack_path = default_input_stack_dir())
        : X_(kVolumeSizeX), Y_(kVolumeSizeY), Z_(kVolumeSizeZ), images_stack_path_(images_stack_path)
    {
        volume_.resize(size_t(X_) * Y_ * Z_);

        std::vector<fs::path> slices;
        slices.reserve(Z_);

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

        if (slices.size() != static_cast<size_t>(Z_))
        {
            throw std::runtime_error("Unexpected number of slices in " + images_stack_path_.string());
        }

        for (int z = 0; z < static_cast<int>(slices.size()); z++)
        {
            cv::Mat gray = read_gray_u16_png(slices[z]);
            if (gray.rows != Y_ || gray.cols != X_)
            {
                throw std::runtime_error("Slice shape mismatch for image: " + slices[z].string());
            }

            for (int y = 0; y < Y_; y++)
            {
                for (int x = 0; x < X_; x++)
                {
                    this->at(x, y, z) = gray.at<uint16_t>(y, x);
                }
            }
        }
    }

    int X() const
    {
        return X_;
    }

    int Y() const
    {
        return Y_;
    }

    int Z() const
    {
        return Z_;
    }

    uint16_t &at(int x, int y, int z)
    {
        return volume_[idx(x, y, z)];
    }

    const uint16_t &at(int x, int y, int z) const
    {
        return volume_[idx(x, y, z)];
    }

    const std::vector<uint16_t> &data() const
    {
        return volume_;
    }

    std::vector<uint16_t> apply_affine_transform(const cv::Matx44d &transform) const
    {
        std::vector<uint16_t> out(volume_.size());

        cv::Matx44d invT = transform.inv();

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

    void save_png_stack(const std::vector<uint16_t> &vol, const fs::path &out_dir) const
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

private:
    int X_;
    int Y_;
    int Z_;
    std::vector<uint16_t> volume_;
    fs::path images_stack_path_;

    size_t idx(int x, int y, int z) const
    {
        return (size_t(z) * size_t(Y_) + size_t(y)) * size_t(X_) + size_t(x);
    }

    uint16_t get0(int x, int y, int z) const
    {
        if (x < 0 || y < 0 || z < 0 || x >= X_ || y >= Y_ || z >= Z_)
            return 0;
        return at(x, y, z);
    }

    uint16_t trilinear_u16(double x, double y, double z) const
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
};

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

    H(0, 1) = sh_xy; // x += sh_xy * y
    H(0, 2) = sh_xz; // x += sh_xz * z

    H(1, 0) = sh_yx; // y += sh_yx * x
    H(1, 2) = sh_yz; // y += sh_yz * z

    H(2, 0) = sh_zx; // z += sh_zx * x
    H(2, 1) = sh_zy; // z += sh_zy * y

    return H;
}

cv::Matx44d make_roi_transform(const cv::Matx44d &T_full,
                               double x0, double y0, double z0)
{
    cv::Matx44d C = cv::Matx44d::eye();
    C(0, 3) = x0;
    C(1, 3) = y0;
    C(2, 3) = z0;

    return C.inv() * T_full * C;
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

int main(int argc, char **argv)
{
    if (argc == 2)
    {
        const std::string mode = argv[1];
        if (mode == "--transform-tag")
        {
            std::cout << experiment_transform_tag() << "\n";
            return 0;
        }
        if (mode == "--transform-description")
        {
            std::cout << experiment_transform_description();
            return 0;
        }
    }

    Image3D img;
    fs::path out_dir = fs::path(UIR_PROJECT_DIR) / "resources" / "identity_copy";
    fs::path artifact_dir = out_dir.parent_path();

    if (argc >= 2)
    {
        out_dir = argv[1];
    }

    if (argc >= 3)
    {
        artifact_dir = argv[2];
    }

    fs::create_directories(out_dir);
    fs::create_directories(artifact_dir);

    const cv::Matx44d T = build_experiment_transform();

    std::cout << "T_full = \n"
              << cv::Mat(T) << "\n\n";
    std::cout << "T_full_inv = \n"
              << cv::Mat(T.inv()) << "\n\n";
    std::cout << "Output dir = " << out_dir << "\n";
    std::cout << "Artifact dir = " << artifact_dir << "\n";

    write_mat44_csv(T, artifact_dir / "T_full_4x4.csv");
    write_mat44_csv(T.inv(), artifact_dir / "T_full_inv_4x4.csv");
    write_mat34_csv(T, artifact_dir / "T_full_3x4.csv");
    write_mat34_csv(T.inv(), artifact_dir / "expected_reg_transform.csv");
    write_text_file(artifact_dir / "transform_info.txt", experiment_transform_description());

    std::vector<uint16_t> out = img.apply_affine_transform(T);
    img.save_png_stack(out, out_dir);

    std::cout << "Done. out size = " << out.size() << "\n";
    return 0;
}
