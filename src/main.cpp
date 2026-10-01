#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "gpimage.hpp"

static void usage() {
    std::puts(
        "GPIMAGE - image upscaler (bicubic / lanczos / EASU + RCAS)\n"
        "\n"
        "Usage: gpimage <input> <output> [options]\n"
        "\n"
        "Size (pick one, default --scale 2):\n"
        "  --scale F         multiply resolution by F (e.g. 2, 1.5, 3)\n"
        "  --width W         target width, height keeps aspect ratio\n"
        "  --height H        target height, width keeps aspect ratio\n"
        "  --size WxH        exact target size (e.g. 3840x2160)\n"
        "\n"
        "Options:\n"
        "  --method M        easu (default) | bicubic | lanczos | bilinear | nearest\n"
        "  --sharpen S       RCAS sharpening strength 0..1 (default 0.75, 0 = off)\n"
        "  --quality Q       JPEG quality 1..100 (default 95)\n"
        "  --threads N       worker threads (default: all cores)\n"
        "  -h, --help        show this help\n"
        "\n"
        "Input: png, jpg, bmp, tga, gif, psd, hdr, pic.  Output type from extension: png, jpg, bmp, tga.\n");
}

static bool endsWith(const std::string& s, const char* ext) {
    size_t n = std::strlen(ext);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
        if (std::tolower((unsigned char)s[s.size() - n + i]) != ext[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    std::string in, out;
    double scale = 0;
    int width = 0, height = 0, quality = 95, threads = 0;
    gp::Options opt;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--scale") scale = std::atof(next("--scale"));
        else if (a == "--width") width = std::atoi(next("--width"));
        else if (a == "--height") height = std::atoi(next("--height"));
        else if (a == "--size") {
            if (std::sscanf(next("--size"), "%dx%d", &width, &height) != 2) {
                std::fprintf(stderr, "--size expects WxH, e.g. 1920x1080\n");
                return 1;
            }
        } else if (a == "--method") {
            std::string m = next("--method");
            if (m == "easu") opt.method = gp::Method::EASU;
            else if (m == "bicubic") opt.method = gp::Method::Bicubic;
            else if (m == "lanczos" || m == "lanczos3") opt.method = gp::Method::Lanczos3;
            else if (m == "bilinear") opt.method = gp::Method::Bilinear;
            else if (m == "nearest") opt.method = gp::Method::Nearest;
            else { std::fprintf(stderr, "unknown method '%s'\n", m.c_str()); return 1; }
        } else if (a == "--sharpen") opt.sharpen = float(std::atof(next("--sharpen")));
        else if (a == "--quality") quality = std::atoi(next("--quality"));
        else if (a == "--threads") threads = std::atoi(next("--threads"));
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option '%s'\n", a.c_str()); return 1; }
        else if (in.empty()) in = a;
        else if (out.empty()) out = a;
        else { std::fprintf(stderr, "too many arguments\n"); return 1; }
    }
    if (in.empty() || out.empty()) { usage(); return 1; }
    opt.threads = threads;

    int w, h, n;
    unsigned char* data = stbi_load(in.c_str(), &w, &h, &n, 4);
    if (!data) {
        std::fprintf(stderr, "cannot read '%s': %s\n", in.c_str(), stbi_failure_reason());
        return 1;
    }

    // target size
    int dw, dh;
    if (width > 0 && height > 0) { dw = width; dh = height; }
    else if (width > 0) { dw = width; dh = int(std::lround(double(h) * width / w)); }
    else if (height > 0) { dh = height; dw = int(std::lround(double(w) * height / h)); }
    else { if (scale <= 0) scale = 2; dw = int(std::lround(w * scale)); dh = int(std::lround(h * scale)); }
    if (dw < 1 || dh < 1 || double(dw) * dh > 400e6) {
        std::fprintf(stderr, "invalid or too large target size %dx%d\n", dw, dh);
        stbi_image_free(data);
        return 1;
    }

    gp::Image src(w, h);
    src.hasAlpha = (n == 4);
    for (size_t i = 0; i < size_t(w) * h * 4; ++i) src.px[i] = data[i] / 255.0f;
    stbi_image_free(data);

    std::printf("%s: %dx%d -> %dx%d  [%s, sharpen %.2f]\n", in.c_str(), w, h, dw, dh, gp::methodName(opt.method), opt.sharpen);
    auto t0 = std::chrono::steady_clock::now();
    gp::Image res = gp::resize(src, dw, dh, opt);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::vector<unsigned char> bytes(size_t(dw) * dh * 4);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (unsigned char)std::lround(std::clamp(res.px[i], 0.f, 1.f) * 255.f);

    int comp = src.hasAlpha ? 4 : 3;
    std::vector<unsigned char> packed;
    const unsigned char* outp = bytes.data();
    if (comp == 3) {
        packed.resize(size_t(dw) * dh * 3);
        for (size_t i = 0, p = 0; i < size_t(dw) * dh; ++i) {
            packed[p++] = bytes[i * 4]; packed[p++] = bytes[i * 4 + 1]; packed[p++] = bytes[i * 4 + 2];
        }
        outp = packed.data();
    }

    int ok = 0;
    if (endsWith(out, ".jpg") || endsWith(out, ".jpeg")) ok = stbi_write_jpg(out.c_str(), dw, dh, comp, outp, quality);
    else if (endsWith(out, ".bmp")) ok = stbi_write_bmp(out.c_str(), dw, dh, comp, outp);
    else if (endsWith(out, ".tga")) ok = stbi_write_tga(out.c_str(), dw, dh, comp, outp);
    else ok = stbi_write_png(out.c_str(), dw, dh, comp, outp, dw * comp);
    if (!ok) { std::fprintf(stderr, "cannot write '%s'\n", out.c_str()); return 1; }

    std::printf("done in %.1f ms -> %s\n", ms, out.c_str());
    return 0;
}
