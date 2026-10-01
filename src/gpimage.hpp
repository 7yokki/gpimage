// GPIMAGE - image upscaler: Nearest / Bilinear / Bicubic / Lanczos3 / EASU + RCAS
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace gp {

// ---------- small vector types ----------
struct Vec2 {
    float x = 0, y = 0;
};

struct Vec3 {
    float r = 0, g = 0, b = 0;
    Vec3() = default;
    Vec3(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
    Vec3 operator+(const Vec3& o) const { return {r + o.r, g + o.g, b + o.b}; }
    Vec3 operator-(const Vec3& o) const { return {r - o.r, g - o.g, b - o.b}; }
    Vec3 operator*(float s) const { return {r * s, g * s, b * s}; }
    Vec3& operator+=(const Vec3& o) { r += o.r; g += o.g; b += o.b; return *this; }
};

inline Vec3 vmin(const Vec3& a, const Vec3& b) { return {std::min(a.r, b.r), std::min(a.g, b.g), std::min(a.b, b.b)}; }
inline Vec3 vmax(const Vec3& a, const Vec3& b) { return {std::max(a.r, b.r), std::max(a.g, b.g), std::max(a.b, b.b)}; }
inline Vec3 vclamp(const Vec3& v, const Vec3& lo, const Vec3& hi) { return vmin(vmax(v, lo), hi); }

// ---------- image: float RGBA, values 0..1, sRGB-encoded ----------
struct Image {
    int w = 0, h = 0;
    bool hasAlpha = false;
    std::vector<float> px;  // w*h*4
    Image() = default;
    Image(int w_, int h_) : w(w_), h(h_), px(size_t(w_) * h_ * 4, 1.0f) {}

    float* at(int x, int y) { return &px[(size_t(y) * w + x) * 4]; }
    const float* at(int x, int y) const { return &px[(size_t(y) * w + x) * 4]; }
    // edge-clamped RGB fetch
    Vec3 rgb(int x, int y) const {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        const float* p = at(x, y);
        return {p[0], p[1], p[2]};
    }
};

enum class Method { Nearest, Bilinear, Bicubic, Lanczos3, EASU };

struct Options {
    Method method = Method::EASU;
    float sharpen = 0.75f;  // RCAS strength 0..1 (0 = off)
    int threads = 0;        // 0 = auto
};

// Resize to (dw, dh) with the chosen method, then optional RCAS.
Image resize(const Image& src, int dw, int dh, const Options& opt);

// Individual stages (public so they can be used on their own)
Image resampleSeparable(const Image& src, int dw, int dh, Method m, int threads);
Image easu(const Image& src, int dw, int dh, int threads);
void rcas(Image& img, float strength, int threads);

const char* methodName(Method m);

}  // namespace gp
