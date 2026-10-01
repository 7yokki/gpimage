#include "gpimage.hpp"

#include <atomic>
#include <thread>

namespace gp {

// ---------- threading helper ----------
template <class F>
static void parallelRows(int rows, int threads, F fn) {
    if (threads <= 0) threads = int(std::max(1u, std::thread::hardware_concurrency()));
    threads = std::max(1, std::min(threads, rows));
    if (threads == 1) {
        for (int y = 0; y < rows; ++y) fn(y);
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            for (;;) {
                int y = next.fetch_add(1);
                if (y >= rows) break;
                fn(y);
            }
        });
    for (auto& t : pool) t.join();
}

const char* methodName(Method m) {
    switch (m) {
        case Method::Nearest: return "nearest";
        case Method::Bilinear: return "bilinear";
        case Method::Bicubic: return "bicubic";
        case Method::Lanczos3: return "lanczos3";
        case Method::EASU: return "easu";
    }
    return "?";
}

// ======================================================================
//  Separable kernels (nearest / bilinear / bicubic Catmull-Rom / lanczos3)
// ======================================================================
namespace {

constexpr float kPi = 3.14159265358979f;

float wNearest(float x) { return (x >= -0.5f && x < 0.5f) ? 1.f : 0.f; }
float wBilinear(float x) { x = std::fabs(x); return x < 1.f ? 1.f - x : 0.f; }
float wBicubic(float x) {  // Catmull-Rom (a = -0.5)
    x = std::fabs(x);
    if (x < 1.f) return 1.5f * x * x * x - 2.5f * x * x + 1.f;
    if (x < 2.f) return -0.5f * x * x * x + 2.5f * x * x - 4.f * x + 2.f;
    return 0.f;
}
float sinc(float x) {
    if (std::fabs(x) < 1e-6f) return 1.f;
    x *= kPi;
    return std::sin(x) / x;
}
float wLanczos3(float x) { x = std::fabs(x); return x < 3.f ? sinc(x) * sinc(x / 3.f) : 0.f; }

struct Kernel {
    float radius;
    float (*fn)(float);
};

Kernel kernelFor(Method m) {
    switch (m) {
        case Method::Nearest: return {0.5f, wNearest};
        case Method::Bilinear: return {1.f, wBilinear};
        case Method::Lanczos3: return {3.f, wLanczos3};
        default: return {2.f, wBicubic};
    }
}

struct Tap {
    int idx;
    float w;
};

// For every destination index, the list of (source index, weight).
// When downscaling the kernel is stretched, which gives proper anti-aliasing.
std::vector<std::vector<Tap>> makeContribs(int srcN, int dstN, const Kernel& k) {
    std::vector<std::vector<Tap>> out(dstN);
    const float ratio = float(srcN) / float(dstN);
    const float scale = std::max(1.f, ratio);
    const float support = k.radius * scale;
    for (int i = 0; i < dstN; ++i) {
        const float center = (i + 0.5f) * ratio - 0.5f;
        int lo = int(std::ceil(center - support));
        int hi = int(std::floor(center + support));
        float sum = 0;
        auto& v = out[i];
        for (int j = lo; j <= hi; ++j) {
            float w = k.fn((j - center) / scale);
            if (w == 0.f) continue;
            v.push_back({std::clamp(j, 0, srcN - 1), w});
            sum += w;
        }
        if (v.empty() || std::fabs(sum) < 1e-8f) {
            v.clear();
            v.push_back({std::clamp(int(std::lround(center)), 0, srcN - 1), 1.f});
            sum = 1.f;
        }
        for (auto& t : v) t.w /= sum;
    }
    return out;
}

inline float sat(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

}  // namespace

Image resampleSeparable(const Image& src, int dw, int dh, Method m, int threads) {
    const Kernel k = kernelFor(m);
    auto cx = makeContribs(src.w, dw, k);
    auto cy = makeContribs(src.h, dh, k);

    Image tmp(dw, src.h);  // horizontal pass
    parallelRows(src.h, threads, [&](int y) {
        for (int x = 0; x < dw; ++x) {
            float acc[4] = {0, 0, 0, 0};
            for (const Tap& t : cx[x]) {
                const float* p = src.at(t.idx, y);
                for (int c = 0; c < 4; ++c) acc[c] += p[c] * t.w;
            }
            float* o = tmp.at(x, y);
            for (int c = 0; c < 4; ++c) o[c] = acc[c];
        }
    });

    Image out(dw, dh);  // vertical pass
    out.hasAlpha = src.hasAlpha;
    parallelRows(dh, threads, [&](int y) {
        for (int x = 0; x < dw; ++x) {
            float acc[4] = {0, 0, 0, 0};
            for (const Tap& t : cy[y]) {
                const float* p = tmp.at(x, t.idx);
                for (int c = 0; c < 4; ++c) acc[c] += p[c] * t.w;
            }
            float* o = out.at(x, y);
            for (int c = 0; c < 4; ++c) o[c] = sat(acc[c]);  // clamp ringing overshoot
        }
    });
    return out;
}

// ======================================================================
//  EASU  (Edge Adaptive Spatial Upsampling, FSR 1 style, CPU port)
//
//  12-tap footprint around the source pixel:
//        b c
//      e f g h
//      i j k l
//        n o
//  Local gradient direction + edge length are estimated from luma, then a
//  rotated / stretched Lanczos-like kernel is evaluated on the 12 taps.
// ======================================================================
namespace {

inline float lumaOf(const Vec3& c) { return c.b * 0.5f + c.g + c.r * 0.5f; }

// Accumulate direction/length for one bilinear quadrant.
inline void easuSet(Vec2& dir, float& len, float w, float lA, float lB, float lC, float lD, float lE) {
    // X axis
    float dc = lD - lC, cb = lC - lB;
    float lenX = std::max(std::fabs(dc), std::fabs(cb));
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = lenX > 1e-10f ? sat(std::fabs(dirX) / lenX) : 0.f;
    lenX *= lenX;
    len += lenX * w;
    // Y axis
    float ec = lE - lC, ca = lC - lA;
    float lenY = std::max(std::fabs(ec), std::fabs(ca));
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = lenY > 1e-10f ? sat(std::fabs(dirY) / lenY) : 0.f;
    lenY *= lenY;
    len += lenY * w;
}

inline void easuTap(Vec3& aC, float& aW, float ox, float oy, const Vec2& dir, const Vec2& len2, float lob, float clp,
                    const Vec3& c) {
    float vx = (ox * dir.x + oy * dir.y) * len2.x;
    float vy = (ox * -dir.y + oy * dir.x) * len2.y;
    float d2 = vx * vx + vy * vy;
    d2 = std::min(d2, clp);
    float wB = 2.f / 5.f * d2 - 1.f;
    float wA = lob * d2 - 1.f;
    wB *= wB;
    wA *= wA;
    wB = 25.f / 16.f * wB - (25.f / 16.f - 1.f);
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

}  // namespace

Image easu(const Image& src, int dw, int dh, int threads) {
    Image out(dw, dh);
    out.hasAlpha = src.hasAlpha;
    const float rx = float(src.w) / dw, ry = float(src.h) / dh;

    // alpha (if any) goes through bicubic
    Image alpha;
    if (src.hasAlpha) alpha = resampleSeparable(src, dw, dh, Method::Bicubic, threads);

    parallelRows(dh, threads, [&](int oy) {
        for (int ox = 0; ox < dw; ++ox) {
            // source-space position
            float ppx = (ox + 0.5f) * rx - 0.5f;
            float ppy = (oy + 0.5f) * ry - 0.5f;
            float fxf = std::floor(ppx), fyf = std::floor(ppy);
            ppx -= fxf;
            ppy -= fyf;
            int fx = int(fxf), fy = int(fyf);

            auto S = [&](int dx, int dy) { return src.rgb(fx + dx, fy + dy); };
            const Vec3 b = S(0, -1), c = S(1, -1), e = S(-1, 0), f = S(0, 0), g = S(1, 0), h = S(2, 0);
            const Vec3 i = S(-1, 1), j = S(0, 1), k = S(1, 1), l = S(2, 1), n = S(0, 2), o = S(1, 2);

            const float bL = lumaOf(b), cL = lumaOf(c), eL = lumaOf(e), fL = lumaOf(f), gL = lumaOf(g), hL = lumaOf(h);
            const float iL = lumaOf(i), jL = lumaOf(j), kL = lumaOf(k), lL = lumaOf(l), nL = lumaOf(n), oL = lumaOf(o);

            // direction & length of the local edge
            Vec2 dir;
            float len = 0.f;
            easuSet(dir, len, (1 - ppx) * (1 - ppy), bL, eL, fL, gL, jL);
            easuSet(dir, len, ppx * (1 - ppy), cL, fL, gL, hL, kL);
            easuSet(dir, len, (1 - ppx) * ppy, fL, iL, jL, kL, nL);
            easuSet(dir, len, ppx * ppy, gL, jL, kL, lL, oL);

            float dir2 = dir.x * dir.x + dir.y * dir.y;
            const bool zro = dir2 < 1.f / 32768.f;
            dir2 = zro ? 1.f : 1.f / std::sqrt(dir2);
            if (zro) dir.x = 1.f;
            dir.x *= dir2;
            dir.y *= dir2;

            len = len * 0.5f;
            len *= len;
            float stretch = (dir.x * dir.x + dir.y * dir.y) / std::max(std::fabs(dir.x), std::fabs(dir.y));
            Vec2 len2{1.f + (stretch - 1.f) * len, 1.f - 0.5f * len};
            float lob = 0.5f + ((1.f / 4.f - 0.04f) - 0.5f) * len;
            float clp = 1.f / lob;

            Vec3 aC;
            float aW = 0.f;
            easuTap(aC, aW, 0 - ppx, -1 - ppy, dir, len2, lob, clp, b);
            easuTap(aC, aW, 1 - ppx, -1 - ppy, dir, len2, lob, clp, c);
            easuTap(aC, aW, -1 - ppx, 1 - ppy, dir, len2, lob, clp, i);
            easuTap(aC, aW, 0 - ppx, 1 - ppy, dir, len2, lob, clp, j);
            easuTap(aC, aW, 0 - ppx, 0 - ppy, dir, len2, lob, clp, f);
            easuTap(aC, aW, -1 - ppx, 0 - ppy, dir, len2, lob, clp, e);
            easuTap(aC, aW, 1 - ppx, 1 - ppy, dir, len2, lob, clp, k);
            easuTap(aC, aW, 2 - ppx, 1 - ppy, dir, len2, lob, clp, l);
            easuTap(aC, aW, 2 - ppx, 0 - ppy, dir, len2, lob, clp, h);
            easuTap(aC, aW, 1 - ppx, 0 - ppy, dir, len2, lob, clp, g);
            easuTap(aC, aW, 1 - ppx, 2 - ppy, dir, len2, lob, clp, o);
            easuTap(aC, aW, 0 - ppx, 2 - ppy, dir, len2, lob, clp, n);

            Vec3 pix = aC * (1.f / aW);
            // anti-ringing: clamp to the 4 nearest source pixels
            Vec3 mn = vmin(vmin(f, g), vmin(j, k));
            Vec3 mx = vmax(vmax(f, g), vmax(j, k));
            pix = vclamp(pix, mn, mx);

            float* p = out.at(ox, oy);
            p[0] = sat(pix.r);
            p[1] = sat(pix.g);
            p[2] = sat(pix.b);
            p[3] = src.hasAlpha ? alpha.at(ox, oy)[3] : 1.f;
        }
    });
    return out;
}

// ======================================================================
//  RCAS  (Robust Contrast Adaptive Sharpening)
//  5-tap cross, per-pixel negative lobe limited so it never clips.
//  strength 0..1  ->  FSR "sharpness stops" = 2 * (1 - strength)
// ======================================================================
void rcas(Image& img, float strength, int threads) {
    if (strength <= 0.f) return;
    strength = std::min(strength, 1.f);
    const float con = std::exp2(-2.f * (1.f - strength));
    constexpr float kLimit = 0.25f - 1.f / 16.f;
    const Image src = img;

    parallelRows(img.h, threads, [&](int y) {
        for (int x = 0; x < img.w; ++x) {
            const Vec3 b = src.rgb(x, y - 1), d = src.rgb(x - 1, y), e = src.rgb(x, y);
            const Vec3 f = src.rgb(x + 1, y), h = src.rgb(x, y + 1);

            const Vec3 mn4 = vmin(vmin(b, d), vmin(f, h));
            const Vec3 mx4 = vmax(vmax(b, d), vmax(f, h));

            auto lobeCh = [](float mn, float mx, float ev) {
                float hitMin = std::min(mn, ev) / std::max(4.f * mx, 1e-6f);
                float den = 4.f * mn - 4.f;
                if (den > -1e-6f) den = -1e-6f;
                float hitMax = (1.f - std::max(mx, ev)) / den;
                return std::max(-hitMin, hitMax);
            };
            float lr = lobeCh(mn4.r, mx4.r, e.r);
            float lg = lobeCh(mn4.g, mx4.g, e.g);
            float lb = lobeCh(mn4.b, mx4.b, e.b);
            float lobe = std::max(-kLimit, std::min(std::min(lr, lg), std::min(lb, 0.f))) * con;

            // noise removal: damp sharpening where the ring is just noise
            const float bL = lumaOf(b), dL = lumaOf(d), eL = lumaOf(e), fL = lumaOf(f), hL = lumaOf(h);
            float nz = 0.25f * (bL + dL + fL + hL) - eL;
            float range = std::max(std::max(bL, dL), std::max(std::max(eL, fL), hL)) -
                          std::min(std::min(bL, dL), std::min(std::min(eL, fL), hL));
            nz = range > 1e-6f ? sat(std::fabs(nz) / range) : 0.f;
            nz = -0.5f * nz + 1.f;
            lobe *= nz;

            const float rcpL = 1.f / (4.f * lobe + 1.f);
            Vec3 pix = ((b + d + f + h) * lobe + e) * rcpL;
            float* p = img.at(x, y);
            p[0] = sat(pix.r);
            p[1] = sat(pix.g);
            p[2] = sat(pix.b);
        }
    });
}

// ======================================================================
Image resize(const Image& src, int dw, int dh, const Options& opt) {
    Image out;
    if (opt.method == Method::EASU)
        out = easu(src, dw, dh, opt.threads);
    else
        out = resampleSeparable(src, dw, dh, opt.method, opt.threads);
    rcas(out, opt.sharpen, opt.threads);
    return out;
}

}  // namespace gp
