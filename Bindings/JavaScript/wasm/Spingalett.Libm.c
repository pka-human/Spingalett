/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The mathematical functions the engine calls that WebAssembly has no instruction for, for its build
 * without a C library: expf, tanhf and erff for the activations, pow, sin and cos for the rotary
 * embeddings' table. Integer models must give the native engine's results bit for bit, so these round
 * as the C library of the platforms the engine is tested on does (glibc):
 *
 * - expf is glibc's (and musl's) algorithm from Arm's optimized routines (MIT, Copyright (c) 2017-2018
 *   Arm Limited), with its table and coefficients: exp(x) = 2^(k/32) 2^(r/32), evaluated in double and
 *   rounded to float once. It gives glibc's results bit for bit, but for 2 of the 2^32 floats, where
 *   glibc on processors with FMA evaluates the polynomial with fused multiply-adds.
 * - tanhf and erff are correctly rounded, as glibc's are from 2.41 on (CORE-MATH's): a polynomial in
 *   double gives the result unless it lies too close to a boundary between two roundings, where a slower
 *   evaluation a few units in the last place of double from the exact value decides. Of the 2^32 floats,
 *   both give glibc 2.43's results for all.
 * - pow, sin and cos (double) are accurate to about 2^-68 (pow, in double-double arithmetic) and to an
 *   ulp (sin, cos); the engine rounds their results to float.
 *
 * The file needs no headers but the compiler's own and compiles as C99 for any target, so that the
 * native tests can link it in place of the C library's functions. It must be compiled without
 * contraction (-ffp-contract=off): the double-double products rely on every operation being rounded.
 */

#include <stdint.h>

float expf(float x);
float tanhf(float x);
float erff(float x);
double pow(double x, double y);
double sin(double x);
double cos(double x);

static inline uint32_t bits_of_float(float x) { uint32_t u; __builtin_memcpy(&u, &x, 4); return u; }
static inline uint64_t bits_of(double x) { uint64_t u; __builtin_memcpy(&u, &x, 8); return u; }
static inline double from_bits(uint64_t u) { double x; __builtin_memcpy(&x, &u, 8); return x; }

/* x rounded to an integer, ties to even, for |x| < 2^51 (adding 1.5 * 2^52 leaves no fraction bits). */
static inline double round_to_int(double x) { return (x + 0x1.8p52) - 0x1.8p52; }

/* ------------------------------------------------------------------------- expf (glibc's) */

/* exp2f_table[i] = the bits of 2^(i/32) minus i << 47, so that adding k << 47 for k = 32 q + i gives
   2^(k/32) */
static const uint64_t exp2f_table[32] = {
    0x3ff0000000000000u, 0x3fefd9b0d3158574u, 0x3fefb5586cf9890fu, 0x3fef9301d0125b51u,
    0x3fef72b83c7d517bu, 0x3fef54873168b9aau, 0x3fef387a6e756238u, 0x3fef1e9df51fdee1u,
    0x3fef06fe0a31b715u, 0x3feef1a7373aa9cbu, 0x3feedea64c123422u, 0x3feece086061892du,
    0x3feebfdad5362a27u, 0x3feeb42b569d4f82u, 0x3feeab07dd485429u, 0x3feea47eb03a5585u,
    0x3feea09e667f3bcdu, 0x3fee9f75e8ec5f74u, 0x3feea11473eb0187u, 0x3feea589994cce13u,
    0x3feeace5422aa0dbu, 0x3feeb737b0cdc5e5u, 0x3feec49182a3f090u, 0x3feed503b23e255du,
    0x3feee89f995ad3adu, 0x3feeff76f2fb5e47u, 0x3fef199bdd85529cu, 0x3fef3720dcef9069u,
    0x3fef5818dcfba487u, 0x3fef7c97337b9b5fu, 0x3fefa4afa2a490dau, 0x3fefd0765b6e4540u,
};

float expf(float x) {
    const uint32_t abstop = (bits_of_float(x) >> 20) & 0x7FFu;
    if (abstop >= 0x42Bu) {                             /* |x| >= 88, or NaN */
        if (bits_of_float(x) == 0xFF800000u) return 0.0f;    /* -inf */
        if (abstop >= 0x7F8u) return x + x;                   /* +inf, NaN */
        if (x > 0x1.62e42ep6f) return (float)(0x1p97 * 0x1p97);   /* overflow: inf */
        if (x < -0x1.9fe368p6f) return 0.0f;                  /* below half the smallest subnormal */
        if (x < -0x1.9d1d9ep6f) return 0x1p-149f;             /* the smallest subnormal */
    }
    /* x 32 / ln 2 = k + r, k an integer and |r| <= 1/2; exp(x) = 2^(k/32) 2^(r/32), the second factor a
       cubic in r */
    const double z = 0x1.71547652b82fep+5 * (double)x;
    double kd = z + 0x1.8p52;                           /* the integer in its low bits */
    const uint64_t ki = bits_of(kd);
    kd -= 0x1.8p52;
    const double r = z - kd;
    const double s = from_bits(exp2f_table[ki % 32u] + (ki << 47));
    const double c0 = 0x1.c6af84b912394p-5 / 32 / 32 / 32, c1 = 0x1.ebfce50fac4f3p-3 / 32 / 32;
    const double c2 = 0x1.62e42ff0c52d6p-1 / 32;
    const double p = c0 * r + c1, r2 = r * r;
    double y = c2 * r + 1.0;
    y = p * r2 + y;
    return (float)(y * s);
}

/* ------------------------------------------------------------------------- double-double arithmetic */

typedef struct {
    double hi, lo;
} Dd;

static inline Dd dd(double hi, double lo) { Dd r; r.hi = hi; r.lo = lo; return r; }

/* a + b exactly, as the rounded sum and its error */
static inline Dd two_sum(double a, double b) {
    const double s = a + b, v = s - a;
    return dd(s, (a - (s - v)) + (b - v));
}
/* the same for |a| >= |b| */
static inline Dd fast_two_sum(double a, double b) {
    const double s = a + b;
    return dd(s, b - (s - a));
}
/* a b exactly (Dekker: each factor split into halves of 26 bits) */
static inline Dd two_prod(double a, double b) {
    const double p = a * b, ca = 134217729.0 * a, cb = 134217729.0 * b;
    const double ah = ca - (ca - a), al = a - ah, bh = cb - (cb - b), bl = b - bh;
    return dd(p, ((ah * bh - p) + ah * bl + al * bh) + al * bl);
}
static inline Dd dd_add(Dd a, Dd b) {
    Dd s = two_sum(a.hi, b.hi);
    return fast_two_sum(s.hi, s.lo + (a.lo + b.lo));
}
static inline Dd dd_mul(Dd a, Dd b) {
    Dd p = two_prod(a.hi, b.hi);
    return fast_two_sum(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi));
}

/* x 2^q, for results that are normal */
static double scale(double x, int q) {
    while (q > 1000) { x *= 0x1p1000; q -= 1000; }
    while (q < -1000) { x *= 0x1p-1000; q += 1000; }
    return x * from_bits((uint64_t)(q + 1023) << 52);
}

/* 2^(j/32) as double-double */
static const double exp2_dd[32][2] = {
    {0x1.0000000000000p+0, 0x0.0p+0},                   {0x1.059b0d3158574p+0, 0x1.d73e2a475b465p-55},
    {0x1.0b5586cf9890fp+0, 0x1.8a62e4adc610bp-54},      {0x1.11301d0125b51p+0, -0x1.6c51039449b3ap-54},
    {0x1.172b83c7d517bp+0, -0x1.19041b9d78a76p-55},     {0x1.1d4873168b9aap+0, 0x1.e016e00a2643cp-54},
    {0x1.2387a6e756238p+0, 0x1.9b07eb6c70573p-54},      {0x1.29e9df51fdee1p+0, 0x1.612e8afad1255p-55},
    {0x1.306fe0a31b715p+0, 0x1.6f46ad23182e4p-55},      {0x1.371a7373aa9cbp+0, -0x1.63aeabf42eae2p-54},
    {0x1.3dea64c123422p+0, 0x1.ada0911f09ebcp-55},      {0x1.44e086061892dp+0, 0x1.89b7a04ef80d0p-59},
    {0x1.4bfdad5362a27p+0, 0x1.d4397afec42e2p-56},      {0x1.5342b569d4f82p+0, -0x1.07abe1db13cadp-55},
    {0x1.5ab07dd485429p+0, 0x1.6324c054647adp-54},      {0x1.6247eb03a5585p+0, -0x1.383c17e40b497p-54},
    {0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54},     {0x1.71f75e8ec5f74p+0, -0x1.16e4786887a99p-55},
    {0x1.7a11473eb0187p+0, -0x1.41577ee04992fp-55},     {0x1.82589994cce13p+0, -0x1.d4c1dd41532d8p-54},
    {0x1.8ace5422aa0dbp+0, 0x1.6e9f156864b27p-54},      {0x1.93737b0cdc5e5p+0, -0x1.75fc781b57ebcp-57},
    {0x1.9c49182a3f090p+0, 0x1.c7c46b071f2bep-56},      {0x1.a5503b23e255dp+0, -0x1.d2f6edb8d41e1p-54},
    {0x1.ae89f995ad3adp+0, 0x1.7a1cd345dcc81p-54},      {0x1.b7f76f2fb5e47p+0, -0x1.5584f7e54ac3bp-56},
    {0x1.c199bdd85529cp+0, 0x1.11065895048ddp-55},      {0x1.cb720dcef9069p+0, 0x1.503cbd1e949dbp-56},
    {0x1.d5818dcfba487p+0, 0x1.2ed02d75b3707p-55},      {0x1.dfc97337b9b5fp+0, -0x1.1a5cd4f184b5cp-54},
    {0x1.ea4afa2a490dap+0, -0x1.e9c23179c2893p-54},     {0x1.f50765b6e4540p+0, 0x1.9d3e12dd8a18bp-54},
};

/* exp(a) = e 2^q for |a.hi| < 746, e in double-double: a = k ln 2 / 32 + r with |r| <= ln 2 / 64, exp(r)
   by its series to r^8 (1 + r + r^2 / 2 in double-double) and 2^(k/32) from the table; about 2^-70 from
   the exact value. */
static Dd exp_dd(Dd a, int *q) {
    const double k = round_to_int(a.hi * 0x1.71547652b82fep+5);
    const int32_t ki = (int32_t)k, j = ki & 31;
    Dd p = two_prod(k, 0x1.62e42fefa39efp-6);           /* k ln 2 / 32 */
    p.lo += k * 0x1.abc9e3b39803fp-61;
    Dd r = two_sum(a.hi, -p.hi);
    r = fast_two_sum(r.hi, r.lo + (a.lo - p.lo));
    const double x = r.hi;
    const Dd square = two_prod(x, x);
    const double cubic = x * x * x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0 / 120 + x * (1.0 / 720 + x * (1.0 / 5040 +
                         x * (1.0 / 40320))))));
    Dd e = fast_two_sum(1.0, x);
    e = dd_add(e, dd(0.5 * square.hi, 0.5 * square.lo + (x * r.lo + r.lo + cubic)));
    *q = (ki - j) / 32;
    return dd_mul(dd(exp2_dd[j][0], exp2_dd[j][1]), e);
}

/* log x for finite x > 0 in double-double, about 2^-69 from the exact value relative to the part
   that does not come from the exponent: x = 2^e m with m in [sqrt(1/2), sqrt(2)), and log m = 2
   atanh(s) with s = (m - 1) / (m + 1), |s| < 0.172. */
static Dd log_dd(double x) {
    uint64_t u = bits_of(x);
    int e = 0;
    if (u < 0x0010000000000000u) {                      /* subnormal */
        u = bits_of(x * 0x1p54);
        e = -54;
    }
    e += (int)(u >> 52) - 1023;
    double m = from_bits((u & 0x000FFFFFFFFFFFFFu) | 0x3FF0000000000000u);
    if (m > 0x1.6a09e667f3bcdp+0) {
        m *= 0.5;
        e++;
    }
    const double f = m - 1.0;                           /* exact */
    const Dd g = two_sum(m, 1.0);                       /* exact */
    const double sh = f / g.hi;
    const Dd t = two_prod(sh, g.hi);
    const Dd s = fast_two_sum(sh, (((f - t.hi) - t.lo) - sh * g.lo) / g.hi);
    const Dd s2 = dd_mul(s, s), s3 = dd_mul(s2, s), s5 = dd_mul(s3, s2);
    /* 2 atanh(s) = 2 s + 2/3 s^3 + 2/5 s^5 + s^5 z (2/7 + 2/9 z + ... + 2/27 z^10), z = s^2 */
    const double z = s2.hi;
    const double tail = z * (2.0 / 7 + z * (2.0 / 9 + z * (2.0 / 11 + z * (2.0 / 13 + z * (2.0 / 15 + z * (2.0 / 17 +
                        z * (2.0 / 19 + z * (2.0 / 21 + z * (2.0 / 23 + z * (2.0 / 25 + z * (2.0 / 27)))))))))));
    Dd odd = dd_add(dd_mul(s3, dd(0x1.5555555555555p-1, 0x1.5555555555555p-55)),         /* 2/3 s^3 */
                    dd_mul(s5, dd(0x1.999999999999ap-2, -0x1.999999999999ap-56)));       /* 2/5 s^5 */
    odd = fast_two_sum(odd.hi, odd.lo + s5.hi * tail);
    const Dd log_m = dd_add(dd(2.0 * s.hi, 2.0 * s.lo), odd);
    Dd el = two_prod((double)e, 0x1.62e42fefa39efp-1);  /* e ln 2 */
    el.lo += (double)e * 0x1.abc9e3b39803fp-56;
    return dd_add(el, log_m);
}

/* ------------------------------------------------------------------------- pow */

/* 1 for odd integers, 2 for even ones, 0 for numbers that are no integer */
static int integer_kind(double y) {
    const double a = y < 0 ? -y : y;
    if (a >= 0x1p53) return 2;
    if (round_to_int(a) != a) return 0;
    return round_to_int(a * 0.5) == a * 0.5 ? 2 : 1;
}

double pow(double x, double y) {
    if (y == 0.0 || x == 1.0) return 1.0;
    if (x != x || y != y) return x + y;
    const double ax = x < 0 ? -x : x, ay = y < 0 ? -y : y;
    if (ay == __builtin_inf()) {
        if (ax == 1.0) return 1.0;
        return (ax > 1.0) == (y > 0) ? __builtin_inf() : 0.0;
    }
    double sign = 1.0;
    if (x < 0 || (x == 0 && bits_of(x) >> 63)) {       /* negative numbers, -0 */
        const int kind = integer_kind(y);
        if (kind == 0 && x != 0) return (x - x) / (x - x);              /* NaN */
        if (kind == 1) sign = -1.0;
    }
    if (ax == 0) return sign * (y < 0 ? __builtin_inf() : 0.0);
    if (ax == __builtin_inf()) return sign * (y < 0 ? 0.0 : __builtin_inf());
    const Dd l = log_dd(ax);
    Dd a = two_prod(l.hi, y);
    a = fast_two_sum(a.hi, a.lo + l.lo * y);
    if (a.hi > 709.8) return sign * __builtin_inf();
    if (a.hi < -745.2) return sign * 0.0;
    int q;
    const Dd e = exp_dd(a, &q);
    return sign * scale(e.hi + e.lo, q);
}

/* ------------------------------------------------------------------------- sin, cos */

/* x = k pi/2 + r with |r| <= pi/4 or a little more, r in double-double; returns k mod 4. pi/2 is the
   sum of three doubles here, enough for |x| < 2^51. */
static int reduce(double x, Dd *r) {
    const double k = round_to_int(x * 0x1.45f306dc9c883p-1);
    const Dd p1 = two_prod(k, 0x1.921fb54442d18p+0), p2 = two_prod(k, 0x1.1a62633145c07p-54);
    Dd t = two_sum(x, -p1.hi);
    t = fast_two_sum(t.hi, t.lo - p1.lo);
    t = dd_add(t, dd(-p2.hi, -p2.lo));
    *r = fast_two_sum(t.hi, t.lo - k * -0x1.f1976b7ed8fbcp-110);
    return (int)((int64_t)k & 3);
}

/* sin(r.hi + r.lo) and cos(r.hi + r.lo) for |r| <= pi/4, by their series to r^21 and r^22 */
static double sin_reduced(Dd r) {
    const double x = r.hi, z = x * x;
    const double tail = z * (-1.0 / 6 + z * (1.0 / 120 + z * (-1.0 / 5040 + z * (1.0 / 362880 + z * (-1.0 / 39916800 +
                        z * (1.0 / 6227020800.0 + z * (-1.0 / 1307674368000.0 + z * (1.0 / 355687428096000.0 + z *
                        (-1.0 / 121645100408832000.0 + z * (1.0 / 51090942171709440000.0))))))))));
    return x + (x * tail + r.lo * (1.0 - 0.5 * z));
}
static double cos_reduced(Dd r) {
    const double x = r.hi, z = x * x, hz = 0.5 * z, w = 1.0 - hz;
    const double tail = z * z * (1.0 / 24 + z * (-1.0 / 720 + z * (1.0 / 40320 + z * (-1.0 / 3628800 + z *
                        (1.0 / 479001600 + z * (-1.0 / 87178291200.0 + z * (1.0 / 20922789888000.0 + z *
                        (-1.0 / 6402373705728000.0 + z * (1.0 / 2432902008176640000.0 + z *
                        (-1.0 / 1124000727777607680000.0))))))))));
    return w + (((1.0 - w) - hz) + (tail - x * r.lo));
}

double sin(double x) {
    const double a = x < 0 ? -x : x;
    if (!(a < 0x1p51)) return x - x;                    /* NaN for inf and NaN; no reduction beyond */
    Dd r;
    switch (reduce(x, &r)) {
        case 0:  return sin_reduced(r);
        case 1:  return cos_reduced(r);
        case 2:  return -sin_reduced(r);
        default: return -cos_reduced(r);
    }
}

double cos(double x) {
    const double a = x < 0 ? -x : x;
    if (!(a < 0x1p51)) return x - x;
    Dd r;
    switch (reduce(x, &r)) {
        case 0:  return cos_reduced(r);
        case 1:  return -sin_reduced(r);
        case 2:  return -cos_reduced(r);
        default: return sin_reduced(r);
    }
}

/* ------------------------------------------------------------------------- tanhf, erff */

/* tanhf and erff first evaluate a polynomial in double, about 2^-47 from the exact value. Its rounding to
   float is the correctly rounded result unless it lies that close to a boundary between two roundings:
   then (for about one input in a million) the accurate evaluation decides. */
static int rounds_alike(double y, double margin, float *f) {
    *f = (float)(y - margin);
    return *f == (float)(y + margin);
}

/* e^y - 1 for y >= 0, to an ulp or two */
static double expm1_positive(double y) {
    if (y < 0x1p-5)
        return y + y * y * (0.5 + y * (1.0 / 6 + y * (1.0 / 24 + y * (1.0 / 120 + y * (1.0 / 720 + y * (1.0 / 5040 +
               y * (1.0 / 40320 + y * (1.0 / 362880))))))));
    int q;
    const Dd e = exp_dd(dd(y, 0.0), &q);
    const Dd d = two_sum(scale(e.hi, q), -1.0);
    return d.hi + (d.lo + scale(e.lo, q));
}

/* tanh a for 0 < a < 9.1, a few ulps of double from the exact value: (e^2a - 1) / (e^2a + 1) */
static double tanh_accurate(double a) {
    const double u = expm1_positive(2.0 * a);
    return u / (u + 2.0);
}

float tanhf(float x) {
    double a = (double)x;
    const double sign = a < 0 ? -1.0 : 1.0;
    a *= sign;
    if (a == 0 || a != a) return x + x;                 /* +-0, NaN */
    if (a > 9.1) return (float)sign;                    /* 1 to float precision from 9.0109 on */
    double y;
    if (a < 0.25) {                                     /* tanh(a) / a by its series in a^2, to a^18 */
        const double u = a * a;
        y = a * (1.0 + u * (-0x1.5555555555555p-2 + u * (0x1.1111111111111p-3 + u * (-0x1.ba1ba1ba1ba1cp-5 +
            u * (0x1.664f4882c10fap-6 + u * (-0x1.226e355e6c23dp-7 + u * (0x1.d6d3d0e157de0p-9 +
            u * (-0x1.7da36452b75e3p-10 + u * (0x1.3558248036744p-11 + u * -0x1.f57d7734d1664p-13)))))))));
    } else {                                            /* 1 - 2 / (e^2a + 1), e^2a = 2^(k/32) e^(r ln 2 / 32) */
        const double z = 2.0 * a * 0x1.71547652b82fep+5;
        const double kd = (z + 0x1.8p52) - 0x1.8p52, r = z - kd;
        const int k = (int)kd, j = k & 31;
        const double s = from_bits(bits_of(exp2_dd[j][0]) + ((uint64_t)(k >> 5) << 52));
        const double e = s + s * (r * (0x1.62e42fefa39efp-6 + r * (0x1.ebfbdff82c58fp-13 + r * (0x1.c6b08d704a0c0p-20 +
                         r * (0x1.3b2ab6fba4e77p-27 + r * (0x1.5d87fe78a6731p-35 + r * 0x1.430912f86c787p-43))))));
        y = 1.0 - 2.0 / (e + 1.0);
    }
    float f;
    if (!rounds_alike(y, y * 0x1p-45, &f)) f = (float)tanh_accurate(a);
    return (float)(sign * f);
}

/* erf(i/8) and 2/sqrt(pi) exp(-(i/8)^2), correctly rounded */
static const double erf_table[33][2] = {
    {0x0.0p+0, 0x1.20dd750429b6dp+0},                   {0x1.1f5e1a35c3b89p-3, 0x1.1c62fa1e869b6p+0},
    {0x1.1af54e232d609p-2, 0x1.0f5d1602f7e41p+0},       {0x1.9dd0d2b721f39p-2, 0x1.f5f0cdaf15313p-1},
    {0x1.0a7ef5c18edd2p-1, 0x1.c1efca49a5011p-1},       {0x1.3f196dcd0f135p-1, 0x1.86e9694134b9ep-1},
    {0x1.6c1c9759d0e5fp-1, 0x1.492e42d78d2c5p-1},       {0x1.91724951b8fc6p-1, 0x1.0cab61f084b93p-1},
    {0x1.af767a741088bp-1, 0x1.a911f096fbc26p-2},       {0x1.c6dad2829ec62p-1, 0x1.45e99bcbb7915p-2},
    {0x1.d8865d98abe01p-1, 0x1.e4652fadcb6b2p-3},       {0x1.e5768c3b4a3fcp-1, 0x1.5ce595c455b0ap-3},
    {0x1.eea5557137ae0p-1, 0x1.e723726b824a9p-4},       {0x1.f4f693b67bd77p-1, 0x1.499d478bca735p-4},
    {0x1.f92d077f8d56dp-1, 0x1.b055303221015p-5},       {0x1.fbe61eef4cf6ap-1, 0x1.12ceb37ff9bc3p-5},
    {0x1.fd9ae142795e3p-1, 0x1.529b9e8cf9a1ep-6},       {0x1.fea4218d6594ap-1, 0x1.94624e78e0fafp-7},
    {0x1.ff404760319b4p-1, 0x1.d4143a9dfe965p-8},       {0x1.ff9960f3eb327p-1, 0x1.06918b6355624p-8},
    {0x1.ffcaa8f4c9beap-1, 0x1.1d83170fbf6fbp-9},       {0x1.ffe514bbdc197p-1, 0x1.2ce898809244ep-10},
    {0x1.fff2cfb0453d9p-1, 0x1.3360ccd23db3ap-11},      {0x1.fff9ba420e834p-1, 0x1.30538fbb77ecdp-12},
    {0x1.fffd1ac4135f9p-1, 0x1.2408e9ba3327fp-13},      {0x1.fffeb3ebb267bp-1, 0x1.0f9e1b4dd36dfp-14},
    {0x1.ffff6f9f67e55p-1, 0x1.e9b5e8d00ce77p-16},      {0x1.ffffc316d9ed0p-1, 0x1.abe09e9144b5ep-17},
    {0x1.ffffe710d565ep-1, 0x1.6a597219a93dap-18},      {0x1.fffff618c3da6p-1, 0x1.296a70f414053p-19},
    {0x1.fffffc2f171e3p-1, 0x1.d9371e2ff7c35p-21},      {0x1.fffffe92ced93p-1, 0x1.6ce1aa3fd7bddp-22},
    {0x1.ffffff7b91176p-1, 0x1.10b1488aeb235p-23},
};

/* hi + lo for hi > 0 and |lo| at most half an ulp of hi, rounded to float: (float)hi, but for an hi
   halfway between two floats, the one on lo's side */
static float round_positive_dd(double hi, double lo) {
    const float f = (float)hi;
    if (lo == 0 || (double)f == hi) return f;
    uint32_t u = bits_of_float(f);
    u = hi > (double)f ? u + 1u : u - 1u;               /* the other float around hi */
    float g;
    __builtin_memcpy(&g, &u, 4);
    if (hi != 0.5 * ((double)f + (double)g)) return f;
    return (lo > 0) == (g > f) ? g : f;
}

/* erf a for 0 < a < 4, rounded to float: erf(c + h) = erf(c) + 2/sqrt(pi) e^(-c^2) sum_k p_k h / (k + 1)
   with c = i/8 nearest a, |h| <= 1/16: e^(-(c + h)^2) = e^(-c^2) sum_k p_k, p_k = (-2 c h p_(k-1) - 2 h^2
   p_(k-2)) / k, which falls at least as 0.5^k / k! (|2 c h| <= 1/2). In double a few ulps from the exact
   value; below 1/16 (c = 0), 2/sqrt(pi) a (1 + t) in double-double, about 2^-62 from it. */
static float erf_accurate(double a) {
    const int i = (int)(a * 8.0 + 0.5);
    const double h = a - i * 0.125, b = -2.0 * (i * 0.125) * h, c = -2.0 * h * h;  /* h exact */
    double p0 = 1.0, p1 = b, t = 0.5 * b;
    for (int k = 2; k < 40; k++) {                      /* (with c = 0 every other term is 0) */
        const double pk = (b * p1 + c * p0) / k;
        t += pk / (k + 1);
        if ((pk < 0 ? -pk : pk) + (p1 < 0 ? -p1 : p1) < 0x1p-60) break;
        p0 = p1;
        p1 = pk;
    }
    if (i > 0) return (float)(erf_table[i][0] + erf_table[i][1] * (h * (1.0 + t)));
    const Dd g = two_prod(0x1.20dd750429b6dp+0, h);    /* 2/sqrt(pi) h, with 2/sqrt(pi) in double-double */
    const Dd y = fast_two_sum(g.hi, g.lo + (0x1.1ae3a914fed80p-56 * h + g.hi * t));
    return round_positive_dd(y.hi, y.lo);
}

/* erf(i/8 + h) for |h| <= 1/16, row i - 1: polynomials of degree 9 in h interpolating erf at the Chebyshev
   nodes of the interval, 2^-51 from it */
static const double erf_poly[32][10] = {
    {0x1.1f5e1a35c3b89p-3, 0x1.1c62fa1e869b6p+0, -0x1.1c62fa1e8662bp-3, -0x1.6f552dbcc3212p-2, 0x1.196c9cd5546f2p-4,
     0x1.aaba623bcb0dcp-4, -0x1.734e7ee786c34p-6, -0x1.892574f401b32p-6, 0x1.6eb288cd177e6p-8, 0x1.271e85baf002ap-8},
    {0x1.1af54e232d609p-2, 0x1.0f5d1602f7e41p+0, -0x1.0f5d1602f7b4bp-2, -0x1.3c974458cbd6ap-2, 0x1.040e8a6a8ca45p-3,
     0x1.47e5cfed290a3p-4, -0x1.4c0b0410fcfdep-5, -0x1.08d93a8f62399p-6, 0x1.3d1f46efaee52p-7, 0x1.54f23ddcb4ae9p-9},
    {0x1.9dd0d2b721f39p-2, 0x1.f5f0cdaf15313p-1, -0x1.78749a434fb19p-2, -0x1.e106c51d1effcp-3, 0x1.5529abc9ca99dp-3,
     0x1.7488b8a8b065fp-5, -0x1.9a7921d819155p-5, -0x1.65c11c33a7a46p-8, 0x1.6ff64786d46efp-7, 0x1.06855df9c8c1dp-13},
    {0x1.0a7ef5c18edd2p-1, 0x1.c1efca49a5011p-1, -0x1.c1efca49a4d8fp-2, -0x1.2bf531866e1a2p-3, 0x1.76f27de587622p-3,
     0x1.dfeeb5bd4fb31p-8, -0x1.99f11f0e177b6p-5, 0x1.623c1a4c70be1p-8, 0x1.48bccf6d960fbp-7, -0x1.1b737aa3269f6p-9},
    {0x1.3f196dcd0f135p-1, 0x1.86e9694134b9ep-1, -0x1.e8a3c39181d58p-2, -0x1.c810502168b8dp-5, 0x1.6963c8a270b8bp-3,
     -0x1.c1242df7519f4p-6, -0x1.52b25966a236cp-5, 0x1.c7cd6b9254148p-7, 0x1.b5b6fedcc16c1p-8, -0x1.db5d500ff161fp-9},
    {0x1.6c1c9759d0e5fp-1, 0x1.492e42d78d2c5p-1, -0x1.edc5644353c73p-2, 0x1.b6e8591f65e67p-6, 0x1.349b5eaa60254p-3,
     -0x1.b42a188cd111fp-5, -0x1.b8478009688f8p-6, 0x1.2e0ae4b476886p-6, 0x1.2deff0037fe08p-9, -0x1.03aa353934ea3p-8},
    {0x1.91724951b8fc6p-1, 0x1.0cab61f084b93p-1, -0x1.d62beb64e85b1p-2, 0x1.7c9d756a1133cp-4, 0x1.cc60568057105p-4,
     -0x1.1350f4b0e35cep-4, -0x1.53bb8a9e8e9eep-7, 0x1.30ac139afa245p-6, -0x1.e1a9016f902c2p-10, -0x1.aa626d88dc02fp-9},
    {0x1.af767a741088bp-1, 0x1.a911f096fbc26p-2, -0x1.a911f096fbe1bp-2, 0x1.1b614b0f527d3p-3, 0x1.1b614b133d8fcp-4,
     -0x1.1b614b0f0c4d9p-4, 0x1.2e44f5d624178p-8, 0x1.f096f74bf1191p-7, -0x1.384891bbb6750p-8, -0x1.edf8b923a7c5dp-10},
    {0x1.c6dad2829ec62p-1, 0x1.45e99bcbb7915p-2, -0x1.6ea6cf452ea10p-2, 0x1.4cb3cf0aa0c2cp-3, 0x1.ca50832538de0p-6,
     -0x1.f65d15f2f0b98p-5, 0x1.fd1c197d1d245p-7, 0x1.3acc84e99b8fdp-7, -0x1.8a86f6619cf88p-8, -0x1.7b7f5b1cd4f05p-12},
    {0x1.d8865d98abe01p-1, 0x1.e4652fadcb6b2p-3, -0x1.2ebf3dcc9f374p-2, 0x1.571d01c5c57c7p-3, -0x1.93a9a792e48cdp-8,
     -0x1.8281ce0d46e68p-5, 0x1.5cffe742fd9ecp-6, 0x1.db442632a3fa3p-9, -0x1.74e538ae6511bp-8, 0x1.ca7a2e5df386ap-11},
    {0x1.e5768c3b4a3fcp-1, 0x1.5ce595c455b0ap-3, -0x1.dfbbadedf5e37p-3, 0x1.4374d82e04d7cp-3, -0x1.f3b8d529102f8p-6,
     -0x1.f572c4cd1d96ep-6, 0x1.6b16e9912f383p-6, -0x1.73fee77750b80p-10, -0x1.170b3b3f852e3p-8, 0x1.9cc5c537c9f51p-10},
    {0x1.eea5557137ae0p-1, 0x1.e723726b824a9p-4, -0x1.6d5a95d0a1b33p-3, 0x1.1c2a02beb6b88p-3, -0x1.6d5a95d138c50p-5,
     -0x1.e723727204c09p-7, 0x1.3ca3da7ad0189p-6, -0x1.36d715ed1409cp-8, -0x1.35cc971c50ad8p-9, 0x1.bf915dfe2f724p-10},
    {0x1.f4f693b67bd77p-1, 0x1.499d478bca735p-4, -0x1.0bcfca21946b8p-3, 0x1.d6631e1a28f6cp-4, -0x1.974c036aad105p-5,
     -0x1.17d430ca371acp-9, 0x1.d8580c3b7b39cp-7, -0x1.954a9ef04acf4p-8, -0x1.3015ca80f5715p-11, 0x1.717b18eec14f3p-10},
    {0x1.f92d077f8d56dp-1, 0x1.b055303221015p-5, -0x1.7a4a8a2bdcb3ep-4, 0x1.7148c3d57c31cp-4, -0x1.8a0da545da712p-5,
     0x1.b22257dcb79d8p-8, 0x1.25b3987b21053p-7, -0x1.8d10fa9e363cep-8, 0x1.7c7e11a5278b8p-11, 0x1.d4c7c1e9e8631p-11},
    {0x1.fbe61eef4cf6ap-1, 0x1.12ceb37ff9bc3p-5, -0x1.01a1c847f9f89p-4, 0x1.143d1c6f4f01bp-4, -0x1.5a31652336cb3p-5,
     0x1.779b1e58e62b8p-7, 0x1.0d09d45eecf6bp-8, -0x1.42fcc533e2163p-8, 0x1.75ffccb2eff72p-10, 0x1.7c9b6826fb1a9p-12},
    {0x1.fd9ae142795e3p-1, 0x1.529b9e8cf9a1ep-6, -0x1.529b9e8cf96c1p-5, 0x1.8b0ae3a4787d0p-5, -0x1.1a2c597729307p-5,
     0x1.ace7404ed22fap-7, 0x1.e195b95aca521p-12, -0x1.bae0c95406cf9p-9, 0x1.a067feec78aefp-10, -0x1.9bee60382a231p-15},
    {0x1.fea4218d6594ap-1, 0x1.94624e78e0fafp-7, -0x1.ada873606ed84p-6, 0x1.0ea475da3bd48p-5, -0x1.afe553fbd7b4dp-6,
     0x1.9973b48c673f2p-7, -0x1.dd78a79be79c2p-10, -0x1.ea03f17ea5236p-10, 0x1.6a523b1d33928p-10, -0x1.2ee2ad20d8ecbp-12},
    {0x1.ff404760319b4p-1, 0x1.d4143a9dfe965p-8, -0x1.074b60f8df2fcp-6, 0x1.63ef61e82427ap-6, -0x1.38a98327af51bp-6,
     0x1.5d3b17bd8bbbap-7, -0x1.7cae09ee1da06p-9, -0x1.5f835e67687d7p-11, 0x1.05fb475109291p-10, -0x1.82a81364f31f9p-12},
    {0x1.ff9960f3eb327p-1, 0x1.06918b6355624p-8, -0x1.37ccd585f5915p-7, 0x1.c1ec102e36352p-7, -0x1.ae59615e296ffp-7,
     0x1.11dae47420357p-7, -0x1.982b36ee2a453p-9, 0x1.02834f97b79a6p-13, 0x1.37c3cd29c1ac6p-11, -0x1.619986f87dc20p-12},
    {0x1.ffcaa8f4c9beap-1, 0x1.1d83170fbf6fbp-9, -0x1.64e3dcd3afcc9p-8, 0x1.119da0c46cc6dp-7, -0x1.1a89b97ae7330p-7,
     0x1.90e812843e93bp-8, -0x1.6ecdd5f72e048p-9, 0x1.1c6105dbdf180p-11, 0x1.122374a723012p-12, -0x1.0663ed6da0220p-12},
    {0x1.ffe514bbdc197p-1, 0x1.2ce898809244ep-10, -0x1.8af14828c0ec9p-9, 0x1.407fbd18f130ep-8, -0x1.62d4c6d0d3ef9p-8,
     0x1.146c4b3da1ec7p-8, -0x1.267f50f436ddap-9, 0x1.64f89d7e629e6p-11, 0x1.28ad05dd42195p-15, -0x1.403cbbd4c95e2p-13},
    {0x1.fff2cfb0453d9p-1, 0x1.3360ccd23db3ap-11, -0x1.a6a519a1163f6p-10, 0x1.69cf466cce29fp-9, -0x1.ab0c273520f06p-9,
     0x1.69359604c628ap-9, -0x1.b2757b85b8aecp-10, 0x1.52b6386434169p-11, -0x1.7316d54b6b629p-14, -0x1.2b61314bcf0f7p-14},
    {0x1.fff9ba420e834p-1, 0x1.30538fbb77ecdp-12, -0x1.b5781e9d7e1a5p-11, 0x1.89e17c074d983p-10, -0x1.ed4ac7d412c7ep-10,
     0x1.c11f270354a5dp-10, -0x1.2add300f56d3ep-10, 0x1.151f894e84a30p-11, -0x1.1bb4e13e76024p-13, -0x1.1131a84d47539p-16},
    {0x1.fffd1ac4135f9p-1, 0x1.2408e9ba3327ep-13, -0x1.b60d5e974e560p-12, 0x1.9db74b1d1e5b5p-11, -0x1.11c85b1b5bbedp-10,
     0x1.0a7b55447f3e7p-10, -0x1.82f247a0b401dp-11, 0x1.998b6087c5788p-12, -0x1.1a53e8a8853e6p-13, 0x1.cf19de52f2dd2p-17},
    {0x1.fffeb3ebb267bp-1, 0x1.0f9e1b4dd36dep-14, -0x1.a8670aa99b284p-13, 0x1.a3737e2a2fd2cp-12, -0x1.24544f0139d51p-11,
     0x1.2e7e763a97916p-11, -0x1.da4977434a254p-12, 0x1.176d050000892p-12, -0x1.d283a01d26a48p-14, 0x1.a9f2a3acabe86p-16},
    {0x1.ffff6f9f67e55p-1, 0x1.e9b5e8d00ce75p-16, -0x1.8de3cd290989cp-14, 0x1.9aa489e3cb688p-13, -0x1.2c7d5ef2389c8p-12,
     0x1.490a4d20b4eadp-12, -0x1.14545f9c008d8p-12, 0x1.647f7f454f26ap-13, -0x1.568cc4d72ab3cp-14, 0x1.b24ad79446ba1p-16},
    {0x1.ffffc316d9ed0p-1, 0x1.abe09e9144b5ep-17, -0x1.690585ca8e363p-15, 0x1.84522fe881a0ep-14, -0x1.298f8d4d7d079p-13,
     0x1.577577874b0fbp-13, -0x1.330a9663e4b0bp-13, 0x1.ac999de836779p-14, -0x1.cc75df723a213p-15, 0x1.64a999def3fa1p-16},
    {0x1.ffffe710d565ep-1, 0x1.6a597219a93dap-18, -0x1.3d0e43d66c7d2p-16, 0x1.62ccea63caafbp-15, -0x1.1c077229f9b40p-14,
     0x1.586bafcb2c3aep-14, -0x1.4615152d67223p-14, 0x1.e827f2dd72b90p-15, -0x1.1fc44a0075bb1p-15, 0x1.0147b86026475p-16},
    {0x1.fffff618c3da6p-1, 0x1.296a70f414056p-19, -0x1.0d88765d25f5fp-17, 0x1.394b1fa66fc59p-16, -0x1.05760aea1a8cdp-15,
     0x1.4c1fe48f9f9fdp-15, -0x1.4b97dbf9f971fp-15, 0x1.085c00c05555dp-15, -0x1.51a62eefa4087p-16, 0x1.51a8d15ab9294p-17},
    {0x1.fffffc2f171e3p-1, 0x1.d9371e2ff7c42p-21, -0x1.bba3ac4cd624fp-19, 0x1.0b6a7b0f18ce7p-17, -0x1.d06f58a4d805ep-17,
     0x1.3436bca9ec48fp-16, -0x1.435755c2a5a12p-16, 0x1.110dc82bf9120p-16, -0x1.76412cd29e73bp-17, 0x1.9a77ca9b1004dp-18},
    {0x1.fffffe92ced93p-1, 0x1.6ce1aa3fd7bf1p-22, -0x1.617a9cedada1dp-20, 0x1.b95fa39b32281p-19, -0x1.8e1fc46bf4194p-18,
     0x1.13717227a10c4p-17, -0x1.2eb2174718d00p-17, 0x1.0d8c0a0b6f6f8p-17, -0x1.899ae40e68fd5p-18, 0x1.d32444afd801ep-19},
    {0x1.ffffff7b91176p-1, 0x1.10b1488aeb250p-23, -0x1.10b1488ab834cp-21, 0x1.603a5308ba8aep-20, -0x1.4980e2b862928p-19,
     0x1.da5f11065e57bp-19, -0x1.104fc4e0d0aa8p-18, 0x1.fd7befb73691ap-19, -0x1.8a0d83a3922c6p-19, 0x1.f5316dc3ac3afp-20},
};

float erff(float x) {
    double a = (double)x;
    const double sign = a < 0 ? -1.0 : 1.0;
    a *= sign;
    if (a == 0 || a != a) return x + x;                 /* +-0, NaN */
    if (!(a < 4.0)) return (float)sign;                 /* 1 to float precision from 3.92 on */
    double y;
    if (a < 0.0625) {                                   /* erf(a) / a by its series in a^2, to a^10 */
        const double u = a * a;
        y = a * (0x1.20dd750429b6dp+0 + u * (-0x1.812746b0379e7p-2 + u * (0x1.ce2f21a042be2p-4 +
            u * (-0x1.b82ce31288b51p-6 + u * (0x1.565bcd0e6a53fp-8 + u * -0x1.c02db40040b86p-11)))));
    } else {
        const int i = (int)(a * 8.0 + 0.5);
        const double h = a - i * 0.125, *c = erf_poly[i - 1];
        y = c[0] + h * (c[1] + h * (c[2] + h * (c[3] + h * (c[4] + h * (c[5] + h * (c[6] + h * (c[7] + h * (c[8] +
            h * c[9]))))))));
    }
    float f;
    if (!rounds_alike(y, y * 0x1p-46, &f)) f = erf_accurate(a);
    return (float)(sign * f);
}
