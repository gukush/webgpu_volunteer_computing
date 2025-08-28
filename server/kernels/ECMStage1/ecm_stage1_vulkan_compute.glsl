#version 450

// ECM Stage 1 Vulkan Compute Shader
// 256-bit big integers using 4x64-bit limbs
// Montgomery curves, x-only arithmetic
// One invocation per curve

#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define LIMBS 4

struct u256 {
    uint64_t v[LIMBS]; // little-endian limbs
};

struct EcmConsts {
    u256 N;       // modulus
    u256 R2;      // R^2 mod N  
    u256 mont_one;// R mod N
    uint64_t n0inv; // -N^{-1} mod 2^64
    uint compute_gcd; // 1 => compute gcd; 0 => return Z only
};

struct CurveIn {
    u256 A24;  // (A+2)/4 mod N (normal domain)
    u256 X1;   // x-coordinate of base point (normal domain)
};

struct CurveOut {
    u256 result; // gcd(Z,N) or Z
    uint status; // 0=OK, 1=trivial, 2=factor found
};

struct PointXZ {
    u256 X;
    u256 Z;
};

// Input/Output buffers
layout(std430, binding = 0) restrict readonly buffer ConstBuffer {
    EcmConsts constants;
};

layout(std430, binding = 1) restrict readonly buffer CurveInBuffer {
    CurveIn curves[];
};

layout(std430, binding = 2) restrict readonly buffer PrimePowersBuffer {
    uint prime_powers[];
};

layout(std430, binding = 3) restrict writeonly buffer CurveOutBuffer {
    CurveOut results[];
};

layout(push_constant) uniform PushConstants {
    uint num_curves;
    uint pp_count;
};

// Helper functions
bool is_zero_u256(u256 a) {
    return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0ul;
}

int cmp_u256(u256 a, u256 b) {
    for (int i = LIMBS-1; i >= 0; --i) {
        if (a.v[i] < b.v[i]) return -1;
        if (a.v[i] > b.v[i]) return 1;
    }
    return 0;
}

void set_zero_u256(inout u256 r) {
    for (int i = 0; i < LIMBS; ++i) r.v[i] = 0ul;
}

void set_one_u256(inout u256 r) {
    r.v[0] = 1ul;
    for (int i = 1; i < LIMBS; ++i) r.v[i] = 0ul;
}

void add_u256(inout u256 r, u256 a, u256 b) {
    uint64_t c = 0ul;
    for (int i = 0; i < LIMBS; ++i) {
        uint64_t s = a.v[i] + b.v[i];
        uint64_t c1 = (s < a.v[i]) ? 1ul : 0ul;
        s += c;
        uint64_t c2 = (s < c) ? 1ul : 0ul;
        r.v[i] = s;
        c = c1 + c2;
    }
}

void sub_u256(inout u256 r, u256 a, u256 b) {
    uint64_t br = 0ul;
    for (int i = 0; i < LIMBS; ++i) {
        uint64_t d = a.v[i] - b.v[i];
        uint64_t b1 = (a.v[i] < b.v[i]) ? 1ul : 0ul;
        uint64_t d2 = d - br;
        uint64_t b2 = (d < br) ? 1ul : 0ul;
        r.v[i] = d2;
        br = b1 | b2;
    }
}

void cond_sub_u256(inout u256 a, u256 N) {
    u256 t;
    sub_u256(t, a, N);
    bool ge = (cmp_u256(a, N) >= 0);
    uint64_t mask = ge ? 0xFFFFFFFFFFFFFFFFul : 0ul;
    for (int i = 0; i < LIMBS; ++i) {
        a.v[i] = (t.v[i] & mask) | (a.v[i] & ~mask);
    }
}

// Multiply with high part - Vulkan doesn't have built-in umul64hi
uint64_t umul64hi(uint64_t a, uint64_t b) {
    // Split into 32-bit parts
    uint64_t a_lo = a & 0xFFFFFFFFul;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFul;
    uint64_t b_hi = b >> 32;
    
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;
    
    uint64_t cy = ((p0 >> 32) + (p1 & 0xFFFFFFFFul) + (p2 & 0xFFFFFFFFul)) >> 32;
    return p3 + (p1 >> 32) + (p2 >> 32) + cy;
}

void mont_mul(inout u256 r, u256 a, u256 b, EcmConsts C) {
    uint64_t t[LIMBS+1];
    for (int i = 0; i <= LIMBS; ++i) t[i] = 0ul;
    
    for (int i = 0; i < LIMBS; ++i) {
        // t += a_i * b
        uint64_t carry = 0ul;
        for (int j = 0; j < LIMBS; ++j) {
            uint64_t lo = a.v[i] * b.v[j];
            uint64_t hi = umul64hi(a.v[i], b.v[j]);
            uint64_t tmp = t[j] + lo;
            uint64_t c1 = (tmp < t[j]) ? 1ul : 0ul;
            tmp += carry;
            uint64_t c2 = (tmp < carry) ? 1ul : 0ul;
            t[j] = tmp;
            carry = hi + c1 + c2;
        }
        t[LIMBS] += carry;
        
        // m = t[0] * n0inv
        uint64_t m = t[0] * C.n0inv;
        
        // t += m * N
        carry = 0ul;
        for (int j = 0; j < LIMBS; ++j) {
            uint64_t lo = m * C.N.v[j];
            uint64_t hi = umul64hi(m, C.N.v[j]);
            uint64_t tmp = t[j] + lo;
            uint64_t c1 = (tmp < t[j]) ? 1ul : 0ul;
            tmp += carry;
            uint64_t c2 = (tmp < carry) ? 1ul : 0ul;
            t[j] = tmp;
            carry = hi + c1 + c2;
        }
        t[LIMBS] += carry;
        
        // shift right by one limb
        for (int k = 0; k < LIMBS; ++k) t[k] = t[k+1];
        t[LIMBS] = 0ul;
    }
    
    for (int i = 0; i < LIMBS; ++i) r.v[i] = t[i];
    cond_sub_u256(r, C.N);
}

void to_mont(inout u256 r, u256 a, EcmConsts C) {
    mont_mul(r, a, C.R2, C);
}

void from_mont(inout u256 r, u256 a, EcmConsts C) {
    mont_mul(r, a, C.mont_one, C);
}

void mont_add(inout u256 r, u256 a, u256 b, EcmConsts C) {
    add_u256(r, a, b);
    cond_sub_u256(r, C.N);
}

void mont_sub(inout u256 r, u256 a, u256 b, EcmConsts C) {
    u256 t;
    sub_u256(t, a, b);
    if (cmp_u256(a, b) < 0) {
        add_u256(r, t, C.N);
    } else {
        r = t;
    }
}

void mont_sqr(inout u256 r, u256 a, EcmConsts C) {
    mont_mul(r, a, a, C);
}

// Montgomery curve operations
void xDBL(inout PointXZ R2, PointXZ R, u256 A24, EcmConsts C) {
    u256 t1, t2, t3, t4;
    mont_add(t1, R.X, R.Z, C);
    mont_sub(t2, R.X, R.Z, C);
    mont_sqr(t1, t1, C);
    mont_sqr(t2, t2, C);
    mont_sub(t3, t1, t2, C);
    mont_mul(R2.X, t1, t2, C);
    mont_mul(t4, t3, A24, C);
    mont_add(t4, t4, t2, C);
    mont_mul(R2.Z, t3, t4, C);
}

void xADD(inout PointXZ R3, PointXZ P, PointXZ Q, PointXZ Diff, EcmConsts C) {
    u256 t1, t2, t3, t4, t5, t6;
    mont_add(t1, P.X, P.Z, C);
    mont_sub(t2, P.X, P.Z, C);
    mont_add(t3, Q.X, Q.Z, C);
    mont_sub(t4, Q.X, Q.Z, C);
    mont_mul(t5, t1, t4, C);
    mont_mul(t6, t2, t3, C);
    mont_add(t1, t5, t6, C);
    mont_sub(t2, t5, t6, C);
    mont_sqr(t1, t1, C);
    mont_sqr(t2, t2, C);
    mont_mul(R3.X, t1, Diff.Z, C);
    mont_mul(R3.Z, t2, Diff.X, C);
}

void cswap_point(inout PointXZ A, inout PointXZ B, uint64_t bit) {
    uint64_t mask = (bit & 1ul) != 0ul ? 0xFFFFFFFFFFFFFFFFul : 0ul;
    for (int i = 0; i < LIMBS; ++i) {
        uint64_t t = mask & (A.X.v[i] ^ B.X.v[i]);
        A.X.v[i] ^= t;
        B.X.v[i] ^= t;
        t = mask & (A.Z.v[i] ^ B.Z.v[i]);
        A.Z.v[i] ^= t;
        B.Z.v[i] ^= t;
    }
}

void mont_ladder(inout PointXZ R0, PointXZ P, uint k, u256 A24, EcmConsts C) {
    PointXZ R1 = P;
    R0.X = constants.mont_one;
    set_zero_u256(R0.Z);
    
    bool started = false;
    for (int i = 31; i >= 0; --i) {
        uint bit = (k >> i) & 1u;
        if (!started && bit == 0u) continue;
        started = true;
        
        cswap_point(R0, R1, uint64_t(1u - bit));
        PointXZ t0, t1;
        xADD(t0, R0, R1, P, C);
        xDBL(t1, R1, A24, C);
        R0 = t0;
        R1 = t1;
    }
}

// GCD helpers
int ctz64(uint64_t x) {
    if (x == 0ul) return 64;
    int n = 0;
    while (((x >> n) & 1ul) == 0ul && n < 64) ++n;
    return n;
}

int ctz_u256(u256 a) {
    int tz = 0;
    for (int i = 0; i < LIMBS; ++i) {
        if (a.v[i] == 0ul) {
            tz += 64;
            continue;
        }
        tz += ctz64(a.v[i]);
        break;
    }
    return tz;
}

void rshiftk_u256(inout u256 a, int k) {
    if (k <= 0) return;
    int limb = k / 64;
    int bits = k % 64;
    
    if (limb > 0) {
        for (int i = 0; i < LIMBS; ++i) {
            a.v[i] = (limb + i < LIMBS) ? a.v[limb + i] : 0ul;
        }
    }
    
    if (bits > 0) {
        uint64_t carry = 0ul;
        for (int i = LIMBS - 1; i >= 0; --i) {
            uint64_t nc = a.v[i] << (64 - bits);
            a.v[i] = (a.v[i] >> bits) | carry;
            carry = nc;
            if (i == 0) break;
        }
    }
}

void lshiftk_u256(inout u256 a, int k) {
    if (k <= 0) return;
    int limb = k / 64;
    int bits = k % 64;
    
    if (limb > 0) {
        for (int i = LIMBS - 1; i >= 0; --i) {
            a.v[i] = (i >= limb) ? a.v[i - limb] : 0ul;
        }
    }
    
    if (bits > 0) {
        uint64_t carry = 0ul;
        for (int i = 0; i < LIMBS; ++i) {
            uint64_t nc = a.v[i] >> (64 - bits);
            a.v[i] = (a.v[i] << bits) | carry;
            carry = nc;
        }
    }
}

u256 gcd_u256(u256 a, u256 b) {
    if (is_zero_u256(a)) return b;
    if (is_zero_u256(b)) return a;
    
    int shift = min(ctz_u256(a), ctz_u256(b));
    rshiftk_u256(a, ctz_u256(a));
    
    while (!is_zero_u256(b)) {
        rshiftk_u256(b, ctz_u256(b));
        if (cmp_u256(a, b) > 0) {
            u256 t = a;
            a = b;
            b = t;
        }
        u256 t;
        sub_u256(t, b, a);
        b = t;
    }
    
    lshiftk_u256(a, shift);
    return a;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= num_curves) return;
    
    CurveOut res;
    res.status = 0u;
    set_zero_u256(res.result);
    
    CurveIn input = curves[gid];
    
    // Convert to Montgomery domain
    u256 A24m, X1m;
    to_mont(A24m, input.A24, constants);
    to_mont(X1m, input.X1, constants);
    
    // Base point P = (X1m : 1)
    PointXZ P;
    P.X = X1m;
    P.Z = constants.mont_one;
    
    // Accumulate multiplications
    PointXZ R = P;
    PointXZ T;
    
    for (uint i = 0u; i < pp_count; ++i) {
        uint s = prime_powers[i];
        if (s <= 1u) continue;
        
        mont_ladder(T, R, s, A24m, constants);
        R = T;
    }
    
    // Compute result
    if (constants.compute_gcd != 0u) {
        u256 Zstd;
        from_mont(Zstd, R.Z, constants);
        u256 g = gcd_u256(Zstd, constants.N);
        res.result = g;
        
        u256 one;
        set_one_u256(one);
        if (!is_zero_u256(g) && cmp_u256(g, constants.N) < 0) {
            res.status = (cmp_u256(g, one) > 0) ? 2u : 1u;
        } else {
            res.status = 1u;
        }
    } else {
        u256 Zstd;
        from_mont(Zstd, R.Z, constants);
        res.result = Zstd;
        res.status = 0u;
    }
    
    results[gid] = res;
}
