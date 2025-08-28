// ecm_stage1_opencl.cl
//
// ECM Stage 1 (Montgomery curves, x-only arithmetic) on OpenCL 1.2+
// - 256-bit big integers using 4x64-bit limbs
// - Montgomery multiplication (CIOS) with precomputed n0inv and R^2 mod N
// - One work-item per curve (one curve per thread).
//
// This is a sibling of the CUDA kernel; the math mirrors the CUDA implementation.
//

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define LIMBS 4

typedef struct { ulong v[LIMBS]; } u256;

typedef struct {
    u256 N;
    u256 R2;
    u256 mont_one;
    ulong n0inv;
    int compute_gcd;
} EcmConsts;

typedef struct { u256 A24; u256 X1; } CurveIn;
typedef struct { u256 result; uchar status; } CurveOut;

// Helpers
inline int cmp_u256(u256 a, u256 b){
    for (int i=LIMBS-1;i>=0;--i){
        if (a.v[i] < b.v[i]) return -1;
        if (a.v[i] > b.v[i]) return 1;
    }
    return 0;
}
inline int is_zero(u256 a){ return (a.v[0]|a.v[1]|a.v[2]|a.v[3])==0ul; }
inline void set_zero(__private u256* r){ for(int i=0;i<LIMBS;++i) r->v[i]=0ul; }
inline void set_one(__private u256* r){ r->v[0]=1ul; r->v[1]=r->v[2]=r->v[3]=0ul; }
inline void add_u256(__private u256* r, u256 a, u256 b){
    ulong c=0;
    for(int i=0;i<LIMBS;++i){
        ulong s = a.v[i] + b.v[i];
        ulong c1 = (s < a.v[i]);
        s += c;
        ulong c2 = (s < c);
        r->v[i]=s; c = c1 + c2;
    }
}
inline void sub_u256(__private u256* r, u256 a, u256 b){
    ulong br=0;
    for(int i=0;i<LIMBS;++i){
        ulong d = a.v[i] - b.v[i];
        ulong b1 = (a.v[i] < b.v[i]);
        ulong d2 = d - br;
        ulong b2 = (d < br);
        r->v[i] = d2; br = b1 | b2;
    }
}
inline void cond_sub(__private u256* a, u256 N){
    u256 t; sub_u256(&t, *a, N);
    int ge = (cmp_u256(*a, N) >= 0);
    for(int i=0;i<LIMBS;++i){
        ulong mask = (ulong)0 - (ulong)ge;
        a->v[i] = (t.v[i] & mask) | (a->v[i] & ~mask);
    }
}

inline void mont_mul(__private u256* r, u256 a, u256 b, EcmConsts C){
    ulong t[LIMBS+1]; for(int i=0;i<=LIMBS;++i) t[i]=0ul;
    for(int i=0;i<LIMBS;++i){
        // t += a_i * b
        ulong carry=0;
        for(int j=0;j<LIMBS;++j){
            ulong lo = a.v[i]*b.v[j];
            ulong hi = mul_hi(a.v[i], b.v[j]);
            ulong tmp = t[j] + lo;
            ulong c1 = (tmp < t[j]);
            tmp += carry;
            ulong c2 = (tmp < carry);
            t[j] = tmp;
            ulong sum_hi = hi + c1 + c2;
            carry = sum_hi;
        }
        t[LIMBS] += carry;
        // m
        ulong m = t[0] * C.n0inv;
        // t += m*N
        carry=0;
        for(int j=0;j<LIMBS;++j){
            ulong lo = m*C.N.v[j];
            ulong hi = mul_hi(m, C.N.v[j]);
            ulong tmp = t[j] + lo;
            ulong c1 = (tmp < t[j]);
            tmp += carry;
            ulong c2 = (tmp < carry);
            t[j] = tmp;
            ulong sum_hi = hi + c1 + c2;
            carry = sum_hi;
        }
        t[LIMBS] += carry;
        // shift
        for(int k=0;k<LIMBS;++k) t[k]=t[k+1];
        t[LIMBS]=0ul;
    }
    for(int i=0;i<LIMBS;++i) r->v[i]=t[i];
    cond_sub(r, C.N);
}
inline void to_mont(__private u256* r, u256 a, EcmConsts C){ mont_mul(r, a, C.R2, C); }
inline void from_mont(__private u256* r, u256 a, EcmConsts C){ mont_mul(r, a, C.mont_one, C); }
inline void mont_add(__private u256* r, u256 a, u256 b, EcmConsts C){ add_u256(r,a,b); cond_sub(r, C.N); }
inline void mont_sub_u(__private u256* r, u256 a, u256 b, EcmConsts C){
    u256 t; sub_u256(&t, a, b);
    if (cmp_u256(a,b) < 0){ add_u256(r, t, C.N); } else *r = t;
}
inline void mont_sqr(__private u256* r, u256 a, EcmConsts C){ mont_mul(r,a,a,C); }

typedef struct { u256 X; u256 Z; } PointXZ;

inline void xDBL(__private PointXZ* R2, PointXZ R, u256 A24, EcmConsts C){
    u256 t1,t2,t3,t4;
    mont_add(&t1, R.X, R.Z, C);
    mont_sub_u(&t2, R.X, R.Z, C);
    mont_sqr(&t1, t1, C);
    mont_sqr(&t2, t2, C);
    mont_sub_u(&t3, t1, t2, C);
    mont_mul(&R2->X, t1, t2, C);
    mont_mul(&t4, t3, A24, C);
    mont_add(&t4, t4, t2, C);
    mont_mul(&R2->Z, t3, t4, C);
}
inline void xADD(__private PointXZ* R3, PointXZ P, PointXZ Q, PointXZ Diff, EcmConsts C){
    u256 t1,t2,t3,t4,t5,t6;
    mont_add(&t1, P.X, P.Z, C);
    mont_sub_u(&t2, P.X, P.Z, C);
    mont_add(&t3, Q.X, Q.Z, C);
    mont_sub_u(&t4, Q.X, Q.Z, C);
    mont_mul(&t5, t1, t4, C);
    mont_mul(&t6, t2, t3, C);
    mont_add(&t1, t5, t6, C);
    mont_sub_u(&t2, t5, t6, C);
    mont_sqr(&t1, t1, C);
    mont_sqr(&t2, t2, C);
    mont_mul(&R3->X, t1, Diff.Z, C);
    mont_mul(&R3->Z, t2, Diff.X, C);
}

// const-time swap
inline void cswap_point(__private PointXZ* A, __private PointXZ* B, ulong bit){
    ulong mask = (ulong)0 - (bit & 1ul);
    for(int i=0;i<LIMBS;++i){ ulong t = mask & (A->X.v[i] ^ B->X.v[i]); A->X.v[i]^=t; B->X.v[i]^=t; }
    for(int i=0;i<LIMBS;++i){ ulong t = mask & (A->Z.v[i] ^ B->Z.v[i]); A->Z.v[i]^=t; B->Z.v[i]^=t; }
}

inline void mont_ladder(__private PointXZ* R0, PointXZ P, uint k, u256 A24, EcmConsts C){
    PointXZ R1 = P;
    R0->X = C.mont_one; set_zero(&R0->Z);
    int started = 0;
    for (int i=31;i>=0;--i){
        uint bit = (k >> i) & 1u;
        if (!started && bit==0) continue;
        started = 1;
        cswap_point(R0, &R1, (ulong)(1u - bit));
        PointXZ t0, t1;
        xADD(&t0, *R0, R1, P, C);
        xDBL(&t1, R1, A24, C);
        *R0 = t0; R1 = t1;
    }
}

inline void mul_small(__private PointXZ* R, PointXZ P, uint m, u256 A24, EcmConsts C){
    mont_ladder(R, P, m, A24, C);
}

// gcd helpers
inline int ctz64(ulong x){
    if (x==0ul) return 64;
    // OpenCL 1.2: clz available; emulate ctz via bit-reverse or loop
    int n=0;
    while(((x>>n)&1ul)==0ul && n<64) ++n;
    return n;
}
inline int ctz_u256(u256 a){
    int tz=0;
    for(int i=0;i<LIMBS;++i){
        if (a.v[i]==0ul) { tz+=64; continue; }
        tz += ctz64(a.v[i]); break;
    }
    return tz;
}
inline void rshiftk(__private u256* a, int k){
    if (k<=0) return;
    int limb = k/64, bits=k%64;
    if (limb){
        for(int i=0;i<LIMBS;++i) a->v[i] = (limb+i<LIMBS)? a->v[limb+i] : 0ul;
    }
    if (bits){
        ulong carry=0;
        for(int i=LIMBS-1;i>=0;--i){
            ulong nc = a->v[i] << (64-bits);
            a->v[i] = (a->v[i] >> bits) | carry;
            carry = nc;
            if (i==0) break;
        }
    }
}
inline void lshiftk(__private u256* a, int k){
    if (k<=0) return;
    int limb = k/64, bits=k%64;
    if (limb){
        for(int i=LIMBS-1;i>=0;--i) a->v[i] = (i>=limb)? a->v[i-limb] : 0ul;
    }
    if (bits){
        ulong carry=0;
        for(int i=0;i<LIMBS;++i){
            ulong nc = a->v[i] >> (64-bits);
            a->v[i] = (a->v[i] << bits) | carry;
            carry = nc;
        }
    }
}
inline u256 gcd_u256(u256 a, u256 b){
    if (is_zero(a)) return b;
    if (is_zero(b)) return a;
    int shift = min(ctz_u256(a), ctz_u256(b));
    rshiftk(&a, ctz_u256(a));
    do {
        rshiftk(&b, ctz_u256(b));
        if (cmp_u256(a,b) > 0){ u256 t=a; a=b; b=t; }
        u256 t; sub_u256(&t, b, a); b=t;
    } while(!is_zero(b));
    lshiftk(&a, shift);
    return a;
}

__kernel void ecm_stage1_kernel(
    EcmConsts C,
    __global const CurveIn* curves,
    __global const uint* prime_powers,
    int pp_count,
    __global CurveOut* out,
    int num_curves
){
    int gid = get_global_id(0);
    if (gid >= num_curves) return;

    CurveOut res; res.status=0; set_zero(&res.result);

    CurveIn in = curves[gid];

    u256 A24m, X1m;
    to_mont(&A24m, in.A24, C);
    to_mont(&X1m, in.X1, C);

    PointXZ P; P.X = X1m; P.Z = C.mont_one;
    PointXZ R = P, T;

    for (int i=0;i<pp_count;++i){
        uint s = prime_powers[i];
        if (s<=1u) continue;
        mul_small(&T, R, s, A24m, C);
        R = T;
    }

    if (C.compute_gcd){
        u256 Zstd; from_mont(&Zstd, R.Z, C);
        u256 g = gcd_u256(Zstd, C.N);
        res.result = g;
        u256 one; set_one(&one);
        if (!is_zero(g) && cmp_u256(g, C.N) < 0){
            res.status = (cmp_u256(g, one) > 0) ? 2 : 1;
        } else res.status = 1;
    } else {
        u256 Zstd; from_mont(&Zstd, R.Z, C);
        res.result = Zstd; res.status=0;
    }

    out[gid] = res;
}

// Notes: same as CUDA file.
