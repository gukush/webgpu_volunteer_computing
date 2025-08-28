// ecm_stage1_webgl.glsl
// WebGL2 (GLSL ES 3.00) fragment shader: ECM Stage 1 per-fragment (one curve per pixel).
// Uses integer textures (RGBA32UI).
// Input textures (all usampler2D):
//  - texA24:   two rows (row 0 => limbs[0..3], row 1 => limbs[4..7]), width=num_curves
//  - texX1:    same layout as texA24
//  - texPows:  1 row, width=pp_count, each texel.x holds prime power s
// Uniforms:
//  - constN[2], constR2[2], constMont1[2]: each is uvec4[2] covering 8 limbs
//  - n0inv32: uint
//  - pp_count, num_curves, compute_gcd
//
// Outputs (MRT):
//  - layout(location=0) out uvec4 out_lo  -> result.limbs[0..3]
//  - layout(location=1) out uvec4 out_hi  -> result.limbs[4..7]
//  - layout(location=2) out uvec4 out_meta-> status in .x
#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp uimage2D;
out uvec4 out_lo;
layout(location=1) out uvec4 out_hi;
layout(location=2) out uvec4 out_meta;

uniform usampler2D texA24;
uniform usampler2D texX1;
uniform usampler2D texPows;

uniform uvec4 constN[2];
uniform uvec4 constR2[2];
uniform uvec4 constMont1[2];
uniform uint n0inv32;
uniform int  pp_count;
uniform int  num_curves;
uniform int  compute_gcd;

int idx() { return int(floor(gl_FragCoord.x)) ; }

uvec4 load_limbs(usampler2D tex, int x, int row){
  return texelFetch(tex, ivec2(x, row), 0);
}
void store_u256(out uvec4 lo, out uvec4 hi, uvec4 L0, uvec4 L1){
  lo = L0; hi = L1;
}

struct U256 { uint l[8]; };

U256 pack(uvec4 a, uvec4 b){
  U256 r;
  r.l[0]=a.x; r.l[1]=a.y; r.l[2]=a.z; r.l[3]=a.w;
  r.l[4]=b.x; r.l[5]=b.y; r.l[6]=b.z; r.l[7]=b.w;
  return r;
}
void unpack(U256 a, out uvec4 lo, out uvec4 hi){
  lo = uvec4(a.l[0],a.l[1],a.l[2],a.l[3]);
  hi = uvec4(a.l[4],a.l[5],a.l[6],a.l[7]);
}

uvec2 mul32x32_64(uint a, uint b){
  uint a0 = a & 0xFFFFu, a1 = a >> 16;
  uint b0 = b & 0xFFFFu, b1 = b >> 16;
  uint p00 = a0*b0;
  uint p01 = a0*b1;
  uint p10 = a1*b0;
  uint p11 = a1*b1;
  uint mid = p01 + p10;
  uint lo  = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16);
  uint carry = (p00 >> 16) + (mid >> 16);
  uint hi = p11 + carry;
  return uvec2(lo, hi);
}

uvec2 addc(uint a, uint b, uint cin){
  uint s = a + b;
  uint c1 = uint(s < a);
  uint s2 = s + cin;
  uint c2 = uint(s2 < cin);
  return uvec2(s2, c1 + c2);
}
uvec2 subb(uint a, uint b, uint bin){
  uint d = a - b;
  uint b1 = uint(a < b);
  uint d2 = d - bin;
  uint b2 = uint(d < bin);
  return uvec2(d2, uint((b1|b2)!=0u));
}

bool is_zero(U256 a){ return (a.l[0]|a.l[1]|a.l[2]|a.l[3]|a.l[4]|a.l[5]|a.l[6]|a.l[7])==0u; }
int cmp(U256 a, U256 b){
  for (int i=7;i>=0;--i){
    if (a.l[i] < b.l[i]) return -1;
    if (a.l[i] > b.l[i]) return 1;
  }
  return 0;
}
U256 add_u256(U256 a, U256 b){
  U256 r; uint c=0u;
  for (int i=0;i<8;i++){ uvec2 ac = addc(a.l[i], b.l[i], c); r.l[i]=ac.x; c=ac.y; }
  return r;
}
U256 sub_u256(U256 a, U256 b){
  U256 r; uint br=0u;
  for (int i=0;i<8;i++){ uvec2 sb = subb(a.l[i], b.l[i], br); r.l[i]=sb.x; br=sb.y; }
  return r;
}
U256 cond_sub_N(U256 a, U256 N){
  return (cmp(a,N) >= 0) ? sub_u256(a,N) : a;
}
U256 load_const(uvec4 c[2]){ return pack(c[0], c[1]); }

U256 mont_mul(U256 a, U256 b, U256 N, uint n0inv){
  uint t[9]; for (int i=0;i<9;i++) t[i]=0u;
  for (int i=0;i<8;i++){
    uint carry=0u;
    for (int j=0;j<8;j++){
      uvec2 prod = mul32x32_64(a.l[i], b.l[j]);
      uvec2 s1 = addc(t[j], prod.x, 0u);
      uvec2 s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    t[8] += carry;

    uint m = t[0] * n0inv;
    carry = 0u;
    for (int j=0;j<8;j++){
      uvec2 prod = mul32x32_64(m, N.l[j]);
      uvec2 s1 = addc(t[j], prod.x, 0u);
      uvec2 s2 = addc(s1.x, carry, 0u);
      t[j] = s2.x;
      carry = prod.y + s1.y + s2.y;
    }
    t[8] += carry;
    for (int k=0;k<8;k++) t[k]=t[k+1];
    t[8]=0u;
  }
  U256 r; for (int i=0;i<8;i++) r.l[i]=t[i];
  return cond_sub_N(r, N);
}
U256 to_mont(U256 a, U256 R2, U256 N, uint n0){ return mont_mul(a,R2,N,n0); }
U256 from_mont(U256 a, U256 one, U256 N, uint n0){ return mont_mul(a,one,N,n0); }
U256 mont_add(U256 a, U256 b, U256 N){ return cond_sub_N(add_u256(a,b), N); }
U256 mont_sub_u(U256 a, U256 b, U256 N){ return (cmp(a,b) >= 0) ? sub_u256(a,b) : add_u256(sub_u256(a,b), N); }
U256 mont_sqr(U256 a, U256 N, uint n0){ return mont_mul(a,a,N,n0); }

struct PointXZ { U256 X; U256 Z; };
PointXZ xDBL(PointXZ R, U256 A24, U256 N, uint n0){
  U256 t1 = mont_add(R.X, R.Z, N);
  U256 t2 = mont_sub_u(R.X, R.Z, N);
  t1 = mont_sqr(t1, N, n0);
  t2 = mont_sqr(t2, N, n0);
  U256 t3 = mont_sub_u(t1, t2, N);
  U256 X2 = mont_mul(t1, t2, N, n0);
  U256 t4 = mont_mul(t3, A24, N, n0);
  t4 = mont_add(t4, t2, N);
  U256 Z2 = mont_mul(t3, t4, N, n0);
  PointXZ P; P.X=X2; P.Z=Z2; return P;
}
PointXZ xADD(PointXZ P, PointXZ Q, PointXZ Diff, U256 N, uint n0){
  U256 t1 = mont_add(P.X, P.Z, N);
  U256 t2 = mont_sub_u(P.X, P.Z, N);
  U256 t3 = mont_add(Q.X, Q.Z, N);
  U256 t4 = mont_sub_u(Q.X, Q.Z, N);
  U256 t5 = mont_mul(t1, t4, N, n0);
  U256 t6 = mont_mul(t2, t3, N, n0);
  t1 = mont_add(t5, t6, N);
  t2 = mont_sub_u(t5, t6, N);
  t1 = mont_sqr(t1, N, n0);
  t2 = mont_sqr(t2, N, n0);
  PointXZ R;
  R.X = mont_mul(t1, Diff.Z, N, n0);
  R.Z = mont_mul(t2, Diff.X, N, n0);
  return R;
}
void cswap(inout PointXZ A, inout PointXZ B, uint bit){
  uint mask = uint(0u - (bit & 1u));
  for (int i=0;i<8;i++){
    uint tx = (A.X.l[i] ^ B.X.l[i]) & mask; A.X.l[i]^=tx; B.X.l[i]^=tx;
    uint tz = (A.Z.l[i] ^ B.Z.l[i]) & mask; A.Z.l[i]^=tz; B.Z.l[i]^=tz;
  }
}
PointXZ ladder(PointXZ P, uint k, U256 A24, U256 N, uint n0){
  U256 one; for (int i=0;i<8;i++) one.l[i]=0u; one.l[0]=1u;
  PointXZ R0; R0.X = one; for(int i=0;i<8;i++) R0.Z.l[i]=0u;
  PointXZ R1 = P;
  bool started=false;
  for (int i=31;i>=0;--i){
    uint bit = (k >> uint(i)) & 1u;
    if (!started && bit==0u) continue;
    started=true;
    cswap(R0, R1, 1u - bit);
    PointXZ T0 = xADD(R0, R1, P, N, n0);
    PointXZ T1 = xDBL(R1, A24, N, n0);
    R0 = T0; R1 = T1;
  }
  return R0;
}

// simple binary gcd (like WGSL version) – helpers:
uint ctz32(uint x){ if (x==0u) return 32u; uint n=0u; uint y=x; while((y & 1u)==0u){ y>>=1u; n++; } return n; }
uint ctz_u256(U256 a){
  uint tz=0u;
  for (int i=0;i<8;i++){ if (a.l[i]==0u) tz+=32u; else { tz+=ctz32(a.l[i]); break; } }
  return tz;
}
U256 rshiftk(U256 a, uint k){
  if (k==0u) return a;
  U256 r; for(int i=0;i<8;i++) r.l[i]=0u;
  uint limb = k/32u, bits=k%32u;
  for (int i=0;i<8;i++){
    uint val=0u;
    if (i+int(limb) < 8){
      val = a.l[i+int(limb)];
      if (bits!=0u){
        uint hi = (i+int(limb)+1<8)? a.l[i+int(limb)+1] : 0u;
        val = (val >> bits) | (hi << (32u-bits));
      }
    }
    r.l[i]=val;
  }
  return r;
}
U256 lshiftk(U256 a, uint k){
  if (k==0u) return a;
  U256 r; for(int i=0;i<8;i++) r.l[i]=0u;
  uint limb=k/32u, bits=k%32u;
  for (int i=7;i>=0;--i){
    uint val=0u;
    if (i>=int(limb)){
      val = a.l[i-int(limb)];
      if (bits!=0u){
        uint lo = (i-1>=int(limb))? a.l[i-1-int(limb)] : 0u;
        val = (val << bits) | (lo >> (32u-bits));
      }
    }
    r.l[i]=val;
  }
  return r;
}
U256 gcd_u256(U256 a, U256 b){
  if (is_zero(a)) return b;
  if (is_zero(b)) return a;
  uint shift = min(ctz_u256(a), ctz_u256(b));
  a = rshiftk(a, ctz_u256(a));
  for(;;){
    b = rshiftk(b, ctz_u256(b));
    if (cmp(a,b) > 0){ U256 t=a; a=b; b=t; }
    b = sub_u256(b,a);
    if (is_zero(b)) break;
  }
  return lshiftk(a, shift);
}

void main(){
  int x = idx();
  if (x >= num_curves) {
    out_lo = uvec4(0); out_hi=uvec4(0); out_meta=uvec4(0); return;
  }

  U256 N = load_const(constN);
  U256 R2 = load_const(constR2);
  U256 mont1 = load_const(constMont1);

  // Load per-curve inputs
  U256 A24 = pack(load_limbs(texA24, x, 0), load_limbs(texA24, x, 1));
  U256 X1  = pack(load_limbs(texX1,  x, 0), load_limbs(texX1,  x, 1));

  U256 A24m = mont_mul(A24, R2, N, n0inv32);
  U256 X1m  = mont_mul(X1,  R2, N, n0inv32);

  PointXZ P; P.X=X1m; P.Z=mont1;
  PointXZ R = P;

  for (int i=0;i<pp_count;i++){
    uint s = texelFetch(texPows, ivec2(i,0), 0).x;
    if (s<=1u) continue;
    PointXZ T = ladder(R, s, A24m, N, n0inv32);
    R = T;
  }

  U256 result; for (int i=0;i<8;i++) result.l[i]=0u;
  uint status=0u;
  if (compute_gcd == 1){
    U256 Zstd = mont_mul(R.Z, mont1, N, n0inv32); // from_mont
    U256 g = gcd_u256(Zstd, N);
    result = g;
    U256 one; for(int i=0;i<8;i++) one.l[i]=0u; one.l[0]=1u;
    if (!is_zero(g) && (cmp(g, N) < 0)){
      status = (cmp(g, one) > 0) ? 2u : 1u;
    } else status = 1u;
  } else {
    result = mont_mul(R.Z, mont1, N, n0inv32);
    status = 0u;
  }

  uvec4 lo, hi; unpack(result, lo, hi);
  out_lo = lo;
  out_hi = hi;
  out_meta = uvec4(status,0u,0u,0u);
}
