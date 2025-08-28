// ECM Stage 1 JavaScript Implementation
// 256-bit big integers using BigInt
// Montgomery curves, x-only arithmetic
// Compatible with Web Workers for parallel processing

class ECMStage1 {
    constructor() {
        this.LIMBS = 4;
        this.LIMB_BITS = 64n;
        this.LIMB_MASK = (1n << this.LIMB_BITS) - 1n;
    }

    // 256-bit integer representation using BigInt
    createU256(v0 = 0n, v1 = 0n, v2 = 0n, v3 = 0n) {
        return [BigInt(v0), BigInt(v1), BigInt(v2), BigInt(v3)];
    }

    // Convert BigInt to u256 limbs (little-endian)
    bigIntToU256(value) {
        const v = BigInt(value);
        return [
            v & this.LIMB_MASK,
            (v >> this.LIMB_BITS) & this.LIMB_MASK,
            (v >> (2n * this.LIMB_BITS)) & this.LIMB_MASK,
            (v >> (3n * this.LIMB_BITS)) & this.LIMB_MASK
        ];
    }

    // Convert u256 limbs to BigInt
    u256ToBigInt(a) {
        return a[0] + (a[1] << this.LIMB_BITS) + (a[2] << (2n * this.LIMB_BITS)) + (a[3] << (3n * this.LIMB_BITS));
    }

    // Helper functions
    isZeroU256(a) {
        return a[0] === 0n && a[1] === 0n && a[2] === 0n && a[3] === 0n;
    }

    cmpU256(a, b) {
        for (let i = this.LIMBS - 1; i >= 0; i--) {
            if (a[i] < b[i]) return -1;
            if (a[i] > b[i]) return 1;
        }
        return 0;
    }

    setZeroU256() {
        return [0n, 0n, 0n, 0n];
    }

    setOneU256() {
        return [1n, 0n, 0n, 0n];
    }

    copyU256(a) {
        return [a[0], a[1], a[2], a[3]];
    }

    addU256(a, b) {
        let c = 0n;
        const r = [0n, 0n, 0n, 0n];
        
        for (let i = 0; i < this.LIMBS; i++) {
            const s = a[i] + b[i] + c;
            r[i] = s & this.LIMB_MASK;
            c = s >> this.LIMB_BITS;
        }
        return r;
    }

    subU256(a, b) {
        let br = 0n;
        const r = [0n, 0n, 0n, 0n];
        
        for (let i = 0; i < this.LIMBS; i++) {
            const d = a[i] - b[i] - br;
            if (d < 0n) {
                r[i] = d + (1n << this.LIMB_BITS);
                br = 1n;
            } else {
                r[i] = d;
                br = 0n;
            }
        }
        return r;
    }

    condSubU256(a, N) {
        const t = this.subU256(a, N);
        const ge = this.cmpU256(a, N) >= 0;
        return ge ? t : a;
    }

    // 64-bit multiplication with high and low parts
    mul64(a, b) {
        const prod = a * b;
        const lo = prod & this.LIMB_MASK;
        const hi = prod >> this.LIMB_BITS;
        return { lo, hi };
    }

    // Montgomery multiplication (CIOS algorithm)
    montMul(a, b, C) {
        const t = new Array(this.LIMBS + 1).fill(0n);
        
        for (let i = 0; i < this.LIMBS; i++) {
            // t += a[i] * b
            let carry = 0n;
            for (let j = 0; j < this.LIMBS; j++) {
                const { lo, hi } = this.mul64(a[i], b[j]);
                const tmp = t[j] + lo + carry;
                t[j] = tmp & this.LIMB_MASK;
                carry = hi + (tmp >> this.LIMB_BITS);
            }
            t[this.LIMBS] += carry;
            
            // m = t[0] * n0inv mod 2^64
            const m = (t[0] * C.n0inv) & this.LIMB_MASK;
            
            // t += m * N
            carry = 0n;
            for (let j = 0; j < this.LIMBS; j++) {
                const { lo, hi } = this.mul64(m, C.N[j]);
                const tmp = t[j] + lo + carry;
                t[j] = tmp & this.LIMB_MASK;
                carry = hi + (tmp >> this.LIMB_BITS);
            }
            t[this.LIMBS] += carry;
            
            // Shift right by one limb
            for (let k = 0; k < this.LIMBS; k++) {
                t[k] = t[k + 1];
            }
            t[this.LIMBS] = 0n;
        }
        
        const r = [t[0], t[1], t[2], t[3]];
        return this.condSubU256(r, C.N);
    }

    toMont(a, C) {
        return this.montMul(a, C.R2, C);
    }

    fromMont(a, C) {
        return this.montMul(a, C.montOne, C);
    }

    montAdd(a, b, C) {
        const r = this.addU256(a, b);
        return this.condSubU256(r, C.N);
    }

    montSub(a, b, C) {
        const t = this.subU256(a, b);
        if (this.cmpU256(a, b) < 0) {
            return this.addU256(t, C.N);
        }
        return t;
    }

    montSqr(a, C) {
        return this.montMul(a, a, C);
    }

    // Montgomery curve point operations
    xDBL(R, A24, C) {
        const t1 = this.montAdd(R.X, R.Z, C);
        const t2 = this.montSub(R.X, R.Z, C);
        const t1_sq = this.montSqr(t1, C);
        const t2_sq = this.montSqr(t2, C);
        const t3 = this.montSub(t1_sq, t2_sq, C);
        const X2 = this.montMul(t1_sq, t2_sq, C);
        const t4 = this.montMul(t3, A24, C);
        const t4_plus_t2 = this.montAdd(t4, t2_sq, C);
        const Z2 = this.montMul(t3, t4_plus_t2, C);
        
        return { X: X2, Z: Z2 };
    }

    xADD(P, Q, Diff, C) {
        const t1 = this.montAdd(P.X, P.Z, C);
        const t2 = this.montSub(P.X, P.Z, C);
        const t3 = this.montAdd(Q.X, Q.Z, C);
        const t4 = this.montSub(Q.X, Q.Z, C);
        const t5 = this.montMul(t1, t4, C);
        const t6 = this.montMul(t2, t3, C);
        const t1_new = this.montAdd(t5, t6, C);
        const t2_new = this.montSub(t5, t6, C);
        const t1_sq = this.montSqr(t1_new, C);
        const t2_sq = this.montSqr(t2_new, C);
        const X3 = this.montMul(t1_sq, Diff.Z, C);
        const Z3 = this.montMul(t2_sq, Diff.X, C);
        
        return { X: X3, Z: Z3 };
    }

    cswapPoint(A, B, bit) {
        const mask = BigInt(bit & 1) === 1n ? this.LIMB_MASK : 0n;
        
        for (let i = 0; i < this.LIMBS; i++) {
            const t = mask & (A.X[i] ^ B.X[i]);
            A.X[i] ^= t;
            B.X[i] ^= t;
            
            const tz = mask & (A.Z[i] ^ B.Z[i]);
            A.Z[i] ^= tz;
            B.Z[i] ^= tz;
        }
    }

    montLadder(P, k, A24, C) {
        let R0 = { X: this.copyU256(C.montOne), Z: this.setZeroU256() };
        let R1 = { X: this.copyU256(P.X), Z: this.copyU256(P.Z) };
        
        let started = false;
        for (let i = 31; i >= 0; i--) {
            const bit = (k >> i) & 1;
            if (!started && bit === 0) continue;
            started = true;
            
            this.cswapPoint(R0, R1, 1 - bit);
            const t0 = this.xADD(R0, R1, P, C);
            const t1 = this.xDBL(R1, A24, C);
            R0 = t0;
            R1 = t1;
        }
        
        return R0;
    }

    // GCD helpers
    ctz64(x) {
        if (x === 0n) return 64;
        let n = 0;
        while ((x & 1n) === 0n && n < 64) {
            x >>= 1n;
            n++;
        }
        return n;
    }

    ctzU256(a) {
        let tz = 0;
        for (let i = 0; i < this.LIMBS; i++) {
            if (a[i] === 0n) {
                tz += 64;
                continue;
            }
            tz += this.ctz64(a[i]);
            break;
        }
        return tz;
    }

    rshiftkU256(a, k) {
        if (k <= 0) return this.copyU256(a);
        
        const limbShift = Math.floor(k / 64);
        const bitShift = k % 64;
        const result = [0n, 0n, 0n, 0n];
        
        // Shift limbs
        for (let i = 0; i < this.LIMBS; i++) {
            if (limbShift + i < this.LIMBS) {
                result[i] = a[limbShift + i];
            }
        }
        
        // Shift bits
        if (bitShift > 0) {
            let carry = 0n;
            for (let i = this.LIMBS - 1; i >= 0; i--) {
                const newCarry = result[i] << (64n - BigInt(bitShift));
                result[i] = (result[i] >> BigInt(bitShift)) | carry;
                carry = newCarry & this.LIMB_MASK;
            }
        }
        
        return result;
    }

    lshiftkU256(a, k) {
        if (k <= 0) return this.copyU256(a);
        
        const limbShift = Math.floor(k / 64);
        const bitShift = k % 64;
        const result = [0n, 0n, 0n, 0n];
        
        // Shift limbs
        for (let i = this.LIMBS - 1; i >= 0; i--) {
            if (i >= limbShift) {
                result[i] = a[i - limbShift];
            }
        }
        
        // Shift bits
        if (bitShift > 0) {
            let carry = 0n;
            for (let i = 0; i < this.LIMBS; i++) {
                const newCarry = result[i] >> (64n - BigInt(bitShift));
                result[i] = ((result[i] << BigInt(bitShift)) | carry) & this.LIMB_MASK;
                carry = newCarry;
            }
        }
        
        return result;
    }

    gcdU256(a_orig, b_orig) {
        let a = this.copyU256(a_orig);
        let b = this.copyU256(b_orig);
        
        if (this.isZeroU256(a)) return b;
        if (this.isZeroU256(b)) return a;
        
        const shift = Math.min(this.ctzU256(a), this.ctzU256(b));
        a = this.rshiftkU256(a, this.ctzU256(a));
        
        while (!this.isZeroU256(b)) {
            b = this.rshiftkU256(b, this.ctzU256(b));
            if (this.cmpU256(a, b) > 0) {
                [a, b] = [b, a];
            }
            b = this.subU256(b, a);
        }
        
        return this.lshiftkU256(a, shift);
    }

    // Main ECM Stage 1 function
    ecmStage1Single(curve, constants, primePowers) {
        const result = { result: this.setZeroU256(), status: 0 };
        
        // Convert inputs to Montgomery domain
        const A24m = this.toMont(curve.A24, constants);
        const X1m = this.toMont(curve.X1, constants);
        
        // Base point P = (X1m : 1)
        let P = { X: X1m, Z: this.copyU256(constants.montOne) };
        let R = { X: this.copyU256(P.X), Z: this.copyU256(P.Z) };
        
        // Multiply by each prime power
        for (const s of primePowers) {
            if (s <= 1) continue;
            R = this.montLadder(R, s, A24m, constants);
        }
        
        // Compute result
        if (constants.computeGcd) {
            const Zstd = this.fromMont(R.Z, constants);
            const g = this.gcdU256(Zstd, constants.N);
            result.result = g;
            
            const one = this.setOneU256();
            if (!this.isZeroU256(g) && this.cmpU256(g, constants.N) < 0) {
                result.status = this.cmpU256(g, one) > 0 ? 2 : 1;
            } else {
                result.status = 1;
            }
        } else {
            const Zstd = this.fromMont(R.Z, constants);
            result.result = Zstd;
            result.status = 0;
        }
        
        return result;
    }

    // Process multiple curves (can be parallelized with Web Workers)
    ecmStage1Batch(curves, constants, primePowers, numWorkers = 1) {
        if (numWorkers === 1 || typeof Worker === 'undefined') {
            // Single-threaded execution
            return curves.map(curve => this.ecmStage1Single(curve, constants, primePowers));
        } else {
            // Multi-threaded execution with Web Workers
            return this.ecmStage1Parallel(curves, constants, primePowers, numWorkers);
        }
    }

    // Parallel execution using Web Workers
    async ecmStage1Parallel(curves, constants, primePowers, numWorkers) {
        const chunkSize = Math.ceil(curves.length / numWorkers);
        const promises = [];
        
        for (let i = 0; i < numWorkers; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, curves.length);
            if (start >= curves.length) break;
            
            const chunk = curves.slice(start, end);
            promises.push(this.processChunkInWorker(chunk, constants, primePowers));
        }
        
        const results = await Promise.all(promises);
        return results.flat();
    }

    processChunkInWorker(curves, constants, primePowers) {
        return new Promise((resolve, reject) => {
            const workerCode = `
                ${this.constructor.toString()}
                
                self.onmessage = function(e) {
                    const { curves, constants, primePowers } = e.data;
                    const ecm = new ECMStage1();
                    const results = curves.map(curve => 
                        ecm.ecmStage1Single(curve, constants, primePowers)
                    );
                    self.postMessage(results);
                };
            `;
            
            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const worker = new Worker(URL.createObjectURL(blob));
            
            worker.onmessage = (e) => {
                resolve(e.data);
                worker.terminate();
                URL.revokeObjectURL(blob);
            };
            
            worker.onerror = (error) => {
                reject(error);
                worker.terminate();
                URL.revokeObjectURL(blob);
            };
            
            worker.postMessage({ curves, constants, primePowers });
        });
    }

    // Utility functions for host preparation
    static computeMontgomeryConstants(N_bigint) {
        const ecm = new ECMStage1();
        const N = ecm.bigIntToU256(N_bigint);
        
        // Compute n0inv = -N^{-1} mod 2^64
        const n0inv = ecm.computeN0Inv(N[0]);
        
        // Compute R = 2^256 mod N and R^2 mod N
        const R = ecm.computeR(N_bigint);
        const R2 = (R * R) % N_bigint;
        
        return {
            N: N,
            R2: ecm.bigIntToU256(R2),
            montOne: ecm.bigIntToU256(R),
            n0inv: n0inv,
            computeGcd: true
        };
    }

    computeN0Inv(n0) {
        // Extended Euclidean algorithm to find modular inverse
        // Returns -N^{-1} mod 2^64
        const mod = 1n << 64n;
        let [old_r, r] = [n0, mod];
        let [old_s, s] = [1n, 0n];
        
        while (r !== 0n) {
            const quotient = old_r / r;
            [old_r, r] = [r, old_r - quotient * r];
            [old_s, s] = [s, old_s - quotient * s];
        }
        
        return (-old_s) & this.LIMB_MASK;
    }

    computeR(N_bigint) {
        // Compute 2^256 mod N
        const R = (1n << 256n) % N_bigint;
        return R;
    }

    // Generate prime powers for ECM Stage 1
    static generatePrimePowers(B1) {
        const primes = ECMStage1.generatePrimes(B1);
        const primePowers = [];
        
        for (const p of primes) {
            let power = p;
            let exp = 1;
            while (power * p <= B1) {
                power *= p;
                exp++;
            }
            primePowers.push(power);
        }
        
        return primePowers;
    }

    static generatePrimes(limit) {
        const sieve = new Array(limit + 1).fill(true);
        sieve[0] = sieve[1] = false;
        
        for (let i = 2; i * i <= limit; i++) {
            if (sieve[i]) {
                for (let j = i * i; j <= limit; j += i) {
                    sieve[j] = false;
                }
            }
        }
        
        return sieve.map((isPrime, num) => isPrime ? num : null)
                   .filter(num => num !== null);
    }
}

// Export for use in Node.js or browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ECMStage1;
} else if (typeof window !== 'undefined') {
    window.ECMStage1 = ECMStage1;
}
