function blockMatrixMultiply(blockA, blockB, blockSize) {
      const result = new Float32Array(blockSize * blockSize);

      // Standard matrix multiplication algorithm
      for (let i = 0; i < blockSize; i++) {
        for (let j = 0; j < blockSize; j++) {
          let sum = 0;
          for (let k = 0; k < blockSize; k++) {
            const aVal = blockA[i * blockSize + k];
            const bVal = blockB[k * blockSize + j];
            sum += aVal * bVal;
          }
          result[i * blockSize + j] = sum;
        }
      }

      return result;
    }
