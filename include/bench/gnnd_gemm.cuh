// CUDA kernels for the GNND-style batched tiny GEMM (one block per point).
//   regular : fp32, 256 threads (16x16), 4x4 register tile, BK-tiled scattered gather.
//   tensor  : fp16-in/fp32-out via nvcuda::wmma 16x16x16, 4 warps in a 2x2 grid,
//             each warp a 32x32 sub-tile (2x2 accumulator fragments).
// NEW is staged transposed in shared so the row-major matrix_b fragment reads NEW^T,
// giving C = OLD . NEW^T (each entry a dot product). Included by gnnd_gemm.hpp.
#pragma once
#include <mma.h>
#include "bench/backend.hpp"
#include "bench/gpu_types.hpp"

namespace bench {
using namespace nvcuda;

// ---- deterministic data generation on device ----
__global__ void gnnd_fill_f32(float* d, long long n){
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = gnnd_dataval(i);
}
__global__ void gnnd_to_f16(const float* in, half* out, long long n){
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

// ---- regular core (fp32) ----
__global__ void gnnd_regular_kernel(const float* __restrict__ data, float* __restrict__ out,
                                    long long npts, int dim){
  constexpr int L = GNND_LIST, BK = GNND_BK, PAD = 1;
  const long long p = blockIdx.x; if (p >= npts) return;
  const int tid = threadIdx.x, tr = tid >> 4, tc = tid & 15;     // 16x16 thread grid
  __shared__ long long oi[L], ni[L];
  for (int s = tid; s < L; s += blockDim.x){
    oi[s] = gnnd_neighbor(p, s, GNND_SALT_OLD, npts);
    ni[s] = gnnd_neighbor(p, s, GNND_SALT_NEW, npts);
  }
  __syncthreads();
  __shared__ float As[L][BK + PAD], Bs[L][BK + PAD];
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  for (int kc = 0; kc < dim; kc += BK){
    for (int e = tid; e < L * BK; e += blockDim.x){
      int r = e / BK, c = e % BK, k = kc + c;
      As[r][c] = (k < dim) ? data[oi[r]*dim + k] : 0.f;
      Bs[r][c] = (k < dim) ? data[ni[r]*dim + k] : 0.f;
    }
    __syncthreads();
    int km = (dim - kc < BK) ? (dim - kc) : BK;
    for (int k = 0; k < km; ++k){
      float a[4], b[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[tr*4 + i][k];
      #pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[tc*4 + j][k];
      #pragma unroll
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  __shared__ float C[L][L + 1];         // C[oldRow][newCol]
  #pragma unroll
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) C[tr*4 + i][tc*4 + j] = acc[i][j];
  __syncthreads();
  for (int r = tid; r < L; r += blockDim.x){          // per-OLD min (over NEW cols)
    float m = C[r][0]; for (int c = 1; c < L; ++c) m = fminf(m, C[r][c]);
    out[p*(2*L) + r] = m;
  }
  for (int c = tid; c < L; c += blockDim.x){          // per-NEW min (over OLD rows)
    float m = C[0][c]; for (int r = 1; r < L; ++r) m = fminf(m, C[r][c]);
    out[p*(2*L) + L + c] = m;
  }
}

// ---- tensor core (fp16-in/fp32-out, WMMA 16x16x16) ----
__global__ void gnnd_tensor_kernel(const half* __restrict__ data, float* __restrict__ out,
                                   long long npts, int dim){
  constexpr int L = GNND_LIST, BK = GNND_BK, PAD = 8;
  const long long p = blockIdx.x; if (p >= npts) return;
  const int tid = threadIdx.x, warp = tid >> 5, wr = warp >> 1, wc = warp & 1;  // 4 warps, 2x2
  __shared__ long long oi[L], ni[L];
  for (int s = tid; s < L; s += blockDim.x){
    oi[s] = gnnd_neighbor(p, s, GNND_SALT_OLD, npts);
    ni[s] = gnnd_neighbor(p, s, GNND_SALT_NEW, npts);
  }
  __syncthreads();
  __shared__ half  As[L][BK + PAD];     // OLD[row][k]
  __shared__ half  Bs[BK][L + PAD];     // NEW transposed: Bs[k][row] = NEW[row][k]
  __shared__ float C[L][L + 4];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  for (int kc = 0; kc < dim; kc += BK){
    for (int e = tid; e < L * BK; e += blockDim.x){
      int r = e / BK, c = e % BK, k = kc + c;
      As[r][c] = (k < dim) ? data[oi[r]*dim + k] : __float2half(0.f);
    }
    for (int e = tid; e < BK * L; e += blockDim.x){
      int c = e / L, r = e % L, k = kc + c;
      Bs[c][r] = (k < dim) ? data[ni[r]*dim + k] : __float2half(0.f);
    }
    __syncthreads();
    #pragma unroll
    for (int ks = 0; ks < BK; ks += 16){
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
      #pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(af[i], &As[wr*32 + i*16][ks], BK + PAD);
      #pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], &Bs[ks][wc*32 + j*16], L + PAD);
      #pragma unroll
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
    wmma::store_matrix_sync(&C[wr*32 + i*16][wc*32 + j*16], acc[i][j], L + 4, wmma::mem_row_major);
  __syncthreads();
  for (int r = tid; r < L; r += blockDim.x){
    float m = C[r][0]; for (int c = 1; c < L; ++c) m = fminf(m, C[r][c]);
    out[p*(2*L) + r] = m;
  }
  for (int c = tid; c < L; c += blockDim.x){
    float m = C[0][c]; for (int r = 1; r < L; ++r) m = fminf(m, C[r][c]);
    out[p*(2*L) + L + c] = m;
  }
}

inline void launch_gnnd_fill(float* d, long long n){
  gnnd_fill_f32<<<(unsigned)((n + 255) / 256), 256>>>(d, n); check_launch();
}
inline void launch_gnnd_to_f16(const float* in, half_t* out, long long n){
  gnnd_to_f16<<<(unsigned)((n + 255) / 256), 256>>>(in, reinterpret_cast<half*>(out), n); check_launch();
}
inline void launch_gnnd_gemm(bool tensor, const float* dF32, const half_t* dF16, float* out,
                             long long npts, int dim, int list){
  (void)list;                                        // fixed at GNND_LIST
  dim3 grid((unsigned)npts);
  if (tensor) gnnd_tensor_kernel<<<grid, 128>>>(reinterpret_cast<const half*>(dF16), out, npts, dim);
  else        gnnd_regular_kernel<<<grid, 256>>>(dF32, out, npts, dim);
  check_launch();
}

}  // namespace bench
