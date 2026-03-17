#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BR 32
#define BC 32
#define MAX_HEAD_DIM 256

__global__ void flash_attn_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                  const float* __restrict__ V, float* __restrict__ output,
                                  int seq_len, int head_dim, float scale) {
    int h = blockIdx.x;
    int qi_block = blockIdx.y;
    int ti = threadIdx.x;
    int qi = qi_block * BR + ti;
    if (qi >= seq_len)
        return;

    const float* Qh = Q + (long)h * seq_len * head_dim;
    const float* Kh = K + (long)h * seq_len * head_dim;
    const float* Vh = V + (long)h * seq_len * head_dim;
    float* Oh = output + (long)h * seq_len * head_dim;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++)
        acc[d] = 0.0f;

    for (int kj_block = 0; (long)kj_block * BC <= qi; kj_block++) {
        int kj_start = kj_block * BC;
        int kj_end = kj_start + BC;
        if (kj_end > seq_len)
            kj_end = seq_len;
        if (kj_end > qi + 1)
            kj_end = qi + 1;
        int actual_len = kj_end - kj_start;

        float s[BC];
        for (int j = 0; j < actual_len; j++) {
            int kj = kj_start + j;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Qh[qi * head_dim + d] * Kh[kj * head_dim + d];
            }
            s[j] = dot * scale;
        }

        float m_tile = -FLT_MAX;
        for (int j = 0; j < actual_len; j++) {
            if (s[j] > m_tile)
                m_tile = s[j];
        }

        float m_new = (m > m_tile) ? m : m_tile;
        float alpha = expf(m - m_new);

        float p[BC];
        float l_tile = 0.0f;
        for (int j = 0; j < actual_len; j++) {
            p[j] = expf(s[j] - m_new);
            l_tile += p[j];
        }
        float l_new = alpha * l + l_tile;

        for (int d = 0; d < head_dim; d++) {
            float pv = 0.0f;
            for (int j = 0; j < actual_len; j++) {
                pv += p[j] * Vh[(kj_start + j) * head_dim + d];
            }
            acc[d] = (alpha * l * acc[d] + pv) / l_new;
        }

        m = m_new;
        l = l_new;
    }

    for (int d = 0; d < head_dim; d++) {
        Oh[qi * head_dim + d] = acc[d];
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int num_heads,
                      int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int num_q_blocks = (seq_len + BR - 1) / BR;
    dim3 grid(num_heads, num_q_blocks);
    dim3 block(BR);
    flash_attn_kernel<<<grid, block>>>(Q, K, V, output, seq_len, head_dim, scale);
    cudaDeviceSynchronize();
}
