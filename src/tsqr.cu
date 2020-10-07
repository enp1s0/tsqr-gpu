#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>
#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/error.hpp>
#include <cutf/thread.hpp>
#include "tsqr.hpp"
#include "tcqr.hpp"
#include "utils.hpp"
#include "validation.hpp"
#include "matrix_copy.cuh"
#include "matrix_operations.cuh"
#include "gemm_core/gemm_core.cuh"
#include "matmul.hpp"
#include "experimental.hpp"

//#define DEBUG
//#define DEBUG_INPUT_MATRIX_PRINT
//#define DEBUG_Q_MATRIX_PRINT
//#define MEASURE_QR_TIME
//#define EVALUATE_EACH_SMALL_Q
//#define EVALUATE_EXPONENT_DISTRIBUTION

// Defining `EMULATE_TF32` enables `FP32-noTC` to emulate NVIDIA A100 TF32 TensorCore
//#define EMULATE_TF32

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ENABLE_TF32
#endif


namespace mtk {
namespace tsqr {
std::size_t get_batch_size_log2(const std::size_t m) {
	return (std::max(5u, static_cast<unsigned>( std::ceil( std::log2(static_cast<float>(m))))) - 5u);
}
std::size_t get_batch_size(const std::size_t m) {
	return 1lu << get_batch_size_log2(m);
}

// Calculating necessary working memory size
std::size_t get_working_q_size(const std::size_t m, const std::size_t n) {
	const auto batch_size = get_batch_size(m);
	const auto working_q_size = n * m + 2 * n * n * (batch_size - 1);

	return working_q_size;
}

std::size_t get_working_r_size(const std::size_t m, const std::size_t n) {
	const auto batch_size = get_batch_size(m);
	const auto working_r_size_0 = n * n * batch_size;
	const auto working_r_size_1 = n * n * batch_size / 2;

	return working_r_size_0 + working_r_size_1;
}

} // namespace tsqr
} // namespace mtk
namespace{
constexpr unsigned warp_size = 32;
template <class Func>
void debug_func(Func func) {
#ifdef DEBUG
	func();
#endif
}

template <mtk::tsqr::compute_mode>
constexpr mtk::tcqr::compute_mode get_tcqr_compute_mode();
#define TSQR_GET_TCQR_COMPUTE_MODE(mode) template<> constexpr mtk::tcqr::compute_mode get_tcqr_compute_mode<mtk::tsqr::compute_mode::mode>() {return mtk::tcqr::compute_mode::mode;}
TSQR_GET_TCQR_COMPUTE_MODE(fp16_notc        );
TSQR_GET_TCQR_COMPUTE_MODE(fp32_notc        );
TSQR_GET_TCQR_COMPUTE_MODE(fp16_tc_nocor    );
TSQR_GET_TCQR_COMPUTE_MODE(fp32_tc_nocor    );
TSQR_GET_TCQR_COMPUTE_MODE(fp32_tc_cor      );
TSQR_GET_TCQR_COMPUTE_MODE(tf32_tc_nocor    );
TSQR_GET_TCQR_COMPUTE_MODE(tf32_tc_cor      );
TSQR_GET_TCQR_COMPUTE_MODE(tf32_tc_nocor_emu);
TSQR_GET_TCQR_COMPUTE_MODE(tf32_tc_cor_emu  );
TSQR_GET_TCQR_COMPUTE_MODE(mixed_tc_cor_emu );

template <mtk::tsqr::compute_mode>
constexpr mtk::matmul::compute_mode get_matmul_compute_mode();
#define TSQR_GET_MATMUL_COMPUTE_MODE(mode) template<> constexpr mtk::matmul::compute_mode get_matmul_compute_mode<mtk::tsqr::compute_mode::mode>() {return mtk::matmul::compute_mode::mode;}
TSQR_GET_MATMUL_COMPUTE_MODE(fp16_notc        );
TSQR_GET_MATMUL_COMPUTE_MODE(fp32_notc        );
TSQR_GET_MATMUL_COMPUTE_MODE(tf32_tc_cor_emu  );
TSQR_GET_MATMUL_COMPUTE_MODE(tf32_tc_nocor_emu);
TSQR_GET_MATMUL_COMPUTE_MODE(mixed_tc_cor_emu );

#ifdef EVALUATE_EXPONENT_DISTRIBUTION
template <mtk::tsqr::compute_mode>
std::string get_tsqr_compute_mode_string();
#define TSQR_GET_TSQR_COMPUTE_MODE_STRING(mode) template<> std::string get_tsqr_compute_mode_string<mtk::tsqr::compute_mode::mode>() {return #mode;}
TSQR_GET_TSQR_COMPUTE_MODE_STRING(fp16_notc        );
TSQR_GET_TSQR_COMPUTE_MODE_STRING(fp32_notc        );
TSQR_GET_TSQR_COMPUTE_MODE_STRING(fp16_tc_nocor    );
TSQR_GET_TSQR_COMPUTE_MODE_STRING(fp32_tc_nocor    );
TSQR_GET_TSQR_COMPUTE_MODE_STRING(fp32_tc_cor      );
TSQR_GET_TSQR_COMPUTE_MODE_STRING(tf32_tc_nocor_emu);
TSQR_GET_TSQR_COMPUTE_MODE_STRING(tf32_tc_cor_emu  );
TSQR_GET_TSQR_COMPUTE_MODE_STRING(mixed_tc_cor_emu );
#endif

template <class DST_T, class SRC_T>
__device__ void copy_32x16(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr auto stride = warp_size;
	const auto y = unique_id & 0x1f;
	for(unsigned i = 0; i < (FRAGMENT_DIM_M * FRAGMENT_DIM_N); i += stride) {
		dst_ptr[i + y] = cutf::type::cast<DST_T>(src_ptr[i + y]);
	}
	__syncthreads();
}

template <class DST_T, class SRC_T>
__device__ void copy_16x16(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 16;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr auto stride = warp_size;
	const auto y = unique_id & 0x1f;
	for(unsigned i = 0; i < (FRAGMENT_DIM_M * FRAGMENT_DIM_N); i += stride) {
		dst_ptr[i + y] = cutf::type::cast<DST_T>(src_ptr[i + y]);
	}
	__syncthreads();
}

// backward except last one layer
template <mtk::tsqr::compute_mode mode, class T>
__global__ void tsqr_backward(
		T* const ac_ptr,
		const T* const b_ptr,
		const unsigned n,
		const std::size_t k
		) {
	constexpr unsigned FRAGMENT_DIM_M = 32;
	constexpr unsigned FRAGMENT_DIM_N = 16;
	constexpr unsigned max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = (1lu << (k)) * 2 * n;

	if(matrix_id >= (1lu << k)) return;

	__shared__ T shared_ac_in[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ T shared_ac_out[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ T shared_b[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	auto shared_ac_in_ptr = shared_ac_in + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	auto shared_ac_out_ptr = shared_ac_out + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	auto shared_b_ptr = shared_b + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// Copy AC(in)
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_in_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	// Copy B
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);

	__syncthreads();

	mtk::matmul::matmul_core_m16n16k16<get_matmul_compute_mode<mode>(), T>(
			shared_ac_out_ptr, FRAGMENT_DIM_M,
			shared_ac_in_ptr, FRAGMENT_DIM_M,
			shared_b_ptr, FRAGMENT_DIM_N,
			tid & 0x1f
			);

	mtk::matmul::matmul_core_m16n16k16<get_matmul_compute_mode<mode>(), T>(
			shared_ac_out_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			shared_ac_in_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			shared_b_ptr, FRAGMENT_DIM_N,
			tid & 0x1f
			);

	__syncthreads();

	mtk::matrix_copy::s2g32x16_1w(
			ac_ptr, matrix_id * 2 * n, ac_m,
			shared_ac_out_ptr, 2 * n, n,
			tid
			);
}

template <>
__global__ void tsqr_backward<mtk::tsqr::compute_mode::fp32_tc_nocor, half>(
		half* const ac_ptr,
		const half* const b_ptr,
		const unsigned n,
		const std::size_t k
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = (1lu << (k)) * 2 * n;

	if(matrix_id >= (1lu << k)) return;

	__shared__ half shared_ac_f16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_ac_f32[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_b_f16[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_fp16_ptr = shared_ac_f16 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_fp32_ptr = shared_ac_f32 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp16_ptr = shared_b_f16 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// Copy A
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp16_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	// Copy B
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp16_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c0, frag_c1;

	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

	nvcuda::wmma::load_matrix_sync(frag_a0, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);

	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			ac_ptr, matrix_id * 2 * n, ac_m,
			shared_ac_fp32_ptr, 2 * n, n,
			tid
			);
}

template <>
__global__ void tsqr_backward<mtk::tsqr::compute_mode::fp16_tc_nocor, half>(
		half* const ac_ptr,
		const half* const b_ptr,
		const unsigned n,
		const std::size_t k
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = (1lu << (k)) * 2 * n;

	if(matrix_id >= (1lu << k)) return;

	__shared__ half shared_ac_f16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_ac_f32[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_b_f16[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_fp16_ptr = shared_ac_f16 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_fp32_ptr = shared_ac_f32 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp16_ptr = shared_b_f16 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// Copy AC
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp16_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	// Copy B
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp16_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c0, frag_c1;

	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

	nvcuda::wmma::load_matrix_sync(frag_a0, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);

	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			ac_ptr, matrix_id * 2 * n, ac_m,
			shared_ac_fp32_ptr, 2 * n, n,
			tid
			);
}

template <>
__global__ void tsqr_backward<mtk::tsqr::compute_mode::fp32_tc_cor, float>(
		float* const ac_ptr,
		const float* const b_ptr,
		const unsigned n,
		const std::size_t k
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	constexpr float correction_rescale = 1024.0f;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = (1lu << (k)) * 2 * n;

	if(matrix_id >= (1lu << k)) return;

	__shared__ half shared_ac_f16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_ac_f32[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_b_f16[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_b_f32[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_fp16_ptr = shared_ac_f16 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_fp32_ptr = shared_ac_f32 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp16_ptr = shared_b_f16 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp32_ptr = shared_b_f32 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp32_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	copy_32x16(shared_ac_fp16_ptr, shared_ac_fp32_ptr, tid);
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp32_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);
	copy_16x16(shared_b_fp16_ptr, shared_b_fp32_ptr, tid);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0_diff, frag_a1_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c0, frag_c1;

	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

	nvcuda::wmma::load_matrix_sync(frag_a0, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	mtk::matrix_operation::diff32x16_1w(shared_ac_fp16_ptr, shared_ac_fp32_ptr, shared_ac_fp16_ptr, correction_rescale, tid);
	mtk::matrix_operation::diff16x16_1w(shared_b_fp16_ptr, shared_b_fp32_ptr, shared_b_fp16_ptr, correction_rescale, tid);

	nvcuda::wmma::load_matrix_sync(frag_a0_diff, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1_diff, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b_diff, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	nvcuda::wmma::mma_sync(frag_c0, frag_a0_diff, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1_diff, frag_b, frag_c1);
	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b_diff, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b_diff, frag_c1);

	for (unsigned i = 0; i < frag_c0.num_elements; i++) {
		frag_c0.x[i] *= 1.0f / correction_rescale;
		frag_c1.x[i] *= 1.0f / correction_rescale;
	}

	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);

	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			ac_ptr, matrix_id * 2 * n, ac_m,
			shared_ac_fp32_ptr, 2 * n, n,
			tid
			);
}

template <>
__global__ void tsqr_backward<mtk::tsqr::compute_mode::tf32_tc_nocor, float>(
		float* const ac_ptr,
		const float* const b_ptr,
		const unsigned n,
		const std::size_t k
		) {
#ifdef ENABLE_TF32
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t FRAGMENT_DIM_K = 8;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = (1lu << (k)) * 2 * n;

	if(matrix_id >= (1lu << k)) return;

	__shared__ float shared_ac[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_b[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_ptr = shared_ac + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_ptr = shared_b + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> frag_c0, frag_c1;

	// TODO: Move into the loop and execute when k = 0.
	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

#pragma unroll
	for (unsigned k = 0; k < FRAGMENT_DIM_N / FRAGMENT_DIM_K; k++) {
		const auto ac_ptr = shared_ac_ptr + k * FRAGMENT_DIM_M * FRAGMENT_DIM_K;
		const auto b_ptr = shared_b_ptr + k * FRAGMENT_DIM_K;

		nvcuda::wmma::load_matrix_sync(frag_a0, ac_ptr, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_a1, ac_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_b, b_ptr, FRAGMENT_DIM_N);

		nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
		nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);
	}

	nvcuda::wmma::store_matrix_sync(shared_ac_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			ac_ptr, matrix_id * 2 * n, ac_m,
			shared_ac_ptr, 2 * n, n,
			tid
			);
#endif
}

template <>
__global__ void tsqr_backward<mtk::tsqr::compute_mode::tf32_tc_cor, float>(
		float* const ac_ptr,
		const float* const b_ptr,
		const unsigned n,
		const std::size_t k
		) {
#ifdef ENABLE_TF32
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t FRAGMENT_DIM_K = 8;
	constexpr std::size_t max_batch_size_per_block = 4;
	constexpr float correction_rescale = 1024.0f;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = (1lu << (k)) * 2 * n;

	if(matrix_id >= (1lu << k)) return;

	__shared__ float shared_ac[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_b[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_ptr = shared_ac + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_ptr = shared_b + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_a0_diff, frag_a1_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_b_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> frag_c0_diff, frag_c1_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> frag_c0, frag_c1;

	// TODO: Move into the loop and execute when k = 0.
	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c0_diff, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1_diff, 0.0f);

#pragma unroll
	for (unsigned k = 0; k < FRAGMENT_DIM_N / FRAGMENT_DIM_K; k++) {
		const auto tmp_ac_ptr = shared_ac_ptr + k * FRAGMENT_DIM_M * FRAGMENT_DIM_K;
		const auto tmp_b_ptr = shared_b_ptr + k * FRAGMENT_DIM_K;

		nvcuda::wmma::load_matrix_sync(frag_a0, tmp_ac_ptr, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_a1, tmp_ac_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_b, tmp_b_ptr, FRAGMENT_DIM_N);

		// compute diff
		for (unsigned i = 0; i < FRAGMENT_DIM_M * FRAGMENT_DIM_K; i+= warp_size) {
			const auto index = i + cutf::thread::get_lane_id();
			const auto v = tmp_ac_ptr[index];
			tmp_ac_ptr[index] = v - cutf::type::cast<nvcuda::wmma::precision::tf32>(v);
		}
		for (unsigned i = 0; i < FRAGMENT_DIM_N * FRAGMENT_DIM_K; i+= warp_size) {
			const auto v_tid = i + cutf::thread::get_lane_id();
			const auto index = v_tid % FRAGMENT_DIM_K + (v_tid / FRAGMENT_DIM_K) * FRAGMENT_DIM_N;
			const auto v = tmp_b_ptr[index];
			tmp_b_ptr[index] = v - cutf::type::cast<nvcuda::wmma::precision::tf32>(v);
		}

		nvcuda::wmma::load_matrix_sync(frag_a0_diff, tmp_ac_ptr, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_a1_diff, tmp_ac_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_b_diff, tmp_b_ptr, FRAGMENT_DIM_N);

		nvcuda::wmma::mma_sync(frag_c0_diff, frag_a0_diff, frag_b, frag_c0_diff);
		nvcuda::wmma::mma_sync(frag_c1_diff, frag_a1_diff, frag_b, frag_c1_diff);
		nvcuda::wmma::mma_sync(frag_c0_diff, frag_a0, frag_b_diff, frag_c0_diff);
		nvcuda::wmma::mma_sync(frag_c1_diff, frag_a1, frag_b_diff, frag_c1_diff);

		nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
		nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);
	}

	for (unsigned i = 0; i < frag_c0.num_elements; i++) {
		frag_c0.x[i] += frag_c0_diff.x[i] / correction_rescale;
		frag_c1.x[i] += frag_c1_diff.x[i] / correction_rescale;
	}

	nvcuda::wmma::store_matrix_sync(shared_ac_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			ac_ptr, matrix_id * 2 * n, ac_m,
			shared_ac_ptr, 2 * n, n,
			tid
			);
#endif
}

// Backword of last layer
template <mtk::tsqr::compute_mode mode, class OUTPUT_T, class INPUT_T>
__global__ void tsqr_backward_layer0(
		OUTPUT_T* const q_ptr, const std::size_t ldq,
		const INPUT_T* const a_ptr,
		const INPUT_T* const b_ptr,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* const q_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = q_start_position[batch_size];
	const auto q_start_pos = q_start_position[matrix_id];
	const auto sub_m = q_start_position[matrix_id + 1] - q_start_pos;

	if(matrix_id >= batch_size) return;

	__shared__ INPUT_T shared_ac_in[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ INPUT_T shared_ac_out[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ INPUT_T shared_b[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_in_ptr = shared_ac_in + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_out_ptr = shared_ac_out + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_ptr = shared_b + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// Copy A(in)
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_in_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	// Copy B
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);

	__syncthreads();

	mtk::matmul::matmul_core_m16n16k16<get_matmul_compute_mode<mode>()>(
			shared_ac_out_ptr, FRAGMENT_DIM_M,
			shared_ac_in_ptr, FRAGMENT_DIM_M,
			shared_b_ptr, FRAGMENT_DIM_N,
			tid & 0x1f
			);

	mtk::matmul::matmul_core_m16n16k16<get_matmul_compute_mode<mode>()>(
			shared_ac_out_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			shared_ac_in_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			shared_b_ptr, FRAGMENT_DIM_N,
			tid & 0x1f
			);

	__syncthreads();

	mtk::matrix_copy::s2g32x16_1w(
			q_ptr, q_start_pos, ldq,
			shared_ac_out_ptr, sub_m, n,
			tid
			);
}

template <>
__global__ void tsqr_backward_layer0<mtk::tsqr::compute_mode::fp16_tc_nocor, half, half>(
		half* const q_ptr, const std::size_t ldq,
		const half* const a_ptr,
		const half* const b_ptr,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* const q_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = q_start_position[batch_size];
	const auto q_start_pos = q_start_position[matrix_id];
	const auto sub_m = q_start_position[matrix_id + 1] - q_start_pos;

	if(matrix_id >= batch_size) return;

	__shared__ half shared_ac_in[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_ac_out[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_b_f16[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_fp16_ptr = shared_ac_in + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_fp32_ptr = shared_ac_out + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp16_ptr = shared_b_f16 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// Copy A
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp16_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	// Copy B
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp16_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c0, frag_c1;

	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

	nvcuda::wmma::load_matrix_sync(frag_a0, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);

	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			q_ptr, q_start_pos, ldq,
			shared_ac_fp32_ptr, sub_m, n,
			tid
			);
}

template <>
__global__ void tsqr_backward_layer0<mtk::tsqr::compute_mode::fp32_tc_nocor, float, half>(
		float* const q_ptr, const std::size_t ldq,
		const half* const a_ptr,
		const half* const b_ptr,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* const q_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = q_start_position[batch_size];
	const auto q_start_pos = q_start_position[matrix_id];
	const auto sub_m = q_start_position[matrix_id + 1] - q_start_pos;

	if(matrix_id >= batch_size) return;

	__shared__ half shared_ac_in[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_ac_out[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_b_f16[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_fp16_ptr = shared_ac_in + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_fp32_ptr = shared_ac_out + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp16_ptr = shared_b_f16 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// Copy A
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp16_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	// Copy B
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp16_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c0, frag_c1;

	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

	nvcuda::wmma::load_matrix_sync(frag_a0, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);

	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			q_ptr, q_start_pos, ldq,
			shared_ac_fp32_ptr, sub_m, n,
			tid
			);
}

template <>
__global__ void tsqr_backward_layer0<mtk::tsqr::compute_mode::fp32_tc_cor, float, float>(
		float* const q_ptr, const std::size_t ldq,
		const float* const a_ptr,
		const float* const b_ptr,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* const q_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	constexpr float correction_rescale = 1024.0f;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = q_start_position[batch_size];
	const auto q_start_pos = q_start_position[matrix_id];
	const auto sub_m = q_start_position[matrix_id + 1] - q_start_pos;

	if(matrix_id >= batch_size) return;

	__shared__ half shared_ac_f16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_ac_f32[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_b_f16[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_b_f32[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_fp16_ptr = shared_ac_f16 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_fp32_ptr = shared_ac_f32 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp16_ptr = shared_b_f16 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp32_ptr = shared_b_f32 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp32_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	copy_32x16(shared_ac_fp16_ptr, shared_ac_fp32_ptr, tid);
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp32_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);
	copy_16x16(shared_b_fp16_ptr, shared_b_fp32_ptr, tid);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> frag_a0_diff, frag_a1_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c0, frag_c1;

	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

	nvcuda::wmma::load_matrix_sync(frag_a0, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	mtk::matrix_operation::diff32x16_1w(shared_ac_fp16_ptr, shared_ac_fp32_ptr, shared_ac_fp16_ptr, correction_rescale, tid);
	mtk::matrix_operation::diff16x16_1w(shared_b_fp16_ptr, shared_b_fp32_ptr, shared_b_fp16_ptr, correction_rescale, tid);

	nvcuda::wmma::load_matrix_sync(frag_a0_diff, shared_ac_fp16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_a1_diff, shared_ac_fp16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(frag_b_diff, shared_b_fp16_ptr, FRAGMENT_DIM_N);

	nvcuda::wmma::mma_sync(frag_c0, frag_a0_diff, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1_diff, frag_b, frag_c1);
	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b_diff, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b_diff, frag_c1);

	for (unsigned i = 0; i < frag_c0.num_elements; i++) {
		frag_c0.x[i] *= 1.0f / correction_rescale;
		frag_c1.x[i] *= 1.0f / correction_rescale;
	}

	nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
	nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);

	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_fp32_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			q_ptr, q_start_pos, ldq,
			shared_ac_fp32_ptr, sub_m, n,
			tid
			);
}

template <>
__global__ void tsqr_backward_layer0<mtk::tsqr::compute_mode::tf32_tc_nocor, float, float>(
		float* const q_ptr, const std::size_t ldq,
		const float* const a_ptr,
		const float* const b_ptr,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* const q_start_position
		) {
#ifdef ENABLE_TF32
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t FRAGMENT_DIM_K = 8;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = q_start_position[batch_size];
	const auto q_start_pos = q_start_position[matrix_id];
	const auto sub_m = q_start_position[matrix_id + 1] - q_start_pos;

	if(matrix_id >= batch_size) return;

	__shared__ float shared_ac[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_b[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_ptr = shared_ac + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_ptr = shared_b + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> frag_c0, frag_c1;

	// TODO: Move into the loop and execute when k = 0.
	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);

#pragma unroll
	for (unsigned k = 0; k < FRAGMENT_DIM_N / FRAGMENT_DIM_K; k++) {
		const auto tmp_ac_ptr = shared_ac_ptr + k * FRAGMENT_DIM_M * FRAGMENT_DIM_K;
		const auto tmp_b_ptr = shared_b_ptr + k * FRAGMENT_DIM_K;

		nvcuda::wmma::load_matrix_sync(frag_a0, tmp_ac_ptr, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_a1, tmp_ac_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_b, tmp_b_ptr, FRAGMENT_DIM_N);

		nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
		nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);
	}

	nvcuda::wmma::store_matrix_sync(shared_ac_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			q_ptr, q_start_pos, ldq,
			shared_ac_ptr, sub_m, n,
			tid
			);
#endif
}

template <>
__global__ void tsqr_backward_layer0<mtk::tsqr::compute_mode::tf32_tc_cor, float, float>(
		float* const q_ptr, const std::size_t ldq,
		const float* const a_ptr,
		const float* const b_ptr,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* const q_start_position
		) {
#ifdef ENABLE_TF32
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t FRAGMENT_DIM_K = 8;
	constexpr std::size_t max_batch_size_per_block = 4;
	constexpr float correction_rescale = 1024.0f;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / warp_size;
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	const auto ac_m = q_start_position[batch_size];
	const auto q_start_pos = q_start_position[matrix_id];
	const auto sub_m = q_start_position[matrix_id + 1] - q_start_pos;

	if(matrix_id >= batch_size) return;

	__shared__ float shared_ac[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_b[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_ptr = shared_ac + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_ptr = shared_b + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_a0, frag_a1;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_b;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_a0_diff, frag_a1_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_b_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> frag_c0_diff, frag_c1_diff;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> frag_c0, frag_c1;

	// TODO: Move into the loop and execute when k = 0.
	nvcuda::wmma::fill_fragment(frag_c0, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c0_diff, 0.0f);
	nvcuda::wmma::fill_fragment(frag_c1_diff, 0.0f);

#pragma unroll
	for (unsigned k = 0; k < FRAGMENT_DIM_N / FRAGMENT_DIM_K; k++) {
		const auto tmp_ac_ptr = shared_ac_ptr + k * FRAGMENT_DIM_M * FRAGMENT_DIM_K;
		const auto tmp_b_ptr = shared_b_ptr + k * FRAGMENT_DIM_K;

		nvcuda::wmma::load_matrix_sync(frag_a0, tmp_ac_ptr, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_a1, tmp_ac_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_b, tmp_b_ptr, FRAGMENT_DIM_N);

		// compute diff
		for (unsigned i = 0; i < FRAGMENT_DIM_M * FRAGMENT_DIM_K; i+= warp_size) {
			const auto index = i + cutf::thread::get_lane_id();
			const auto v = tmp_ac_ptr[index];
			tmp_ac_ptr[index] = v - cutf::type::cast<nvcuda::wmma::precision::tf32>(v);
		}
		for (unsigned i = 0; i < FRAGMENT_DIM_N * FRAGMENT_DIM_K; i+= warp_size) {
			const auto v_tid = i + cutf::thread::get_lane_id();
			const auto index = v_tid % FRAGMENT_DIM_K + (v_tid / FRAGMENT_DIM_K) * FRAGMENT_DIM_N;
			const auto v = tmp_b_ptr[index];
			tmp_b_ptr[index] = v - cutf::type::cast<nvcuda::wmma::precision::tf32>(v);
		}

		nvcuda::wmma::load_matrix_sync(frag_a0_diff, tmp_ac_ptr, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_a1_diff, tmp_ac_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
		nvcuda::wmma::load_matrix_sync(frag_b_diff, tmp_b_ptr, FRAGMENT_DIM_N);

		nvcuda::wmma::mma_sync(frag_c0_diff, frag_a0_diff, frag_b, frag_c0_diff);
		nvcuda::wmma::mma_sync(frag_c1_diff, frag_a1_diff, frag_b, frag_c1_diff);
		nvcuda::wmma::mma_sync(frag_c0_diff, frag_a0, frag_b_diff, frag_c0_diff);
		nvcuda::wmma::mma_sync(frag_c1_diff, frag_a1, frag_b_diff, frag_c1_diff);

		nvcuda::wmma::mma_sync(frag_c0, frag_a0, frag_b, frag_c0);
		nvcuda::wmma::mma_sync(frag_c1, frag_a1, frag_b, frag_c1);
	}

	for (unsigned i = 0; i < frag_c0.num_elements; i++) {
		frag_c0.x[i] += frag_c0_diff.x[i] / correction_rescale;
		frag_c1.x[i] += frag_c1_diff.x[i] / correction_rescale;
	}

	nvcuda::wmma::store_matrix_sync(shared_ac_ptr, frag_c0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(shared_ac_ptr + FRAGMENT_DIM_N, frag_c1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::matrix_copy::s2g32x16_1w(
			q_ptr, q_start_pos, ldq,
			shared_ac_ptr, sub_m, n,
			tid
			);
#endif
}
} // noname namespace

template <mtk::tsqr::compute_mode mode, class T>
void tsqr16_geq32(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const T* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename mtk::tsqr::get_working_q_type<mode>::type* const working_q_ptr, typename mtk::tsqr::get_working_r_type<mode>::type* const working_r_ptr,
		unsigned* const d_sub_m_list,
		unsigned* const h_sub_m_list,
		cudaStream_t const cuda_stream) {

	const std::size_t max_batch_size_per_block = 4;
	const auto batch_size_log2 = mtk::tsqr::get_batch_size_log2(m);
	const auto batch_size = 1lu << batch_size_log2;
	typename mtk::tsqr::get_working_r_type<mode>::type* const working_r_ptrs[2] = {working_r_ptr, working_r_ptr + n * n * batch_size};
	const std::size_t ldrs[2] = {n * batch_size, n * batch_size / 2};

	debug_func([&m, &n]() {std::printf("%s : matrix size = %lu x %lu\n", __func__, m, n);});
	debug_func([&batch_size]() {std::printf("%s : batch_size = %lu\n", __func__, batch_size);});
	debug_func([&working_r_ptrs]() {std::printf("%s : working_r_ptr[0] = %p\n", __func__, working_r_ptrs[0]);});
	debug_func([&working_r_ptrs]() {std::printf("%s : working_r_ptr[1] = %p\n", __func__, working_r_ptrs[1]);});
	debug_func([&working_q_ptr]() {std::printf("%s : working_q_ptr    = %p\n", __func__, working_q_ptr);});

	// Fisrt QR Factorization, whose matrix sizes are special
	h_sub_m_list[0] = 0;
	for(std::size_t i = 1; i < batch_size; i++) {
		h_sub_m_list[i] = m * i / batch_size;
	}
	h_sub_m_list[batch_size] = m;
	cutf::memory::copy_async(d_sub_m_list, h_sub_m_list, batch_size + 1, cuda_stream);

#ifdef MEASURE_QR_TIME
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto t0 = std::chrono::system_clock::now();
#endif

	debug_func([&batch_size_log2]() {std::printf("%s : %lu bQR\n", __func__, batch_size_log2);});
	debug_func([]() {std::printf("%s : a -> wr[0]\n", __func__);});
	mtk::tcqr::qr32x16_batched<get_tcqr_compute_mode<mode>()>(
			working_q_ptr, m,
			working_r_ptrs[0], n * batch_size,
			a_ptr, lda, m, n,
			batch_size, d_sub_m_list,
			cuda_stream
			);
#ifdef EVALUATE_EXPONENT_DISTRIBUTION
	mtk::validation::exponent_distribution(working_q_ptr, m * n, get_tsqr_compute_mode_string<mode>().c_str(), std::to_string(batch_size_log2).c_str(), cuda_stream);
#endif
	cudaStreamSynchronize(cuda_stream);

	// Rest QR Factorization, whose matrix sizes are n x n
	for(std::size_t i = 0; i < batch_size / 2 + 1; i++) {
		h_sub_m_list[i] = 2 * n * i;
	}
	cutf::memory::copy_async(d_sub_m_list, h_sub_m_list, batch_size / 2 + 1, cuda_stream);
	cudaStreamSynchronize(cuda_stream);

	for(std::size_t k = batch_size_log2 - 1; k > 0; k--) {
		debug_func([&k]() {std::printf("%s : %lu bQR\n", __func__, k);});
		const auto local_batch_size = 1lu << k;	
		const auto working_q_sride = 2 * n * n * (batch_size - (1lu << (k + 1))) + m * n;
		const auto working_r_index = 1lu - (batch_size_log2 - k) % 2;
		debug_func([&working_r_index, local_batch_size]() {std::printf("%s : a(wr[%lu]) -> a(wr[%lu]) [l_bs : %lu]\n", __func__, working_r_index, 1-working_r_index, local_batch_size);});

#ifdef DEBUG_INPUT_MATRIX_PRINT
		{
			auto h_tmp = cutf::memory::get_host_unique_ptr<T>(2 * n * n * local_batch_size);
			cutf::memory::copy(h_tmp.get(), working_r_ptrs[working_r_index], 2 * n * n * local_batch_size);
			mtk::utils::print_matrix(h_tmp.get(), 2 * n * local_batch_size, n, "input");
		}
#endif

		mtk::tcqr::qr32x16_batched<get_tcqr_compute_mode<mode>()>(
				working_q_ptr + working_q_sride, 2 * n * local_batch_size,
				working_r_ptrs[1 - working_r_index], ldrs[1 - working_r_index],
				working_r_ptrs[working_r_index], ldrs[working_r_index],
				2 * n * local_batch_size,
				n, 
				local_batch_size, d_sub_m_list,
				cuda_stream
				);
#ifdef EVALUATE_EXPONENT_DISTRIBUTION
		mtk::validation::exponent_distribution(working_q_ptr + working_q_sride, 2 * n * local_batch_size * n, get_tsqr_compute_mode_string<mode>().c_str(), std::to_string(k).c_str(), cuda_stream);
#endif

		debug_func([]() {CUTF_CHECK_ERROR(cudaGetLastError());});

#ifdef DEBUG_Q_MATRIX_PRINT
		{
			auto h_tmp = cutf::memory::get_host_unique_ptr<typename mtk::tsqr::get_working_q_type<mode>::type>(2 * n * n * local_batch_size);
			cutf::memory::copy(h_tmp.get(), working_q_ptr + working_q_sride, 2 * n * n * local_batch_size);
			mtk::utils::print_matrix(h_tmp.get(), 2 * n * local_batch_size, n, ("Q (" + std::to_string(k) + ")").c_str());
		}
#endif

	}

	// Store final R to `r_ptr`
	debug_func([]() {std::printf("%s : 1 bQR\n", __func__);});
	debug_func([&batch_size_log2]() {std::printf("%s : a(wr[%lu]) -> r\n", __func__, (batch_size_log2 % 2));});
	const auto working_q_sride = 2 * n * n * (batch_size - 2) + m * n;
	mtk::tcqr::qr32x16<get_tcqr_compute_mode<mode>()>(
			working_q_ptr + working_q_sride, 2 * n,
			r_ptr, ldr,
			working_r_ptrs[1 - (batch_size_log2 % 2)], ldrs[1 - (batch_size_log2 % 2)],
			2 * n,
			n,
			cuda_stream
			);
#ifdef EVALUATE_EXPONENT_DISTRIBUTION
	mtk::validation::exponent_distribution(working_q_ptr + working_q_sride, 2 * n * n, get_tsqr_compute_mode_string<mode>().c_str(), std::to_string(0).c_str(), cuda_stream);
#endif

	cudaStreamSynchronize(cuda_stream);

	// experimental force underflow
	// mtk::experimental::min_exponent<typename mtk::tsqr::get_working_q_type<mode>::type>(working_q_ptr, -16, mtk::tsqr::get_working_q_size(m, n), cuda_stream);

	debug_func([]() {std::printf("%s : last Q\n", __func__);});
#ifdef DEBUG_Q_MATRIX_PRINT
	{
		auto h_tmp = cutf::memory::get_host_unique_ptr<typename mtk::tsqr::get_working_q_type<mode>::type>(2 * n * n);
		cutf::memory::copy(h_tmp.get(), working_q_ptr + working_q_sride, 2 * n * n);
		mtk::utils::print_matrix(h_tmp.get(), 2 * n, n, "Q");
	}
#endif
#ifdef DEBUG
	{
		auto h_tmp = cutf::memory::get_host_unique_ptr<typename mtk::tsqr::get_working_r_type<mode>::type>(n * n);
		cutf::memory::copy(h_tmp.get(), r_ptr, n * n);
		mtk::utils::print_matrix(h_tmp.get(), n, n, "R (result)");
	}
#endif

#ifdef MEASURE_QR_TIME
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto t1 = std::chrono::system_clock::now();
#endif
	debug_func([]() {std::printf("%s : Backword\n", __func__);});

	// Backward
	for(std::size_t k = 1; k < batch_size_log2; k++) {
		debug_func([&k]() {std::printf("%s : %lu\n", __func__, k);});
		const auto working_q_sride = 2 * n * n * (batch_size - (1lu << (k + 1))) + m * n;
		const auto grid_size = ((1lu<<k) + max_batch_size_per_block - 1) / max_batch_size_per_block;
		const auto block_size = max_batch_size_per_block * warp_size;
#ifdef DEBUG_Q_MATRIX_PRINT
		{
			const auto local_batch_size = 1lu << k;	
			auto h_tmp = cutf::memory::get_host_unique_ptr<typename mtk::tsqr::get_working_q_type<mode>::type>(2 * n * n * local_batch_size);
			cutf::memory::copy(h_tmp.get(), working_q_ptr + working_q_sride, 2 * n * n * local_batch_size);
			mtk::utils::print_matrix(h_tmp.get(), 2 * n * local_batch_size, n, "Q (before backwarding)");
		}
#endif
		cudaStreamSynchronize(cuda_stream);
#ifdef EVALUATE_EACH_SMALL_Q
		mtk::validation::multi_orthogonality(working_q_ptr + working_q_sride, 2 * n, n, 2 * n * (1lu << k), 1lu << k, cuda_stream);
		cudaStreamSynchronize(cuda_stream);
#endif
		tsqr_backward<mode, typename mtk::tsqr::get_working_q_type<mode>::type><<<grid_size, block_size, 0, cuda_stream>>>(
				working_q_ptr + working_q_sride,
				working_q_ptr + working_q_sride + (1lu << k) * 2 * n * n,
				n,
				k
				);
		cudaStreamSynchronize(cuda_stream);
	}
	// the each matrix size of last layer is different from other layers
	h_sub_m_list[0] = 0;
	for(std::size_t i = 1; i < batch_size; i++) {
		h_sub_m_list[i] = m * i / batch_size;
	}
	h_sub_m_list[batch_size] = m;
	cutf::memory::copy_async(d_sub_m_list, h_sub_m_list, batch_size + 1, cuda_stream);
	const auto grid_size = (batch_size + max_batch_size_per_block - 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * warp_size;
#ifdef DEBUG_Q_MATRIX_PRINT
	{
		auto h_tmp = cutf::memory::get_host_unique_ptr<typename mtk::tsqr::get_working_q_type<mode>::type>(n * m);
		cutf::memory::copy(h_tmp.get(), working_q_ptr, m * n);
		mtk::utils::print_matrix(h_tmp.get(), m, n, "Q (before backwarding)");
	}
#endif
#ifdef EVALUATE_EACH_SMALL_Q
	mtk::validation::multi_orthogonality(working_q_ptr + m * n, 2 * n, n, 2 * n * batch_size, batch_size, cuda_stream);
	cudaStreamSynchronize(cuda_stream);
#endif
	cudaStreamSynchronize(cuda_stream);
	tsqr_backward_layer0<mode, T, typename mtk::tsqr::get_working_q_type<mode>::type><<<grid_size, block_size, 0, cuda_stream>>>(
			q_ptr, ldq,
			working_q_ptr,
			working_q_ptr + m * n,
			n,
			batch_size,
			d_sub_m_list
			);
	cudaStreamSynchronize(cuda_stream);
	debug_func([]() {CUTF_CHECK_ERROR(cudaDeviceSynchronize());});
#ifdef DEBUG_Q_MATRIX_PRINT
	{
		auto h_tmp = cutf::memory::get_host_unique_ptr<T>(n * m);
		cutf::memory::copy(h_tmp.get(), q_ptr, ldq * n);
		mtk::utils::print_matrix(h_tmp.get(), m, n, "Q (result)");
	}
#endif
#ifdef MEASURE_QR_TIME
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto t2 = std::chrono::system_clock::now();

	// analyze
	const auto computing_q_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0;
	const auto computing_r_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0;
	std::printf("computing_q_time,computing_r_time\n");
	std::printf("%e,%e\n", computing_q_time, computing_r_time);
#endif
}

template <mtk::tsqr::compute_mode mode>
void mtk::tsqr::tsqr16(
		typename mtk::tsqr::get_io_type<mode>::type* const q_ptr, const std::size_t ldq,
		typename mtk::tsqr::get_io_type<mode>::type* const r_ptr, const std::size_t ldr,
		const typename mtk::tsqr::get_io_type<mode>::type* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		typename get_working_q_type<mode>::type* const working_q_ptr, typename get_working_r_type<mode>::type* const working_r_ptr,
		unsigned* const d_working_l_ptr,
		unsigned* const h_working_l_ptr,
		cudaStream_t const cuda_stream) {
	if(m > 32) {
		tsqr16_geq32<mode>(
				q_ptr, ldq,
				r_ptr, ldr,
				a_ptr, lda,
				m, n,
				working_q_ptr, working_r_ptr,
				d_working_l_ptr,
				h_working_l_ptr,
				cuda_stream);
	}else {
		mtk::tcqr::qr32x16<get_tcqr_compute_mode<mode>()>(
				q_ptr, ldq,
				r_ptr, ldr,
				a_ptr, lda,
				m, n,
				cuda_stream
				);
	}
}

// (T *const q_ptr, T *const r_ptr, const T *const a_ptr, const std::size_t m, const std::size_t n, T *const working_memory_ptr)
#define TSQR_TEMPLATE_INSTANCE(mode) template void mtk::tsqr::tsqr16<mode>(mtk::tsqr::get_io_type<mode>::type* const, const std::size_t, mtk::tsqr::get_io_type<mode>::type* const, const std::size_t, const mtk::tsqr::get_io_type<mode>::type* const, const std::size_t, const std::size_t, const std::size_t, typename mtk::tsqr::get_working_q_type<mode>::type* const, typename mtk::tsqr::get_working_r_type<mode>::type* const, unsigned* const, unsigned* const, cudaStream_t const)
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::fp16_notc        );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::fp32_notc        );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::fp16_tc_nocor    );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::fp32_tc_nocor    );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::fp32_tc_cor      );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::tf32_tc_cor      );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::tf32_tc_nocor    );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::tf32_tc_cor_emu  );
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::tf32_tc_nocor_emu);
TSQR_TEMPLATE_INSTANCE(mtk::tsqr::compute_mode::mixed_tc_cor_emu );
