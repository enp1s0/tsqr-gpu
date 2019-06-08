#include <algorithm>
#include <cmath>
#include <vector>
#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/error.hpp>
#include "tsqr.hpp"
#include "tcqr.hpp"
#include "utils.hpp"
#include "matrix_copy.cuh"
#include "matrix_operations.cuh"
#include "gemm_core/gemm_core.cuh"

//#define DEBUG
//#define DEBUG_INPUT_MATRIX_PRINT
//#define DEBUG_Q_MATRIX_PRINT

namespace{
constexpr unsigned warp_size = 32;
template <class Func>
void debug_func(Func func) {
#ifdef DEBUG
	func();
#endif
}
std::size_t get_batch_size_log2(const std::size_t m) {
	return (std::max(5u, static_cast<unsigned>( std::ceil( std::log2(static_cast<float>(m))))) - 5u);
}
std::size_t get_batch_size(const std::size_t m) {
	return 1lu << get_batch_size_log2(m);
}

// backward 1層目以外
template <bool UseTC, class T>
__global__ void tsqr_backward(
		T* const ac_ptr,
		const T* const b_ptr,
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

	__shared__ T shared_ac_in[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ T shared_ac_out[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ T shared_b[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_in_ptr = shared_ac_in + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_out_ptr = shared_ac_out + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_ptr = shared_b + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// AC(in)のコピー
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_in_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	// Bのコピー
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);
	// AC(out)の初期化
	mtk::matrix_operation::make_zero_matrix<T, FRAGMENT_DIM_M, FRAGMENT_DIM_N, 1>(
			shared_ac_out_ptr, tid);

	__syncthreads();

	mtk::gemm_core16x16<T, 1>(
			shared_ac_out_ptr, FRAGMENT_DIM_M,
			shared_ac_in_ptr, FRAGMENT_DIM_M,
			shared_b_ptr, FRAGMENT_DIM_N,
			tid & 0x1f
			);

	mtk::gemm_core16x16<T, 1>(
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
__global__ void tsqr_backward<true, half>(
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

	// ACのコピー
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp16_ptr, 2 * n, n,
			ac_ptr, matrix_id * 2 * n, ac_m,
			tid
			);
	// Bのコピー
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp16_ptr, n, n,
			b_ptr, matrix_id * n, ac_m / 2,
			tid
			);

	// TCによる行列積
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

template <bool UseTC, class OUTPUT_T, class INPUT_T>
__global__ void tsqr_backward_layer0(
		OUTPUT_T* const q_ptr,
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

	// A(in) のコピー
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_in_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	// AC(out)の初期化
	mtk::matrix_operation::make_zero_matrix<INPUT_T, FRAGMENT_DIM_M, FRAGMENT_DIM_N, 1>(
			shared_ac_out_ptr, tid);
	// Bのコピー
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);

	__syncthreads();

	mtk::gemm_core16x16<INPUT_T, 1>(
			shared_ac_out_ptr, FRAGMENT_DIM_M,
			shared_ac_in_ptr, FRAGMENT_DIM_M,
			shared_b_ptr, FRAGMENT_DIM_N,
			tid & 0x1f
			);

	mtk::gemm_core16x16<INPUT_T, 1>(
			shared_ac_out_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			shared_ac_in_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			shared_b_ptr, FRAGMENT_DIM_N,
			tid & 0x1f
			);

	__syncthreads();

	mtk::matrix_copy::s2g32x16_1w(
			q_ptr, q_start_pos, ac_m,
			shared_ac_out_ptr, sub_m, n,
			tid
			);
}

template <>
__global__ void tsqr_backward_layer0<true, float, half>(
		float* const q_ptr,
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

	__shared__ half shared_ac_f16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_ac_f32[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_b_f16[FRAGMENT_DIM_N * FRAGMENT_DIM_N * max_batch_size_per_block];

	const auto shared_ac_fp16_ptr = shared_ac_f16 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_ac_fp32_ptr = shared_ac_f32 + FRAGMENT_DIM_M * FRAGMENT_DIM_N * shared_memory_id;
	const auto shared_b_fp16_ptr = shared_b_f16 + FRAGMENT_DIM_N * FRAGMENT_DIM_N * shared_memory_id;

	// A のコピー
	mtk::matrix_copy::g2s32x16_1w(
			shared_ac_fp16_ptr, sub_m, n,
			a_ptr, q_start_pos, ac_m,
			tid
			);
	// Bのコピー
	mtk::matrix_copy::g2s16x16_1w(
			shared_b_fp16_ptr, n, n,
			b_ptr, matrix_id * n, n * batch_size,
			tid
			);

	// TCによる行列積
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
			q_ptr, q_start_pos, ac_m,
			shared_ac_fp32_ptr, sub_m, n,
			tid
			);
}
}

// 必要な作業用メモリ
std::size_t mtk::tsqr::get_working_q_size(const std::size_t m, const std::size_t n) {
	const auto batch_size = get_batch_size(m);
	const auto working_q_size = n * m + 2 * n * n * (batch_size - 1);

	return working_q_size;
}
std::size_t mtk::tsqr::get_working_r_size(const std::size_t m, const std::size_t n) {
	const auto batch_size = get_batch_size(m);
	const auto working_r_size_0 = n * n * batch_size;
	const auto working_r_size_1 = n * n * batch_size / 2;

	return working_r_size_0 + working_r_size_1;
}

template <bool UseTC, class T>
void mtk::tsqr::tsqr16(
		T* const q_ptr, T* const r_ptr, 
		const T* const a_ptr, const std::size_t m, const std::size_t n,
		typename get_working_q_type<T, UseTC>::type* const working_q_ptr, typename get_working_r_type<T, UseTC>::type* const working_r_ptr) {

	const std::size_t max_batch_size_per_block = 4;
	const auto batch_size_log2 = get_batch_size_log2(m);
	const auto batch_size = 1lu << batch_size_log2;
	typename get_working_r_type<T, UseTC>::type* const working_r_ptrs[2] = {working_r_ptr, working_r_ptr + n * n * batch_size};

	debug_func([&m, &n]() {std::printf("%s : matrix size = %lu x %lu\n", __func__, m, n);});
	debug_func([&batch_size]() {std::printf("%s : batch_size = %lu\n", __func__, batch_size);});
	debug_func([&working_r_ptrs]() {std::printf("%s : working_r_ptr[0] = 0x%x\n", __func__, working_r_ptrs[0]);});
	debug_func([&working_r_ptrs]() {std::printf("%s : working_r_ptr[1] = 0x%x\n", __func__, working_r_ptrs[1]);});
	debug_func([&working_q_ptr]() {std::printf("%s : working_q_ptr    = 0x%x\n", __func__, working_q_ptr);});

	const auto d_sub_m_list = cutf::memory::get_device_unique_ptr<unsigned>(batch_size + 1);
	const auto h_sub_m_list = cutf::memory::get_host_unique_ptr<unsigned>(batch_size + 1);

	// 1層目はsub_mが特殊なので別途計算を行う
	h_sub_m_list.get()[0] = 0;
	for(std::size_t i = 1; i < batch_size; i++) {
		h_sub_m_list.get()[i] = m * i / batch_size;
	}
	h_sub_m_list.get()[batch_size] = m;
	cutf::memory::copy(d_sub_m_list.get(), h_sub_m_list.get(), batch_size + 1);

	debug_func([&batch_size_log2]() {std::printf("%s : %lu bQR\n", __func__, batch_size_log2);});
	debug_func([]() {std::printf("%s : a -> wr[0]\n", __func__);});
	mtk::tcqr::qr32x16_batched<UseTC>(
			working_q_ptr,
			working_r_ptrs[0],
			a_ptr, m, n,
			batch_size, d_sub_m_list.get()
			);

	// 2層目からはsub matrixの大きさが 2n * n となるので，一度計算しGPUに転送しておけばOK
	for(std::size_t i = 0; i < batch_size / 2 + 1; i++) {
		h_sub_m_list.get()[i] = 2 * n * i;
	}
	cutf::memory::copy(d_sub_m_list.get(), h_sub_m_list.get(), batch_size / 2 + 1);

	// 再帰的QR分解のfor展開
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

		mtk::tcqr::qr32x16_batched<UseTC>(
				working_q_ptr + working_q_sride,
				working_r_ptrs[1 - working_r_index],
				working_r_ptrs[working_r_index],
				2 * n * local_batch_size,
				n, 
				local_batch_size, d_sub_m_list.get()
				);

		debug_func([]() {CUTF_HANDLE_ERROR(cudaGetLastError());});

#ifdef DEBUG_Q_MATRIX_PRINT
		{
			auto h_tmp = cutf::memory::get_host_unique_ptr<typename get_working_q_type<T, UseTC>::type>(2 * n * n * local_batch_size);
			cutf::memory::copy(h_tmp.get(), working_q_ptr + working_q_sride, 2 * n * n * local_batch_size);
			mtk::utils::print_matrix(h_tmp.get(), 2 * n * local_batch_size, n, "Q");
		}
#endif

	}

	// 最終層はrの保存先が異なる
	debug_func([]() {std::printf("%s : 1 bQR\n", __func__);});
	debug_func([&batch_size_log2]() {std::printf("%s : a(wr[%lu]) -> r\n", __func__, (batch_size_log2 % 2));});
	const auto working_q_sride = 2 * n * n * (batch_size - 2) + m * n;
	mtk::tcqr::qr32x16<UseTC>(
			working_q_ptr + working_q_sride,
			r_ptr,
			working_r_ptrs[1 - (batch_size_log2 % 2)],
			2 * n,
			n
			);

	debug_func([]() {std::printf("%s : last Q\n", __func__);});
#ifdef DEBUG_Q_MATRIX_PRINT
	{
		auto h_tmp = cutf::memory::get_host_unique_ptr<typename get_working_q_type<T, UseTC>::type>(2 * n * n);
		cutf::memory::copy(h_tmp.get(), working_q_ptr + working_q_sride, 2 * n * n);
		mtk::utils::print_matrix(h_tmp.get(), 2 * n, n, "Q");
	}
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
			auto h_tmp = cutf::memory::get_host_unique_ptr<typename get_working_q_type<T, UseTC>::type>(2 * n * n * local_batch_size);
			cutf::memory::copy(h_tmp.get(), working_q_ptr + working_q_sride, 2 * n * n * local_batch_size);
			mtk::utils::print_matrix(h_tmp.get(), 2 * n * local_batch_size, n, "Q (before backwarding)");
		}
#endif
		tsqr_backward<UseTC><<<grid_size, block_size>>>(
				working_q_ptr + working_q_sride,
				working_q_ptr + working_q_sride + (1lu << k) * 2 * n * n,
				n,
				k
				);

	}
	// 1層目はsub_mが特殊なので別途計算を行う
	h_sub_m_list.get()[0] = 0;
	for(std::size_t i = 1; i < batch_size; i++) {
		h_sub_m_list.get()[i] = m * i / batch_size;
	}
	h_sub_m_list.get()[batch_size] = m;
	cutf::memory::copy(d_sub_m_list.get(), h_sub_m_list.get(), batch_size + 1);
	const auto grid_size = (batch_size + max_batch_size_per_block - 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * warp_size;
#ifdef DEBUG_Q_MATRIX_PRINT
	{
		auto h_tmp = cutf::memory::get_host_unique_ptr<typename get_working_q_type<T, UseTC>::type>(n * m);
		cutf::memory::copy(h_tmp.get(), working_q_ptr, m * n);
		mtk::utils::print_matrix(h_tmp.get(), m, n, "Q (before backwarding)");
	}
#endif
	tsqr_backward_layer0<UseTC><<<grid_size, block_size>>>(
			q_ptr,
			working_q_ptr,
			working_q_ptr + m * n,
			n,
			batch_size,
			d_sub_m_list.get()
			);
	debug_func([]() {CUTF_HANDLE_ERROR(cudaDeviceSynchronize());});
#ifdef DEBUG_Q_MATRIX_PRINT
	{
		auto h_tmp = cutf::memory::get_host_unique_ptr<T>(n * m);
		cutf::memory::copy(h_tmp.get(), q_ptr, m * n);
		mtk::utils::print_matrix(h_tmp.get(), m, n, "Q (result)");
	}
#endif
}

// (T *const q_ptr, T *const r_ptr, const T *const a_ptr, const std::size_t m, const std::size_t n, T *const working_memory_ptr)
template void mtk::tsqr::tsqr16<true, float>(float* const, float* const, const float* const, const std::size_t, const std::size_t, typename mtk::tsqr::get_working_q_type<float, true>::type* const, typename mtk::tsqr::get_working_r_type<float, true>::type* const);
template void mtk::tsqr::tsqr16<false, float>(float* const, float* const, const float* const, const std::size_t, const std::size_t, typename mtk::tsqr::get_working_q_type<float, false>::type* const, typename mtk::tsqr::get_working_r_type<float, false>::type* const);
template void mtk::tsqr::tsqr16<true, half>(half* const, half* const, const half* const, const std::size_t, const std::size_t, typename mtk::tsqr::get_working_q_type<half, false>::type* const, typename mtk::tsqr::get_working_r_type<half, false>::type* const);
template void mtk::tsqr::tsqr16<false, half>(half* const, half* const, const half* const, const std::size_t, const std::size_t, typename mtk::tsqr::get_working_q_type<half, false>::type* const, typename mtk::tsqr::get_working_r_type<half, false>::type* const);
