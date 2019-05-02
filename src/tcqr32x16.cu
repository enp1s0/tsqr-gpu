#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include <stdio.h>
#include "tcqr.hpp"
#include "utils.hpp"
#include "matrix_copy.cuh"
#include "matrix_operations.cuh"

//#define DEBUG

namespace {
constexpr unsigned warp_size = 32;

template <class Func>
__device__ void debug_func(unsigned unique_id, Func run_func){
#ifdef DEBUG
	if(unique_id == 0){
		run_func();
	}
#endif
}

template <class INPUT_T, class OUTPUT_T>
__device__ OUTPUT_T get_norm2_32(
		INPUT_T* const ptr, const unsigned size,
	   	unsigned warp_id){
	auto tmp = cutf::cuda::type::cast<OUTPUT_T>(0.0f);

	if(warp_id < size){
		tmp = cutf::cuda::type::cast<OUTPUT_T>(ptr[warp_id]);
		tmp = tmp * tmp;
	}

	for(auto mask = (warp_size >> 1); mask > 0; mask >>= 1){
		tmp += __shfl_xor_sync(0xffffffff, tmp, mask);
	}

	return cutf::cuda::type::cast<OUTPUT_T>(tmp);
}

template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ void copy_32x16(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr,
		const unsigned unique_id
		){
	constexpr auto stride = 2 * warp_size;
	for(unsigned i = 0; i < (FRAGMENT_DIM_M * FRAGMENT_DIM_N) / stride; i++){
		dst_ptr[i * stride + unique_id] = cutf::cuda::type::cast<DST_T>(src_ptr[i * stride + unique_id]);
	}
}

template <class T, class U_T, std::size_t FRAGMENT_DIM_M = 32>
__device__ void make_h(
		T* const h_ptr, const unsigned m, 
		const U_T* const u_ptr, const U_T norm2_u_1, 
		const unsigned unique_id){
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	for(unsigned k = 0; k < FRAGMENT_DIM_M; k+= 2){
		const auto x = k + lane;
		U_T tmp;
		if(x == y){
			tmp = cutf::cuda::type::cast<U_T>(1.0f);
		}else{
			tmp = cutf::cuda::type::cast<U_T>(0.0f);
		}
		tmp -= cutf::cuda::type::cast<U_T>(2.0f) * u_ptr[y] * u_ptr[x] / norm2_u_1;

		h_ptr[x * FRAGMENT_DIM_M + y] = cutf::cuda::type::cast<T>(tmp);
	}
}
template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ void update_qr_f32tc(
		float* const q32_ptr, float* const r32_ptr,
		const half* const q16_ptr, const half* const r16_ptr,
		half* const h16_ptr,
		const unsigned unique_id
		){
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h16_0_frag, h16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> r16_0_frag, r16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> q16_0_frag, q16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> r32_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> q32_0_frag, q32_1_frag;

	nvcuda::wmma::fill_fragment(r32_frag, 0.0f);
	nvcuda::wmma::fill_fragment(q32_0_frag, 0.0f);
	nvcuda::wmma::fill_fragment(q32_1_frag, 0.0f);

	// load h
	nvcuda::wmma::load_matrix_sync(h16_0_frag, h16_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(h16_1_frag, h16_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);

	/*  Q 0 */
	// load q
	nvcuda::wmma::load_matrix_sync(q16_0_frag, q16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_frag, q16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_frag, q16_0_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_frag, q16_1_frag, q32_0_frag);
	/*  Q 1 */
	// load q
	nvcuda::wmma::load_matrix_sync(q16_0_frag, q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_frag, q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_frag, q16_0_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_frag, q16_1_frag, q32_1_frag);
	/*  R */
	// load r
	nvcuda::wmma::load_matrix_sync(r16_0_frag, r16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(r16_1_frag, r16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(r32_frag, h16_0_frag, r16_0_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_1_frag, r16_1_frag, r32_frag);

	// store
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_N, q32_0_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_1_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(r32_ptr + lane * FRAGMENT_DIM_N, r32_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ void qr32x16_f32tc_core(
		float* const q32_ptr, float* const r32_ptr,
		half* const q16_ptr, half* const r16_ptr,
		float* const u32_ptr, half* h16_ptr,
		const unsigned m, const unsigned n,
		const unsigned tid
		){
	const auto unique_id = tid & 0x3f;
	for(unsigned k = 0; k < n ; k++){
		debug_func(
				unique_id,
				[&k](){printf("/* -------- %u ---------\n", k);}
				);
		__syncthreads();
		debug_func(
				unique_id,
				[&r32_ptr, &m, &n](){mtk::utils::print_matrix_32x16(r32_ptr, m, n, "R");}
				);
		__syncthreads();
		debug_func(
				unique_id,
				[&q32_ptr, &m](){mtk::utils::print_matrix_32x16(q32_ptr, m, m, "Q");}
				);
		__syncthreads();
		// copy u
		// TODO ; 0埋めとデータロードを異なるwarpでできないか検証
		if(unique_id < FRAGMENT_DIM_M){
			u32_ptr[unique_id] = 0.0f;
			if(unique_id >= k){
				u32_ptr[unique_id] = r32_ptr[FRAGMENT_DIM_M * k + unique_id];
			}
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u32_ptr, &m](){mtk::utils::print_matrix(u32_ptr, 1, m, "u");}
				);
		// compute |u|
		// TODO : どうせ0埋めされているなら32個で和をとってしまってもいい気がするので検証
		const auto norm_u_0 = cutf::cuda::math::sqrt<float>(get_norm2_32<float, float>(u32_ptr, m, unique_id & 0x1f));
		__syncthreads();
		debug_func(
				unique_id,
				[&norm_u_0](){printf("norm_u_0 = %.5f\n", norm_u_0);}
				);
		// update u
		if(unique_id == k){
			u32_ptr[unique_id] += cutf::cuda::math::sign(u32_ptr[unique_id]) * norm_u_0;
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u32_ptr, &m](){mtk::utils::print_matrix(u32_ptr, 1, m, "u`");}
				);
		// recompute |u|
		const auto norm2_u_1 = get_norm2_32<float, float>(u32_ptr, m, unique_id & 0x1f);
		debug_func(
				unique_id,
				[&norm2_u_1](){printf("norm_u_1^2 = %.5f\n", norm2_u_1);}
				);
		// compute h
		make_h(
				h16_ptr, m,
				u32_ptr, norm2_u_1,
				unique_id
				);
		debug_func(
				unique_id,
				[&h16_ptr, &m](){mtk::utils::print_matrix_32x16(h16_ptr, m, m, "H");}
				);
		// copy f32 to f16
		copy_32x16(r16_ptr, r32_ptr, unique_id);
		copy_32x16(q16_ptr, q32_ptr, unique_id);
		copy_32x16(q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
		debug_func(
				unique_id,
				[&r16_ptr, &m, &n](){mtk::utils::print_matrix_32x16(r16_ptr, 32, 16, "R (before update)");}
				);
		debug_func(
				unique_id,
				[&q16_ptr, &m](){mtk::utils::print_matrix_32x16(q16_ptr, 32, 32, "Q (before update)");}
				);
		__syncthreads();
		// update q, r
		update_qr_f32tc(
				q32_ptr, r32_ptr,
				q16_ptr, r16_ptr,
				h16_ptr,
				unique_id
				);
		__syncthreads();
	}
}

template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16, std::size_t max_batch_size_per_block = 4>
__global__ void qr32x16_f32_batched_kernel(
		float* const q32_ptr,
		float* const r32_ptr,
		const float* const a32_ptr,
		const std::size_t m,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / (warp_size * 2);
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	if(matrix_id >= batch_size) return;

	__shared__ float shared_q32[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ float shared_r32[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_q16[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ half shared_r16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_h16[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ float shared_u32[FRAGMENT_DIM_M * max_batch_size_per_block];

	const auto shared_q32_ptr = shared_q32 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r32_ptr = shared_r32 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_q16_ptr = shared_q16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r16_ptr = shared_r16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_h16_ptr = shared_h16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_u32_ptr = shared_u32 + shared_memory_id * FRAGMENT_DIM_M;

	const auto sub_a_position = a_start_position[matrix_id];
	const auto sub_a_m = a_start_position[matrix_id + 1] - sub_a_position;

	// init shared memory
	mtk::matrix_copy::g2s32x16(
			shared_r32_ptr, sub_a_m, n,
			a32_ptr, sub_a_position, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix(
			shared_q32_ptr,
			tid
			);

	// qr core
	qr32x16_f32tc_core(
			shared_q32_ptr, shared_r32_ptr,
			shared_q16_ptr, shared_r16_ptr,
			shared_u32_ptr, shared_h16_ptr,
			sub_a_m, n,
			tid
			);

	// store result
	mtk::matrix_copy::s2g16x32_t(
			q32_ptr, sub_a_position, m,
			shared_q32_ptr, n, sub_a_m,
			tid
			);
	mtk::matrix_copy::s2g32x16(
			r32_ptr, n * matrix_id, n * batch_size,
			shared_r32_ptr, n, n,
			tid
			);
}

template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__global__ void qr32x16_f32_kernel(
		float* const q32_ptr,
		float* const r32_ptr,
		const float* const a32_ptr,
		const unsigned m,
		const unsigned n
		){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float shared_q32[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ float shared_r32[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ half shared_q16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ half shared_r16[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ half shared_h16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ float shared_u32[FRAGMENT_DIM_M];

	// init shared memory
	mtk::matrix_copy::g2s32x16(
			shared_r32, m, n,
			a32_ptr, 0, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix(
			shared_q32,
			tid
			);

	// qr core
	qr32x16_f32tc_core(
			shared_q32, shared_r32,
			shared_q16, shared_r16,
			shared_u32, shared_h16,
			m, n,
			tid
			);
	// store result
	mtk::matrix_copy::s2g16x32_t(
			q32_ptr, 0, m,
			shared_q32, n, m,
			tid
			);
	mtk::matrix_copy::s2g32x16(
			r32_ptr, 0, n,
			shared_r32, n, n,
			tid
			);
}
}

void mtk::tcqr::qr32x16_f32tc_batched(
		float *const q, float *const r,
		const float *const a, const unsigned int m, const unsigned int n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		){
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto grid_size = (batch_size + max_batch_size_per_block + 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * 2 * warp_size;

	qr32x16_f32_batched_kernel<<<grid_size, block_size>>>(
			q, r,
			a, m, n,
			batch_size,
			a_start_position
			);
}
void mtk::tcqr::qr32x16_f32tc(
		float *const q, float *const r,
		const float *const a, const unsigned int m, const unsigned int n
		){
	qr32x16_f32_kernel<<<1, 2 * warp_size>>>(
			q, r,
			a, m, n
			);
}
