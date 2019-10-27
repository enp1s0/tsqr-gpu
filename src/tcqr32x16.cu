#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include <wmma_extension.hpp>
#include <stdio.h>
#include "tcqr.hpp"
#include "utils.hpp"
#include "matrix_copy.cuh"
#include "matrix_operations.cuh"
#include "gemm_core/gemm_core.cuh"

//#define DEBUG
//#define MEASURE_CLOCK
// clock : make_u,norm1,update_u,norm2,make_h,mem_init,update_qr,mem_swap

namespace {
constexpr unsigned warp_size = 32;

template <class Func>
__device__ void debug_func(unsigned unique_id, Func run_func) {
#ifdef DEBUG
	if(unique_id == 0) {
		run_func();
	}
#endif
}

template <class INPUT_T>
__device__ float get_norm2_32(
		INPUT_T* const ptr, const unsigned size,
		unsigned warp_id) {
	float tmp;

	if(warp_id < size) {
		tmp = cutf::type::cast<float>(ptr[warp_id]);
		tmp = tmp * tmp;
	} else {
		tmp = 0.0f;
	}

	for(auto mask = (warp_size >> 1); mask > 0; mask >>= 1) {
		tmp += __shfl_xor_sync(0xffffffff, tmp, mask);
	}

	return tmp;
}

template <class DST_T, class SRC_T>
__device__ void copy_32x16(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr auto stride = 2 * warp_size;
	for(unsigned i = 0; i < (FRAGMENT_DIM_M * FRAGMENT_DIM_N); i += stride) {
		dst_ptr[i + unique_id] = cutf::type::cast<DST_T>(src_ptr[i + unique_id]);
	}
}

template <class T, class U_T>
__device__ void make_h(
		T* const h_ptr, const unsigned m,
		const U_T* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	for(unsigned k = 0; k < FRAGMENT_DIM_M; k+= 2) {
		const auto x = k + lane;
		float tmp = 0.0f;
		if(x == y) {
			tmp = 1.0f;
		}
		if(x < m && y < m)
			tmp -= 2.0f * cutf::type::cast<float>(u_ptr[y]) * cutf::type::cast<float>(u_ptr[x]) / norm2_u_1;

		h_ptr[x * FRAGMENT_DIM_M + y] = cutf::type::cast<T>(tmp);
	}
}

template <class T>
__device__ void make_h_tc32(
		half* const h_ptr, const unsigned m,
		T* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> u_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> ut_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> h_frag_0, h_frag_1, i_frag;

	nvcuda::wmma::fill_fragment(h_frag_0, cutf::type::cast<half>(0.0f));
	nvcuda::wmma::fill_fragment(h_frag_1, cutf::type::cast<half>(0.0f));

	const auto alpha = 2.0f / norm2_u_1;
	mtk::wmma::load_vector_sync(u_frag, u_ptr + lane * 16, alpha);

	mtk::wmma::make_identity_matrix(i_frag);


	mtk::wmma::load_vector_sync(ut_frag, u_ptr);
	nvcuda::wmma::mma_sync(h_frag_0, u_frag, ut_frag, h_frag_0);

	mtk::wmma::load_vector_sync(ut_frag, u_ptr + 16);
	nvcuda::wmma::mma_sync(h_frag_1, u_frag, ut_frag, h_frag_1);

	if(lane == 0) {
		for(unsigned i = 0; i < i_frag.num_elements; i++) {
			h_frag_0.x[i] = i_frag.x[i] - h_frag_0.x[i];
			h_frag_1.x[i] = - h_frag_1.x[i];
		}
	} else {
		for(unsigned i = 0; i < i_frag.num_elements; i++) {
			h_frag_0.x[i] = - h_frag_0.x[i];
			h_frag_1.x[i] = i_frag.x[i] - h_frag_1.x[i];
		}
	}

	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16, h_frag_0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16 + FRAGMENT_DIM_M * 16, h_frag_1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

__device__ void make_h_tc32_refine(
		float* const h_ptr, const unsigned m,
		float* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> u_frag, u_diff_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> ut_frag_0, ut_frag_1, ut_diff_frag_0, ut_diff_frag_1;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> h_frag_0, h_frag_1, i_frag;

	nvcuda::wmma::fill_fragment(h_frag_0, cutf::type::cast<half>(0.0f));
	nvcuda::wmma::fill_fragment(h_frag_1, cutf::type::cast<half>(0.0f));
	
	mtk::wmma::make_identity_matrix(i_frag);

	const auto alpha = cutf::math::sqrt(2.0f / norm2_u_1);
	if (lane == 0) {
		u_ptr[unique_id] *= alpha;
	}
	__syncthreads();

	// load original u
	mtk::wmma::load_vector_sync(u_frag, u_ptr + lane * 16);
	mtk::wmma::load_vector_sync(ut_frag_0, u_ptr);
	mtk::wmma::load_vector_sync(ut_frag_1, u_ptr + 16);

	// compute diff
	if (lane == 0) {
		u_ptr[unique_id] -= cutf::type::cast<float>(cutf::type::cast<half>(u_ptr[unique_id]));
	}
	__syncthreads();

	// load diff u
	mtk::wmma::load_vector_sync(u_diff_frag, u_ptr + lane * 16);
	mtk::wmma::load_vector_sync(ut_diff_frag_0, u_ptr);
	mtk::wmma::load_vector_sync(ut_diff_frag_1, u_ptr + 16);

	nvcuda::wmma::mma_sync(h_frag_0, u_frag, ut_frag_0, h_frag_0);
	nvcuda::wmma::mma_sync(h_frag_1, u_frag, ut_frag_1, h_frag_1);
	nvcuda::wmma::mma_sync(h_frag_0, u_diff_frag, ut_frag_0, h_frag_0);
	nvcuda::wmma::mma_sync(h_frag_1, u_diff_frag, ut_frag_1, h_frag_1);
	nvcuda::wmma::mma_sync(h_frag_0, u_frag, ut_diff_frag_0, h_frag_0);
	nvcuda::wmma::mma_sync(h_frag_1, u_frag, ut_diff_frag_1, h_frag_1);

	if(lane == 0) {
		for(unsigned i = 0; i < i_frag.num_elements; i++) {
			h_frag_0.x[i] = i_frag.x[i] - h_frag_0.x[i];
			h_frag_1.x[i] = - h_frag_1.x[i];
		}
	} else {
		for(unsigned i = 0; i < i_frag.num_elements; i++) {
			h_frag_0.x[i] = - h_frag_0.x[i];
			h_frag_1.x[i] = i_frag.x[i] - h_frag_1.x[i];
		}
	}

	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16, h_frag_0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16 + FRAGMENT_DIM_M * 16, h_frag_1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <class T>
__device__ void make_h_tc16(
		half* const h_ptr, const unsigned m,
		T* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> u_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> ut_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> h_frag_0, h_frag_1, i_frag;

	nvcuda::wmma::fill_fragment(h_frag_0, cutf::type::cast<half>(0.0f));
	nvcuda::wmma::fill_fragment(h_frag_1, cutf::type::cast<half>(0.0f));

	mtk::wmma::make_identity_matrix(i_frag);

	const auto alpha = cutf::math::sqrt(2.0f / norm2_u_1);

	if(lane == 0) {
		u_ptr[unique_id] *= alpha;
	}
	__syncthreads();

	mtk::wmma::load_vector_sync(u_frag, u_ptr + lane * 16);
	mtk::wmma::load_vector_sync(ut_frag, u_ptr);
	nvcuda::wmma::mma_sync(h_frag_0, u_frag, ut_frag, h_frag_0);

	mtk::wmma::load_vector_sync(ut_frag, u_ptr + 16);
	nvcuda::wmma::mma_sync(h_frag_1, u_frag, ut_frag, h_frag_1);

	if(lane == 0) {
		for(unsigned i = 0; i < i_frag.num_elements; i++) {
			h_frag_0.x[i] = i_frag.x[i] - h_frag_0.x[i];
			h_frag_1.x[i] = - h_frag_1.x[i];
		}
	} else {
		for(unsigned i = 0; i < i_frag.num_elements; i++) {
			h_frag_0.x[i] = - h_frag_0.x[i];
			h_frag_1.x[i] = i_frag.x[i] - h_frag_1.x[i];
		}
	}

	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16, h_frag_0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16 + FRAGMENT_DIM_M * 16, h_frag_1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

__device__ void update_qr_f32tc(
		float* const q32_ptr, float* const r32_ptr,
		const half* const q16_ptr, const half* const r16_ptr,
		half* const h16_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
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

__device__ void update_qr_f32tc_refine(
		float* const q32_ptr, float* const r32_ptr,
		half* const q16_ptr, half* const r16_ptr,
		float* const h32_ptr, half* const h16_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h16_0_frag, h16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> r16_0_frag, r16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> q16_0_frag, q16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h16_0_diff_frag, h16_1_diff_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> r16_0_diff_frag, r16_1_diff_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> q16_0_diff_frag, q16_1_diff_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> r32_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> q32_0_frag, q32_1_frag;

	nvcuda::wmma::fill_fragment(r32_frag, 0.0f);
	nvcuda::wmma::fill_fragment(q32_0_frag, 0.0f);
	nvcuda::wmma::fill_fragment(q32_1_frag, 0.0f);

	// load h
	nvcuda::wmma::load_matrix_sync(h16_0_frag, h16_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(h16_1_frag, h16_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// load h diff
	mtk::matrix_operation::diff32x16_2w(h16_ptr, h32_ptr, h16_ptr, unique_id);
	mtk::matrix_operation::diff32x16_2w(h16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N,
			h32_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N,
			h16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N,
			unique_id);
	nvcuda::wmma::load_matrix_sync(h16_0_diff_frag, h16_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(h16_1_diff_frag, h16_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);

	/*  Q 0 */
	// load q
	nvcuda::wmma::load_matrix_sync(q16_0_frag, q16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_frag, q16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_frag, q16_0_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_frag, q16_1_frag, q32_0_frag);
	// load q diff
	mtk::matrix_operation::diff32x16_2w(q16_ptr, q32_ptr, q16_ptr, unique_id);
	nvcuda::wmma::load_matrix_sync(q16_0_diff_frag, q16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_diff_frag, q16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// diff mma
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_diff_frag, q16_0_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_frag, q16_0_diff_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_diff_frag, q16_1_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_frag, q16_1_diff_frag, q32_0_frag);

	/*  Q 1 */
	// load q
	nvcuda::wmma::load_matrix_sync(q16_0_frag, q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_frag, q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_frag, q16_0_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_frag, q16_1_frag, q32_1_frag);
	// load q diff
	mtk::matrix_operation::diff32x16_2w(q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
	nvcuda::wmma::load_matrix_sync(q16_0_diff_frag, q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_diff_frag, q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// diff mma
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_diff_frag, q16_0_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_frag, q16_0_diff_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_diff_frag, q16_1_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_frag, q16_1_diff_frag, q32_1_frag);

	/*  R */
	// load r
	nvcuda::wmma::load_matrix_sync(r16_0_frag, r16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(r16_1_frag, r16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(r32_frag, h16_0_frag, r16_0_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_1_frag, r16_1_frag, r32_frag);
	// load r diff
	mtk::matrix_operation::diff32x16_2w(r16_ptr, r32_ptr, r16_ptr, unique_id);
	nvcuda::wmma::load_matrix_sync(r16_0_diff_frag, r16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(r16_1_diff_frag, r16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// diff mma
	nvcuda::wmma::mma_sync(r32_frag, h16_0_diff_frag, r16_0_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_0_frag, r16_0_diff_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_1_diff_frag, r16_1_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_1_frag, r16_1_diff_frag, r32_frag);

	// store
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_N, q32_0_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_1_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(r32_ptr + lane * FRAGMENT_DIM_N, r32_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

__device__ void update_qr_f16tc(
		half* const q16_ptr, half* const r16_ptr,
		half* const h16_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h16_0_frag, h16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> r16_0_frag, r16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> q16_0_frag, q16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> r32_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> q32_0_frag, q32_1_frag;

	nvcuda::wmma::fill_fragment(r32_frag, cutf::type::cast<half>(0.0f));
	nvcuda::wmma::fill_fragment(q32_0_frag, cutf::type::cast<half>(0.0f));
	nvcuda::wmma::fill_fragment(q32_1_frag, cutf::type::cast<half>(0.0f));

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
	__syncthreads();
	nvcuda::wmma::store_matrix_sync(q16_ptr + lane * FRAGMENT_DIM_N, q32_0_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(q16_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_1_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(r16_ptr + lane * FRAGMENT_DIM_N, r32_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <class T>
__device__ void update_qr(
		T* const out_q_ptr, T* const out_r_ptr,
		const T* const in_q_ptr, const T* const in_r_ptr,
		T* const h_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;

	/* mma q 0 */
	mtk::gemm_core16x16<T, 1>(
		out_q_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M,
		in_q_ptr, FRAGMENT_DIM_M,
		unique_id & 0x1f);
	mtk::gemm_core16x16<T, 1>(
		out_q_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		h_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		in_q_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		unique_id & 0x1f);

	/* mma q 1 */
	mtk::gemm_core16x16<T, 1>(
		out_q_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M,
		in_q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		unique_id & 0x1f);
	mtk::gemm_core16x16<T, 1>(
		out_q_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		h_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		in_q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		unique_id & 0x1f);

	/*  R */
	mtk::gemm_core16x16<T, 1>(
			out_r_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M,
			in_r_ptr, FRAGMENT_DIM_M,
			unique_id & 0x1f);
	mtk::gemm_core16x16<T, 1>(
			out_r_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			h_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			in_r_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			unique_id & 0x1f);
	__syncthreads();
}

__device__ void qr32x16_f32tc_refine_core(
		float* const q32_ptr, float* const r32_ptr,
		half* const q16_ptr, half* const r16_ptr,
		float* const u32_ptr,
		float* const h32_ptr, half* h16_ptr,
		const unsigned m, const unsigned n,
		const unsigned tid
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto unique_id = tid & 0x3f;
	for(unsigned k = 0; k < n; k++) {
		debug_func(
				unique_id,
				[&k]() {printf("/* -------- %u ---------\n", k);}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&r32_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r32_ptr, m, n, "R");}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&q32_ptr, &m]() {mtk::utils::print_matrix_32x16(q32_ptr, m, m, "Q");}
				);
		debug_func(0, []() {__syncthreads();});
		// copy u
		// TODO ; 0埋めとデータロードを異なるwarpでできないか検証
#ifdef MEASURE_CLOCK
		const auto t1 = clock64();
#endif
		if(unique_id < FRAGMENT_DIM_M) {
			u32_ptr[unique_id] = 0.0f;
			if(unique_id >= k && unique_id < m) {
				u32_ptr[unique_id] = r32_ptr[FRAGMENT_DIM_M * k + unique_id];
			}
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u32_ptr, &m]() {mtk::utils::print_matrix(u32_ptr, 1, m, "u");}
				);
		// compute |u|
		// TODO : どうせ0埋めされているなら32個で和をとってしまってもいい気がするので検証
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t2 = clock64();
#endif
		const auto norm_u_0 = cutf::math::sqrt(get_norm2_32(u32_ptr, m, unique_id & 0x1f));
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t3 = clock64();
#endif
		debug_func(
				unique_id,
				[&norm_u_0]() {printf("norm_u_0 = %.5f\n", norm_u_0);}
				);
		// update u
		if(unique_id == k) {
			u32_ptr[unique_id] += cutf::math::sign(u32_ptr[unique_id]) * norm_u_0;
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u32_ptr, &m]() {mtk::utils::print_matrix(u32_ptr, 1, m, "u`");}
				);
		// recompute |u|
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t4 = clock64();
#endif
		const auto norm2_u_1 = get_norm2_32(u32_ptr, m, unique_id & 0x1f);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t5 = clock64();
#endif
		debug_func(
				unique_id,
				[&norm2_u_1]() {printf("norm_u_1^2 = %.5f\n", norm2_u_1);}
				);
		// compute h
		make_h_tc32_refine(
				h32_ptr, m,
				u32_ptr, norm2_u_1,
				unique_id
				);
		debug_func(
				unique_id,
				[&h16_ptr, &m]() {mtk::utils::print_matrix_32x16(h16_ptr, m, m, "H");}
				);
#ifdef MEASURE_CLOCK
		const auto t6 = clock64();
#endif
		// copy f32 to f16
		copy_32x16(r16_ptr, r32_ptr, unique_id);
		copy_32x16(q16_ptr, q32_ptr, unique_id);
		copy_32x16(q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
		copy_32x16(h16_ptr, h32_ptr, unique_id);
		copy_32x16(h16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, h32_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
#ifdef MEASURE_CLOCK
		const auto t7 = clock64();
#endif
		debug_func(
				unique_id,
				[&r16_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r16_ptr, 32, 16, "R (before update)");}
				);
		debug_func(
				unique_id,
				[&q16_ptr, &m]() {mtk::utils::print_matrix_32x16(q16_ptr, 32, 32, "Q (before update)");}
				);
		__syncthreads();
		// update q, r
		update_qr_f32tc_refine(
				q32_ptr, r32_ptr,
				q16_ptr, r16_ptr,
				h32_ptr, h16_ptr,
				unique_id
				);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t8 = clock64();
		if(tid == 0)
			printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,0\n",
					t2 - t1,
					t3 - t2,
					t4 - t3,
					t5 - t4,
					t6 - t5,
					t7 - t6,
					t8 - t7);
#endif
	}
}

__device__ void qr32x16_f32tc_core(
		float* const q32_ptr, float* const r32_ptr,
		half* const q16_ptr, half* const r16_ptr,
		float* const u32_ptr, half* h16_ptr,
		const unsigned m, const unsigned n,
		const unsigned tid
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto unique_id = tid & 0x3f;
	for(unsigned k = 0; k < n; k++) {
		debug_func(
				unique_id,
				[&k]() {printf("/* -------- %u ---------\n", k);}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&r32_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r32_ptr, m, n, "R");}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&q32_ptr, &m]() {mtk::utils::print_matrix_32x16(q32_ptr, m, m, "Q");}
				);
		debug_func(0, []() {__syncthreads();});
		// copy u
		// TODO ; 0埋めとデータロードを異なるwarpでできないか検証
#ifdef MEASURE_CLOCK
		const auto t1 = clock64();
#endif
		if(unique_id < FRAGMENT_DIM_M) {
			u32_ptr[unique_id] = 0.0f;
			if(unique_id >= k && unique_id < m) {
				u32_ptr[unique_id] = r32_ptr[FRAGMENT_DIM_M * k + unique_id];
			}
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u32_ptr, &m]() {mtk::utils::print_matrix(u32_ptr, 1, m, "u");}
				);
		// compute |u|
		// TODO : どうせ0埋めされているなら32個で和をとってしまってもいい気がするので検証
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t2 = clock64();
#endif
		const auto norm_u_0 = cutf::math::sqrt(get_norm2_32(u32_ptr, m, unique_id & 0x1f));
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t3 = clock64();
#endif
		debug_func(
				unique_id,
				[&norm_u_0]() {printf("norm_u_0 = %.5f\n", norm_u_0);}
				);
		// update u
		if(unique_id == k) {
			u32_ptr[unique_id] += cutf::math::sign(u32_ptr[unique_id]) * norm_u_0;
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u32_ptr, &m]() {mtk::utils::print_matrix(u32_ptr, 1, m, "u`");}
				);
		// recompute |u|
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t4 = clock64();
#endif
		const auto norm2_u_1 = get_norm2_32(u32_ptr, m, unique_id & 0x1f);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t5 = clock64();
#endif
		debug_func(
				unique_id,
				[&norm2_u_1]() {printf("norm_u_1^2 = %.5f\n", norm2_u_1);}
				);
		// compute h
		make_h_tc32(
				h16_ptr, m,
				u32_ptr, norm2_u_1,
				unique_id
				);
		debug_func(
				unique_id,
				[&h16_ptr, &m]() {mtk::utils::print_matrix_32x16(h16_ptr, m, m, "H");}
				);
#ifdef MEASURE_CLOCK
		const auto t6 = clock64();
#endif
		// copy f32 to f16
		copy_32x16(r16_ptr, r32_ptr, unique_id);
		copy_32x16(q16_ptr, q32_ptr, unique_id);
		copy_32x16(q16_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
#ifdef MEASURE_CLOCK
		const auto t7 = clock64();
#endif
		debug_func(
				unique_id,
				[&r16_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r16_ptr, 32, 16, "R (before update)");}
				);
		debug_func(
				unique_id,
				[&q16_ptr, &m]() {mtk::utils::print_matrix_32x16(q16_ptr, 32, 32, "Q (before update)");}
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
#ifdef MEASURE_CLOCK
		const auto t8 = clock64();
		if(tid == 0)
			printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,0\n",
					t2 - t1,
					t3 - t2,
					t4 - t3,
					t5 - t4,
					t6 - t5,
					t7 - t6,
					t8 - t7);
#endif
	}
}

__device__ void qr32x16_f16tc_core(
		half* const q16_ptr, half* const r16_ptr,
		half* const u16_ptr, half* h16_ptr,
		const unsigned m, const unsigned n,
		const unsigned tid
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto unique_id = tid & 0x3f;
	for(unsigned k = 0; k < n; k++) {
		debug_func(
				unique_id,
				[&k]() {printf("/* -------- %u ---------\n", k);}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&r16_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r16_ptr, m, n, "R");}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&q16_ptr, &m]() {mtk::utils::print_matrix_32x16(q16_ptr, m, m, "Q");}
				);
		debug_func(0, []() {__syncthreads();});
#ifdef MEASURE_CLOCK
		const auto t1 = clock64();
#endif
		// copy u
		// TODO ; 0埋めとデータロードを異なるwarpでできないか検証
		if(unique_id < FRAGMENT_DIM_M) {
			u16_ptr[unique_id] = cutf::type::cast<half>(0.0f);
			if(unique_id >= k && unique_id < m) {
				u16_ptr[unique_id] = r16_ptr[FRAGMENT_DIM_M * k + unique_id];
			}
		}
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t2 = clock64();
#endif
		debug_func(
				unique_id,
				[&u16_ptr, &m]() {mtk::utils::print_matrix(u16_ptr, 1, m, "u");}
				);
		// compute |u|
		// TODO : どうせ0埋めされているなら32個で和をとってしまってもいい気がするので検証
		const auto norm_u_0 = cutf::type::cast<half>(cutf::math::sqrt(get_norm2_32(u16_ptr, m, unique_id & 0x1f)));
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t3 = clock64();
#endif
		debug_func(
				unique_id,
				[&norm_u_0]() {printf("norm_u_0 = %.5f\n", cutf::type::cast<float>(norm_u_0));}
				);
		// update u
		if(unique_id == k) {
			u16_ptr[unique_id] += cutf::math::sign(u16_ptr[unique_id]) * norm_u_0;
		}
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t4 = clock64();
#endif
		debug_func(
				unique_id,
				[&u16_ptr, &m]() {mtk::utils::print_matrix(u16_ptr, 1, m, "u`");}
				);
		// recompute |u|
		const auto norm2_u_1 = get_norm2_32(u16_ptr, m, unique_id & 0x1f);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t5 = clock64();
#endif
		debug_func(
				unique_id,
				[&norm2_u_1]() {printf("norm_u_1^2 = %.5f\n", cutf::type::cast<float>(norm2_u_1));}
				);
		// compute h
		make_h_tc16(
				h16_ptr, m,
				u16_ptr, norm2_u_1,
				unique_id
				);
		debug_func(
				unique_id,
				[&h16_ptr, &m]() {mtk::utils::print_matrix_32x16(h16_ptr, m, m, "H");}
				);
		debug_func(
				unique_id,
				[&r16_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r16_ptr, 32, 16, "R (before update)");}
				);
		debug_func(
				unique_id,
				[&q16_ptr, &m]() {mtk::utils::print_matrix_32x16(q16_ptr, 32, 32, "Q (before update)");}
				);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t6 = clock64();
#endif
		// update q, r
		update_qr_f16tc(
				q16_ptr, r16_ptr,
				h16_ptr,
				unique_id
				);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t7 = clock64();
		if(tid == 0)
			printf("%lu,%lu,%lu,%lu,%lu,0,%lu,0\n",
					t2 - t1,
					t3 - t2,
					t4 - t3,
					t5 - t4,
					t6 - t5,
					t7 - t6);
#endif
	}
}

template <class T>
__device__ void qr32x16_core(
		T* const q_ptr0, T* const r_ptr0,
		T* const q_ptr1, T* const r_ptr1,
		T* const u_ptr, T* h_ptr,
		const unsigned m, const unsigned n,
		const unsigned tid
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto unique_id = tid & 0x3f;
	for(unsigned k = 0; k < n; k++) {
		debug_func(
				unique_id,
				[&k]() {printf("/* -------- %u ---------\n", k);}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&r_ptr0, &m, &n]() {mtk::utils::print_matrix_32x16(r_ptr0, m, n, "R");}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&q_ptr0, &m]() {mtk::utils::print_matrix_32x16(q_ptr0, m, m, "Q");}
				);
		debug_func(0, []() {__syncthreads();});
#ifdef MEASURE_CLOCK
		const auto t1 = clock64();
#endif
		// copy u
		// TODO ; 0埋めとデータロードを異なるwarpでできないか検証
		if(unique_id < FRAGMENT_DIM_M) {
			u_ptr[unique_id] = 0.0f;
			if(unique_id >= k && unique_id < m) {
				u_ptr[unique_id] = r_ptr0[FRAGMENT_DIM_M * k + unique_id];
			}
		}
#ifdef MEASURE_CLOCK
		const auto t2 = clock64();
#endif
		__syncthreads();
		debug_func(
				unique_id,
				[&u_ptr, &m]() {mtk::utils::print_matrix(u_ptr, 1, m, "u");}
				);
		// compute |u|
		// TODO : どうせ0埋めされているなら32個で和をとってしまってもいい気がするので検証
		const auto norm_u_0 = cutf::type::cast<T>(cutf::math::sqrt(get_norm2_32(u_ptr, m, unique_id & 0x1f)));
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t3 = clock64();
#endif
		debug_func(
				unique_id,
				[&norm_u_0]() {printf("norm_u_0 = %.5f\n", cutf::type::cast<float>(norm_u_0));}
				);
		// update u
		if(unique_id == k) {
			u_ptr[unique_id] += cutf::math::sign(u_ptr[unique_id]) * norm_u_0;
		}
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t4 = clock64();
#endif
		debug_func(
				unique_id,
				[&u_ptr, &m]() {mtk::utils::print_matrix(u_ptr, 1, m, "u`");}
				);
		// recompute |u|
		const auto norm2_u_1 = get_norm2_32(u_ptr, m, unique_id & 0x1f);
		debug_func(
				unique_id,
				[&norm2_u_1]() {printf("norm_u_1^2 = %.5f\n", cutf::type::cast<float>(norm2_u_1));}
				);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t5 = clock64();
#endif
		// compute h
		make_h(
				h_ptr, m,
				u_ptr, norm2_u_1,
				unique_id
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&h_ptr, &m]() {mtk::utils::print_matrix_32x16(h_ptr, m, m, "H");}
				);
		debug_func(
				unique_id,
				[&r_ptr0, &m, &n]() {mtk::utils::print_matrix_32x16(r_ptr0, 32, 16, "R (before update)");}
				);
		debug_func(
				unique_id,
				[&q_ptr0, &m]() {mtk::utils::print_matrix_32x16(q_ptr0, 32, 32, "Q (before update)");}
				);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t6 = clock64();
#endif
		// initialize *1
		mtk::matrix_operation::make_zero_matrix<T, FRAGMENT_DIM_M, FRAGMENT_DIM_M>(q_ptr1, tid);
		mtk::matrix_operation::make_zero_matrix<T, FRAGMENT_DIM_M, FRAGMENT_DIM_N>(r_ptr1, tid);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t7 = clock64();
		__syncthreads();
#endif
		// update q, r
		update_qr<T>(
				q_ptr1, r_ptr1,
				q_ptr0, r_ptr0,
				h_ptr,
				unique_id
				);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t8 = clock64();
#endif
		// copy f32 to f16
		copy_32x16(r_ptr0, r_ptr1, unique_id);
		copy_32x16(q_ptr0, q_ptr1, unique_id);
		copy_32x16(q_ptr0 + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q_ptr1 + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
		__syncthreads();
#ifdef MEASURE_CLOCK
		const auto t9 = clock64();
		if(tid == 0)
			printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n",
					t2 - t1,
					t3 - t2,
					t4 - t3,
					t5 - t4,
					t6 - t5,
					t7 - t6,
					t8 - t7,
					t9 - t8);
#endif
	}
}

__global__ void qr32x16_f32tc_refine_batched_kernel(
		float* const q32_ptr,
		float* const r32_ptr,
		const float* const a32_ptr,
		const std::size_t m,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 2;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / (warp_size * 2);
	const auto shared_memory_id = matrix_id & (max_batch_size_per_block - 1);
	if(matrix_id >= batch_size) return;

	__shared__ float shared_q32[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ float shared_r32[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ float shared_h32[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ half shared_q16[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ half shared_r16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_h16[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ float shared_u32[FRAGMENT_DIM_M * max_batch_size_per_block];

	const auto shared_q32_ptr = shared_q32 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r32_ptr = shared_r32 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_h32_ptr = shared_h32 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_q16_ptr = shared_q16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r16_ptr = shared_r16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_h16_ptr = shared_h16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_u32_ptr = shared_u32 + shared_memory_id * FRAGMENT_DIM_M;

	const auto sub_a_position = a_start_position[matrix_id];
	const auto sub_a_m = a_start_position[matrix_id + 1] - sub_a_position;

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r32_ptr, sub_a_m, n,
			a32_ptr, sub_a_position, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<float, FRAGMENT_DIM_M>(
			shared_q32_ptr,
			tid
			);

	// qr core
	qr32x16_f32tc_refine_core(
			shared_q32_ptr, shared_r32_ptr,
			shared_q16_ptr, shared_r16_ptr,
			shared_u32_ptr,
			shared_h32_ptr, shared_h16_ptr,
			sub_a_m, n,
			tid
			);

	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q32_ptr, sub_a_position, m,
			shared_q32_ptr, n, sub_a_m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r32_ptr, n * matrix_id, n * batch_size,
			shared_r32_ptr, n, n,
			tid
			);
}

template <class Q_T, class R_T>
__global__ void qr32x16_f32tc_batched_kernel(
		Q_T* const q32_ptr,
		R_T* const r32_ptr,
		const float* const a32_ptr,
		const std::size_t m,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / (warp_size * 2);
	const auto shared_memory_id = matrix_id & (max_batch_size_per_block - 1);
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
	mtk::matrix_copy::g2s32x16_2w(
			shared_r32_ptr, sub_a_m, n,
			a32_ptr, sub_a_position, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<float, FRAGMENT_DIM_M>(
			shared_q32_ptr,
			tid
			);

	// qr core
	qr32x16_f32tc_core(
			shared_q32_ptr, shared_r32_ptr,
			shared_q16_ptr, shared_r16_ptr,
			shared_u32_ptr,
			shared_h16_ptr,
			sub_a_m, n,
			tid
			);

	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q32_ptr, sub_a_position, m,
			shared_q32_ptr, n, sub_a_m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r32_ptr, n * matrix_id, n * batch_size,
			shared_r32_ptr, n, n,
			tid
			);
}

__global__ void qr32x16_f16tc_batched_kernel(
		half* const q16_ptr,
		half* const r16_ptr,
		const half* const a16_ptr,
		const std::size_t m,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / (warp_size * 2);
	const auto shared_memory_id = matrix_id & (max_batch_size_per_block - 1);
	if(matrix_id >= batch_size) return;

	__shared__ half shared_q16[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ half shared_r16[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ half shared_h16[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ half shared_u16[FRAGMENT_DIM_M * max_batch_size_per_block];

	const auto shared_q16_ptr = shared_q16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r16_ptr = shared_r16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_h16_ptr = shared_h16 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_u16_ptr = shared_u16 + shared_memory_id * FRAGMENT_DIM_M;

	const auto sub_a_position = a_start_position[matrix_id];
	const auto sub_a_m = a_start_position[matrix_id + 1] - sub_a_position;

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r16_ptr, sub_a_m, n,
			a16_ptr, sub_a_position, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<half, FRAGMENT_DIM_M>(
			shared_q16_ptr,
			tid
			);

	// qr core
	qr32x16_f16tc_core(
			shared_q16_ptr, shared_r16_ptr,
			shared_u16_ptr, shared_h16_ptr,
			sub_a_m, n,
			tid
			);

	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q16_ptr, sub_a_position, m,
			shared_q16_ptr, n, sub_a_m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r16_ptr, n * matrix_id, n * batch_size,
			shared_r16_ptr, n, n,
			tid
			);
}

template <class Q_T, class R_T>
__global__ void qr32x16_f32tc_kernel(
		Q_T* const q32_ptr,
		R_T* const r32_ptr,
		const float* const a32_ptr,
		const unsigned m,
		const unsigned n
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float shared_q32[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ float shared_r32[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ half shared_q16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ half shared_r16[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ half shared_h16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ float shared_u32[FRAGMENT_DIM_M];

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r32, m, n,
			a32_ptr, 0, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<float, FRAGMENT_DIM_M>(
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
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q32_ptr, 0, m,
			shared_q32, n, m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r32_ptr, 0, n,
			shared_r32, n, n,
			tid
			);
}

__global__ void qr32x16_f32tc_refine_kernel(
		float* const q32_ptr,
		float* const r32_ptr,
		const float* const a32_ptr,
		const unsigned m,
		const unsigned n
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float shared_q32[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ float shared_r32[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ float shared_h32[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ half shared_q16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ half shared_r16[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ half shared_h16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ float shared_u32[FRAGMENT_DIM_M];

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r32, m, n,
			a32_ptr, 0, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<float, FRAGMENT_DIM_M>(
			shared_q32,
			tid
			);

	// qr core
	qr32x16_f32tc_refine_core(
			shared_q32, shared_r32,
			shared_q16, shared_r16,
			shared_u32,
			shared_h32, shared_h16,
			m, n,
			tid
			);
	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q32_ptr, 0, m,
			shared_q32, n, m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r32_ptr, 0, n,
			shared_r32, n, n,
			tid
			);
}

__global__ void qr32x16_f16tc_kernel(
		half* const q16_ptr,
		half* const r16_ptr,
		const half* const a16_ptr,
		const unsigned m,
		const unsigned n
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ half shared_q16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ half shared_r16[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ half shared_h16[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ half shared_u16[FRAGMENT_DIM_M];

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r16, m, n,
			a16_ptr, 0, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<half, FRAGMENT_DIM_M>(
			shared_q16,
			tid
			);

	// qr core
	qr32x16_f16tc_core(
			shared_q16, shared_r16,
			shared_u16, shared_h16,
			m, n,
			tid
			);
	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q16_ptr, 0, m,
			shared_q16, n, m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r16_ptr, 0, n,
			shared_r16, n, n,
			tid
			);
}

template <class Q_T, class R_T, class A_T>
__global__ void qr32x16_batched_kernel(
		Q_T* const q_ptr,
		R_T* const r_ptr,
		const A_T* const a_ptr,
		const std::size_t m,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = 2;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / (warp_size * 2);
	const auto shared_memory_id = matrix_id % max_batch_size_per_block;
	if(matrix_id >= batch_size) return;

	__shared__ A_T shared_q0[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ A_T shared_r0[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ A_T shared_q1[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ A_T shared_r1[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ A_T shared_h[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ A_T shared_u[FRAGMENT_DIM_M * max_batch_size_per_block];

	const auto shared_q0_ptr = shared_q0 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r0_ptr = shared_r0 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_q1_ptr = shared_q1 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r1_ptr = shared_r1 + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_h_ptr = shared_h + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_u_ptr = shared_u + shared_memory_id * FRAGMENT_DIM_M;

	const auto sub_a_position = a_start_position[matrix_id];
	const auto sub_a_m = a_start_position[matrix_id + 1] - sub_a_position;

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r0_ptr, sub_a_m, n,
			a_ptr, sub_a_position, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<A_T, FRAGMENT_DIM_M>(
			shared_q0_ptr,
			tid
			);

	// qr core
	qr32x16_core<A_T>(
			shared_q0_ptr, shared_r0_ptr,
			shared_q1_ptr, shared_r1_ptr,
			shared_u_ptr, shared_h_ptr,
			sub_a_m, n,
			tid
			);

	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q_ptr, sub_a_position, m,
			shared_q1_ptr, n, sub_a_m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r_ptr, n * matrix_id, n * batch_size,
			shared_r1_ptr, n, n,
			tid
			);
}

template <class Q_T, class R_T, class A_T>
__global__ void qr32x16_kernel(
		Q_T* const q_ptr,
		R_T* const r_ptr,
		const A_T* const a_ptr,
		const unsigned m,
		const unsigned n
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ A_T shared_q0[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ A_T shared_r0[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ A_T shared_q1[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ A_T shared_r1[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ A_T shared_h[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ A_T shared_u[FRAGMENT_DIM_M];

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r0, m, n,
			a_ptr, 0, m,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<A_T, FRAGMENT_DIM_M>(
			shared_q0,
			tid
			);

	// qr core
	qr32x16_core<A_T>(
			shared_q0, shared_r0,
			shared_q1, shared_r1,
			shared_u, shared_h,
			m, n,
			tid
			);
	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q_ptr, 0, m,
			shared_q1, n, m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r_ptr, 0, n,
			shared_r1, n, n,
			tid
			);
}
}

template <bool UseTC, bool Refine, class Q_T, class R_T, class A_T>
void mtk::tcqr::qr32x16_batched(
		Q_T* const q, R_T* const r,
		const A_T* const a, const unsigned int m, const unsigned int n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t max_batch_size_per_block = 2;
	const auto grid_size = (batch_size + max_batch_size_per_block + 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * 2 * warp_size;

	qr32x16_batched_kernel<Q_T, R_T, A_T><<<grid_size, block_size>>>(
			q, r,
			a, m, n,
			batch_size,
			a_start_position
			);
}
template void mtk::tcqr::qr32x16_batched<false, false, float, float, float>(float* const q, float* const r, const float* const a, const unsigned int m, const unsigned int n, const std::size_t batch_size, const unsigned* a_start_position);
template void mtk::tcqr::qr32x16_batched<false, false, half, half, half>(half* const q, half* const r, const half* const a, const unsigned int m, const unsigned int n, const std::size_t batch_size, const unsigned* a_start_position);

template <> void mtk::tcqr::qr32x16_batched<true, false, float, float, float>(
		float* const q, float* const r,
		const float* const a, const unsigned int m, const unsigned int n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto grid_size = (batch_size + max_batch_size_per_block + 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * 2 * warp_size;

	qr32x16_f32tc_batched_kernel<float, float><<<grid_size, block_size>>>(
			q, r,
			a, m, n,
			batch_size,
			a_start_position
			);
}

template <> void mtk::tcqr::qr32x16_batched<true, true, float, float, float>(
		float* const q, float* const r,
		const float* const a, const unsigned int m, const unsigned int n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t max_batch_size_per_block = 2;
	const auto grid_size = (batch_size + max_batch_size_per_block + 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * 2 * warp_size;

	qr32x16_f32tc_refine_batched_kernel<<<grid_size, block_size>>>(
			q, r,
			a, m, n,
			batch_size,
			a_start_position
			);
}

template <> void mtk::tcqr::qr32x16_batched<true, false, half, half, half>(
		half* const q, half* const r,
		const half* const a, const unsigned int m, const unsigned int n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto grid_size = (batch_size + max_batch_size_per_block + 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * 2 * warp_size;

	qr32x16_f16tc_batched_kernel<<<grid_size, block_size>>>(
			q, r,
			a, m, n,
			batch_size,
			a_start_position
			);
}
template <> void mtk::tcqr::qr32x16_batched<true, false, half, float, float>(
		half* const q, float* const r,
		const float* const a, const unsigned int m, const unsigned int n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t max_batch_size_per_block = 4;
	const auto grid_size = (batch_size + max_batch_size_per_block + 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * 2 * warp_size;

	qr32x16_f32tc_batched_kernel<half, float><<<grid_size, block_size>>>(
			q, r,
			a, m, n,
			batch_size,
			a_start_position
			);
}

template <bool UseTC, bool Refine, class Q_T, class R_T, class A_T>
void mtk::tcqr::qr32x16(
		Q_T* const q, R_T* const r,
		const A_T* const a, const unsigned int m, const unsigned int n
		) {
	qr32x16_kernel<Q_T, R_T, A_T><<<1, 2 * warp_size>>>(
			q, r,
			a, m, n
			);
}

template void mtk::tcqr::qr32x16<false, false, float, float, float>(float* const, float* const, const float* const, const unsigned int, const unsigned int);
template void mtk::tcqr::qr32x16<false, false, half, half, half>(half* const, half* const, const half* const, const unsigned int, const unsigned int);

template<> void mtk::tcqr::qr32x16<true, false, half, half, half>(half* const q, half* const r, const half* const a, const unsigned int m, const unsigned int n) {
	qr32x16_f16tc_kernel<<<1, 2 * warp_size>>>(
			q, r,
			a, m, n
			);
}

template<> void mtk::tcqr::qr32x16<true, false, half, float, float>(half* const q, float* const r, const float* const a, const unsigned int m, const unsigned int n) {
	qr32x16_f32tc_kernel<half, float><<<1, 2 * warp_size>>>(
			q, r,
			a, m, n
			);
}

template<> void mtk::tcqr::qr32x16<true, false, float, float, float>(float* const q, float* const r, const float* const a, const unsigned int m, const unsigned int n) {
	qr32x16_f32tc_kernel<float, float><<<1, 2 * warp_size>>>(
			q, r,
			a, m, n
			);
}

template<> void mtk::tcqr::qr32x16<true, true, float, float, float>(float* const q, float* const r, const float* const a, const unsigned int m, const unsigned int n) {
	qr32x16_f32tc_refine_kernel<<<1, 2 * warp_size>>>(
			q, r,
			a, m, n
			);
}
