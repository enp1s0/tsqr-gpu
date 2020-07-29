#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include <cutf/experimental/tf32.hpp>
#include <wmma_extension.hpp>
#include <stdio.h>
#include "tcqr.hpp"
#include "utils.hpp"
#include "matrix_copy.cuh"
#include "matrix_operations.cuh"
#include "gemm_core/gemm_core.cuh"
#include "matmul.hpp"

//#define DEBUG
//#define MEASURE_CLOCK
//#define IMPLICIT_H
//#define THREE_TERMS_CORRECTION

// Defining `EMULATE_TF32` enables `FP32-noTC` to emulate NVIDIA A100 TF32 TensorCore
//#define EMULATE_TF32

// clock : make_u,norm1,update_u,norm2,make_h,mem_init,update_qr,mem_swap
// clock : make_u,norm1,update_u,norm2,update_qr_with_u

namespace {
constexpr unsigned warp_size = 32;

template <mtk::tcqr::compute_mode mode>
struct h_mat_t {using type = void;};
template <> struct h_mat_t<mtk::tcqr::compute_mode::fp32_tc_nocor    > {using type = half;};
template <> struct h_mat_t<mtk::tcqr::compute_mode::fp32_notc        > {using type = float;};
template <> struct h_mat_t<mtk::tcqr::compute_mode::fp32_tc_cor      > {using type = float;};
template <> struct h_mat_t<mtk::tcqr::compute_mode::fp16_notc        > {using type = half;};
template <> struct h_mat_t<mtk::tcqr::compute_mode::fp16_tc_nocor    > {using type = half;};
template <> struct h_mat_t<mtk::tcqr::compute_mode::tf32_tc_nocor_emu> {using type = float;};
template <> struct h_mat_t<mtk::tcqr::compute_mode::tf32_tc_cor_emu  > {using type = float;};

template <mtk::tcqr::compute_mode mode>
constexpr unsigned get_max_batch_size_per_block() {return 4u;}

template <mtk::tcqr::compute_mode>
constexpr mtk::matmul::compute_mode get_matmul_compute_mode();
#define TCQR_GET_MATMUL_COMPUTE_MODE(mode) template<> constexpr mtk::matmul::compute_mode get_matmul_compute_mode<mtk::tcqr::compute_mode::mode>() {return mtk::matmul::compute_mode::mode;}
TCQR_GET_MATMUL_COMPUTE_MODE(fp16_notc        );
TCQR_GET_MATMUL_COMPUTE_MODE(fp32_notc        );
TCQR_GET_MATMUL_COMPUTE_MODE(tf32_tc_cor_emu  );
TCQR_GET_MATMUL_COMPUTE_MODE(tf32_tc_nocor_emu);
TCQR_GET_MATMUL_COMPUTE_MODE(mixed_tc_cor     );

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
	__syncthreads();
}

#ifndef IMPLICIT_H
//
// Generating Householder matrix from the vector `u`
//
// `u_ptr` is not `const` pointer because the values are destoried in `<mtk::tcqr::compute_mode::fp32_tc_cor, float, float>`
template <mtk::tcqr::compute_mode mode, class T, class U_T>
__device__ void make_h(
		T* const h_ptr, const unsigned m,
		U_T* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	const auto uy = 2.0f * cutf::type::cast<float>(u_ptr[y]) / norm2_u_1;
	for(unsigned k = 0; k < FRAGMENT_DIM_M; k += 2) {
		const auto x = k + lane;
		float tmp = 0.0f;
		if(x == y) {
			tmp = 1.0f;
		}
		if(x < m && y < m)
			tmp -= uy * cutf::type::cast<float>(u_ptr[x]);

		h_ptr[x * FRAGMENT_DIM_M + y] = cutf::type::cast<T>(tmp);
	}
}

template <>
__device__ void make_h<mtk::tcqr::compute_mode::fp16_tc_nocor, half, half>(
		half* const h_ptr, const unsigned m,
		half* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
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

	mtk::wmma::load_vector_sync(u_frag, u_ptr + lane * FRAGMENT_DIM_N);
	mtk::wmma::load_vector_sync(ut_frag, u_ptr);
	nvcuda::wmma::mma_sync(h_frag_0, u_frag, ut_frag, h_frag_0);

	mtk::wmma::load_vector_sync(ut_frag, u_ptr + FRAGMENT_DIM_N);
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

	nvcuda::wmma::store_matrix_sync(h_ptr + lane * FRAGMENT_DIM_N, h_frag_0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(h_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, h_frag_1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <>
__device__ void make_h<mtk::tcqr::compute_mode::fp32_tc_nocor, half, float>(
		half* const h_ptr, const unsigned m,
		float* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
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

	mtk::wmma::load_vector_sync(ut_frag, u_ptr + FRAGMENT_DIM_N);
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

	nvcuda::wmma::store_matrix_sync(h_ptr + lane * FRAGMENT_DIM_N, h_frag_0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(h_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, h_frag_1, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <>
__device__ void make_h<mtk::tcqr::compute_mode::fp32_tc_cor, float, float>(
		float* const h_ptr, const unsigned m,
		float* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> u_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> ut_frag_0;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> h_frag_0;

	half* const u16_ptr = reinterpret_cast<half*>(u_ptr);
	half* const du16_ptr = reinterpret_cast<half*>(u_ptr + FRAGMENT_DIM_M / 2);

	const auto alpha = cutf::math::sqrt(2.0f / norm2_u_1);
	if (lane == 0) {
		const float uf = u_ptr[unique_id] * alpha;
		const half uh = cutf::type::cast<half>(uf);
		u16_ptr[unique_id] = uh;
		const half duh = cutf::type::cast<half>(uf - cutf::type::cast<float>(uh));
		du16_ptr[unique_id] = duh;
	}
	__syncthreads();

	// load original u
	mtk::wmma::make_direct_product_fragment_c3(u_frag, u16_ptr + lane * 16, du16_ptr + lane * 16);

	mtk::wmma::make_direct_product_fragment_c3(ut_frag_0, u16_ptr, du16_ptr);
	mtk::wmma::fill_zero(h_frag_0);
	nvcuda::wmma::mma_sync(h_frag_0, u_frag, ut_frag_0, h_frag_0);
	for(unsigned i = 0; i < 8; i++) {
		h_frag_0.x[i] = - h_frag_0.x[i];
	}
	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16, h_frag_0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	mtk::wmma::make_direct_product_fragment_c3(ut_frag_0, u16_ptr + 16, du16_ptr + 16);
	mtk::wmma::fill_zero(h_frag_0);
	nvcuda::wmma::mma_sync(h_frag_0, u_frag, ut_frag_0, h_frag_0);
	for(unsigned i = 0; i < 8; i++) {
		h_frag_0.x[i] = - h_frag_0.x[i];
	}
	nvcuda::wmma::store_matrix_sync(h_ptr + lane * 16 + FRAGMENT_DIM_M * 16, h_frag_0, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	__syncthreads();
	if (unique_id < FRAGMENT_DIM_M) {
		h_ptr[unique_id * (FRAGMENT_DIM_M + 1)] += 1.0f;
	}
}

template <>
__device__ void make_h<mtk::tcqr::compute_mode::tf32_tc_cor_emu, float, float>(
		float* const h_ptr, const unsigned m,
		float* const u_ptr, const float norm2_u_1,
		const unsigned unique_id) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	for(unsigned k = 0; k < FRAGMENT_DIM_M; k += 2) {
		const auto x = k + lane;
		float tmp = 0.0f;
		if(x == y) {
			tmp = 1.0f;
		}
		if(x < m && y < m) {
			const auto y_v = cutf::experimental::tf32::to_tf32(u_ptr[y]);
			const auto x_v = cutf::experimental::tf32::to_tf32(u_ptr[x]);
			const auto y_dv = cutf::experimental::tf32::to_tf32(u_ptr[y] - y_v);
			const auto x_dv = cutf::experimental::tf32::to_tf32(u_ptr[x] - x_v);
			tmp -= 2.0f * (x_dv * y_v + x_v * y_dv + x_v * y_v) / norm2_u_1;
		}

		h_ptr[x * FRAGMENT_DIM_M + y] = tmp;
	}
}

//
// Updating Q and R
//
template <mtk::tcqr::compute_mode mode, class Q_T, class R_T, class H_T>
__device__ void update_qr(
		Q_T* const q_ptr, R_T* const r_ptr,
		H_T* const h_ptr,
		half* const working_memory,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;

	/* mma q 0 */
	mtk::matmul::matmul_core_m16n16k32<get_matmul_compute_mode<mode>()>(
		q_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M,
		q_ptr, FRAGMENT_DIM_M,
		unique_id & 0x1f);

	/* mma q 1 */
	mtk::matmul::matmul_core_m16n16k32<get_matmul_compute_mode<mode>()>(
		q_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M,
		q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
		unique_id & 0x1f);

	/*  R */
	mtk::matmul::matmul_core_m16n16k32<get_matmul_compute_mode<mode>()>(
			r_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M,
			h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M,
			r_ptr, FRAGMENT_DIM_M,
			unique_id & 0x1f);
	__syncthreads();
}

template <>
__device__ void update_qr<mtk::tcqr::compute_mode::fp32_tc_nocor, float, float, half>(
		float* const q_ptr, float* const r_ptr,
		half* const h_ptr,
		half* const working_memory,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h_0_frag, h_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> in_r_0_frag, in_r_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> in_q_0_frag, in_q_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> out_r_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> out_q_0_frag, out_q_1_frag;

	nvcuda::wmma::fill_fragment(out_r_frag, 0.0f);
	nvcuda::wmma::fill_fragment(out_q_0_frag, 0.0f);
	nvcuda::wmma::fill_fragment(out_q_1_frag, 0.0f);

	// load h
	nvcuda::wmma::load_matrix_sync(h_0_frag, h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(h_1_frag, h_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);

	__syncthreads();
	copy_32x16(working_memory, q_ptr, unique_id);
	__syncthreads();

	/*  Q 0 */
	// load q
	nvcuda::wmma::load_matrix_sync(in_q_0_frag, working_memory, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_q_1_frag, working_memory + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_q_0_frag, h_0_frag, in_q_0_frag, out_q_0_frag);
	nvcuda::wmma::mma_sync(out_q_0_frag, h_1_frag, in_q_1_frag, out_q_0_frag);
	/*  Q 1 */

	__syncthreads();
	copy_32x16(working_memory, q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
	__syncthreads();
	// load q
	nvcuda::wmma::load_matrix_sync(in_q_0_frag, working_memory, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_q_1_frag, working_memory + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_q_1_frag, h_0_frag, in_q_0_frag, out_q_1_frag);
	nvcuda::wmma::mma_sync(out_q_1_frag, h_1_frag, in_q_1_frag, out_q_1_frag);
	/*  R */
	__syncthreads();
	copy_32x16(working_memory, r_ptr, unique_id);
	__syncthreads();
	// load r
	nvcuda::wmma::load_matrix_sync(in_r_0_frag, working_memory, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_r_1_frag, working_memory + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_r_frag, h_0_frag, in_r_0_frag, out_r_frag);
	nvcuda::wmma::mma_sync(out_r_frag, h_1_frag, in_r_1_frag, out_r_frag);

	// store
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N, out_q_0_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, out_q_1_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(r_ptr + lane * FRAGMENT_DIM_N, out_r_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <>
__device__ void update_qr<mtk::tcqr::compute_mode::fp32_tc_nocor, half, float, half>(
		half* const q_ptr, float* const r_ptr,
		half* const h_ptr,
		half* const working_memory,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h_0_frag, h_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> in_r_0_frag, in_r_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> in_q_0_frag, in_q_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> out_r_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> out_q_0_frag, out_q_1_frag;

	nvcuda::wmma::fill_fragment(out_r_frag, 0.0f);
	nvcuda::wmma::fill_fragment(out_q_0_frag, 0.0f);
	nvcuda::wmma::fill_fragment(out_q_1_frag, 0.0f);

	// load h
	nvcuda::wmma::load_matrix_sync(h_0_frag, h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(h_1_frag, h_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);

	/*  Q 0 */
	// load q
	nvcuda::wmma::load_matrix_sync(in_q_0_frag, q_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_q_1_frag, q_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_q_0_frag, h_0_frag, in_q_0_frag, out_q_0_frag);
	nvcuda::wmma::mma_sync(out_q_0_frag, h_1_frag, in_q_1_frag, out_q_0_frag);
	/*  Q 1 */
	// load q
	nvcuda::wmma::load_matrix_sync(in_q_0_frag, q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_q_1_frag, q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_q_1_frag, h_0_frag, in_q_0_frag, out_q_1_frag);
	nvcuda::wmma::mma_sync(out_q_1_frag, h_1_frag, in_q_1_frag, out_q_1_frag);
	/*  R */
	__syncthreads();
	copy_32x16(working_memory, r_ptr, unique_id);
	__syncthreads();
	// load r
	nvcuda::wmma::load_matrix_sync(in_r_0_frag, working_memory, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_r_1_frag, working_memory + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_r_frag, h_0_frag, in_r_0_frag, out_r_frag);
	nvcuda::wmma::mma_sync(out_r_frag, h_1_frag, in_r_1_frag, out_r_frag);

	// store
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N, out_q_0_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, out_q_1_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(r_ptr + lane * FRAGMENT_DIM_N, out_r_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <>
__device__ void update_qr<mtk::tcqr::compute_mode::fp16_tc_nocor, half, half, half>(
		half* const q_ptr, half* const r_ptr,
		half* const h_ptr,
		half* const working_memory,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h_0_frag, h_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> in_r_0_frag, in_r_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> in_q_0_frag, in_q_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> out_r_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> out_q_0_frag, out_q_1_frag;

	nvcuda::wmma::fill_fragment(out_r_frag, 0.0f);
	nvcuda::wmma::fill_fragment(out_q_0_frag, 0.0f);
	nvcuda::wmma::fill_fragment(out_q_1_frag, 0.0f);

	// load h
	nvcuda::wmma::load_matrix_sync(h_0_frag, h_ptr + FRAGMENT_DIM_N * lane, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(h_1_frag, h_ptr + FRAGMENT_DIM_N * lane + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);

	/*  Q 0 */
	// load q
	nvcuda::wmma::load_matrix_sync(in_q_0_frag, q_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_q_1_frag, q_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_q_0_frag, h_0_frag, in_q_0_frag, out_q_0_frag);
	nvcuda::wmma::mma_sync(out_q_0_frag, h_1_frag, in_q_1_frag, out_q_0_frag);
	/*  Q 1 */
	// load q
	nvcuda::wmma::load_matrix_sync(in_q_0_frag, q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_q_1_frag, q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_q_1_frag, h_0_frag, in_q_0_frag, out_q_1_frag);
	nvcuda::wmma::mma_sync(out_q_1_frag, h_1_frag, in_q_1_frag, out_q_1_frag);
	/*  R */
	// load r
	nvcuda::wmma::load_matrix_sync(in_r_0_frag, r_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(in_r_1_frag, r_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// mma
	nvcuda::wmma::mma_sync(out_r_frag, h_0_frag, in_r_0_frag, out_r_frag);
	nvcuda::wmma::mma_sync(out_r_frag, h_1_frag, in_r_1_frag, out_r_frag);

	// store
	__syncthreads();
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N, out_q_0_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, out_q_1_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::store_matrix_sync(r_ptr + lane * FRAGMENT_DIM_N, out_r_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <>
__device__ void update_qr<mtk::tcqr::compute_mode::fp32_tc_cor, float, float, float>(
		float* const q_ptr, float* const r_ptr,
		float* const h_ptr,
		half* const working_memory,
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

	// load h
	__syncthreads();
	mtk::wmma::foreach(
			h16_0_frag,
			[&](const unsigned frag_index, const unsigned mem_index) {
				const auto m = (mem_index & 0xf);
				const auto n = mem_index >> 4;
				const auto mem = m + FRAGMENT_DIM_M * n;

				const auto v0_f32 = h_ptr[FRAGMENT_DIM_N * lane + mem];
				const auto v0_f16 = cutf::type::cast<half>(v0_f32);
				h16_0_frag.x[frag_index] = v0_f16;
				h16_0_diff_frag.x[frag_index] = cutf::type::cast<half>(v0_f32 - cutf::type::cast<float>(v0_f16));
				const auto v1_f32 = h_ptr[FRAGMENT_DIM_N * lane + mem + FRAGMENT_DIM_N * FRAGMENT_DIM_M];
				const auto v1_f16 = cutf::type::cast<half>(v1_f32);
				h16_1_frag.x[frag_index] = v1_f16;
				h16_1_diff_frag.x[frag_index] = cutf::type::cast<half>(v1_f32 - cutf::type::cast<float>(v1_f16));
			});


	/*  Q 0 */
	// load q
	half* const q16_ptr = reinterpret_cast<half*>(working_memory);
	copy_32x16(q16_ptr, q_ptr, unique_id);
	mtk::wmma::fill_zero(q32_0_frag);
	__syncthreads();
	nvcuda::wmma::load_matrix_sync(q16_0_frag, q16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_frag, q16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	// make q diff
	__syncthreads();
	mtk::matrix_operation::diff32x16_2w(q16_ptr, q_ptr, q16_ptr, unique_id);
	// diff mma
	nvcuda::wmma::load_matrix_sync(q16_0_diff_frag, q16_ptr, FRAGMENT_DIM_M);
#ifdef THREE_TERMS_CORRECTION
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_diff_frag, q16_0_diff_frag, q32_0_frag);
#endif
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_diff_frag, q16_0_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_frag, q16_0_diff_frag, q32_0_frag);
	nvcuda::wmma::load_matrix_sync(q16_1_diff_frag, q16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
#ifdef THREE_TERMS_CORRECTION
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_diff_frag, q16_1_diff_frag, q32_0_frag);
#endif
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_diff_frag, q16_1_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_frag, q16_1_diff_frag, q32_0_frag);
	// mma
	nvcuda::wmma::mma_sync(q32_0_frag, h16_0_frag, q16_0_frag, q32_0_frag);
	nvcuda::wmma::mma_sync(q32_0_frag, h16_1_frag, q16_1_frag, q32_0_frag);

	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N, q32_0_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	__syncthreads();

	/*  Q 1 */
	// load q
	copy_32x16(q16_ptr, q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, unique_id);
	mtk::wmma::fill_zero(q32_1_frag);
	__syncthreads();
	nvcuda::wmma::load_matrix_sync(q16_0_frag, q16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(q16_1_frag, q16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	__syncthreads();
	// load q diff
	mtk::matrix_operation::diff32x16_2w(q16_ptr, q_ptr + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q16_ptr, unique_id);
	nvcuda::wmma::load_matrix_sync(q16_0_diff_frag, q16_ptr, FRAGMENT_DIM_M);
	// diff mma
#ifdef THREE_TERMS_CORRECTION
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_diff_frag, q16_0_diff_frag, q32_1_frag);
#endif
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_diff_frag, q16_0_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_frag, q16_0_diff_frag, q32_1_frag);
	nvcuda::wmma::load_matrix_sync(q16_1_diff_frag, q16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
#ifdef THREE_TERMS_CORRECTION
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_diff_frag, q16_1_diff_frag, q32_1_frag);
#endif
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_diff_frag, q16_1_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_frag, q16_1_diff_frag, q32_1_frag);
	// mma
	nvcuda::wmma::mma_sync(q32_1_frag, h16_0_frag, q16_0_frag, q32_1_frag);
	nvcuda::wmma::mma_sync(q32_1_frag, h16_1_frag, q16_1_frag, q32_1_frag);
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_N + FRAGMENT_DIM_M * FRAGMENT_DIM_N, q32_1_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	__syncthreads();

	/*  R */
	// load r
	__syncthreads();
	half* const r16_ptr = reinterpret_cast<half*>(working_memory);
	copy_32x16(r16_ptr, r_ptr, unique_id);
	mtk::wmma::fill_zero(r32_frag);
	__syncthreads();
	nvcuda::wmma::load_matrix_sync(r16_0_frag, r16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::load_matrix_sync(r16_1_frag, r16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	__syncthreads();
	// load r diff
	mtk::matrix_operation::diff32x16_2w(r16_ptr, r_ptr, r16_ptr, unique_id);
	__syncthreads();
	nvcuda::wmma::load_matrix_sync(r16_0_diff_frag, r16_ptr, FRAGMENT_DIM_M);
	// diff mma
#ifdef THREE_TERMS_CORRECTION
	nvcuda::wmma::mma_sync(r32_frag, h16_0_diff_frag, r16_0_diff_frag, r32_frag);
#endif
	nvcuda::wmma::mma_sync(r32_frag, h16_0_diff_frag, r16_0_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_0_frag, r16_0_diff_frag, r32_frag);
	nvcuda::wmma::load_matrix_sync(r16_1_diff_frag, r16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
#ifdef THREE_TERMS_CORRECTION
	nvcuda::wmma::mma_sync(r32_frag, h16_1_diff_frag, r16_1_diff_frag, r32_frag);
#endif
	nvcuda::wmma::mma_sync(r32_frag, h16_1_diff_frag, r16_1_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_1_frag, r16_1_diff_frag, r32_frag);
	// mma
	nvcuda::wmma::mma_sync(r32_frag, h16_0_frag, r16_0_frag, r32_frag);
	nvcuda::wmma::mma_sync(r32_frag, h16_1_frag, r16_1_frag, r32_frag);
	nvcuda::wmma::store_matrix_sync(r_ptr + lane * FRAGMENT_DIM_N, r32_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}
#else // IMPLICIT_H

// update q and r not making H explicitly
__device__ void update_qr_f32tc_correction_with_u(
		float* const q32_ptr, float* const r32_ptr,
		half* const q16_ptr, half* const r16_ptr,
		float* const u_ptr, const float norm_u2,
		float* const tmp_vec,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;

	float* const u_tmp_vec = tmp_vec;
	float* const q_tmp_vec = u_tmp_vec + FRAGMENT_DIM_M;
	float* const r_tmp_vec = q_tmp_vec + FRAGMENT_DIM_M;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> u_0_frag, u_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::row_major> ut_0_frag, ut_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::row_major> ut_diff_0_frag, ut_diff_1_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> q_0_frag, q_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> q_diff_0_frag, q_diff_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> r_0_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> r_diff_0_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, float> tmp_vec_acc_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::row_major> tmp_vec_mb_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, float> mma_result_frag;

	__syncthreads();
	if (unique_id < FRAGMENT_DIM_M) {
		const auto u = u_ptr[unique_id] / cutf::math::sqrt(norm_u2);
		u_ptr[unique_id] = u;
		u_tmp_vec[unique_id] = u - cutf::type::cast<float>(cutf::type::cast<half>(u));
	}
	__syncthreads();

	// Q (first step)
	mtk::wmma::fill_zero(tmp_vec_acc_frag);
	mtk::wmma::foreach(
			q_0_frag,
			[&](const unsigned frag_index, const unsigned mem_index) {
				const auto m = (mem_index & 0xf);
				const auto n = mem_index >> 4;
				const auto mem = m + FRAGMENT_DIM_M * n;

				const auto v0_f32 = q32_ptr[FRAGMENT_DIM_N * FRAGMENT_DIM_M * lane + mem];
				const auto v0_f16 = cutf::type::cast<half>(v0_f32);
				q_0_frag.x[frag_index] = v0_f16;
				q_diff_0_frag.x[frag_index] = cutf::type::cast<half>(v0_f32 - cutf::type::cast<float>(v0_f16));
				const auto v1_f32 = q32_ptr[FRAGMENT_DIM_N * FRAGMENT_DIM_M * lane + mem + FRAGMENT_DIM_N];
				const auto v1_f16 = cutf::type::cast<half>(v1_f32);
				q_1_frag.x[frag_index] = v1_f16;
				q_diff_1_frag.x[frag_index] = cutf::type::cast<half>(v1_f32 - cutf::type::cast<float>(v1_f16));
			});


	mtk::wmma::load_vector_sync(ut_diff_0_frag, u_tmp_vec);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_diff_0_frag, q_0_frag, tmp_vec_acc_frag);
	mtk::wmma::load_vector_sync(ut_0_frag, u_ptr);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, q_diff_0_frag, tmp_vec_acc_frag);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, q_0_frag, tmp_vec_acc_frag);

	mtk::wmma::load_vector_sync(ut_diff_1_frag, u_tmp_vec + FRAGMENT_DIM_N);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_diff_1_frag, q_1_frag, tmp_vec_acc_frag);
	mtk::wmma::load_vector_sync(ut_1_frag, u_ptr + FRAGMENT_DIM_N);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, q_diff_1_frag, tmp_vec_acc_frag);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, q_1_frag, tmp_vec_acc_frag);

	mtk::wmma::store_vector_sync(q_tmp_vec + lane * FRAGMENT_DIM_N, tmp_vec_acc_frag, -2.0f, nvcuda::wmma::mem_row_major);

	// R (first step)
	mtk::wmma::foreach(
			r_0_frag,
			[&](const unsigned frag_index, const unsigned mem_index) {
				const auto m = (mem_index & 0xf);
				const auto n = mem_index >> 4;
				const auto mem = m + FRAGMENT_DIM_M * n;

				const auto v0_f32 = r32_ptr[FRAGMENT_DIM_N * lane + mem];
				const auto v0_f16 = cutf::type::cast<half>(v0_f32);
				r_0_frag.x[frag_index] = v0_f16;
				r_diff_0_frag.x[frag_index] = cutf::type::cast<half>(v0_f32 - cutf::type::cast<float>(v0_f16));
			});

	mtk::wmma::fill_zero(tmp_vec_acc_frag);
	if (lane == 0) {
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_diff_0_frag, r_0_frag, tmp_vec_acc_frag);
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, r_diff_0_frag, tmp_vec_acc_frag);
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, r_0_frag, tmp_vec_acc_frag);
	} else {
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_diff_1_frag, r_0_frag, tmp_vec_acc_frag);
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, r_diff_0_frag, tmp_vec_acc_frag);
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, r_0_frag, tmp_vec_acc_frag);
	}
	mtk::wmma::store_vector_sync(r_tmp_vec + lane * FRAGMENT_DIM_N, tmp_vec_acc_frag, -2.0f, nvcuda::wmma::mem_row_major);

	__syncthreads();
	if (unique_id < FRAGMENT_DIM_N) {
		r_tmp_vec[unique_id] += r_tmp_vec[unique_id + FRAGMENT_DIM_N];
	}

	// R (second step)
	__syncthreads();
	if (unique_id < FRAGMENT_DIM_N) {
		r_tmp_vec[unique_id + FRAGMENT_DIM_N] = r_tmp_vec[unique_id] - cutf::type::cast<float>(cutf::type::cast<half>(r_tmp_vec[unique_id]));
	}

	__syncthreads();
	mtk::wmma::make_direct_product_fragment_c3(tmp_vec_mb_frag, r_tmp_vec, r_tmp_vec + FRAGMENT_DIM_N);
	mtk::wmma::make_direct_product_fragment_c3(u_0_frag, u_ptr, u_tmp_vec);
	mtk::wmma::make_direct_product_fragment_c3(u_1_frag, u_ptr + FRAGMENT_DIM_N, u_tmp_vec + FRAGMENT_DIM_N);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, r32_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	if (lane == 0) {
		nvcuda::wmma::mma_sync(mma_result_frag, u_0_frag, tmp_vec_mb_frag, mma_result_frag);
	} else {
		nvcuda::wmma::mma_sync(mma_result_frag, u_1_frag, tmp_vec_mb_frag, mma_result_frag);
	}
	__syncthreads();
	nvcuda::wmma::store_matrix_sync(r32_ptr + lane * FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	// Q (second step)
	__syncthreads();
	if (unique_id < FRAGMENT_DIM_M) {
		u_tmp_vec[unique_id] = q_tmp_vec[unique_id] - cutf::type::cast<float>(cutf::type::cast<half>(q_tmp_vec[unique_id]));
	}
	__syncthreads();
	mtk::wmma::make_direct_product_fragment_c3(tmp_vec_mb_frag, q_tmp_vec + lane * FRAGMENT_DIM_N, u_tmp_vec + lane * FRAGMENT_DIM_N);

	// mma
	nvcuda::wmma::load_matrix_sync(mma_result_frag, q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::mma_sync(mma_result_frag, u_0_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::mma_sync(mma_result_frag, u_1_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

__device__ void update_qr_f32tc_with_u(
		float* const q32_ptr, float* const r32_ptr,
		half* const q16_ptr, half* const r16_ptr,
		float* const u_ptr, const float norm_u2,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;

	float* const tmp_vec_ptr = u_ptr + lane * FRAGMENT_DIM_N;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> u_0_frag, u_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::row_major> ut_0_frag, ut_1_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> q_0_frag, q_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> r_0_frag, r_1_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, float> tmp_vec_acc_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::row_major> tmp_vec_mb_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, float> mma_result_frag;

	const auto r_norm_u = 1.0f / cutf::math::sqrt(norm_u2);
	if (unique_id < FRAGMENT_DIM_M) {
		u_ptr[unique_id] *= r_norm_u;
	}
	__syncthreads();

	mtk::wmma::load_vector_sync(u_0_frag, u_ptr);
	mtk::wmma::load_vector_sync(u_1_frag, u_ptr + FRAGMENT_DIM_N);
	mtk::wmma::load_vector_sync(ut_0_frag, u_ptr);
	mtk::wmma::load_vector_sync(ut_1_frag, u_ptr + FRAGMENT_DIM_N);

	__syncthreads();

	/* Q */
	mtk::wmma::fill_zero(tmp_vec_acc_frag);
	nvcuda::wmma::load_matrix_sync(q_0_frag, q16_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, q_0_frag, tmp_vec_acc_frag);
	nvcuda::wmma::load_matrix_sync(q_1_frag, q16_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, q_1_frag, tmp_vec_acc_frag);
	mtk::wmma::store_vector_sync(tmp_vec_ptr, tmp_vec_acc_frag, -2.0f, nvcuda::wmma::mem_row_major);
	mtk::wmma::load_vector_sync(tmp_vec_mb_frag, tmp_vec_ptr);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::mma_sync(mma_result_frag, u_0_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::mma_sync(mma_result_frag, u_1_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(q32_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	/* R */
	mtk::wmma::fill_zero(tmp_vec_acc_frag);
	nvcuda::wmma::load_matrix_sync(r_0_frag, r16_ptr, FRAGMENT_DIM_M);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, r_0_frag, tmp_vec_acc_frag);
	nvcuda::wmma::load_matrix_sync(r_1_frag, r16_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, r_1_frag, tmp_vec_acc_frag);
	mtk::wmma::store_vector_sync(tmp_vec_ptr, tmp_vec_acc_frag, -2.0f, nvcuda::wmma::mem_row_major);
	mtk::wmma::load_vector_sync(tmp_vec_mb_frag, tmp_vec_ptr);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, r32_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	if (lane == 0)
		nvcuda::wmma::mma_sync(mma_result_frag, u_0_frag, tmp_vec_mb_frag, mma_result_frag);
	else
		nvcuda::wmma::mma_sync(mma_result_frag, u_1_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(r32_ptr + lane * FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

__device__ void update_qr_f16tc_with_u(
		half* const q_ptr, half* const r_ptr,
		half* const u_ptr, const float norm_u2,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;

	half* const tmp_vec_ptr = u_ptr + lane * FRAGMENT_DIM_N;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> u_0_frag, u_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::row_major> ut_0_frag, ut_1_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> q_0_frag, q_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::col_major> r_0_frag, r_1_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half> tmp_vec_acc_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half, nvcuda::wmma::row_major> tmp_vec_mb_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAGMENT_DIM_N, FRAGMENT_DIM_N, FRAGMENT_DIM_N, half> mma_result_frag;

	const auto r_norm_u = 1.0f / cutf::math::sqrt(norm_u2);
	if (unique_id < FRAGMENT_DIM_M) {
		u_ptr[unique_id] *= r_norm_u;
	}
	__syncthreads();

	mtk::wmma::load_vector_sync(u_0_frag, u_ptr);
	mtk::wmma::load_vector_sync(u_1_frag, u_ptr + FRAGMENT_DIM_N);
	mtk::wmma::load_vector_sync(ut_0_frag, u_ptr);
	mtk::wmma::load_vector_sync(ut_1_frag, u_ptr + FRAGMENT_DIM_N);

	__syncthreads();

	/* Q */
	mtk::wmma::fill_zero(tmp_vec_acc_frag);
	nvcuda::wmma::load_matrix_sync(q_0_frag, q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, q_0_frag, tmp_vec_acc_frag);
	nvcuda::wmma::load_matrix_sync(q_1_frag, q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
	nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, q_1_frag, tmp_vec_acc_frag);
	mtk::wmma::store_vector_sync(tmp_vec_ptr, tmp_vec_acc_frag, cutf::type::cast<half>(-2.0f), nvcuda::wmma::mem_row_major);
	mtk::wmma::load_vector_sync(tmp_vec_mb_frag, tmp_vec_ptr);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::mma_sync(mma_result_frag, u_0_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	nvcuda::wmma::mma_sync(mma_result_frag, u_1_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);

	/* R */
	if (lane == 0) {
		mtk::wmma::fill_zero(tmp_vec_acc_frag);
		nvcuda::wmma::load_matrix_sync(r_0_frag, r_ptr, FRAGMENT_DIM_M);
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_0_frag, r_0_frag, tmp_vec_acc_frag);
		nvcuda::wmma::load_matrix_sync(r_1_frag, r_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M);
		nvcuda::wmma::mma_sync(tmp_vec_acc_frag, ut_1_frag, r_1_frag, tmp_vec_acc_frag);
		mtk::wmma::store_vector_sync(tmp_vec_ptr, tmp_vec_acc_frag, cutf::type::cast<half>(-2.0f), nvcuda::wmma::mem_row_major);
	}
	__syncthreads();

	mtk::wmma::load_vector_sync(tmp_vec_mb_frag, u_ptr);

	nvcuda::wmma::load_matrix_sync(mma_result_frag, r_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
	if (lane == 0)
		nvcuda::wmma::mma_sync(mma_result_frag, u_0_frag, tmp_vec_mb_frag, mma_result_frag);
	else
		nvcuda::wmma::mma_sync(mma_result_frag, u_1_frag, tmp_vec_mb_frag, mma_result_frag);
	nvcuda::wmma::store_matrix_sync(r_ptr + lane * FRAGMENT_DIM_N, mma_result_frag, FRAGMENT_DIM_M, nvcuda::wmma::mem_col_major);
}

template <class T>
__device__ void update_qr_with_u(
		T* const q_ptr, T* const r_ptr,
		T* const u_ptr, const float norm_u2,
		T* const tmp_vec_ptr,
		const unsigned unique_id
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto lane = unique_id >> 5;

	if (std::is_same<half, T>::value) {
		if (unique_id < FRAGMENT_DIM_M)
			u_ptr[unique_id] /= cutf::math::sqrt(norm_u2);
		__syncthreads();
	}

	/* Q */
	if (unique_id < FRAGMENT_DIM_M)
		tmp_vec_ptr[unique_id] = cutf::type::cast<T>(0.0f);
	__syncthreads();
	mtk::gevm_core16x16<T, 1>(
			tmp_vec_ptr + lane * FRAGMENT_DIM_N,
			u_ptr,
			q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			unique_id & 0x1f
			);
	mtk::gevm_core16x16<T, 1>(
			tmp_vec_ptr + lane * FRAGMENT_DIM_N,
			u_ptr + FRAGMENT_DIM_N,
			q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			unique_id & 0x1f
			);

	__syncthreads();
	if (std::is_same<half, T>::value) {
		if (unique_id < FRAGMENT_DIM_M)
			tmp_vec_ptr[unique_id] *= -2.0f;
	} else {
		if (unique_id < FRAGMENT_DIM_M)
			tmp_vec_ptr[unique_id] *= -2.0f / norm_u2;
	}
	__syncthreads();

	mtk::ger_core16x16<T, 1>(
			q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			u_ptr,
			tmp_vec_ptr + lane * FRAGMENT_DIM_N,
			unique_id & 0x1f
			);
	mtk::ger_core16x16<T, 1>(
			q_ptr + lane * FRAGMENT_DIM_M * FRAGMENT_DIM_N + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			u_ptr + FRAGMENT_DIM_N,
			tmp_vec_ptr + lane * FRAGMENT_DIM_N,
			unique_id & 0x1f
			);

	/* R */
	if (unique_id < FRAGMENT_DIM_N)
		tmp_vec_ptr[unique_id] = cutf::type::cast<T>(0.0f);
	__syncthreads();
	if (lane == 0) {
		mtk::gevm_core16x16<T, 1>(
				tmp_vec_ptr,
				u_ptr,
				r_ptr, FRAGMENT_DIM_M,
				unique_id
				);
		mtk::gevm_core16x16<T, 1>(
				tmp_vec_ptr,
				u_ptr + FRAGMENT_DIM_N,
				r_ptr + FRAGMENT_DIM_N, FRAGMENT_DIM_M,
				unique_id
				);
	}

	if (std::is_same<half, T>::value) {
		if (unique_id < FRAGMENT_DIM_N)
			tmp_vec_ptr[unique_id] *= -2.0f;
	} else {
		if (unique_id < FRAGMENT_DIM_N)
			tmp_vec_ptr[unique_id] *= -2.0f / norm_u2;
	}
	__syncthreads();

	mtk::ger_core16x16<T, 1>(
			r_ptr + lane * FRAGMENT_DIM_N, FRAGMENT_DIM_M,
			u_ptr + lane * FRAGMENT_DIM_N,
			tmp_vec_ptr,
			unique_id & 0x1f
			);
}

#endif // IMPLICIT_H

template <mtk::tcqr::compute_mode mode, class Q_T, class R_T, class A_T, class WORK_T>
__device__ void qr32x16_core(
		Q_T* const q_ptr, R_T* const r_ptr,
		A_T* const u_ptr,
		WORK_T* const work,
		const unsigned m, const unsigned n,
		const unsigned tid
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	const auto unique_id = tid & 0x3f;
	for (unsigned k = 0; k < n; k++) {
		debug_func(
				unique_id,
				[&k]() {printf("/* -------- %u ---------\n", k);}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&r_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r_ptr, m, n, "R");}
				);
		debug_func(0, []() {__syncthreads();});
		debug_func(
				unique_id,
				[&q_ptr, &m]() {mtk::utils::print_matrix_32x16(q_ptr, m, m, "Q");}
				);
		debug_func(0, []() {__syncthreads();});
		// copy u
		if(unique_id < FRAGMENT_DIM_M) {
			u_ptr[unique_id] = cutf::type::cast<A_T>(0.0f);
			if(unique_id >= k && unique_id < m) {
				u_ptr[unique_id] = cutf::type::cast<A_T>(r_ptr[FRAGMENT_DIM_M * k + unique_id]);
			}
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u_ptr, &m]() {mtk::utils::print_matrix(u_ptr, 1, m, "u");}
				);
		// compute |u|
		__syncthreads();
		const auto norm_u_0 = cutf::math::sqrt(get_norm2_32(u_ptr, m, unique_id & 0x1f));
		__syncthreads();

		debug_func(
				unique_id,
				[&norm_u_0]() {printf("norm_u_0 = %.5f\n", norm_u_0);}
				);
		// update u
		if(unique_id == k) {
			u_ptr[unique_id] += cutf::type::cast<A_T>(cutf::type::cast<float>(cutf::math::sign(u_ptr[unique_id])) * norm_u_0);
		}
		__syncthreads();
		debug_func(
				unique_id,
				[&u_ptr, &m]() {mtk::utils::print_matrix(u_ptr, 1, m, "u`");}
				);
		// recompute |u|
		__syncthreads();
		const auto norm2_u_1 = get_norm2_32(u_ptr, m, unique_id & 0x1f);
		__syncthreads();

		debug_func(
				unique_id,
				[&norm2_u_1]() {printf("norm_u_1^2 = %.5f\n", norm2_u_1);}
				);
		// compute h

		__syncthreads();
		auto *h_ptr = reinterpret_cast<typename h_mat_t<mode>::type*>(work);
		make_h<mode>(
				h_ptr, m,
				u_ptr, norm2_u_1,
				unique_id
				);
		__syncthreads();
		debug_func(
				unique_id,
				[&h_ptr, &m]() {mtk::utils::print_matrix_32x16(h_ptr, m, m, "H");}
				);
		debug_func(
				unique_id,
				[&r_ptr, &m, &n]() {mtk::utils::print_matrix_32x16(r_ptr, 32, 16, "R (before update)");}
				);
		debug_func(
				unique_id,
				[&q_ptr, &m]() {mtk::utils::print_matrix_32x16(q_ptr, 32, 32, "Q (before update)");}
				);
		__syncthreads();
		// update q, r
		update_qr<mode>(
				q_ptr, r_ptr,
				h_ptr, reinterpret_cast<half*>(h_ptr),
				unique_id
				);
		__syncthreads();
	}
}

template <mtk::tcqr::compute_mode mode, class Q_T, class R_T, class A_T>
__global__ void qr32x16_batched_kernel(
		Q_T* const q_ptr, const std::size_t ldq,
		R_T* const r_ptr, const std::size_t ldr,
		const A_T* const a_ptr, const std::size_t lda,
		const std::size_t m,
		const unsigned n,
		const std::size_t batch_size,
		const unsigned* a_start_position
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	constexpr std::size_t max_batch_size_per_block = get_max_batch_size_per_block<mode>();
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_id = tid / (warp_size * 2);
	const auto shared_memory_id = matrix_id & (max_batch_size_per_block - 1);
	if(matrix_id >= batch_size) return;

	__shared__ Q_T shared_q[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ R_T shared_r[FRAGMENT_DIM_M * FRAGMENT_DIM_N * max_batch_size_per_block];
	__shared__ A_T shared_u[FRAGMENT_DIM_M * max_batch_size_per_block];
	__shared__ typename h_mat_t<mode>::type shared_w[FRAGMENT_DIM_M * FRAGMENT_DIM_M * max_batch_size_per_block];

	const auto shared_q_ptr = shared_q + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_r_ptr = shared_r + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_N;
	const auto shared_w_ptr = shared_w + shared_memory_id * FRAGMENT_DIM_M * FRAGMENT_DIM_M;
	const auto shared_u_ptr = shared_u + shared_memory_id * FRAGMENT_DIM_M;

	const auto sub_a_position = a_start_position[matrix_id];
	const auto sub_a_m = a_start_position[matrix_id + 1] - sub_a_position;

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r_ptr, sub_a_m, n,
			a_ptr, sub_a_position, lda,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<Q_T, FRAGMENT_DIM_M>(
			shared_q_ptr,
			tid
			);

	// qr core
	qr32x16_core<mode> (
			shared_q_ptr, shared_r_ptr,
			shared_u_ptr, shared_w_ptr,
			sub_a_m, n,
			tid
			);

	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q_ptr, sub_a_position, ldq,
			shared_q_ptr, n, sub_a_m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r_ptr, n * matrix_id, ldr,
			shared_r_ptr, n, n,
			tid
			);
}

template <mtk::tcqr::compute_mode mode, class Q_T, class R_T, class A_T>
__global__ void qr32x16_kernel(
		Q_T* const q_ptr, const std::size_t ldq,
		R_T* const r_ptr, const std::size_t ldr,
		const A_T* const a_ptr, const std::size_t lda,
		const unsigned m,
		const unsigned n
		) {
	constexpr std::size_t FRAGMENT_DIM_M = 32;
	constexpr std::size_t FRAGMENT_DIM_N = 16;
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ Q_T shared_q[FRAGMENT_DIM_M * FRAGMENT_DIM_M];
	__shared__ R_T shared_r[FRAGMENT_DIM_M * FRAGMENT_DIM_N];
	__shared__ A_T shared_u[FRAGMENT_DIM_M];
	__shared__ typename h_mat_t<mode>::type shared_w[FRAGMENT_DIM_M * FRAGMENT_DIM_M];

	// init shared memory
	mtk::matrix_copy::g2s32x16_2w(
			shared_r, m, n,
			a_ptr, 0, lda,
			tid
			);
	mtk::matrix_operation::make_identity_matrix<Q_T, FRAGMENT_DIM_M>(
			shared_q,
			tid
			);

	// qr core
	qr32x16_core<mode> (
			shared_q, shared_r,
			shared_u, shared_w,
			m, n,
			tid
			);

	// store result
	mtk::matrix_copy::s2g32x32_16x32_t_2w(
			q_ptr, 0, ldq,
			shared_q, n, m,
			tid
			);
	mtk::matrix_copy::s2g32x16_2w(
			r_ptr, 0, ldr,
			shared_r, n, n,
			tid
			);
}
}

template <mtk::tcqr::compute_mode mode, class Q_T, class R_T, class A_T>
void mtk::tcqr::qr32x16_batched(
		Q_T* const q, const std::size_t ldq,
		R_T* const r, const std::size_t ldr,
		const A_T* const a, const std::size_t lda,
		const unsigned int m, const unsigned int n,
		const std::size_t batch_size,
		const unsigned* a_start_position,
		cudaStream_t const cuda_stream
		) {
	constexpr std::size_t max_batch_size_per_block = get_max_batch_size_per_block<mode>();
	const auto grid_size = (batch_size + max_batch_size_per_block + 1) / max_batch_size_per_block;
	const auto block_size = max_batch_size_per_block * 2 * warp_size;

	qr32x16_batched_kernel<mode, Q_T, R_T, A_T><<<grid_size, block_size, 0, cuda_stream>>>(
			q, ldq,
			r, ldr,
			a, lda,
			m, n,
			batch_size,
			a_start_position
			);
}

template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::fp32_tc_cor       , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);
template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::fp32_notc         , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);
template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::fp32_tc_nocor     , half , float, float>(half * const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);
template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::fp32_tc_nocor     , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);
template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::fp16_notc         , half , half , half >(half * const, const std::size_t, half * const, const std::size_t, const half * const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);
template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::fp16_tc_nocor     , half , half , half >(half * const, const std::size_t, half * const, const std::size_t, const half * const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);
template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::tf32_tc_nocor_emu , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);
template void mtk::tcqr::qr32x16_batched<mtk::tcqr::compute_mode::tf32_tc_cor_emu   , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, const std::size_t, const unsigned*, cudaStream_t const);

template <mtk::tcqr::compute_mode mode, class Q_T, class R_T, class A_T>
void mtk::tcqr::qr32x16(
		Q_T* const q, const std::size_t ldq,
		R_T* const r, const std::size_t ldr,
		const A_T* const a, const std::size_t lda,
		const unsigned int m, const unsigned int n,
		cudaStream_t const cuda_stream
		){
	qr32x16_kernel<mode, Q_T, R_T, A_T><<<1, 2 * warp_size, 0, cuda_stream>>>(
			q, ldq,
			r, ldr,
			a, lda,
			m, n
			);
}

template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::fp32_tc_cor       , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::fp32_notc         , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::fp32_tc_nocor     , half , float, float>(half * const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::fp32_tc_nocor     , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::fp16_notc         , half , half , half >(half * const, const std::size_t, half * const, const std::size_t, const half * const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::fp16_tc_nocor     , half , half , half >(half * const, const std::size_t, half * const, const std::size_t, const half * const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::tf32_tc_nocor_emu , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
template void mtk::tcqr::qr32x16<mtk::tcqr::compute_mode::tf32_tc_cor_emu   , float, float, float>(float* const, const std::size_t, float* const, const std::size_t, const float* const, const std::size_t, const unsigned int, const unsigned int, cudaStream_t const);
