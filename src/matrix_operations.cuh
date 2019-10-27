#ifndef __MATRIX_OPERATIONS_CUH__
#define __MATRIX_OPERATIONS_CUH__
#include <cutf/type.hpp>
namespace mtk {
namespace matrix_operation {
template <class T, std::size_t FRAGMENT_DIM_M, std::size_t FRAGMENT_DIM_N, std::size_t num_warps = 2>
__device__ inline void make_zero_matrix(
		T* const target_ptr,
		const unsigned tid
		) {
	constexpr unsigned warp_size = 32;
	constexpr auto stride = num_warps * warp_size;
	const auto unique_id = tid & (warp_size * num_warps - 1);
	for(unsigned i = 0; i < (FRAGMENT_DIM_M * FRAGMENT_DIM_N); i += stride) {
		target_ptr[i + unique_id] = cutf::type::cast<T>(0.0f);
	}
	__syncthreads();
}

template <class T, std::size_t FRAGMENT_DIM_M, std::size_t num_warps = 2>
__device__ inline void make_identity_matrix(
		T* const target_ptr,
		const unsigned tid,
		const T value
		) {
	constexpr unsigned warp_size = 32;
	const auto unique_id = tid & (warp_size * num_warps - 1);
	make_zero_matrix<T, FRAGMENT_DIM_M, FRAGMENT_DIM_M, num_warps>(target_ptr, unique_id);
	if(unique_id < FRAGMENT_DIM_M)
		target_ptr[unique_id * (1 + FRAGMENT_DIM_M)] = value;
	__syncthreads();
}

template <class T, std::size_t FRAGMENT_DIM_M, std::size_t num_warps = 2>
__device__ inline void make_identity_matrix(
		T* const target_ptr,
		const unsigned tid
		) {
	make_identity_matrix<T, FRAGMENT_DIM_M, num_warps>(target_ptr, tid, cutf::type::cast<T>(1.0f));
}

template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ inline void diff32x16_2w(
		half* const dst,
		const float* const src_fp32, const half* const src_fp16,
		const unsigned tid
		) {
	const auto unique_id = tid & 0x3f;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	for(std::size_t x = lane; x < FRAGMENT_DIM_N; x += 2) {
		const auto shared_index = FRAGMENT_DIM_M * x + y;

		dst[shared_index] = cutf::type::cast<half>(src_fp32[shared_index] - cutf::type::cast<float>(src_fp16[shared_index]));
	}
	__syncthreads();
}
} // namespace matrix_operation
} // namespace mtk
#endif /* end of include guard */
