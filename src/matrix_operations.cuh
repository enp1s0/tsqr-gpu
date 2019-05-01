#ifndef __MATRIX_OPERATIONS_CUH__
#define __MATRIX_OPERATIONS_CUH__
#include <cutf/type.hpp>
namespace mtk {
namespace matrix_operation {
template <class T, std::size_t FRAGMENT_DIM_M = 32>
__device__ inline void make_zero_matrix(
		T* const target_ptr,
		const unsigned tid
		){
	const auto unique_id = tid & 0x3f;
	constexpr auto stride = 2 * warp_size;
	for(unsigned i = 0; i < (FRAGMENT_DIM_M * FRAGMENT_DIM_M) / stride; i++){
		target_ptr[i * stride + unique_id] = cutf::cuda::type::cast<T>(0.0f);
	}
}

template <class T, std::size_t FRAGMENT_DIM_M = 32>
__device__ inline void make_identity_matrix(
		T* const target_ptr,
		const unsigned tid
		){
	const auto unique_id = tid & 0x3f;
	make_zero_matrix(target_ptr, unique_id);
	target_ptr[unique_id * (1 + FRAGMENT_DIM_M)] = cutf::cuda::type::cast<T>(1.0f);
}
} // namespace matrix_operation
} // namespace mtk
#endif /* end of include guard */
