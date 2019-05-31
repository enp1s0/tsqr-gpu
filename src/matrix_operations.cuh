#ifndef __MATRIX_OPERATIONS_CUH__
#define __MATRIX_OPERATIONS_CUH__
#include <cutf/type.hpp>
namespace mtk {
namespace matrix_operation {
template <class T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 32>
__device__ inline void make_zero_matrix(
		T* const target_ptr,
		const unsigned tid
		){
	constexpr unsigned warp_size = 32;
	constexpr auto stride = 2 * warp_size;
	const auto unique_id = tid & 0x3f;
	for(unsigned i = 0; i < (FRAGMENT_DIM_M * FRAGMENT_DIM_M) / stride; i++){
		target_ptr[i * stride + unique_id] = cutf::type::cast<T>(0.0f);
	}
	__syncthreads();
}

template <class T, std::size_t FRAGMENT_DIM_M = 32>
__device__ inline void make_identity_matrix(
		T* const target_ptr,
		const unsigned tid
		){
	const auto unique_id = tid & 0x3f;
	make_zero_matrix<T, FRAGMENT_DIM_M, FRAGMENT_DIM_M>(target_ptr, unique_id);
	if(unique_id < FRAGMENT_DIM_M)
		target_ptr[unique_id * (1 + FRAGMENT_DIM_M)] = cutf::type::cast<T>(1.0f);
	__syncthreads();
}
} // namespace matrix_operation
} // namespace mtk
#endif /* end of include guard */
