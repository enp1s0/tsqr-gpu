#ifndef __MATRIX_COPY_CUH__
#define __MATRIX_COPY_CUH__
#include <cstddef>
namespace mtk {
namespace matrix_copy {
template <class T, std::size_t FRAGMENT_DIM = 16>
__device__ inline void g2s16x16(
		T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const unsigned tid
		){
	constexpr auto load_size = FRAGMENT_DIM >> 1;
	const auto unique_id = tid & 0x1f;
	const auto x = unique_id >> 1;
	if(x >= shared_n) return;

	const auto start_y = (unique_id & 0b1) * load_size;
	for(std::size_t i = 0; i < load_size; i++){
		const auto y = start_y + i;
		if(y >= shared_m) return;

		// copy
		const auto shared_index = x * FRAGMENT_DIM + y;
		const auto global_index = x * global_ld + y + global_p_y;

		shared_ptr[shared_index] = global_ptr[global_index];
	}
}
template <class T, std::size_t FRAGMENT_DIM = 16>
__device__ inline void s2g16x16(
		T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const unsigned tid
		){
	constexpr auto load_size = FRAGMENT_DIM >> 1;
	const auto unique_id = tid & 0x1f;
	const auto x = unique_id >> 1;
	if(x >= shared_n) return;

	const auto start_y = (unique_id & 0b1) * load_size;
	for(std::size_t i = 0; i < load_size; i++){
		const auto y = start_y + i;
		if(y >= shared_m) return;

		// copy
		const auto shared_index = x * FRAGMENT_DIM + y;
		const auto global_index = x * global_ld + y + global_p_y;

		global_ptr[global_index] = shared_ptr[shared_index];
	}
}
template <class T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ inline void g2s32x16(
		T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const unsigned tid
		){
	const auto unique_id = tid & 0x3f;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	if(y >= shared_m) return;
	for(std::size_t i = 0; i < FRAGMENT_DIM_N; i+=2){
		const auto x = i + lane;
		if(x >= shared_n) return;
		const auto shared_index = FRAGMENT_DIM_M * x + y;
		const auto global_index = global_ld * x + y + global_p_y;

		shared_ptr[shared_index] = global_ptr[global_index];
	}
}
template <class T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ inline void s2g32x16(
		T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const unsigned tid
		){
	const auto unique_id = tid & 0x3f;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	if(y >= shared_m) return;
	for(std::size_t i = 0; i < FRAGMENT_DIM_N; i+=2){
		const auto x = i + lane;
		if(x >= shared_n) return;
		const auto shared_index = FRAGMENT_DIM_M * x + y;
		const auto global_index = global_ld * x + y + global_p_y;

		global_ptr[global_index] = shared_ptr[shared_index];
	}
}
} // namespace matrix_copy
} // namespace mtk

#endif /* end of include guard */
