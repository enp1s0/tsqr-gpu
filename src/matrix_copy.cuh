#ifndef __MATRIX_COPY_CUH__
#define __MATRIX_COPY_CUH__
#include <cstddef>
#include <cutf/type.hpp>
namespace mtk {
namespace matrix_copy {
template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM = 16>
__device__ inline void g2s16x16_1w(
		DST_T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const SRC_T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const unsigned tid
		) {
	constexpr auto load_size = FRAGMENT_DIM >> 1;
	const auto unique_id = tid & 0x1f;
	const auto x = unique_id >> 1;

	const auto start_y = (unique_id & 0b1) * load_size;
	for(std::size_t i = 0; i < load_size; i++) {
		const auto y = start_y + i;
		DST_T val;
		if(y < shared_m && x < shared_n) {
			// copy
			const auto global_index = x * global_ld + y + global_p_y;
			val = cutf::type::cast<DST_T>( global_ptr[global_index] );
		} else {
			val = cutf::type::cast<DST_T>(0.0f);
		}
		const auto shared_index = x * FRAGMENT_DIM + y;
		shared_ptr[shared_index] = val;
	}
}
template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM = 16>
__device__ inline void s2g16x16_1w(
		DST_T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const SRC_T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const unsigned tid
		) {
	constexpr auto load_size = FRAGMENT_DIM >> 1;
	const auto unique_id = tid & 0x1f;
	const auto x = unique_id >> 1;
	if(x >= shared_n) return;

	const auto start_y = (unique_id & 0b1) * load_size;
	for(std::size_t i = 0; i < load_size; i++) {
		const auto y = start_y + i;
		if(y >= shared_m) return;

		// copy
		const auto shared_index = x * FRAGMENT_DIM + y;
		const auto global_index = x * global_ld + y + global_p_y;

		global_ptr[global_index] = cutf::type::cast<DST_T>( shared_ptr[shared_index] );
	}
}
template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ inline void s2g32x16_1w(
		DST_T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const SRC_T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const unsigned tid
		) {
	const auto y = tid & 0x1f;
	if(y >= shared_m) return;
	for(std::size_t x = 0; x < FRAGMENT_DIM_N; x++) {
		if(x >= shared_n) return;
		const auto shared_index = FRAGMENT_DIM_M * x + y;
		const auto global_index = global_ld * x + y + global_p_y;

		global_ptr[global_index] = cutf::type::cast<DST_T>( shared_ptr[shared_index] );
	}
}
template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ inline void g2s32x16_1w(
		DST_T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const SRC_T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const unsigned tid
		) {
	const auto y = tid & 0x1f;
	for(std::size_t x = 0; x < FRAGMENT_DIM_N; x++) {
		DST_T val;
		if(y < shared_m && x < shared_n) {
			// copy
			const auto global_index = x * global_ld + y + global_p_y;
			val = cutf::type::cast<DST_T>( global_ptr[global_index] );
		} else {
			val = cutf::type::cast<DST_T>(0.0f);
		}
		const auto shared_index = x * FRAGMENT_DIM_M + y;
		shared_ptr[shared_index] = val;
	}
}



// for 2 warps
template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ inline void g2s32x16_2w(
		DST_T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const SRC_T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const unsigned tid
		) {
	const auto unique_id = tid & 0x3f;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	if(y >= shared_m) return;
	for(std::size_t x = lane; x < FRAGMENT_DIM_N; x += 2) {
		//const auto x = i + lane;
		DST_T val;
		if(x < shared_n) {
			// copy
			const auto global_index = x * global_ld + y + global_p_y;
			val = cutf::type::cast<DST_T>( global_ptr[global_index] );
		} else {
			val = cutf::type::cast<DST_T>(0.0f);
		}
		const auto shared_index = x * FRAGMENT_DIM_M + y;
		shared_ptr[shared_index] = val;
	}
}
template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ inline void s2g32x16_2w(
		DST_T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const SRC_T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const unsigned tid
		) {
	const auto unique_id = tid & 0x3f;
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	if(y >= shared_m) return;
	for(std::size_t x = lane; x < FRAGMENT_DIM_N; x += 2) {
		if(x >= shared_n) return;
		const auto shared_index = FRAGMENT_DIM_M * x + y;
		const auto global_index = global_ld * x + y + global_p_y;

		global_ptr[global_index] = cutf::type::cast<DST_T>( shared_ptr[shared_index] );
	}
}
template <class DST_T, class SRC_T, std::size_t FRAGMENT_DIM_M = 16, std::size_t FRAGMENT_DIM_N = 32>
__device__ inline void s2g32x32_16x32_t_2w(
		DST_T* const global_ptr, const std::size_t global_p_y, const std::size_t global_ld,
		const SRC_T* const shared_ptr, const std::size_t shared_m, const std::size_t shared_n,
		const unsigned tid
		) {
	__syncthreads();
	constexpr unsigned warp_size = 32;
	constexpr unsigned stride = (2 * warp_size) / FRAGMENT_DIM_M;
	const auto unique_id = tid & 0x3f;
	const auto y = unique_id % FRAGMENT_DIM_M;
	if(y >= shared_m) return;
	const auto lane = unique_id / FRAGMENT_DIM_M;
	for(unsigned x = lane; x < FRAGMENT_DIM_N; x += stride) {
		if(x >= shared_n) continue;

		const auto shared_index = FRAGMENT_DIM_N * x + y;
		const auto global_index = global_ld * y + x + global_p_y;

		global_ptr[global_index] = cutf::type::cast<DST_T>( shared_ptr[shared_index] );
	}
}
} // namespace matrix_copy
} // namespace mtk

#endif /* end of include guard */
