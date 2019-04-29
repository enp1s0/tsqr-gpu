#ifndef __MATRIX_COPY_CUH__
#define __MATRIX_COPY_CUH__
#include <cstddef>
namespace mtk {
namespace matrix_copy {
template <class T, std::size_t FRAGMENT_DIM = 16>
__device__ inline void g2s(
		T* const dst_ptr, const std::size_t dst_m, const std::size_t dst_n,
		const T* const src_ptr, const std::size_t src_p_y, const std::size_t src_ld,
		const unsigned tid
		){
	constexpr auto load_size = FRAGMENT_DIM >> 1;
	const auto x = tid >> 1;
	if(x >= dst_n) return;

	const auto start_y = (tid & 0b1) * load_size;
	for(std::size_t i = 0; i < load_size; i++){
		const auto y = start_y + i;
		if(y >= dst_m) return;

		// copy
		const auto dst_index = x * FRAGMENT_DIM + y;
		const auto src_index = x * src_ld + y + src_p_y;

		dst_ptr[dst_index] = src_ptr[src_index];
	}
}
} // namespace matrix_copy
} // namespace mtk

#endif /* end of include guard */
