#ifndef __LATMS_HPP__
#define __LATMS_HPP__
#include <cstddef>

namespace mtk {
namespace utils {
template <class T>
void latms(T* const mat_ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t rank,
		const T* const s_array,
		const unsigned long long seed = 0llu);

template <class T>
T get_cond(
		T* const mat,
		const std::size_t m, const std::size_t n
		);
} // namespace utils
} // namespace mtk


#endif /* end of include guard */
