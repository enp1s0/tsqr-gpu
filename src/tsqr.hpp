#ifndef __TSQR_HPP__
#define __TSQR_HPP__
#include <cstddef>
namespace mtk {
namespace tsqr {
template <class T, bool UseTC>
void tsqr16(T* const q_ptr, T* const r_ptr, const T* const a_ptr, const std::size_t m, const std::size_t n, T* const working_memory_ptr);
std::size_t get_working_memory_size(const std::size_t m, const std::size_t n);
} // namespace tsqr
} // namespace mtk

#endif /* end of include guard */
