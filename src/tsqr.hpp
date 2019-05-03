#ifndef __TSQR_HPP__
#define __TSQR_HPP__
#include <cstddef>
namespace mtk {
namespace tsqr {
void tsqr16(float* const q_ptr, float* const r_ptr, const float* const a_ptr, const std::size_t m, const std::size_t n, float* const working_memory_ptr);
std::size_t get_working_memory_size(const std::size_t m, const std::size_t n);
} // namespace tsqr
} // namespace mtk

#endif /* end of include guard */
