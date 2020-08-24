#ifndef __VALIDATION_HPP__
#define __VALIDATION_HPP__
#include <cstddef>

namespace mtk {
namespace validation {
template <class T>
double check_orthogonality16(const T* const matrix, const std::size_t m, const unsigned n);
template <class T>
void check_submatrix_orthogonality(const T* const matrix, const std::size_t m, const unsigned n);
template <class T>
void multi_orthogonality(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const std::size_t size, cudaStream_t stream = 0);
template <class T>
void exponent_distribution(const T* const ptr, const std::size_t size, const char* const csv_item_name, cudaStream stream = 0);
} // namespace validation
} // namespace mtk

#endif /* end of include guard */
