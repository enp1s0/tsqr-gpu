#ifndef __VALIDATION_HPP__
#define __VALIDATION_HPP__
#include <cstddef>

namespace mtk {
namespace validation {
template <class T>
float check_orthogonality16(const T* const matrix, const std::size_t m, const unsigned n);
} // namespace validation
} // namespace mtk

#endif /* end of include guard */
