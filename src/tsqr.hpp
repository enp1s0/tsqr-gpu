#ifndef __TSQR_HPP__
#define __TSQR_HPP__
#include <cstddef>
namespace mtk {
namespace tsqr {
// get working memory type
template <class T, bool UseTC>
template <bool UseTC, class T>
void tsqr16(T* const q_ptr, T* const r_ptr, const T* const a_ptr, const std::size_t m, const std::size_t n, typename get_working_q_type<T, UseTC>::type* const working_q_ptr, typename get_working_r_type<T, UseTC>::type* const working_r_ptr);
} // namespace tsqr
} // namespace mtk

#endif /* end of include guard */
