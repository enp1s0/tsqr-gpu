#ifndef __TEST_HPP__
#define __TEST_HPP__
#include <cstddef>
#include <type_traits>
namespace mtk {
namespace test {
template <bool UseTC, bool Refine, class T, class CORE_T = T>
void precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n);
template <bool UseTC, bool Refine, class T, class CORE_T = T>
void speed(const std::size_t min_m, const std::size_t max_m, const std::size_t n);
template <class T>
void cusolver_precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n);
template <class T>
void cusolver_speed(const std::size_t min_m, const std::size_t max_m, const std::size_t n);
} // namespace test
} // namespace mtk

#endif /* end of include guard */
