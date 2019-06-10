#ifndef __TEST_HPP__
#define __TEST_HPP__
#include <cstddef>
namespace mtk {
namespace test {
template <bool UseTC, class T>
void precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n);
template <bool UseTC, class T>
void speed(const std::size_t min_m, const std::size_t max_m, const std::size_t n);
} // namespace test
} // namespace mtk

#endif /* end of include guard */
