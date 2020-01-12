#ifndef __TEST_HPP__
#define __TEST_HPP__
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
namespace mtk {
namespace test_qr {
template <bool UseTC, bool Refine, class T, class CORE_T = T>
void precision(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector);
template <bool UseTC, bool Refine, class T, class CORE_T = T>
void speed(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector);
template <class T>
void cusolver_precision(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector);
template <class T>
void cusolver_speed(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector);
} // namespace test_qr
} // namespace mtk

#endif /* end of include guard */
