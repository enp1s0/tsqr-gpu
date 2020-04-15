#ifndef __TEST_HPP__
#define __TEST_HPP__
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>
namespace mtk {
namespace test_qr {
template <bool UseTC, bool Refine, bool Reorthogonalize, class T, class CORE_T = T>
void precision(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector, const std::size_t C = 16);
template <bool UseTC, bool Refine, bool Reorthogonalize, class T, class CORE_T = T>
void speed(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector, const std::size_t C = 16);
template <class T>
void cusolver_precision(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector, const std::size_t C = 16);
template <class T>
void cusolver_speed(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector, const std::size_t C = 16);

template <bool UseTC, bool Refine, bool Reorthogonalize, class T, class CORE_T = T>
void precision_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& size_pair_vector, const std::size_t C = 16);
template <class T>
void cusolver_precision_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& size_pair_vector, const std::size_t C = 16);
} // namespace test_qr
} // namespace mtk

#endif /* end of include guard */
