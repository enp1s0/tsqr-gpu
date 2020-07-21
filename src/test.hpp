#ifndef __TEST_HPP__
#define __TEST_HPP__
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>
namespace mtk {
namespace test_qr {

enum compute_mode {
	fp16_notc,
	fp32_notc,
	fp16_tc_nocor,
	fp32_tc_nocor,
	tf32_tc_nocor,
	fp32_tc_cor,
	tf32_tc_cor,
	tf32_tc_cor_emu,
	mixed_tc_cor,
};

template <mtk::test_qr::compute_mode mode, bool Reorthogonalize>
void accuracy(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16);
template <mtk::test_qr::compute_mode mode, bool Reorthogonalize>
void speed(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16);

template <class T>
void cusolver_accuracy(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16);
template <class T>
void cusolver_speed(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16);

template <mtk::test_qr::compute_mode mode, bool Reorthogonalize>
void accuracy_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16);
template <class T>
void cusolver_accuracy_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16);
} // namespace test_qr
} // namespace mtk

#endif /* end of include guard */
