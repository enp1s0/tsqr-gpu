#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>
#include <git_status.hpp>
#include "test.hpp"

constexpr std::size_t test_count = 16;

void qr_test(const std::vector<std::tuple<std::size_t, std::size_t, float>>& test_matrix_config_list) {
	std::cout << "# accuracy test" << std::endl;
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp16_notc        , false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp16_tc_nocor    , false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp32_notc        , false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp32_tc_nocor    , false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp32_tc_cor      , false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::tf32_tc_nocor_emu, false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::tf32_tc_cor_emu  , false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::mixed_tc_cor_emu , false>(test_matrix_config_list, test_count);

	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp16_notc        , true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp16_tc_nocor    , true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp32_notc        , true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp32_tc_nocor    , true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::fp32_tc_cor      , true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::tf32_tc_nocor_emu, true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::tf32_tc_cor_emu  , true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<mtk::test_qr::compute_mode::mixed_tc_cor_emu , true >(test_matrix_config_list, test_count);

	mtk::test_qr::cusolver_accuracy<float>(test_matrix_config_list                  , test_count);
	mtk::test_qr::cusolver_accuracy<double>(test_matrix_config_list                 , test_count);

	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp16_notc    , false>(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp16_tc_nocor, false>(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp32_notc    , false>(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp32_tc_nocor, false>(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp32_tc_cor  , false>(test_matrix_config_list, test_count);

	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp16_notc    , true >(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp16_tc_nocor, true >(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp32_notc    , true >(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp32_tc_nocor, true >(test_matrix_config_list, test_count);
	mtk::test_qr::speed<mtk::test_qr::compute_mode::fp32_tc_cor  , true >(test_matrix_config_list, test_count);

	mtk::test_qr::cusolver_speed<float>(test_matrix_config_list                  , test_count);
	mtk::test_qr::cusolver_speed<double>(test_matrix_config_list                 , test_count);
}


void qr_test_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& test_matrix_config_list) {
	std::cout << "# condition number test" << std::endl;
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp16_notc        , false>(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp16_tc_nocor    , false>(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp32_notc        , false>(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp32_tc_nocor    , false>(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp32_tc_cor      , false>(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::tf32_tc_nocor_emu, false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::tf32_tc_cor_emu  , false>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::mixed_tc_cor_emu , false>(test_matrix_config_list, test_count);

	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp16_notc        , true >(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp16_tc_nocor    , true >(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp32_notc        , true >(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp32_tc_nocor    , true >(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::fp32_tc_cor      , true >(test_matrix_config_list);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::tf32_tc_nocor_emu, true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::tf32_tc_cor_emu  , true >(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy_cond<mtk::test_qr::compute_mode::mixed_tc_cor_emu , true >(test_matrix_config_list, test_count);

	mtk::test_qr::cusolver_accuracy_cond<float>(test_matrix_config_list                  , test_count);
	mtk::test_qr::cusolver_accuracy_cond<double>(test_matrix_config_list                 , test_count);
}

int main() {
	mtk::runtime_status::git::print_info(true);
	{
		std::vector<std::tuple<std::size_t, std::size_t, float>> test_matrix_config_list;
		for (std::size_t m = 10; m <= 15; m++) {
			for (std::size_t n = 10; n <= m; n++) {
				test_matrix_config_list.push_back(std::make_tuple(1lu << m, 1lu << n, 1.0f));
			}
		}
		qr_test(test_matrix_config_list);
	}

	{
		constexpr std::size_t m = 1lu << 15;
		constexpr std::size_t n = 1lu << 7;
		std::vector<std::tuple<std::size_t, std::size_t, float>> test_matrix_config_list;
		for (std::size_t c = 2; c <= 15; c++) {
			test_matrix_config_list.push_back(std::make_tuple(m, n, 1lu << c));
		}
		qr_test_cond(test_matrix_config_list);
	}
}
