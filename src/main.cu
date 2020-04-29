#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>
#include "test.hpp"

constexpr std::size_t test_count = 16;

void qr_test(const std::vector<std::tuple<std::size_t, std::size_t, float>>& test_matrix_config_list) {
	std::cout << "# accuracy test" << std::endl;
	mtk::test_qr::accuracy<true , false, false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<true , false, false, half >(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<true , false, false, float, half>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<false, false, false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<false, false, false, half >(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<true , true , false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<true , false, true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<true , false, true , half >(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<true , false, true , float, half>(test_matrix_config_list, test_count);
	mtk::test_qr::accuracy<false, false, true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<false, false, true , half >(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy<true , true , true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::cusolver_accuracy<float>(test_matrix_config_list                  , test_count);
	mtk::test_qr::cusolver_accuracy<double>(test_matrix_config_list                 , test_count);
	std::cout << "# speed test" << std::endl;
	mtk::test_qr::speed<true , false, false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<true , false, false, half >(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<true , false, false, float, half>(test_matrix_config_list, test_count);
	mtk::test_qr::speed<false, false, false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<false, false, false, half >(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<true , true , false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<true , false, true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<true , false, true , half >(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<true , false, true , float, half>(test_matrix_config_list, test_count);
	mtk::test_qr::speed<false, false, true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<false, false, true , half >(test_matrix_config_list      , test_count);
	mtk::test_qr::speed<true , true , true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::cusolver_speed<float>(test_matrix_config_list                  , test_count);
	mtk::test_qr::cusolver_speed<double>(test_matrix_config_list                 , test_count);
}


void qr_test_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& test_matrix_config_list) {
	std::cout << "# condition number test" << std::endl;
	mtk::test_qr::accuracy_cond<true , true , false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<true , false, false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<false, false, false, float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<true , false, false, half >(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<false, false, false, half >(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<true , true , true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<true , false, true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<false, false, true , float>(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<true , false, true , half >(test_matrix_config_list      , test_count);
	mtk::test_qr::accuracy_cond<false, false, true , half >(test_matrix_config_list      , test_count);
	mtk::test_qr::cusolver_accuracy_cond<float>(test_matrix_config_list                  , test_count);
	mtk::test_qr::cusolver_accuracy_cond<double>(test_matrix_config_list                 , test_count);
}

int main() {
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
