#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>
#include "test.hpp"

constexpr std::size_t test_count = 16;

void qr_test(const std::vector<std::pair<std::size_t, std::size_t>>& test_matrix_size_array) {
	std::cout << "# precision test" << std::endl;
	mtk::test_qr::precision<true , false, false, float>(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<true , false, false, half >(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<true , false, false, float, half>(test_matrix_size_array, test_count);
	mtk::test_qr::precision<false, false, false, float>(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<false, false, false, half >(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<true , true , false, float>(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<true , false, true , float>(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<true , false, true , half >(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<true , false, true , float, half>(test_matrix_size_array, test_count);
	mtk::test_qr::precision<false, false, true , float>(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<false, false, true , half >(test_matrix_size_array      , test_count);
	mtk::test_qr::precision<true , true , true , float>(test_matrix_size_array      , test_count);
	mtk::test_qr::cusolver_precision<float>(test_matrix_size_array);
	mtk::test_qr::cusolver_precision<double>(test_matrix_size_array);
	std::cout << "# speed test" << std::endl;
	mtk::test_qr::speed<true , false, false, float>(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<true , false, false, half >(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<true , false, false, float, half>(test_matrix_size_array, test_count);
	mtk::test_qr::speed<false, false, false, float>(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<false, false, false, half >(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<true , true , false, float>(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<true , false, true , float>(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<true , false, true , half >(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<true , false, true , float, half>(test_matrix_size_array, test_count);
	mtk::test_qr::speed<false, false, true , float>(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<false, false, true , half >(test_matrix_size_array      , test_count);
	mtk::test_qr::speed<true , true , true , float>(test_matrix_size_array      , test_count);
	mtk::test_qr::cusolver_speed<float>(test_matrix_size_array);
	mtk::test_qr::cusolver_speed<double>(test_matrix_size_array);
}

int main() {
	std::vector<std::pair<std::size_t, std::size_t>> test_matrix_size_array;
	for (std::size_t m = 10; m <= 15; m++) {
		for (std::size_t n = 10; n <= m; n++) {
			test_matrix_size_array.push_back(std::make_pair(1lu << m, 1lu << n));
		}
	}
	qr_test(test_matrix_size_array);
}
