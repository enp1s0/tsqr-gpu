#include <iostream>
#include <cuda_fp16.h>
#include "test.hpp"

constexpr std::size_t min_m = 1 << 10;
constexpr std::size_t max_m = 1 << 24;
constexpr std::size_t n = 16;

int main() {
	std::cout<<"# precision test"<<std::endl;
	mtk::test::precision<true, float>(min_m, max_m, n);
	mtk::test::precision<true, half>(min_m, max_m, n);
	mtk::test::precision<false, float>(min_m, max_m, n);
	mtk::test::precision<false, half>(min_m, max_m, n);

	std::cout<<"# speed test"<<std::endl;
	mtk::test::speed<true, float>(min_m, max_m, n);
	mtk::test::speed<true, half>(min_m, max_m, n);
	mtk::test::speed<false, float>(min_m, max_m, n);
	mtk::test::speed<false, half>(min_m, max_m, n);
}
