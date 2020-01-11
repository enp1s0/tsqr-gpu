#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>
#include "test.hpp"

void tsqr_test(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	std::cout<<"# precision test"<<std::endl;
	try{
		mtk::test_tsqr::precision<true, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::precision<true, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::precision<true, false, float, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::precision<false, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::precision<false, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::precision<true, true, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::cusolver_precision<float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::cusolver_precision<double>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};

	std::cout<<"# speed test"<<std::endl;
	try{
		mtk::test_tsqr::speed<true, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::speed<true, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::speed<true, false, float, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::speed<false, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::speed<false, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::speed<true, true, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::cusolver_speed<float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_tsqr::cusolver_speed<double>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
}

void blockqr_test(const std::size_t min_m, const std::size_t max_m) {
	std::cout<<"# precision test"<<std::endl;
	try{
		mtk::test_blockqr::precision<true, false, float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::precision<true, false, half>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::precision<true, false, float, half>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::precision<false, false, float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::precision<false, false, half>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::precision<true, true, float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::cusolver_precision<float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::cusolver_precision<double>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};

	std::cout<<"# speed test"<<std::endl;
	try{
		mtk::test_blockqr::speed<true, false, float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::speed<true, false, half>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::speed<true, false, float, half>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::speed<false, false, float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::speed<false, false, half>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::speed<true, true, float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::cusolver_speed<float>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_blockqr::cusolver_speed<double>(min_m, max_m);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
}

int main() {
	blockqr_test(1lu << 10, 1lu << 27);
	tsqr_test(1lu << 5, 1lu << 20, 1lu << 5);}
