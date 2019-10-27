#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>
#include "test.hpp"

constexpr std::size_t min_m = 1 << 10;
constexpr std::size_t max_m = 1 << 27;
constexpr std::size_t n = 16;

int main() {
	std::cout<<"# precision test"<<std::endl;
	try{
		mtk::test::precision<true, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::precision<true, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::precision<false, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::precision<false, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::cusolver_precision<float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::cusolver_precision<double>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};

	std::cout<<"# speed test"<<std::endl;
	try{
		mtk::test::speed<true, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::speed<true, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::speed<false, false, float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::speed<false, false, half>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::cusolver_speed<float>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test::cusolver_speed<double>(min_m, max_m, n);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
}
