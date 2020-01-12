#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>
#include "test.hpp"

void qr_test(const std::vector<std::pair<std::size_t, std::size_t>>& test_matrix_size_array) {
	std::cout<<"# precision test"<<std::endl;
	try{
		mtk::test_qr::precision<true, false, float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::precision<true, false, half>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::precision<true, false, float, half>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::precision<false, false, float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::precision<false, false, half>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::precision<true, true, float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::cusolver_precision<float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::cusolver_precision<double>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};

	std::cout<<"# speed test"<<std::endl;
	try{
		mtk::test_qr::speed<true, false, float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::speed<true, false, half>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::speed<true, false, float, half>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::speed<false, false, float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::speed<false, false, half>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::speed<true, true, float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::cusolver_speed<float>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
	try{
		mtk::test_qr::cusolver_speed<double>(test_matrix_size_array);
	}catch(std::runtime_error& e){
		std::cerr<<e.what()<<std::endl;
	};
}

int main() {
	std::vector<std::pair<std::size_t, std::size_t>> test_matrix_size_array;
	qr_test(test_matrix_size_array);
}
