#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cutf/type.hpp>

namespace mtk {
namespace utils {
template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * m + i]);
			if(val < 0.0f) {
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}
template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, std::size_t ldm, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * ldm + i]);
			if(val < 0.0f) {
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}
template <class T>
__device__ __host__ inline void print_matrix_16x16(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * 16 + i]);
			if(val < 0.0f) {
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}
template <class T>
__device__ __host__ inline void print_matrix_32x16(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * 32 + i]);
			if(val < 0.0f) {
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}
template <class T>
__device__ __host__ inline void print_matrix_diag(const T* const ptr, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int j = 0; j < n; j++) {
		const auto val = cutf::type::cast<float>(ptr[j * (n + 1)]);
		if(val < 0.0f) {
			printf("%.5f ", val);
		}else{
			printf(" %.5f ", val);
		}
	}
	printf("\n");
}
template <class T>
__device__ __host__ inline void print_matrix_diag_16x16(const T* const ptr, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int j = 0; j < n; j++) {
		const auto val = cutf::type::cast<float>(ptr[j * (16 + 1)]);
		if(val < 0.0f) {
			printf("%.5f ", val);
		}else{
			printf(" %.5f ", val);
		}
	}
	printf("\n");
}

// millisecond
template <class RunFunc>
inline double get_elapsed_time(RunFunc run_func) {
	const auto start_clock = std::chrono::system_clock::now();
	run_func();
	const auto end_clock = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / 1000.0 / 1000.0;
}

template <class T>
inline void print_value(const T val, const std::string name, const std::string unit = "") {
	std::cout<<std::left<<std::setw(25)<<name<<" : "<<val;
	if(unit != "")std::cout<<" ["<<unit<<"]";
	std::cout<<std::endl;
}

template <class T>
inline double get_error(const T* const matrix_a, const T* const matrix_b, const std::size_t m, const std::size_t n) {
	double norm = 0.0;
	double norm_a = 0.0;
	for(std::size_t i = 0; i < m * n; i++) {
		const auto tmp = cutf::type::cast<double>(matrix_a[i]) - cutf::type::cast<double>(matrix_b[i]);
		norm += tmp * tmp;
		const auto tmp_a = cutf::type::cast<double>(matrix_a[i]);
		norm_a += tmp_a * tmp_a;
	}
	return std::sqrt(norm / norm_a);
}
} // namespace utils
} // namespace mtk

#endif /* end of include guard */
