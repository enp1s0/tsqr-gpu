#include <cutf/type.hpp>
#include <cutf/error.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include "test.hpp"
#include "tcqr.hpp"
#include "utils.hpp"
#include "blockqr.hpp"
#include "validation.hpp"
#include "latms.hpp"

namespace {
template <class T>
void get_rand_matrix_with_cond_number(
		T* const mat, std::size_t m, std::size_t n,
	   	const float cond_number, unsigned long seed, const float max_abs_v = 16.0f) {
	assert(cond_number >= 1.0f);
	assert(m >= n);

	const std::size_t rank = std::min(m, n);

	auto s_array = std::make_unique<T[]>(rank);

	s_array.get()[0] = std::sqrt(cond_number);
	s_array.get()[rank - 1] = 1.0f;
	std::mt19937 mt(seed);
	std::uniform_real_distribution<T> dist(1.0f, std::sqrt(cond_number));
	for (unsigned i = 0; i < rank; i++) {
		s_array.get()[i] = 1 / s_array.get()[i];
	}
	T* tmp_mat;
	T* d_tmp_mat_0;
	T* d_tmp_mat_1;
	cudaMallocHost(&tmp_mat, sizeof(T) * m * n);
	cudaMalloc(&d_tmp_mat_0, sizeof(T) * m * n);
	cudaMalloc(&d_tmp_mat_1, sizeof(T) * m * n);

	T cond_ratio = 0.0;
	do {
		for (unsigned i = 1; i < rank - 1; i++) {
			s_array.get()[i] = dist(mt);
		}
		std::sort(s_array.get(), s_array.get() + rank, std::greater<T>());
		mtk::utils::latms(
				d_tmp_mat_0,
				m, n,
				rank,
				s_array.get()
				);
		cutf::memory::copy(d_tmp_mat_1, d_tmp_mat_0, m * n);
		const auto real_cond = mtk::utils::get_cond(d_tmp_mat_0, m, n);
		cond_ratio = real_cond / cond_number;
	} while (cond_ratio < 0.9);

	cutf::memory::copy(tmp_mat, d_tmp_mat_1, m * n);

	T abs_max = 0.0f;
	cudaDeviceSynchronize();
	for (unsigned i = 0; i < m * n; i++) {
		abs_max = std::max(abs_max, std::abs(tmp_mat[i]));
	}
#pragma omp parallel for
	for (unsigned i = 0; i < m * n; i++) {
		mat[i] = tmp_mat[i];// / abs_max * max_abs_v;
	}
	cudaFree(d_tmp_mat_0);
	cudaFree(d_tmp_mat_1);
	cudaFreeHost(tmp_mat);
}

template <class T> std::string get_type_name();
template <> std::string get_type_name<double>() {return "double";}
template <> std::string get_type_name<float>() {return "float";}
template <> std::string get_type_name<half>() {return "half";}

template <class T>
__global__ void cut_r(T* const dst, const T* const src, const std::size_t m, const std::size_t n) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	const auto x = tid / n;
	const auto y = tid % n;

	if(y > x) return;

	dst[tid] = src[m * x + y];
}

template <class DST_T, class SRC_T>
__global__ void convert_copy(DST_T* const dst, const SRC_T* const src, const std::size_t size){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= size) return;
	dst[tid] = cutf::type::cast<DST_T>(src[tid]);
}

template <class DST_T>
__global__ void make_zero(DST_T* const dst, const std::size_t size){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= size) return;
	dst[tid] = cutf::type::cast<DST_T>(0);
}

void print_accuracy_head() {
	std::cout << "m,n,cond,type,core_type,tc,correction,reorthogonalization,error,error_deviation,orthogonality,orthogonality_deviation" << std::endl;
	std::cout.flush();
}
} // namespace

template <bool UseTC, bool Refine, bool Reorthogonalize, class T, class CORE_T = T>
void mtk::test_qr::accuracy_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& test_case_tuple_vector, const std::size_t C) {
	constexpr std::size_t block_size = 256;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	print_accuracy_head();

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	for(const auto &test_case_tuple : test_case_tuple_vector) {
		try {
			const std::size_t m = std::get<0>(test_case_tuple);
			const std::size_t n = std::get<1>(test_case_tuple);
			const float condition_number = std::get<2>(test_case_tuple);
			auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_a_test = cutf::memory::get_device_unique_ptr<float>(m * n);
			auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto d_q_test = cutf::memory::get_device_unique_ptr<float>(m * n);
			auto d_r_test = cutf::memory::get_device_unique_ptr<float>(n * n);

			mtk::qr::buffer<T, UseTC, Refine, Reorthogonalize> buffer;
			buffer.allocate(m, n);

			auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_a_test = cutf::memory::get_host_unique_ptr<float>(m * n);
			auto h_q = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);

			std::vector<float> error_list;
			std::vector<float> orthogonality_list;

			for(std::size_t c = 0; c < C; c++) {
				float norm_a = 0.0f;
				get_rand_matrix_with_cond_number(h_a_test.get(), m, n, condition_number, mt(), 1.0f);
				CUTF_HANDLE_ERROR(cudaDeviceSynchronize());
				for(std::size_t i = 0; i < m * n; i++) {
					const auto tmp = h_a_test.get()[i];
					h_a.get()[i] = tmp;
					norm_a += tmp * tmp;
				}
				cutf::memory::copy(d_a.get(), h_a.get(), m * n);
				cutf::memory::copy(d_a_test.get(), h_a_test.get(), m * n);
				make_zero<T><<<(n * n + block_size - 1) / block_size, block_size>>>(d_r.get(), n * n);

				CUTF_HANDLE_ERROR(cudaDeviceSynchronize());
				mtk::qr::qr<UseTC, Refine, Reorthogonalize, T, CORE_T>(
						d_q.get(), m,
						d_r.get(), n,
						d_a.get(), m,
						m, n,
						buffer,
						*cublas_handle.get()
						);
				CUTF_HANDLE_ERROR(cudaDeviceSynchronize());

				convert_copy<float, T><<<(m * n + block_size - 1) / block_size, block_size>>>(d_q_test.get(), d_q.get(), m * n);
				convert_copy<float, T><<<(n * n + block_size - 1) / block_size, block_size>>>(d_r_test.get(), d_r.get(), n * n);
				CUTF_HANDLE_ERROR(cudaDeviceSynchronize());

				// verify
				const float alpha = 1.0f, beta = -1.0f;
				cutf::cublas::gemm(
						*cublas_handle.get(),
						CUBLAS_OP_N, CUBLAS_OP_N,
						m, n, n,
						&alpha,
						d_q_test.get(), m,
						d_r_test.get(), n,
						&beta,
						d_a_test.get(), m
						);

				cutf::memory::copy(h_a_test.get(), d_a_test.get(), m * n);
				float norm_diff = 0.0f;
				for(std::size_t i = 0; i < m * n; i++) {
					const auto tmp = h_a_test.get()[i];
					norm_diff += tmp * tmp;
				}
				error_list.push_back(std::sqrt(norm_diff/norm_a));
				orthogonality_list.push_back(mtk::validation::check_orthogonality16(d_q.get(), m, n));
			}
			float error = 0.0f;
			float orthogonality = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				error += error_list[c];
				orthogonality += orthogonality_list[c];
			}
			error /= C;
			orthogonality /= C;

			float error_deviation = 0.0f;
			float orthogonality_deviation = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				error_deviation += (error_list[c] - error) * (error_list[c] - error);
				orthogonality_deviation += (orthogonality_list[c] - orthogonality) * (orthogonality_list[c] - orthogonality);
			}
			error_deviation = std::sqrt(error_deviation / C);
			orthogonality_deviation = std::sqrt(orthogonality_deviation / C);

			std::cout << m << ","
				<< n << ","
				<< condition_number << ","
				<< get_type_name<T>() << ","
				<< get_type_name<CORE_T>() << ","
				<< (UseTC ? "1" : "0") << ","
				<< (Refine ? "1" : "0") << ","
				<< (Reorthogonalize ? "1" : "0") << ","
				<< error << ","
				<< error_deviation << ","
				<< orthogonality << ","
				<< orthogonality_deviation << std::endl;
			std::cout.flush();
		} catch (std::runtime_error& e) {
			std::cerr<<e.what()<<std::endl;
			continue;
		}
	}
}

template void mtk::test_qr::accuracy_cond<true , false, false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<true , false, false, half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<false, false, false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<false, false, false, half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<true , true , false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<true , false, false, float, half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<true , false, true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<true , false, true , half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<false, false, true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<false, false, true , half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<true , true , true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy_cond<true , false, true , float, half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);


template <class T>
void mtk::test_qr::cusolver_accuracy_cond(const std::vector<std::tuple<std::size_t, std::size_t, float>>& test_case_tuple_vector, const std::size_t C) {
	constexpr std::size_t block_size = 1 << 8;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	print_accuracy_head();

	for(const auto &test_case_tuple : test_case_tuple_vector) {
		try {
			const std::size_t m = std::get<0>(test_case_tuple);
			const std::size_t n = std::get<1>(test_case_tuple);
			const float condition_number = std::get<2>(test_case_tuple);
			auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_a_cond = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto d_tau = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);

			std::vector<T> error_list;
			std::vector<T> orthogonality_list;

			auto cusolver = cutf::cusolver::get_cusolver_dn_unique_ptr();

			// working memory
			int geqrf_working_memory_size, gqr_working_memory_size;
			CUTF_HANDLE_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
						*cusolver.get(), m, n,
						d_a.get(), m, &geqrf_working_memory_size
						));
			CUTF_HANDLE_ERROR(cutf::cusolver::dn::gqr_buffer_size(
						*cusolver.get(), m, n, n,
						d_a.get(), m, d_tau.get(), &gqr_working_memory_size
						));

			auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<T>(geqrf_working_memory_size);
			auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<T>(gqr_working_memory_size);
			auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

			for(std::size_t c = 0; c < C; c++) {
				T norm_a = 0.0f;
				get_rand_matrix_with_cond_number(h_a.get(), m, n, condition_number, mt(), 1.0f);
				CUTF_HANDLE_ERROR(cudaDeviceSynchronize());
				for(std::size_t i = 0; i < m * n; i++) {
					const auto tmp = h_a.get()[i];
					norm_a += tmp * tmp;
				}
				cutf::memory::copy(d_a.get(), h_a.get(), m * n);
				CUTF_HANDLE_ERROR(cudaDeviceSynchronize());

				CUTF_HANDLE_ERROR(cutf::cusolver::dn::geqrf(
							*cusolver.get(), m, n,
							d_a.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
							geqrf_working_memory_size, d_info.get()
							));
				cut_r<<<(n * n + block_size - 1) / block_size, block_size>>>(d_r.get(), d_a.get(), m, n);

				CUTF_HANDLE_ERROR(cutf::cusolver::dn::gqr(
							*cusolver.get(), m, n, n,
							d_a.get(), m,
							d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
							d_info.get()
							));

				cutf::memory::copy(d_q.get(), d_a.get(), n * m);
				cutf::memory::copy(d_a.get(), h_a.get(), m * n);
				//cutf::memory::copy(h_r.get(), d_r.get(), n * n);
				//mtk::utils::print_matrix(h_r.get(), n, n, "R");

				// verify
				auto cublas = cutf::cublas::get_cublas_unique_ptr();
				const T alpha = 1.0f, beta = -1.0f;
				cutf::cublas::gemm(
						*cublas.get(),
						CUBLAS_OP_N, CUBLAS_OP_N,
						m, n, n,
						&alpha,
						d_q.get(), m,
						d_r.get(), n,
						&beta,
						d_a.get(), m
						);

				cutf::memory::copy(h_a.get(), d_a.get(), m * n);
				T norm_diff = 0.0f;
				for(std::size_t i = 0; i < m * n; i++) {
					const auto tmp = cutf::type::cast<T>(h_a.get()[i]);
					norm_diff += tmp * tmp;
				}
				error_list.push_back(std::sqrt(norm_diff/norm_a));
				orthogonality_list.push_back(mtk::validation::check_orthogonality16(d_q.get(), m, n));
			}
			T error = 0.0f;
			T orthogonality = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				error += error_list[c];
				orthogonality += orthogonality_list[c];
			}
			error /= C;
			orthogonality /= C;

			T error_deviation = 0.0f;
			T orthogonality_deviation = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				error_deviation += (error_list[c] - error) * (error_list[c] - error);
				orthogonality_deviation += (orthogonality_list[c] - orthogonality) * (orthogonality_list[c] - orthogonality);
			}
			error_deviation /= C;
			orthogonality_deviation /= C;

			std::cout << m << ","
				<< n << ","
				<< condition_number << ","
				<< get_type_name<T>() << ","
				<< get_type_name<T>() << ","
				<< "cusolver" << ","
				<< "0" << ","
				<< "0" << ","
				<< error << ","
				<< error_deviation << ","
				<< orthogonality << ","
				<< orthogonality_deviation << std::endl;
			std::cout.flush();
		} catch (std::runtime_error& e) {
			std::cerr<<e.what()<<std::endl;
			continue;
		}
	}
}

template void mtk::test_qr::cusolver_accuracy_cond<float >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::cusolver_accuracy_cond<double>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
