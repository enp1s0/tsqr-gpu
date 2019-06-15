#include <cutf/type.hpp>
#include <cutf/error.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include "test.hpp"
#include "tcqr.hpp"
#include "tsqr.hpp"
#include "utils.hpp"
#include "validation.hpp"

template <class T> std::string get_type_name();
template <> std::string get_type_name<float>() {return "float";}
template <> std::string get_type_name<half>() {return "half";}

namespace {
__global__ void cut_r(float* const dst, const float* const src, const std::size_t m, const std::size_t n) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	const auto x = tid / n;
	const auto y = tid % n;

	if(y > x) return;

	dst[tid] = src[m * x + y];
}
} // namespace 
template <class DST_T, class SRC_T>
__global__ void convert_copy(DST_T* const dst, const SRC_T* const src, const std::size_t size){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= size) return;
	dst[tid] = cutf::type::cast<DST_T>(src[tid]);
}

template <bool UseTC, class T>
void mtk::test::precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t block_size = 256;
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	std::cout<<"m,n,type,tc,error,error_deviation,orthogonality,orthogonality_deviation"<<std::endl;
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_a_test = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
		auto d_q_test = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_r_test = cutf::memory::get_device_unique_ptr<float>(n * n);
		auto d_working_q = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_q_type<T, UseTC>::type>(
				mtk::tsqr::get_working_q_size(m, n));
		auto d_working_r = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_r_type<T, UseTC>::type>(
				mtk::tsqr::get_working_r_size(m, n));
		auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto h_a_test = cutf::memory::get_host_unique_ptr<float>(m * n);
		auto h_q = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);

		std::vector<float> error_list;
		std::vector<float> orthogonality_list;

		for(std::size_t c = 0; c < C; c++) {
			float norm_a = 0.0f;
			for(std::size_t i = 0; i < m * n; i++) {
				const auto tmp = dist(mt);
				h_a.get()[i] = cutf::type::cast<T>(tmp);
				h_a_test.get()[i] = tmp;
				norm_a += tmp * tmp;
			}
			cutf::memory::copy(d_a.get(), h_a.get(), m * n);
			cutf::memory::copy(d_a_test.get(), h_a_test.get(), m * n);

			mtk::tsqr::tsqr16<UseTC, T>(
					d_q.get(), d_r.get(),
					d_a.get(), m, n,
					d_working_q.get(),
					d_working_r.get()
					);

			cutf::memory::copy(h_r.get(), d_r.get(), n * n);

			convert_copy<float, T><<<(m * n + block_size - 1) / block_size, block_size>>>(d_q_test.get(), d_q.get(), m * n);
			convert_copy<float, T><<<(n * n + block_size - 1) / block_size, block_size>>>(d_r_test.get(), d_r.get(), n * n);
			cut_r<<<(n * n + block_size - 1) / block_size, block_size>>>(d_r_test.get(), d_r_test.get(), n, n);

			// verify
			auto cublas = cutf::cublas::get_cublas_unique_ptr();
			const float alpha = 1.0f, beta = -1.0f;
			cutf::cublas::gemm(
					*cublas.get(),
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

		std::cout<<m<<","<<n<<","<<get_type_name<T>()<<","<<(UseTC ? "1" : "0")<<","<<error<<","<<error_deviation<<","<<orthogonality<<","<<orthogonality_deviation<<std::endl;
	}
}

template void mtk::test::precision<true, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::precision<true, half>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::precision<false, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::precision<false, half>(const std::size_t, const std::size_t, const std::size_t);

template <bool UseTC, class T>
void mtk::test::speed(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	auto get_qr_complexity = [](const std::size_t m, const std::size_t n) {
		return 2 * n * (m * m * n + m * m * m);
	};

	std::cout<<"m,n,type,tc,elapsed_time,tflops,working_memory_size"<<std::endl;
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
		auto d_working_q = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_q_type<T, UseTC>::type>(
				mtk::tsqr::get_working_q_size(m, n));
		auto d_working_r = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_r_type<T, UseTC>::type>(
				mtk::tsqr::get_working_r_size(m, n));
		auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto h_q = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);
		for(std::size_t i = 0; i < m * n; i++) {
			const auto tmp = dist(mt);
			h_a.get()[i] = cutf::type::cast<T>(tmp);
		}
		cutf::memory::copy(d_a.get(), h_a.get(), m * n);

		// for cache
		mtk::tsqr::tsqr16<UseTC, T>(
				d_q.get(), d_r.get(),
				d_a.get(), m, n,
				d_working_q.get(),
				d_working_r.get()
				);

		const auto elapsed_time = mtk::utils::get_elapsed_time([&](){
				for(std::size_t c = 0; c < C; c++) {
				mtk::tsqr::tsqr16<UseTC, T>(
						d_q.get(), d_r.get(),
						d_a.get(), m, n,
						d_working_q.get(),
						d_working_r.get()
						);
				}}) / C;

		const auto batch_size = mtk::tsqr::get_batch_size(m);
		const auto complexity = batch_size * get_qr_complexity(m / batch_size, n) + (batch_size - 1) * get_qr_complexity(2 * n, n) + (batch_size - 1) * 4 * n * n * n + 4 * n * n * m;

		std::cout<<m<<","<<n<<","<<get_type_name<T>()<<","<<(UseTC ? "1" : "0")<<","<<elapsed_time<<","<<(complexity / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0))<<","<<
			(mtk::tsqr::get_working_q_size(m, n) * sizeof(typename mtk::tsqr::get_working_q_type<T, UseTC>::type) + mtk::tsqr::get_working_r_size(m, n) * sizeof(typename mtk::tsqr::get_working_r_type<T, UseTC>::type))<<std::endl;
	}
}

template void mtk::test::speed<true, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<true, half>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<false, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<false, half>(const std::size_t, const std::size_t, const std::size_t);

void mtk::test::cusolver_precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t block_size = 1 << 8;
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	std::cout<<"m,n,type,tc,error,error_deviation,orthogonality,orthogonality_deviation"<<std::endl;
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_q = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_r = cutf::memory::get_device_unique_ptr<float>(n * n);
		auto d_tau = cutf::memory::get_device_unique_ptr<float>(n * n);
		auto h_a = cutf::memory::get_host_unique_ptr<float>(m * n);
		auto h_r = cutf::memory::get_host_unique_ptr<float>(n * n);

		std::vector<float> error_list;
		std::vector<float> orthogonality_list;

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

		auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<float>(geqrf_working_memory_size);
		auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<float>(gqr_working_memory_size);
		auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

		for(std::size_t c = 0; c < C; c++) {
			float norm_a = 0.0f;
			for(std::size_t i = 0; i < m * n; i++) {
				const auto tmp = dist(mt);
				h_a.get()[i] = cutf::type::cast<float>(tmp);
				norm_a += tmp * tmp;
			}
			cutf::memory::copy(d_a.get(), h_a.get(), m * n);

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
			const auto alpha = 1.0f, beta = -1.0f;
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
			float norm_diff = 0.0f;
			for(std::size_t i = 0; i < m * n; i++) {
				const auto tmp = cutf::type::cast<float>(h_a.get()[i]);
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
		error_deviation /= C;
		orthogonality_deviation /= C;


		std::cout<<m<<","<<n<<",float,cusolver,"<<error<<","<<error_deviation<<","<<orthogonality<<","<<orthogonality_deviation<<std::endl;
	}
}

void mtk::test::cusolver_speed(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t block_size = 256;
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	auto get_qr_complexity = [](const std::size_t m, const std::size_t n) {
		return 2 * n * (m * m * n + m * m * m);
	};

	std::cout<<"m,n,type,tc,elapsed_time,tflops,working_memory_size"<<std::endl;
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_q = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_r = cutf::memory::get_device_unique_ptr<float>(n * n);
		auto d_tau = cutf::memory::get_device_unique_ptr<float>(n * n);
		auto h_a = cutf::memory::get_host_unique_ptr<float>(m * n);

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

		auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<float>(geqrf_working_memory_size);
		auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<float>(gqr_working_memory_size);
		auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

		for(std::size_t i = 0; i < m * n; i++) {
			const auto tmp = dist(mt);
			h_a.get()[i] = tmp;
		}
		cutf::memory::copy(d_a.get(), h_a.get(), m * n);

		// for cache
		CUTF_HANDLE_ERROR(cutf::cusolver::dn::geqrf(
					*cusolver.get(), m, n,
					d_a.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
					geqrf_working_memory_size, d_info.get()
					));

		CUTF_HANDLE_ERROR(cutf::cusolver::dn::gqr(
					*cusolver.get(), m, n, n,
					d_a.get(), m,
					d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
					d_info.get()
					));

		const auto elapsed_time = mtk::utils::get_elapsed_time([&](){
				for(std::size_t c = 0; c < C; c++) {
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
				}
				CUTF_HANDLE_ERROR(cudaDeviceSynchronize());
				}) / C;

		const auto batch_size = mtk::tsqr::get_batch_size(m);
		const auto complexity = batch_size * get_qr_complexity(m / batch_size, n) + (batch_size - 1) * get_qr_complexity(2 * n, n) + (batch_size - 1) * 4 * n * n * n + 4 * n * n * m;

		std::cout<<m<<","<<n<<",float,cusolver,"<<elapsed_time<<","<<(complexity / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0))<<","<<((geqrf_working_memory_size + gqr_working_memory_size) * sizeof(float))<<std::endl;
	}
}
