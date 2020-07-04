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
#include "utils.hpp"
#include "blockqr.hpp"
#include "validation.hpp"

namespace {
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
	std::cout << "m,n,rand_range,type,core_type,tc,correction,reorthogonalization,residual,residual_variance,orthogonality,orthogonality_variance" << std::endl;
	std::cout.flush();
}

void print_speed_head() {
	std::cout << "m,n,rand_range,type,core_type,tc,correction,reorthogonalization,elapsed_time,tflops,working_memory_size" << std::endl;
	std::cout.flush();
}
} // namespace

template <bool UseTC, bool Correction, bool Reorthogonalize, class T, class CORE_T>
void mtk::test_qr::accuracy(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C) {
	constexpr std::size_t block_size = 256;
	std::mt19937 mt(std::random_device{}());

	print_accuracy_head();

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	for(const auto &matrix_config : matrix_config_list) {
		try {
			const std::size_t m = std::get<0>(matrix_config);
			const std::size_t n = std::get<1>(matrix_config);
			const float rand_range_abs = std::get<2>(matrix_config);
			std::uniform_real_distribution<> dist(-rand_range_abs, rand_range_abs);
			auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_a_test = cutf::memory::get_device_unique_ptr<float>(m * n);
			auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto d_q_test = cutf::memory::get_device_unique_ptr<float>(m * n);
			auto d_r_test = cutf::memory::get_device_unique_ptr<float>(n * n);

			mtk::qr::buffer<T, UseTC, Correction, Reorthogonalize> buffer;
			buffer.allocate(m, n);

			auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_a_test = cutf::memory::get_host_unique_ptr<float>(m * n);
			auto h_q = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);

			std::vector<float> residual_list;
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
				make_zero<T><<<(n * n + block_size - 1) / block_size, block_size>>>(d_r.get(), n * n);

				CUTF_CHECK_ERROR(cudaDeviceSynchronize());
				mtk::qr::qr<UseTC, Correction, Reorthogonalize, T, CORE_T>(
						d_q.get(), m,
						d_r.get(), n,
						d_a.get(), m,
						m, n,
						buffer,
						*cublas_handle.get()
						);
				CUTF_CHECK_ERROR(cudaDeviceSynchronize());

				convert_copy<float, T><<<(m * n + block_size - 1) / block_size, block_size>>>(d_q_test.get(), d_q.get(), m * n);
				convert_copy<float, T><<<(n * n + block_size - 1) / block_size, block_size>>>(d_r_test.get(), d_r.get(), n * n);
				CUTF_CHECK_ERROR(cudaDeviceSynchronize());

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
				residual_list.push_back(std::sqrt(norm_diff/norm_a));
				orthogonality_list.push_back(mtk::validation::check_orthogonality16(d_q.get(), m, n));
			}
			float residual = 0.0f;
			float orthogonality = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				residual += residual_list[c];
				orthogonality += orthogonality_list[c];
			}
			residual /= C;
			orthogonality /= C;

			float residual_variance = 0.0f;
			float orthogonality_variance = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				residual_variance += (residual_list[c] - residual) * (residual_list[c] - residual);
				orthogonality_variance += (orthogonality_list[c] - orthogonality) * (orthogonality_list[c] - orthogonality);
			}
			residual_variance = residual_variance / C;
			orthogonality_variance = orthogonality_variance / C;

			std::cout << m << ","
				<< n << ","
				<< rand_range_abs << ","
				<< get_type_name<T>() << ","
				<< get_type_name<CORE_T>() << ","
				<< (UseTC ? "1" : "0") << ","
				<< (Correction ? "1" : "0") << ","
				<< (Reorthogonalize ? "1" : "0") << ","
				<< residual << ","
				<< residual_variance << ","
				<< orthogonality << ","
				<< orthogonality_variance << std::endl;
			std::cout.flush();
		} catch (std::runtime_error& e) {
			std::cerr<<e.what()<<std::endl;
			continue;
		}
	}
}

template void mtk::test_qr::accuracy<true , false, false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<true , false, false, half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<false, false, false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<false, false, false, half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<true , true , false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<true , false, false, float, half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<true , false, true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<true , false, true , half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<false, false, true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<false, false, true , half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<true , true , true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::accuracy<true , false, true , float, half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);

template <bool UseTC, bool Correction, bool Reorthogonalize, class T, class CORE_T>
void mtk::test_qr::speed(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C) {
	std::mt19937 mt(std::random_device{}());

	print_speed_head();

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	for(const auto &matrix_config : matrix_config_list) {
		try {
			const std::size_t m = std::get<0>(matrix_config);
			const std::size_t n = std::get<1>(matrix_config);
			const float rand_range_abs = std::get<2>(matrix_config);
			std::uniform_real_distribution<> dist(-rand_range_abs, rand_range_abs);
			auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);

			mtk::qr::buffer<T, UseTC, Correction, Reorthogonalize> buffer;
			buffer.allocate(m, n);

			auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_q = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);
			for(std::size_t i = 0; i < m * n; i++) {
				const auto tmp = dist(mt);
				h_a.get()[i] = cutf::type::cast<T>(tmp);
			}
			cutf::memory::copy(d_a.get(), h_a.get(), m * n);

			// for cache
			mtk::qr::qr<UseTC, Correction, Reorthogonalize, T, CORE_T>(
					d_q.get(), m,
					d_r.get(), n,
					d_a.get(), m,
					m, n,
					buffer,
					*cublas_handle.get()
					);

			const auto elapsed_time = mtk::utils::get_elapsed_time([&](){
					for(std::size_t c = 0; c < C; c++) {
					mtk::qr::qr<UseTC, Correction, Reorthogonalize, T, CORE_T>(
							d_q.get(), m,
							d_r.get(), n,
							d_a.get(), m,
							m, n,
							buffer,
							*cublas_handle.get()
							);
					}}) / C;

			std::size_t complexity = 0;
			const auto batch_size = mtk::tsqr::get_batch_size(m);
			const auto pannel_block_size = (n + 16 - 1) / 16;

			auto get_qr_complexity = [](const std::size_t m, const std::size_t n) {
				return 2 * n * (m * m * n + m * m * m);
			};
			const auto tsqr_comprexity = [&get_qr_complexity, &batch_size](const std::size_t m, const std::size_t n) {
				return batch_size * get_qr_complexity(m / batch_size, n) + (batch_size - 1) * get_qr_complexity(2 * n, n) + (batch_size - 1) * 4 * n * n * n + 4 * n * n * m;
			};

			for (std::size_t i = 0; i < pannel_block_size; i++) {
				const auto local_n = std::min(16lu, n - i * 16lu);
				complexity += tsqr_comprexity(m, local_n);
				complexity += 2 * 2 * 16 * 16 * i * m;
			}

			std::cout << m << ","
				<< n << ","
				<< rand_range_abs << ","
				<< get_type_name<T>() << ","
				<< get_type_name<CORE_T>() << ","
				<< (UseTC ? "1" : "0") << ","
				<< (Correction ? "1" : "0") << ","
				<< (Reorthogonalize ? "1" : "0") << ","
				<< elapsed_time << ","
				<< (complexity / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0)) << ","
				<< buffer.get_device_memory_size() << std::endl;
			std::cout.flush();
		} catch (std::runtime_error& e) {
			std::cerr<<e.what()<<std::endl;
			continue;
		}
	}
}

template void mtk::test_qr::speed<true , false, false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<true , false, false, half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<false, false, false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<false, false, false, half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<true , true , false, float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<true , false, false, float, half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<true , false, true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<true , false, true , half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<false, false, true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<false, false, true , half , half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<true , true , true , float, float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::speed<true , false, true , float, half >(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);

template <class T>
void mtk::test_qr::cusolver_accuracy(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C) {
	constexpr std::size_t block_size = 1 << 8;
	std::mt19937 mt(std::random_device{}());

	print_accuracy_head();

	for(const auto &matrix_config : matrix_config_list) {
		try {
			const std::size_t m = std::get<0>(matrix_config);
			const std::size_t n = std::get<1>(matrix_config);
			const float rand_range_abs = std::get<2>(matrix_config);
			std::uniform_real_distribution<> dist(-rand_range_abs, rand_range_abs);
			auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto d_tau = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
			auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);

			std::vector<T> residual_list;
			std::vector<T> orthogonality_list;

			auto cusolver = cutf::cusolver::get_cusolver_dn_unique_ptr();

			// working memory
			int geqrf_working_memory_size, gqr_working_memory_size;
			CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
						*cusolver.get(), m, n,
						d_a.get(), m, &geqrf_working_memory_size
						));
			CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr_buffer_size(
						*cusolver.get(), m, n, n,
						d_a.get(), m, d_tau.get(), &gqr_working_memory_size
						));

			auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<T>(geqrf_working_memory_size);
			auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<T>(gqr_working_memory_size);
			auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

			for(std::size_t c = 0; c < C; c++) {
				T norm_a = 0.0f;
				for(std::size_t i = 0; i < m * n; i++) {
					const auto tmp = dist(mt);
					h_a.get()[i] = cutf::type::cast<T>(tmp);
					norm_a += tmp * tmp;
				}
				cutf::memory::copy(d_a.get(), h_a.get(), m * n);

				CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
							*cusolver.get(), m, n,
							d_a.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
							geqrf_working_memory_size, d_info.get()
							));
				cut_r<<<(n * n + block_size - 1) / block_size, block_size>>>(d_r.get(), d_a.get(), m, n);

				CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr(
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
				residual_list.push_back(std::sqrt(norm_diff/norm_a));
				orthogonality_list.push_back(mtk::validation::check_orthogonality16(d_q.get(), m, n));
			}
			T residual = 0.0f;
			T orthogonality = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				residual += residual_list[c];
				orthogonality += orthogonality_list[c];
			}
			residual /= C;
			orthogonality /= C;

			T residual_variance = 0.0f;
			T orthogonality_variance = 0.0f;
			for(std::size_t c = 0; c < C; c++) {
				residual_variance += (residual_list[c] - residual) * (residual_list[c] - residual);
				orthogonality_variance += (orthogonality_list[c] - orthogonality) * (orthogonality_list[c] - orthogonality);
			}
			residual_variance /= C;
			orthogonality_variance /= C;

			std::cout << m << ","
				<< n << ","
				<< rand_range_abs << ","
				<< get_type_name<T>() << ","
				<< get_type_name<T>() << ","
				<< "cusolver" << ","
				<< "0" << ","
				<< "0" << ","
				<< residual << ","
				<< residual_variance << ","
				<< orthogonality << ","
				<< orthogonality_variance << std::endl;
			std::cout.flush();
		} catch (std::runtime_error& e) {
			std::cerr<<e.what()<<std::endl;
			continue;
		}
	}
}

template void mtk::test_qr::cusolver_accuracy<float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::cusolver_accuracy<double>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);

template <class T>
void mtk::test_qr::cusolver_speed(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C) {
	constexpr std::size_t block_size = 256;
	std::mt19937 mt(std::random_device{}());

	auto get_qr_complexity = [](const std::size_t m, const std::size_t n) {
		return 2 * n * (m * m * n + m * m * m);
	};

	for(const auto &matrix_config : matrix_config_list) {
		try {
			const std::size_t m = std::get<0>(matrix_config);
			const std::size_t n = std::get<1>(matrix_config);
			const float rand_range_abs = std::get<2>(matrix_config);
			std::uniform_real_distribution<> dist(-rand_range_abs, rand_range_abs);
			auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
			auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto d_tau = cutf::memory::get_device_unique_ptr<T>(n * n);
			auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);

			auto cusolver = cutf::cusolver::get_cusolver_dn_unique_ptr();

			// working memory
			int geqrf_working_memory_size, gqr_working_memory_size;
			CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
						*cusolver.get(), m, n,
						d_a.get(), m, &geqrf_working_memory_size
						));
			CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr_buffer_size(
						*cusolver.get(), m, n, n,
						d_a.get(), m, d_tau.get(), &gqr_working_memory_size
						));

			auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<T>(geqrf_working_memory_size);
			auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<T>(gqr_working_memory_size);
			auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

			for(std::size_t i = 0; i < m * n; i++) {
				const auto tmp = dist(mt);
				h_a.get()[i] = tmp;
			}
			cutf::memory::copy(d_a.get(), h_a.get(), m * n);

			// for cache
			CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
						*cusolver.get(), m, n,
						d_a.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
						geqrf_working_memory_size, d_info.get()
						));

			CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr(
						*cusolver.get(), m, n, n,
						d_a.get(), m,
						d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
						d_info.get()
						));
			const auto start_clock = std::chrono::system_clock::now();
			for(std::size_t c = 0; c < C; c++) {
				CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
							*cusolver.get(), m, n,
							d_a.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
							geqrf_working_memory_size, d_info.get()
							));
				cut_r<<<(n * n + block_size - 1) / block_size, block_size>>>(d_r.get(), d_a.get(), m, n);
				CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr(
							*cusolver.get(), m, n, n,
							d_a.get(), m,
							d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
							d_info.get()
							));
				cutf::memory::copy(d_q.get(), d_a.get(), n * m);
			}
			CUTF_CHECK_ERROR(cudaDeviceSynchronize());
			const auto end_clock = std::chrono::system_clock::now();
			const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / C;

			const auto batch_size = mtk::tsqr::get_batch_size(m);
			const auto complexity = batch_size * get_qr_complexity(m / batch_size, n) + (batch_size - 1) * get_qr_complexity(2 * n, n) + (batch_size - 1) * 4 * n * n * n + 4 * n * n * m;

			std::cout << m << ","
				<< n << ","
				<< rand_range_abs << ","
				<< get_type_name<T>() << ","
				<< get_type_name<T>() << ","
				<< "cusolver" << ","
				<< "0" << ","
				<< "0" << ","
				<< elapsed_time << ","
				<< (complexity / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0)) << ","
				<< ((geqrf_working_memory_size + gqr_working_memory_size) * sizeof(T)) << std::endl;
			std::cout.flush();
		} catch (std::runtime_error& e) {
			std::cerr<<e.what()<<std::endl;
			continue;
		}
	}
}
template void mtk::test_qr::cusolver_speed<float>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
template void mtk::test_qr::cusolver_speed<double>(const std::vector<std::tuple<std::size_t, std::size_t, float>>&, const std::size_t);
