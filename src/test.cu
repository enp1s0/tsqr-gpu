#include <cutf/type.hpp>
#include <cutf/error.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include "test.hpp"
#include "tcqr.hpp"
#include "tsqr.hpp"
#include "utils.hpp"
#include "blockqr.hpp"
#include "validation.hpp"

template <class T> std::string get_type_name();
template <> std::string get_type_name<double>() {return "double";}
template <> std::string get_type_name<float>() {return "float";}
template <> std::string get_type_name<half>() {return "half";}

namespace {
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
} // namespace

template <bool UseTC, bool Refine, class T>
void mtk::test::precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t block_size = 256;
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	std::string filename = "precision-" + get_type_name<T>() + (UseTC ? "-TC" : "") + (Refine ? "-R" : "") + ".csv";
	std::ofstream ost(filename);

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	ost<<"m,n,type,tc,refinement,error,error_deviation,orthogonality,orthogonality_deviation"<<std::endl;
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_a_test = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
		auto d_q_test = cutf::memory::get_device_unique_ptr<float>(m * n);
		auto d_r_test = cutf::memory::get_device_unique_ptr<float>(n * n);
		auto d_working_q = cutf::memory::get_device_unique_ptr<typename mtk::qr::get_working_q_type<T, UseTC, Refine>::type>(
				mtk::qr::get_working_q_size(m));
		auto d_working_r = cutf::memory::get_device_unique_ptr<typename mtk::qr::get_working_r_type<T, UseTC, Refine>::type>(
				mtk::qr::get_working_r_size(m));
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
			make_zero<T><<<(n * n + block_size - 1) / block_size, block_size>>>(d_r.get(), n * n);

			CUTF_HANDLE_ERROR(cudaDeviceSynchronize());
			mtk::qr::qr<UseTC, Refine>(
					d_q.get(), m,
					d_r.get(), n,
					d_a.get(), m,
					m, n,
					d_working_q.get(),
					d_working_r.get(),
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

		ost<<m<<","<<n<<","<<get_type_name<T>()<<","<<(UseTC ? "1" : "0")<<","<<(Refine ? "1" : "0")<<","<<error<<","<<error_deviation<<","<<orthogonality<<","<<orthogonality_deviation<<std::endl;
	}
	ost.close();
}

template void mtk::test::precision<true, false, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::precision<true, false, half>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::precision<false, false, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::precision<false, false, half>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::precision<true, true, float>(const std::size_t, const std::size_t, const std::size_t);

template <bool UseTC, bool Refine, class T>
void mtk::test::speed(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	std::string filename = "speed-" + get_type_name<T>() + (UseTC ? "-TC" : "") + (Refine ? "-R" : "") + ".csv";
	std::ofstream ost(filename);

	auto get_qr_complexity = [](const std::size_t m, const std::size_t n) {
		return 2 * n * (m * m * n + m * m * m);
	};

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	ost<<"m,n,type,tc,refinement,elapsed_time,tflops,working_memory_size\n";
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
		auto d_working_q = cutf::memory::get_device_unique_ptr<typename mtk::qr::get_working_q_type<T, UseTC, Refine>::type>(
				mtk::qr::get_working_q_size(m));
		auto d_working_r = cutf::memory::get_device_unique_ptr<typename mtk::qr::get_working_r_type<T, UseTC, Refine>::type>(
				mtk::qr::get_working_r_size(m));
		auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto h_q = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto h_r = cutf::memory::get_host_unique_ptr<T>(n * n);
		for(std::size_t i = 0; i < m * n; i++) {
			const auto tmp = dist(mt);
			h_a.get()[i] = cutf::type::cast<T>(tmp);
		}
		cutf::memory::copy(d_a.get(), h_a.get(), m * n);

		// for cache
		mtk::qr::qr<UseTC, Refine, T>(
				d_q.get(), m,
				d_r.get(), n,
				d_a.get(), m,
				m, n,
				d_working_q.get(),
				d_working_r.get(),
				*cublas_handle.get()
				);

		const auto elapsed_time = mtk::utils::get_elapsed_time([&](){
				for(std::size_t c = 0; c < C; c++) {
				mtk::qr::qr<UseTC, Refine, T>(
						d_q.get(), m,
						d_r.get(), n,
						d_a.get(), m,
						m, n,
						d_working_q.get(),
						d_working_r.get(),
						*cublas_handle.get()
						);
				}}) / C;

		const auto batch_size = mtk::tsqr::get_batch_size(m);
		const auto complexity = batch_size * get_qr_complexity(m / batch_size, n) + (batch_size - 1) * get_qr_complexity(2 * n, n) + (batch_size - 1) * 4 * n * n * n + 4 * n * n * m;

		ost<<m<<","<<n<<","<<get_type_name<T>()<<","<<(UseTC ? "1" : "0")<<","<<(Refine ? "1" : "0")<<","<<elapsed_time<<","<<(complexity / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0))<<","<<
			(mtk::tsqr::get_working_q_size(m, n) * sizeof(typename mtk::tsqr::get_working_q_type<T, UseTC, Refine>::type) + mtk::tsqr::get_working_r_size(m, n) * sizeof(typename mtk::tsqr::get_working_r_type<T, UseTC, Refine>::type))<<"\n";
	}
	ost.close();
}

template void mtk::test::speed<true, false, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<true, false, half>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<false, false, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<false, false, half>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<true, true, float>(const std::size_t, const std::size_t, const std::size_t);

template <class T>
void mtk::test::cusolver_precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t block_size = 1 << 8;
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	std::string filename = "precision-" + get_type_name<T>() + "-cusolver.csv";
	std::ofstream ost(filename);

	ost<<"m,n,type,tc,refinement,error,error_deviation,orthogonality,orthogonality_deviation"<<std::endl;
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
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
			for(std::size_t i = 0; i < m * n; i++) {
				const auto tmp = dist(mt);
				h_a.get()[i] = cutf::type::cast<T>(tmp);
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


		ost<<m<<","<<n<<",float,cusolver,0,"<<error<<","<<error_deviation<<","<<orthogonality<<","<<orthogonality_deviation<<std::endl;
	}
	ost.close();
}

template void mtk::test::cusolver_precision<float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::cusolver_precision<double>(const std::size_t, const std::size_t, const std::size_t);

template <class T>
void mtk::test::cusolver_speed(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t block_size = 256;
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	std::string filename = "speed-" + get_type_name<T>() + "-cusolver.csv";
	std::ofstream ost(filename);

	auto get_qr_complexity = [](const std::size_t m, const std::size_t n) {
		return 2 * n * (m * m * n + m * m * m);
	};

	ost<<"m,n,type,tc,refinement,elapsed_time,tflops,working_memory_size"<<std::endl;
	for(std::size_t m = min_m; m <= max_m; m <<= 1) {
		auto d_a = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_q = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto d_r = cutf::memory::get_device_unique_ptr<T>(n * n);
		auto d_tau = cutf::memory::get_device_unique_ptr<T>(n * n);
		auto h_a = cutf::memory::get_host_unique_ptr<T>(m * n);

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

		ost<<m<<","<<n<<",T,cusolver,0,"<<elapsed_time<<","<<(complexity / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0))<<","<<((geqrf_working_memory_size + gqr_working_memory_size) * sizeof(T))<<std::endl;
	}
	ost.close();
}
template void mtk::test::cusolver_speed<float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::cusolver_speed<double>(const std::size_t, const std::size_t, const std::size_t);
