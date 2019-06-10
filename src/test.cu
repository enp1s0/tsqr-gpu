#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include "test.hpp"
#include "tcqr.hpp"
#include "tsqr.hpp"
#include "utils.hpp"
#include "validation.hpp"

template <class T> std::string get_type_name();
template <> std::string get_type_name<float>() {return "float";}
template <> std::string get_type_name<half>() {return "half";}

template <bool UseTC, class T>
void mtk::test::precision(const std::size_t min_m, const std::size_t max_m, const std::size_t n) {
	constexpr std::size_t C = 16;
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	std::cout<<"m,n,type,tc,error,orthogonality"<<std::endl;
	for(std::size_t m = min_m; m < max_m; m <<= 1) {
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

		float error = 0.0f;
		float orthogonality = 0.0f;

		for(std::size_t c = 0; c < C; c++) {
			float norm_a = 0.0f;
			for(std::size_t i = 0; i < m * n; i++) {
				const auto tmp = dist(mt);
				h_a.get()[i] = cutf::type::cast<T>(tmp);
				norm_a += tmp * tmp;
			}
			cutf::memory::copy(d_a.get(), h_a.get(), m * n);

			mtk::tsqr::tsqr16<UseTC, T>(
					d_q.get(), d_r.get(),
					d_a.get(), m, n,
					d_working_q.get(),
					d_working_r.get()
					);

			cutf::memory::copy(h_r.get(), d_r.get(), n * n);

			// verify
			auto cublas = cutf::cublas::get_cublas_unique_ptr();
			const auto alpha = cutf::type::cast<T>(1.0f), beta = cutf::type::cast<T>(-1.0f);
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
			error += std::sqrt(norm_diff/norm_a);
			orthogonality += mtk::validation::check_orthogonality16(d_q.get(), m, n);
		}

		std::cout<<m<<","<<n<<","<<get_type_name<T>()<<","<<(UseTC ? "1" : "0")<<","<<(error / C)<<","<<(orthogonality / C)<<std::endl;
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

	std::cout<<"m,n,type,tc,elapsed_time,tflops"<<std::endl;
	for(std::size_t m = min_m; m < max_m; m <<= 1) {
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

		std::cout<<m<<","<<n<<","<<get_type_name<T>()<<","<<(UseTC ? "1" : "0")<<","<<elapsed_time<<","<<(complexity / elapsed_time / (1024.0 * 1024.0 * 1024.0 * 1024.0))<<std::endl;
	}
}

template void mtk::test::speed<true, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<true, half>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<false, float>(const std::size_t, const std::size_t, const std::size_t);
template void mtk::test::speed<false, half>(const std::size_t, const std::size_t, const std::size_t);
