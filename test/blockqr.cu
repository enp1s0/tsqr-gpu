#include <iostream>
#include <stdexcept>
#include <random>
#include <functional>
#include <cuda_fp16.h>
#include "blockqr.hpp"
#include "utils.hpp"
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/cublas.hpp>

constexpr std::size_t m = 1lu << 15;
constexpr std::size_t n = 1lu << 8;

using compute_t = float;
constexpr bool UseTC = false;
constexpr bool Refinement = false;

int main() {
	std::mt19937 mt(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	const auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	auto d_a = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_r = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto h_a = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_r = cutf::memory::get_host_unique_ptr<float>(n * n);

	for(std::size_t i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}
	const auto base_norm2 = std::accumulate(h_a.get(), h_a.get() + m * n, 0.0f, [](const compute_t a, const float b){return cutf::type::cast<float>(b) * cutf::type::cast<float>(b) + a;});

	cutf::memory::copy(d_a.get(), h_a.get(), m * n);

	const auto working_memory_q_size = mtk::tsqr::get_working_q_size(m, n);
	auto d_wq = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_q_type<float, UseTC, Refinement>::type>(working_memory_q_size);
	const auto working_memory_r_size = mtk::tsqr::get_working_r_size(m, n);
	auto d_wr = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_r_type<float, UseTC, Refinement>::type>(working_memory_r_size);

	mtk::qr::qr<UseTC, Refinement>(
			d_q.get(), m,
			d_r.get(), n,
			d_a.get(), m,
			m, n,
			d_wq.get(), d_wr.get(),
			*cublas_handle.get()
			);

	cutf::memory::copy(h_r.get(), d_r.get(), n * n);
	//mtk::utils::print_matrix(h_r.get(), n, n, n, "R");
	cutf::memory::copy(h_q.get(), d_q.get(), m * n);
	//mtk::utils::print_matrix(h_q.get(), m, n, m, "Q");

	// orthogonality
	auto qtq = cutf::memory::get_host_unique_ptr<float>(n * n);
	{
	const float alpah = 1.0f, beta = 0.0f;
	cutf::cublas::gemm(
			*cublas_handle.get(),
			CUBLAS_OP_T, CUBLAS_OP_N,
			n, n, m,
			&alpah,
			d_q.get(), m,
			d_q.get(), m,
			&beta,
			qtq.get(), n
			);
	}
	cudaDeviceSynchronize();
	//mtk::utils::print_matrix(qtq.get(), n, n, n, "QtQ");

	// diff QR - A
	{
	const float alpah = 1.0f, beta = -1.0f;
	cutf::cublas::gemm(
			*cublas_handle.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			m, n, n,
			&alpah,
			d_q.get(), m,
			d_r.get(), n,
			&beta,
			h_a.get(), m
			);
	}
	cudaDeviceSynchronize();
	//mtk::utils::print_matrix(h_a.get(), m, n, m, "QR-A");
	const auto diff_norm2 = std::accumulate(h_a.get(), h_a.get() + m * n, 0.0f, [](const compute_t a, const float b){return cutf::type::cast<float>(b) * cutf::type::cast<float>(b) + a;});
	std::printf("residual = %e\n", std::sqrt(diff_norm2 / base_norm2));
}
