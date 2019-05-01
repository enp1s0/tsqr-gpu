#include <cutf/memory.hpp>
#include <iostream>
#include <random>
#include "tcqr.hpp"
#include "utils.hpp"

int main(){
	constexpr unsigned m = 30;
	constexpr unsigned n = 16;

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	auto h_a = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_r = cutf::cuda::memory::get_host_unique_ptr<float>(n * n);
	auto d_a = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_r = cutf::cuda::memory::get_device_unique_ptr<float>(n * n);

	for(unsigned i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}
	mtk::utils::print_matrix_32x16(
			h_a.get(), m, n, "A"
			);

	cutf::cuda::memory::copy(d_a.get(), h_a.get(), m * n);

	mtk::tcqr::qr32x16_f32tc(
		d_q.get(), d_r.get(),
		d_a.get(), m, n
		);

	cutf::cuda::memory::copy(h_q.get(), d_q.get(), m * n);
	cutf::cuda::memory::copy(h_r.get(), d_r.get(), n * n);

	mtk::utils::print_matrix_32x16(
			h_q.get(), m, n, "Q"
			);
	mtk::utils::print_matrix_32x16(
			h_r.get(), n, n, "R"
			);
}
