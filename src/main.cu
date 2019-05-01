#include <cutf/memory.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include "tcqr.hpp"
#include "utils.hpp"

int main(){
	constexpr unsigned m = 27437;
	constexpr unsigned n = 16;

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	const std::size_t batch_size = 1u << (std::max(5u, static_cast<unsigned>( std::ceil( std::log2(static_cast<float>(m))))) - 5u);
	auto h_a_start_position = cutf::cuda::memory::get_host_unique_ptr<unsigned>(batch_size + 1);
	auto d_a_start_position = cutf::cuda::memory::get_device_unique_ptr<unsigned>(batch_size + 1);

	auto h_a = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_r = cutf::cuda::memory::get_host_unique_ptr<float>(n * n * batch_size);
	auto d_a = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_r = cutf::cuda::memory::get_device_unique_ptr<float>(n * n * batch_size);

	for(unsigned i = 0; i < batch_size; i++){
		h_a_start_position.get()[i] = i * m / batch_size;
	}
	h_a_start_position.get()[batch_size] = m;

	std::cout<<"batch size : "<<batch_size<<std::endl;
	std::cout<<"size of each batches : ";
	for(unsigned i = 0; i < batch_size; i++){
		std::cout<<(h_a_start_position.get()[i + 1] - h_a_start_position.get()[i])<<" ";
	}
	std::cout<<std::endl;
	cutf::cuda::memory::copy(d_a_start_position.get(), h_a_start_position.get(), batch_size + 1);


	for(unsigned i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}
	mtk::utils::print_matrix(
			h_a.get(), m, n, "A"
			);

	cutf::cuda::memory::copy(d_a.get(), h_a.get(), m * n);

	mtk::tcqr::qr32x16_f32tc_batched(
		d_q.get(), d_r.get(),
		d_a.get(), m, n,
		batch_size,
		d_a_start_position.get()
		);

	cutf::cuda::memory::copy(h_q.get(), d_q.get(), m * n);
	cutf::cuda::memory::copy(h_r.get(), d_r.get(), n * n * batch_size);

	mtk::utils::print_matrix(
			h_q.get(), m, n, "Q"
			);
	mtk::utils::print_matrix(
			h_r.get(), n * batch_size, n, "R"
			);
}
