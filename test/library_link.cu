#include <cutf/memory.hpp>
#include <tsqr.hpp>
#include <iostream>
#include <random>

constexpr std::size_t m = 1 << 24;
constexpr std::size_t n = 1 << 4;

int main(){
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	auto d_a = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_r = cutf::cuda::memory::get_device_unique_ptr<float>(n * n);
	auto h_a = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_r = cutf::cuda::memory::get_host_unique_ptr<float>(n * n);

	for(std::size_t i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}

	cutf::cuda::memory::copy(d_a.get(), h_a.get(), m * n);

	const auto working_memory_size = mtk::tsqr::get_working_memory_size(m, n);
	auto d_w = cutf::cuda::memory::get_device_unique_ptr<float>(working_memory_size);

	mtk::tsqr::tsqr16(
			d_q.get(), d_r.get(),
			d_a.get(), m, n, 
			d_w.get()
			);
}
