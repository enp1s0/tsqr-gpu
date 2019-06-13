#include <cutf/memory.hpp>
#include <tsqr.hpp>
#include <iostream>
#include <random>

constexpr std::size_t m = 1 << 14;
constexpr std::size_t n = 1 << 4;

int main(){
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	auto d_a = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_r = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto h_a = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_r = cutf::memory::get_host_unique_ptr<float>(n * n);

	for(std::size_t i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}

	cutf::memory::copy(d_a.get(), h_a.get(), m * n);

	const auto working_memory_q_size = mtk::tsqr::get_working_q_size(m, n);
	auto d_wq = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_q_type<float, true>::type>(working_memory_q_size);
	const auto working_memory_r_size = mtk::tsqr::get_working_r_size(m, n);
	auto d_wr = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_r_type<float, true>::type>(working_memory_r_size);

	mtk::tsqr::tsqr16<true>(
			d_q.get(), d_r.get(),
			d_a.get(), m, n, 
			d_wq.get(), d_wr.get()
			);
}
