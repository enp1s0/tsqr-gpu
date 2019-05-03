#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include "tcqr.hpp"
#include "tsqr.hpp"
#include "utils.hpp"

constexpr std::size_t m = 1 << 24;
constexpr std::size_t n = 16;

int main(){
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	auto d_a = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::cuda::memory::get_device_unique_ptr<float>(m * n);
	auto d_r = cutf::cuda::memory::get_device_unique_ptr<float>(n * n);
	auto d_working_memory = cutf::cuda::memory::get_device_unique_ptr<float>(
			mtk::tsqr::get_working_memory_size(m, n));
	auto h_a = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::cuda::memory::get_host_unique_ptr<float>(m * n);
	auto h_r = cutf::cuda::memory::get_host_unique_ptr<float>(n * n);
	auto h_working_memory = cutf::cuda::memory::get_host_unique_ptr<float>(
			mtk::tsqr::get_working_memory_size(m, n));

	std::cout<<" A ("<<m<<" x "<<n<<") : "<<(m * n /1024.0/1024.0 * sizeof(float))<<"MB"<<std::endl
		<<" Working memory : "<<(mtk::tsqr::get_working_memory_size(m, n) / 1024.0 / 1024.0 * sizeof(float))<<"MB"<<std::endl;

	for(std::size_t i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}
	cutf::cuda::memory::copy(d_a.get(), h_a.get(), m * n);

	std::cout<<std::endl<<"# Start TSQR test"<<std::endl;
	mtk::tsqr::tsqr16(
			d_q.get(), d_r.get(),
			d_a.get(), m, n,
			d_working_memory.get()
			);
	std::cout<<"# Done"<<std::endl;

	cutf::cuda::memory::copy(h_r.get(), d_r.get(), n * n);
	mtk::utils::print_matrix(h_r.get(), n, n, "R");
}
