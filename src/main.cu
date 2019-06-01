#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include "tcqr.hpp"
#include "tsqr.hpp"
#include "utils.hpp"
#include "validation.hpp"

constexpr std::size_t m = 1 << 10;
constexpr std::size_t n = 16;

int main(){
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<> dist(-1.0f, 1.0f);

	auto d_a = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_r = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto d_working_memory = cutf::memory::get_device_unique_ptr<float>(
			mtk::tsqr::get_working_memory_size(m, n));
	auto h_a = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_r = cutf::memory::get_host_unique_ptr<float>(n * n);

	std::cout<<" A ("<<m<<" x "<<n<<") : "<<(m * n /1024.0/1024.0 * sizeof(float))<<"MB"<<std::endl
		<<" Working memory : "<<(mtk::tsqr::get_working_memory_size(m, n) / 1024.0 / 1024.0 * sizeof(float))<<"MB"<<std::endl;

	float norm_a = 0.0f;
	for(std::size_t i = 0; i < m * n; i++){
		const auto tmp = dist(mt);
		h_a.get()[i] = tmp;
		norm_a += tmp * tmp;
	}
	cutf::memory::copy(d_a.get(), h_a.get(), m * n);

	std::cout<<std::endl<<"# Start TSQR test"<<std::endl;
	const auto elapsed_time = mtk::utils::get_elapsed_time(
			[&d_q, &d_r, &d_a, &d_working_memory](){
			mtk::tsqr::tsqr16<float, false>(
					d_q.get(), d_r.get(),
					d_a.get(), m, n,
					d_working_memory.get()
					);
			}
			);
	std::cout<<"# Done"<<std::endl;
	std::cout<<"Elapsed time : "<<elapsed_time<<" [ms]"<<std::endl;

	cutf::memory::copy(h_r.get(), d_r.get(), n * n);
	mtk::utils::print_matrix(h_r.get(), n, n, "R");
	/*cutf::memory::copy(h_q.get(), d_q.get(), m * n);
	mtk::utils::print_matrix(h_q.get(), m, n, "Q");*/

	// verify
	auto cublas = cutf::cublas::get_cublas_unique_ptr();
	const float alpha = 1.0f, beta = -1.0f;
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
	for(std::size_t i = 0; i < m * n; i++){
		norm_diff += h_a.get()[i] * h_a.get()[i];
	}
	std::cout<<"error : "<<std::sqrt(norm_diff/norm_a)<<std::endl;

	const auto orthogonality = mtk::validation::check_orthogonality16(d_q.get(), m, n);
	std::cout<<"orthogonality : "<<orthogonality<<std::endl;
}
