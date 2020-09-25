#include <iostream>
#include <random>
#include "tsqr.hpp"

constexpr std::size_t m = 1lu << 20;
constexpr std::size_t n = 16;

constexpr auto compute_mode = mtk::tsqr::compute_mode::fp32_tc_cor;

int main() {
	// Allocate device memory
	float *a;
	float *q, *r;
	cudaMalloc(&a, m * n * sizeof(float));
	cudaMalloc(&q, m * n * sizeof(float));
	cudaMalloc(&r, n * n * sizeof(float));

	// Init A
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.f, 1.f);
	float *ha;
	cudaMallocHost(&ha, m * n * sizeof(float));
	for (std::size_t i = 0; i < m * n; i++) ha[i] = dist(mt);
	cudaMemcpy(a, ha, sizeof(float) * m * n, cudaMemcpyDefault);

	// Allocate working memory
	mtk::tsqr::buffer<compute_mode> working_memory;
	working_memory.allocate(m, n);

	// TSQR
	mtk::tsqr::tsqr16<compute_mode>(
			q, m, // Q, leading dimension
			r, n, // R, leading dimension
			a, m, // A, leading dimension
			m, n, // size of A
			working_memory, 0);

	cudaFree(a);
	cudaFree(r);
	cudaFree(q);
}
