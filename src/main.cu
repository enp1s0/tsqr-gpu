#include <iostream>
#include <random>
#include <cutf/cusolver.hpp>
#include <cutf/error.hpp>
#include "tsqr.hpp"

constexpr std::size_t m = 1lu << 20;
constexpr std::size_t n = 16;

//#define USE_CUSOLVER

constexpr auto compute_mode = mtk::tsqr::compute_mode::fp32_tc_cor;

void qr_tc(float* const q, float* const r, const float* const a) {
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
}

void qr_cusolver(float* const q, float* const r, float* const a) {
	auto cusolver = cutf::cusolver::get_cusolver_dn_unique_ptr();

	float *d_tau;
	CUTF_CHECK_ERROR(cudaMalloc(&d_tau, sizeof(float) * n * n));

	// working memory
	int geqrf_working_memory_size, gqr_working_memory_size;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
				*cusolver.get(), m, n,
				a, m, &geqrf_working_memory_size
				));
	CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr_buffer_size(
				*cusolver.get(), m, n, n,
				a, m, d_tau, &gqr_working_memory_size
				));

	float *d_geqrf_working_memory;
	float *d_gqr_working_memory;
	int *d_info;
	CUTF_CHECK_ERROR(cudaMalloc(&d_geqrf_working_memory, sizeof(float) * geqrf_working_memory_size));
	CUTF_CHECK_ERROR(cudaMalloc(&d_gqr_working_memory, sizeof(float) * gqr_working_memory_size));
	CUTF_CHECK_ERROR(cudaMalloc(&d_info, sizeof(int)));

	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
				*cusolver.get(), m, n,
				a, m, d_tau, d_geqrf_working_memory,
				geqrf_working_memory_size, d_info
				));

	CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr(
				*cusolver.get(), m, n, n,
				a, m,
				d_tau, d_gqr_working_memory, gqr_working_memory_size,
				d_info
				));

	cudaFree(d_geqrf_working_memory);
	cudaFree(d_gqr_working_memory);
	cudaFree(d_tau);
	cudaFree(d_info);
}

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

#ifdef USE_CUSOLVER
	qr_cusolver(q, r, a);
#else
	qr_tc(q, r, a);
#endif

	cudaFree(a);
	cudaFree(r);
	cudaFree(q);
}
