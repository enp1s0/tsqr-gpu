#ifndef __TEST_COMPARE_HPP__
#define __TEST_COMPARE_HPP__
#include "test.hpp"
#include "blockqr.hpp"
#include <random>
#include <numeric>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>

namespace {
template <class T>
__global__ void cut_r(T* const dst, const T* const src, const std::size_t m, const std::size_t n) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	const auto x = tid / n;
	const auto y = tid % n;

	if(y > x) return;

	dst[tid] = src[m * x + y];
}

template <mtk::test_qr::compute_mode compute_mode>
struct get_compute_type {using type = float;};
template <> struct get_compute_type<mtk::test_qr::compute_mode::fp16_notc      > {using type = half;};
template <> struct get_compute_type<mtk::test_qr::compute_mode::fp16_tc_nocor  > {using type = half;};

template <mtk::test_qr::compute_mode>
constexpr mtk::qr::compute_mode get_qr_compute_mode();
#define TEST_QR_GET_TSQR_COMPUTE_MODE(mode) template<> constexpr mtk::qr::compute_mode get_qr_compute_mode<mtk::test_qr::compute_mode::mode>() {return mtk::qr::compute_mode::mode;}
TEST_QR_GET_TSQR_COMPUTE_MODE(fp16_notc        );
TEST_QR_GET_TSQR_COMPUTE_MODE(fp32_notc        );
TEST_QR_GET_TSQR_COMPUTE_MODE(fp16_tc_nocor    );
TEST_QR_GET_TSQR_COMPUTE_MODE(fp32_tc_nocor    );
TEST_QR_GET_TSQR_COMPUTE_MODE(tf32_tc_nocor    );
TEST_QR_GET_TSQR_COMPUTE_MODE(fp32_tc_cor      );
TEST_QR_GET_TSQR_COMPUTE_MODE(tf32_tc_cor      );
TEST_QR_GET_TSQR_COMPUTE_MODE(tf32_tc_cor_emu  );
TEST_QR_GET_TSQR_COMPUTE_MODE(tf32_tc_nocor_emu);
TEST_QR_GET_TSQR_COMPUTE_MODE(mixed_tc_cor_emu );

} // namespace


namespace mtk {
namespace test_qr {

template <compute_mode A_compute_mode, bool A_Reorthogonalization, compute_mode B_compute_mode, bool B_Reorthogonalization>
__inline__ void compare(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16) {
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	std::mt19937 mt(std::random_device{}());

	using T = typename get_compute_type<A_compute_mode>::type;

	for (const auto& matrix_config : matrix_config_list) {
		const auto m = std::get<0>(matrix_config);
		const auto n = std::get<1>(matrix_config);
		const auto rand_range = std::get<2>(matrix_config);
		std::uniform_real_distribution<float> dist(-rand_range, rand_range);

		auto dA = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto dQ = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto dR = cutf::memory::get_device_unique_ptr<T>(n * n);

		auto hA = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto hQ_A = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto hQ_B = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto hR_A = cutf::memory::get_host_unique_ptr<T>(n * n);
		auto hR_B = cutf::memory::get_host_unique_ptr<T>(n * n);

		std::vector<float> Q_residual_list;
		std::vector<float> R_residual_list;

		for (std::size_t c = 0; c < C; c++) {
			for (std::size_t i = 0; i < m * n; i++) {
				hA.get()[i] = cutf::type::cast<T>(dist(mt));
			}
			cutf::memory::copy(dA.get(), hA.get(), m * n);

			// A
			{
				mtk::qr::buffer<get_qr_compute_mode<A_compute_mode>(), A_Reorthogonalization> buffer;
				buffer.allocate(m, n);
				mtk::qr::qr<get_qr_compute_mode<A_compute_mode>(), A_Reorthogonalization>(
						dQ.get(), m,
						dR.get(), n,
						dA.get(), m,
						m, n,
						buffer,
						*cublas_handle.get()
						);
			}
			cutf::memory::copy(hQ_A.get(), dQ.get(), m * n);
			cutf::memory::copy(hR_A.get(), dR.get(), n * n);

			// B
			{
				mtk::qr::buffer<get_qr_compute_mode<B_compute_mode>(), B_Reorthogonalization> buffer;
				buffer.allocate(m, n);
				mtk::qr::qr<get_qr_compute_mode<B_compute_mode>(), B_Reorthogonalization>(
						dQ.get(), m,
						dR.get(), n,
						dA.get(), m,
						m, n,
						buffer,
						*cublas_handle.get()
						);
			}
			cutf::memory::copy(hQ_B.get(), dQ.get(), m * n);
			cutf::memory::copy(hR_B.get(), dR.get(), n * n);


			// compare
			float base_norm2_Q = 0.0f;
			float diff_norm2_Q = 0.0f;
#pragma omp parallel for reduction(+: base_norm2_Q) reduction(+: diff_norm2_Q)
			for (std::size_t i = 0; i < m * n; i++) {
				const auto diff = hQ_A.get()[i] - hQ_B.get()[i];
				base_norm2_Q += hQ_A.get()[i] * hQ_A.get()[i];
				diff_norm2_Q += diff * diff;
			}
			Q_residual_list.push_back(std::sqrt(diff_norm2_Q / base_norm2_Q));

			float base_norm2_R = 0.0f;
			float diff_norm2_R = 0.0f;
#pragma omp parallel for reduction(+: base_norm2_R) reduction(+: diff_norm2_R)
			for (std::size_t i = 0; i < n * n; i++) {
				const auto diff = hR_A.get()[i] - hR_B.get()[i];
				base_norm2_R += hR_A.get()[i] * hR_A.get()[i];
				diff_norm2_R += diff * diff;
			}
			R_residual_list.push_back(std::sqrt(diff_norm2_R / base_norm2_R));
		}
		const auto Q_residual = std::accumulate(Q_residual_list.begin(), Q_residual_list.end(), 0.0f) / C;
		const auto R_residual = std::accumulate(R_residual_list.begin(), R_residual_list.end(), 0.0f) / C;
		std::printf("%e,%e\n", Q_residual, R_residual);
	}
}

template <compute_mode A_compute_mode, bool A_Reorthogonalization>
__inline__ void compare_to_cusolver_double(const std::vector<std::tuple<std::size_t, std::size_t, float>>& matrix_config_list, const std::size_t C = 16) {
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	auto cusolver_handle = cutf::cusolver::get_cusolver_dn_unique_ptr();

	std::mt19937 mt(std::random_device{}());

	using T = typename get_compute_type<A_compute_mode>::type;

	std::printf("m,n,compute_mode,reorth,Q,R\n");

	for (const auto& matrix_config : matrix_config_list) {
		const auto m = std::get<0>(matrix_config);
		const auto n = std::get<1>(matrix_config);
		const auto rand_range = std::get<2>(matrix_config);
		std::uniform_real_distribution<float> dist(-rand_range, rand_range);


		auto dA = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto dQ = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto dR = cutf::memory::get_device_unique_ptr<T>(n * n);

		auto hA = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto hQ_A = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto hR_A = cutf::memory::get_host_unique_ptr<T>(n * n);

		auto hAd = cutf::memory::get_host_unique_ptr<double>(m * n);
		auto hQ_Ad = cutf::memory::get_host_unique_ptr<double>(m * n);
		auto hR_Ad = cutf::memory::get_host_unique_ptr<double>(n * n);
		auto d_tau = cutf::memory::get_device_unique_ptr<double>(n * n);

		std::vector<float> Q_residual_list;
		std::vector<float> R_residual_list;

		int geqrf_working_memory_size, gqr_working_memory_size;
		CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
					*cusolver_handle.get(), m, n,
					hAd.get(), m, &geqrf_working_memory_size
					));
		CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr_buffer_size(
					*cusolver_handle.get(), m, n, n,
					hAd.get(), m, d_tau.get(), &gqr_working_memory_size
					));

		auto d_geqrf_working_memory = cutf::memory::get_host_unique_ptr<double>(geqrf_working_memory_size);
		auto d_gqr_working_memory = cutf::memory::get_host_unique_ptr<double>(gqr_working_memory_size);
		auto d_info = cutf::memory::get_host_unique_ptr<int>(1);

		for (std::size_t c = 0; c < C; c++) {
			for (std::size_t i = 0; i < m * n; i++) {
				hAd.get()[i] = hA.get()[i] = cutf::type::cast<T>(dist(mt));
			}
			cutf::memory::copy(dA.get(), hA.get(), m * n);

			// A
			{
				mtk::qr::buffer<get_qr_compute_mode<A_compute_mode>(), A_Reorthogonalization> buffer;
				buffer.allocate(m, n);
				mtk::qr::qr<get_qr_compute_mode<A_compute_mode>(), A_Reorthogonalization>(
						dQ.get(), m,
						dR.get(), n,
						dA.get(), m,
						m, n,
						buffer,
						*cublas_handle.get()
						);
			}
			cutf::memory::copy(hQ_A.get(), dQ.get(), m * n);
			cutf::memory::copy(hR_A.get(), dR.get(), n * n);

			// B
			{
				CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
							*cusolver_handle.get(), m, n,
							hAd.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
							geqrf_working_memory_size, d_info.get()
							));
				constexpr std::size_t block_size = 256;
				cudaDeviceSynchronize();
				cut_r<<<(n * n + block_size - 1) / block_size, block_size>>>(hR_Ad.get(), hAd.get(), m, n);
				cudaDeviceSynchronize();
				CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr(
							*cusolver_handle.get(), m, n, n,
							hAd.get(), m,
							d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
							d_info.get()
							));


			}
			cudaDeviceSynchronize();


			// compare
			float base_norm2_Q = 0.0f;
			float diff_norm2_Q = 0.0f;
#pragma omp parallel for reduction(+: base_norm2_Q) reduction(+: diff_norm2_Q)
			for (std::size_t i = 0; i < m * n; i++) {
				const auto diff = std::abs(hQ_A.get()[i]) - std::abs(hAd.get()[i]);
				base_norm2_Q += hAd.get()[i] * hAd.get()[i];
				diff_norm2_Q += diff * diff;
			}
			Q_residual_list.push_back(std::sqrt(diff_norm2_Q / base_norm2_Q));

			float base_norm2_R = 0.0f;
			float diff_norm2_R = 0.0f;
#pragma omp parallel for reduction(+: base_norm2_R) reduction(+: diff_norm2_R)
			for (std::size_t i = 0; i < n * n; i++) {
				const auto diff = std::abs(hR_A.get()[i]) - std::abs(hR_Ad.get()[i]);
				base_norm2_R += hR_Ad.get()[i] * hR_Ad.get()[i];
				diff_norm2_R += diff * diff;
			}
			R_residual_list.push_back(std::sqrt(diff_norm2_R / base_norm2_R));
		}
		const auto Q_residual = std::accumulate(Q_residual_list.begin(), Q_residual_list.end(), 0.0f) / C;
		const auto R_residual = std::accumulate(R_residual_list.begin(), R_residual_list.end(), 0.0f) / C;
		std::printf("%lu,%lu,%s,%d,%e,%e\n",
				m, n,
				get_compute_mode_name_string<A_compute_mode>().c_str(),
				(A_Reorthogonalization ? 1 : 0),
				Q_residual, R_residual);
	}
}


} // namespace test_qr
} // namespace mtk
#endif /* end of include guard */
