#ifndef __TEST_COMPARE_HPP__
#define __TEST_COMPARE_HPP__
#include "test.hpp"
#include "blockqr.hpp"
#include <random>
#include <numeric>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/cublas.hpp>

namespace mtk {
namespace test_qr {

template <bool A_UseTC, bool A_Refine, bool A_Reorthogonalization, bool B_UseTC, bool B_Refine, bool B_Reorthogonalization, class T>
__inline__ void compare(const std::vector<std::pair<std::size_t, std::size_t>>& size_pair_vector, const std::size_t C) {
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (const auto& size_pair : size_pair_vector) {
		const auto m = size_pair.first;
		const auto n = size_pair.second;

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
				mtk::qr::buffer<T, A_UseTC, A_Refine, A_Reorthogonalization> buffer;
				buffer.allocate(m, n);
				mtk::qr::qr<A_UseTC, A_Refine, A_Reorthogonalization, T, T>(
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
				mtk::qr::buffer<T, B_UseTC, B_Refine, B_Reorthogonalization> buffer;
				buffer.allocate(m, n);
				mtk::qr::qr<B_UseTC, B_Refine, B_Reorthogonalization, T, T>(
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

} // namespace test_qr
} // namespace mtk
#endif /* end of include guard */
