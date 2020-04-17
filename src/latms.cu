#include "latms.hpp"
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <cutf/curand.hpp>

template <class T>
void mtk::utils::latms(
		T* const mat_ptr,
		const std::size_t m,
		const std::size_t n,
		const std::size_t rank,
		const T* const s_array,
		const unsigned long long seed
		) {
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	auto hs_ptr = cutf::memory::get_host_unique_ptr<T>(rank * rank);
	auto s_ptr = cutf::memory::get_device_unique_ptr<T>(rank * rank);
	auto u_ptr = cutf::memory::get_device_unique_ptr<T>(rank * m);
	auto v_ptr = cutf::memory::get_device_unique_ptr<T>(rank * n);
	auto tmp_ptr = cutf::memory::get_device_unique_ptr<T>(rank * m);

	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed);
	CUTF_HANDLE_ERROR(cutf::curand::generate_normal(*cugen.get(), u_ptr.get(), m * rank, 0.0f, 1.0f));
	CUTF_HANDLE_ERROR(cutf::curand::generate_normal(*cugen.get(), v_ptr.get(), n * rank, 0.0f, 1.0f));

	auto d_tau = cutf::memory::get_device_unique_ptr<T>(n * n);
	auto h_a = cutf::memory::get_device_unique_ptr<T>(m * n);

	auto cusolver = cutf::cusolver::get_cusolver_dn_unique_ptr();

	// working memory
	int u_geqrf_working_memory_size, u_gqr_working_memory_size;
	CUTF_HANDLE_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
				*cusolver.get(), m, rank,
				u_ptr.get(), m, &u_geqrf_working_memory_size
				));
	CUTF_HANDLE_ERROR(cutf::cusolver::dn::gqr_buffer_size(
				*cusolver.get(), m, rank, rank,
				u_ptr.get(), m, d_tau.get(), &u_gqr_working_memory_size
				));

	int v_geqrf_working_memory_size, v_gqr_working_memory_size;
	CUTF_HANDLE_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
				*cusolver.get(), n, rank,
				v_ptr.get(), n, &v_geqrf_working_memory_size
				));
	CUTF_HANDLE_ERROR(cutf::cusolver::dn::gqr_buffer_size(
				*cusolver.get(), n, rank, rank,
				v_ptr.get(), n, d_tau.get(), &v_gqr_working_memory_size
				));

	int geqrf_working_memory_size = std::max(v_geqrf_working_memory_size, u_geqrf_working_memory_size);
	int gqr_working_memory_size = std::max(v_gqr_working_memory_size, u_gqr_working_memory_size);

	auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<T>(geqrf_working_memory_size);
	auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<T>(gqr_working_memory_size);
	auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

	CUTF_HANDLE_ERROR(cutf::cusolver::dn::geqrf(
				*cusolver.get(), m, rank,
				u_ptr.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
				geqrf_working_memory_size, d_info.get()
				));

	CUTF_HANDLE_ERROR(cutf::cusolver::dn::gqr(
				*cusolver.get(), m, rank, rank,
				u_ptr.get(), m,
				d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
				d_info.get()
				));

	CUTF_HANDLE_ERROR(cutf::cusolver::dn::geqrf(
				*cusolver.get(), n, rank,
				v_ptr.get(), n, d_tau.get(), d_geqrf_working_memory.get(),
				geqrf_working_memory_size, d_info.get()
				));

	CUTF_HANDLE_ERROR(cutf::cusolver::dn::gqr(
				*cusolver.get(), n, rank, rank,
				v_ptr.get(), n,
				d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
				d_info.get()
				));

	for (std::size_t i = 0; i < rank * rank; i++) {
		if (i % (rank + 1) == 0) {
			hs_ptr.get()[i] = s_array[i / (rank + 1)];
		} else {
			hs_ptr.get()[i] = cutf::type::cast<T>(0.0f);
		}
	}

	cutf::memory::copy(s_ptr.get(), hs_ptr.get(), rank * rank);

	// merge
	const T one = cutf::type::cast<T>(1.0f);
	const T zero = cutf::type::cast<T>(0.0f);
	cutf::cublas::gemm(
			*cublas_handle.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			m, rank, rank,
			&one,
			u_ptr.get(), m,
			s_ptr.get(), rank,
			&zero,
			tmp_ptr.get(), m
			);
	cutf::cublas::gemm(
			*cublas_handle.get(),
			CUBLAS_OP_N, CUBLAS_OP_T,
			m, n, rank,
			&one,
			tmp_ptr.get(), m,
			v_ptr.get(), n,
			&zero,
			mat_ptr, m
			);
}

template void mtk::utils::latms<float >(float * const, const std::size_t, const std::size_t, const std::size_t, const float * const, const unsigned long long);
template void mtk::utils::latms<double>(double* const, const std::size_t, const std::size_t, const std::size_t, const double* const, const unsigned long long);



template <class T>
T mtk::utils::get_cond(
		T* const mat,
		const std::size_t m, const std::size_t n
		) {
	const auto rank = std::min(m, n);
	auto cusolver = cutf::cusolver::get_cusolver_dn_unique_ptr();
	auto dVT   = cutf::memory::get_device_unique_ptr<T>(rank * n);
	auto dS    = cutf::memory::get_device_unique_ptr<T>(rank);
	auto dU    = cutf::memory::get_device_unique_ptr<T>(m * rank);
	auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

	int svd_w_size;
	CUTF_HANDLE_ERROR(
			cutf::cusolver::dn::gesvd_buffer_size<T>(
				*cusolver.get(),
				m, n,
				&svd_w_size
				)
			);

	auto dwsvd = cutf::memory::get_device_unique_ptr<T>(svd_w_size);
	auto dwrsvd = cutf::memory::get_device_unique_ptr<T>(std::min(m, n) - 1);
	CUTF_HANDLE_ERROR(
			cutf::cusolver::dn::gesvd(
				*cusolver.get(),
				'S', 'S',
				m, n,
				mat, m,
				dS.get(),
				dU.get(), m,
				dVT.get(), rank,
				dwsvd.get(),
				svd_w_size,
				dwrsvd.get(),
				dInfo.get()
				)
			);
	auto hS = cutf::memory::get_host_unique_ptr<T>(rank);
	cutf::memory::copy(hS.get(), dS.get(), rank);

	return hS.get()[0] / hS.get()[rank - 1];
}

template float  mtk::utils::get_cond(float*  const mat, const std::size_t m, const std::size_t n);
template double mtk::utils::get_cond(double* const mat, const std::size_t m, const std::size_t n);
