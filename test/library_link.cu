#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/cublas.hpp>
#include <blockqr.hpp>
#include <iostream>
#include <random>

constexpr std::size_t m = 1 << 10;
constexpr std::size_t n = 1 << 4;
constexpr std::size_t ldq = m * 2;

constexpr bool UseTC = true;
constexpr bool Refinement = true;

template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, std::size_t ldm, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * ldm + i]);
			if(val == 0.0f) {
				printf(" %.5f ", 0.0);
			}else if (val < 0.0){
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}

int main(){
	std::mt19937 mt(0);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	auto d_a = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::memory::get_device_unique_ptr<float>(ldq * n);
	auto d_r = cutf::memory::get_device_unique_ptr<float>(n * n);
	auto h_a = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::memory::get_host_unique_ptr<float>(ldq * n);
	auto h_r = cutf::memory::get_host_unique_ptr<float>(n * n);

	for(std::size_t i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}

	cutf::memory::copy(d_a.get(), h_a.get(), m * n);

	const auto working_memory_q_size = mtk::qr::get_working_q_size(m);
	auto d_wq = cutf::memory::get_device_unique_ptr<typename mtk::qr::get_working_q_type<float, UseTC, Refinement>::type>(working_memory_q_size);
	const auto working_memory_r_size = mtk::qr::get_working_r_size(m);
	auto d_wr = cutf::memory::get_device_unique_ptr<typename mtk::qr::get_working_r_type<float, UseTC, Refinement>::type>(working_memory_r_size);

	auto cusolver_handle = cutf::cublas::get_cublas_unique_ptr();

	mtk::qr::qr<UseTC, Refinement, float>(
			d_q.get(), ldq,
			d_r.get(), n,
			d_a.get(), m,
			m, n,
			d_wq.get(), d_wr.get(),
			*cusolver_handle.get()
			);

	cutf::memory::copy(h_r.get(), d_r.get(), n * n);
	print_matrix(h_r.get(), n, n, n, "R");
	cutf::memory::copy(h_q.get(), d_q.get(), ldq * n);
	print_matrix(h_q.get(), ldq, n, ldq, "Q");

	// orthogonality
	auto qtq = cutf::memory::get_host_unique_ptr<float>(n * n);
	const auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	const float alpah = 1.0f, beta = 0.0f;
	cutf::cublas::gemm(
			*cublas_handle.get(),
			CUBLAS_OP_T, CUBLAS_OP_N,
			n, n, m,
			&alpah,
			d_q.get(), ldq,
			d_q.get(), ldq,
			&beta,
			qtq.get(), n
			);
	cudaDeviceSynchronize();
	print_matrix(qtq.get(), n, n, n, "QtQ");
}
