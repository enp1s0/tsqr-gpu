#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <tsqr.hpp>
#include <iostream>
#include <random>

constexpr std::size_t m = 1 << 14;
constexpr std::size_t n = 1 << 4;

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
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	auto d_a = cutf::memory::get_device_unique_ptr<float>(m * n);
	auto d_q = cutf::memory::get_device_unique_ptr<float>((m + 1) * n);
	auto d_r = cutf::memory::get_device_unique_ptr<float>((n + 1) * n);
	auto h_a = cutf::memory::get_host_unique_ptr<float>(m * n);
	auto h_q = cutf::memory::get_host_unique_ptr<float>((m + 1) * n);
	auto h_r = cutf::memory::get_host_unique_ptr<float>((n + 1) * n);

	for(std::size_t i = 0; i < m * n; i++){
		h_a.get()[i] = dist(mt);
	}

	cutf::memory::copy(d_a.get(), h_a.get(), m * n);

	const auto working_memory_q_size = mtk::tsqr::get_working_q_size(m, n);
	auto d_wq = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_q_type<float, true, true>::type>(working_memory_q_size);
	const auto working_memory_r_size = mtk::tsqr::get_working_r_size(m, n);
	auto d_wr = cutf::memory::get_device_unique_ptr<typename mtk::tsqr::get_working_r_type<float, true, true>::type>(working_memory_r_size);

	mtk::tsqr::tsqr16<true, true>(
			d_q.get(), m + 1,
			d_r.get(), n + 1,
			d_a.get(), m,
			m, n,
			d_wq.get(), d_wr.get()
			);

	cutf::memory::copy(h_r.get(), d_r.get(), (n + 1) * n);
	print_matrix(h_r.get(), n, n, n + 1, "R");
}
