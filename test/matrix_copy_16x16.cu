#include <iostream>
#include <matrix_copy.cuh>
#include <utils.hpp>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>

constexpr std::size_t warp_size = 32;
constexpr std::size_t batch_per_block = 8;
constexpr std::size_t fragment_dim = 16;
constexpr std::size_t g_size_m = 62;
constexpr std::size_t g_size_n = fragment_dim;
constexpr std::size_t s_size_m = 12;
constexpr std::size_t s_size_n = 6;
constexpr std::size_t batch_size = (g_size_m + s_size_m -1) / s_size_m;
using test_t = float;

__global__ void kernel16x16(test_t* ptr){
	__shared__ test_t s_mem[fragment_dim * fragment_dim * batch_per_block];

	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto matrix_index = tid / warp_size;

	if(matrix_index >= batch_size) return;

	const auto s_mem_ptr = s_mem + fragment_dim * fragment_dim * matrix_index;
	const auto m = min(s_size_m, g_size_m - s_size_m * matrix_index);

	mtk::matrix_copy::g2s16x16_1w(
			s_mem_ptr, 2, 2,
			ptr, matrix_index * s_size_m, g_size_m,
			tid
			);

	for(std::size_t i = 0; i < batch_size; i++){
		if(matrix_index == i && tid % 32 == 0){
			mtk::utils::print_matrix_16x16(s_mem_ptr, m, s_size_n, "mat");
		}
		__syncthreads();
	}

	mtk::matrix_copy::s2g16x16_1w(
			ptr, matrix_index * s_size_m, g_size_m,
			s_mem_ptr, m, s_size_n,
			tid
			);
}

int main(){
	std::cout<<"test : "<<__FILE__<<std::endl;

	auto h_mem_0 = cutf::memory::get_host_unique_ptr<test_t>(g_size_m * g_size_n);
	auto h_mem_1 = cutf::memory::get_host_unique_ptr<test_t>(g_size_m * g_size_n);
	auto g_mem = cutf::memory::get_device_unique_ptr<test_t>(g_size_m * g_size_n);

	for(std::size_t i = 0; i < g_size_m * g_size_n; i++){
		h_mem_0.get()[i] = cutf::type::cast<test_t>(static_cast<float>(i));
	}
	mtk::utils::print_matrix(h_mem_0.get(), g_size_m, g_size_n, "g");
	cutf::memory::copy(g_mem.get(), h_mem_0.get(), g_size_m * g_size_n);

	constexpr auto grid_size = (batch_size + batch_per_block - 1) / batch_per_block;

	kernel16x16<<<grid_size, batch_per_block * warp_size>>>(g_mem.get());

	cutf::memory::copy(h_mem_1.get(), g_mem.get(), g_size_m * g_size_n);
	mtk::utils::print_matrix(h_mem_1.get(), g_size_m, g_size_n, "g (g2s2g)");

	cudaDeviceSynchronize();
}
