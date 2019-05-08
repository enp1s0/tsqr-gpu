#include <cmath>
#include <cutf/memory.hpp>
#include "validation.hpp"
#include "matrix_copy.cuh"

namespace{
constexpr unsigned warp_size = 32;
template <unsigned num_warps_per_block, unsigned DIM_M = 32, unsigned DIM_N = 16>
__global__ void check_orthogonality16_kernel(
		float *orthogonality2,
		const float* const matrix,
		const std::size_t m,
		const unsigned n
		){
	__shared__ float mat_a_shared[DIM_M * DIM_N * num_warps_per_block];
	__shared__ float mat_b_shared[DIM_M * DIM_N * num_warps_per_block];
	__shared__ float norm2_sum[num_warps_per_block];
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto unique_id = tid & 0x1f;
	const std::size_t num_submatrix = (m + DIM_M - 1) / DIM_M;
	const auto position = tid >> 5;
	if(position >= num_submatrix) return;
	const auto submatrix_a = position / num_submatrix;
	const auto submatrix_b = position % num_submatrix;

	mtk::matrix_copy::g2s32x16_1w(
			mat_a_shared, min(static_cast<std::size_t>(DIM_M), m - submatrix_a * DIM_M), n,
			matrix, submatrix_a * DIM_M, m,
			tid
			);
	mtk::matrix_copy::g2s32x16_1w(
			mat_b_shared, min(static_cast<std::size_t>(DIM_M), m - submatrix_b * DIM_M), n,
			matrix, submatrix_b * DIM_M, m,
			tid
			);

	float norm2 = 0.0f;
	for(unsigned i = 0; i < DIM_M; i++){
		float sum = 0.0f;
		if(submatrix_a == submatrix_b && i == unique_id) sum = -1.0f;
		for(unsigned k = 0; k < DIM_N; k++){
			sum += mat_a_shared[unique_id + k * DIM_M] * mat_b_shared[i * DIM_N + k];
		}
		norm2 += sum * sum;
	}
	atomicAdd(orthogonality2, norm2);
}
}

float mtk::validation::check_orthogonality16(
		const float* const matrix,
		const std::size_t m,
		const unsigned n
		){
	constexpr std::size_t num_warps_per_block = 8;
	constexpr std::size_t DIM_M = 32;
	auto d_orthogonality = cutf::cuda::memory::get_device_unique_ptr<float>(1);
	float h_orthogonality = 0.0f;
	cutf::cuda::memory::copy(d_orthogonality.get(), &h_orthogonality, 1);

	const auto block_size = num_warps_per_block * warp_size;
	const auto grid_size = ((m + DIM_M - 1) / DIM_M + num_warps_per_block - 1) / num_warps_per_block;

	check_orthogonality16_kernel<num_warps_per_block><<<grid_size, block_size>>>(
			d_orthogonality.get(),
			matrix, m, n
			);

	cutf::cuda::memory::copy(&h_orthogonality, d_orthogonality.get(), 1);


	return std::sqrt(h_orthogonality / m);
}
