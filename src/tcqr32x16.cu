#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>

namespace {
constexpr unsigned warp_size = 32;

template <class INPUT_T, class OUTPUT_T>
__device__ OUTPUT_T get_norm2_32(
		INPUT_T* const ptr, const unsigned size,
	   	unsigned warp_id){
	auto tmp = cutf::cuda::type::cast<OUTPUT_T>(0.0f);

	if(warp_id < size){
		tmp = cutf::cuda::type::cast<OUTPUT_T>(ptr[warp_id]);
		tmp = tmp * tmp;
	}

	for(auto mask = (warp_size >> 1); mask > 0; mask >>= 1){
		tmp += __shfl_xor_sync(0xffffffff, tmp, mask);
	}

	return cutf::cuda::type::cast<OUTPUT_T>(tmp);
}

template <class T, class U_T, std::size_t FRAGMENT_DIM_M = 32>
__device__ void make_h(
		T* const h_ptr, const unsigned m, 
		const U_T* const u_ptr, const U_T norm2_u_1, 
		const unsigned unique_id){
	const auto y = unique_id & 0x1f;
	const auto lane = unique_id >> 5;
	for(unsigned k = 0; k < FRAGMENT_DIM_M; k+= 2){
		const auto x = k + lane;
		U_T tmp;
		if(x == y){
			tmp = cutf::cuda::type::cast<U_T>(1.0f);
		}else{
			tmp = cutf::cuda::type::cast<U_T>(0.0f);
		}
		tmp -= cutf::cuda::type::cast<U_T>(2.0f) * u_ptr[y] * u_ptr[x] / norm2_u_1;

		h_ptr[x * FRAGMENT_DIM_M + y] = cutf::cuda::type::cast<T>(tmp);
	}
}
template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ void update_qr(
		float* const q32_ptr, float* const r32_ptr,
		const half* const q16_ptr, const half* const r16_ptr,
		half* h16_ptr,
		const unsigned tid
		){
	const auto lane = unique_id >> 5;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> h16_0_frag, h16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> r16_0_frag, r16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> q16_0_frag, q16_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, nvcuda::wmma::col_major> r32_0_frag, r32_1_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, nvcuda::wmma::col_major> q32_0_frag, q32_1_frag;
	for(unsigned i = 0; i < 2; i++){
		
	}
}

template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__device__ void qr32x16_f32tc_core(
		float* const q32_ptr, float* const r32_ptr,
		half* const q16_ptr, half* const r16_ptr,
		float* const u32_ptr, half* h16_ptr,
		const unsigned m, const unsigned n,
		const unsigned tid
		){
	const auto unique_id = tid & 0x3f;
	cutf::cuda::math::ceil(
	for(unsigned k = 0; k < n - 1; k++){
		// copy u
		// TODO ; 0埋めとデータロードを異なるwarpでできないか検証
		if(unique_id < FRAGMENT_DIM_M){
			u32_ptr[unique_id] = 0.0f;
			if(unique_id >= k){
				u32_ptr[unique_id] = r32_ptr[FRAGMENT_DIM_M * k + unique_id];
			}
		}
		__syncthreads();
		// compute |u|
		// TODO : どうせ0埋めされているなら32個で和をとってしまってもいい気がするので検証
		const auto norm_u_0 = cutf::cuda::math::sqrt<float>(get_norm2_32<float, float>(u32_ptr, m, unique_id & 0x1f));
		__syncthreads();
		// update u
		if(unique_id == k){
			u32_ptr[unique_id] += cutf::cuda::math::sign(u32_ptr[unique_id]) * norm_u_0;
		}
		__syncthreads();
		// recompute |u|
		const auto norm2_u_1 = get_norm2_32<float, float>(u32_ptr, m, unique_id & 0x1f);
		// compute h
		make_h(
				h16_ptr, m,
				u32_ptr, norm2_u_1,
				unique_id
				);
		// copy f32 to f16
		// update q, r
	}
}

template <std::size_t FRAGMENT_DIM_M = 32, std::size_t FRAGMENT_DIM_N = 16>
__global__ void qr32x16_f32_batched_kernel(
		float* const q32_ptr,
		float* const r32_ptr,
		const float* const a32_ptr,
		const unsigned m,
		const unsigned n,
		std::size_t batch_size
		){

}
}
