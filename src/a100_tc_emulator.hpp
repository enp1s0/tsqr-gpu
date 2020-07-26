#ifndef __A100_TC_EMULATOR_HPP__
#define __A100_TC_EMULATOR_HPP__
#include <cutf/experimental/tf32.hpp>
#include <cutf/type.hpp>
#include "gemm_core/gemm_core.cuh"

namespace mtk {
namespace matmul {
enum compute_mode {
	fp16_notc,
	fp32_notc,
	tf32_tc_cor_emu,
	tf32_tc_nocor_emu,
	mixed_tc_cor,
};

template <mtk::matmul::compute_mode mode, class T>
__device__ inline void matmul_core_m16n16k32(T* const c, const unsigned ldm_c, const T* const a, const unsigned ldm_a, const T* const b, const unsigned ldm_b, const unsigned unique_id) {
	mtk::matmul_core16x16<32>(c, ldm_c, a, ldm_a, b, ldm_b, unique_id);
}

template <> __device__ inline void matmul_core_m16n16k32<mtk::matmul::compute_mode::tf32_tc_cor_emu, float>(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_a[16];
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i++){
		tmp_a[i] = a[y + ldm_a * i];
	}

	for (auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		float sum_dab = 0.0f;
		float sum_adb = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::experimental::tf32::to_tf32(tmp_a[k]);
			const auto b_v = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k]);
			const auto a_dv = cutf::experimental::tf32::to_tf32(tmp_a[k] - a_v);
			const auto b_dv = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k] - b_v);
			sum_ab = fmaf(a_v, b_v, sum_ab);
			sum_dab = fmaf(a_dv, b_v, sum_dab);
			sum_adb = fmaf(a_v, b_dv, sum_adb);
		}
		tmp_c[i / 2] += sum_adb + sum_dab + sum_ab;
	}

	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}

template <> __device__ inline void matmul_core_m16n16k32<mtk::matmul::compute_mode::mixed_tc_cor, float>(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_a[16];
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i++){
		tmp_a[i] = a[y + ldm_a * i];
	}

	for (auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		float sum_dab = 0.0f;
		float sum_adb = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::type::cast<float>(cutf::type::cast<half>(tmp_a[k]));
			const auto b_v = cutf::type::cast<float>(cutf::type::cast<half>(b[x * ldm_b + k]));
			const auto a_dv = cutf::experimental::tf32::to_tf32(tmp_a[k] - a_v);
			const auto b_dv = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k] - b_v);
			sum_ab = fmaf(a_v, b_v, sum_ab);
			sum_dab = fmaf(a_dv, b_v, sum_dab);
			sum_adb = fmaf(a_v, b_dv, sum_adb);
		}
		tmp_c[i / 2] += sum_adb + sum_dab + sum_ab;
	}

	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}

template <> __device__ inline void matmul_core_m16n16k32<mtk::matmul::compute_mode::tf32_tc_nocor_emu, float>(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_a[16];
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i++){
		tmp_a[i] = a[y + ldm_a * i];
	}

	for (auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		float sum_dab = 0.0f;
		float sum_adb = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::experimental::tf32::to_tf32(tmp_a[k]);
			const auto b_v = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k]);
			sum_ab = fmaf(a_v, b_v, sum_ab);
		}
		tmp_c[i / 2] += sum_adb + sum_dab + sum_ab;
	}

	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}

template <mtk::matmul::compute_mode mode, class T>
__device__ inline void matmul_core_m16n16k16(T* const c, const unsigned ldm_c, const T* const a, const unsigned ldm_a, const T* const b, const unsigned ldm_b, const unsigned unique_id) {
	mtk::matmul_core16x16<16>(c, ldm_c, a, ldm_a, b, ldm_b, unique_id);
}

template <> __device__ inline void matmul_core_m16n16k16<mtk::matmul::compute_mode::tf32_tc_cor_emu, float>(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_a[16];
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i++){
		tmp_a[i] = a[y + ldm_a * i];
	}

	for (auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		float sum_dab = 0.0f;
		float sum_adb = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::experimental::tf32::to_tf32(tmp_a[k]);
			const auto b_v = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k]);
			const auto a_dv = cutf::experimental::tf32::to_tf32(tmp_a[k] - a_v);
			const auto b_dv = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k] - b_v);
			sum_ab = fmaf(a_v, b_v, sum_ab);
			sum_dab = fmaf(a_dv, b_v, sum_dab);
			sum_adb = fmaf(a_v, b_dv, sum_adb);
		}
		tmp_c[i / 2] += sum_adb + sum_dab + sum_ab;
	}

	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}

template <> __device__ inline void matmul_core_m16n16k16<mtk::matmul::compute_mode::mixed_tc_cor, float>(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_a[16];
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i++){
		tmp_a[i] = a[y + ldm_a * i];
	}

	for (auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		float sum_dab = 0.0f;
		float sum_adb = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::type::cast<float>(cutf::type::cast<half>(tmp_a[k]));
			const auto b_v = cutf::type::cast<float>(cutf::type::cast<half>(b[x * ldm_b + k]));
			const auto a_dv = cutf::experimental::tf32::to_tf32(tmp_a[k] - a_v);
			const auto b_dv = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k] - b_v);
			sum_ab = fmaf(a_v, b_v, sum_ab);
			sum_dab = fmaf(a_dv, b_v, sum_dab);
			sum_adb = fmaf(a_v, b_dv, sum_adb);
		}
		tmp_c[i / 2] += sum_adb + sum_dab + sum_ab;
	}

	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}

template <> __device__ inline void matmul_core_m16n16k16<mtk::matmul::compute_mode::tf32_tc_nocor_emu, float>(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id) {
	constexpr unsigned warp_size = 32;
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	float tmp_a[16];
	float tmp_c[16 * 16 / warp_size];

	for (auto i = 0; i < 16; i++){
		tmp_a[i] = a[y + ldm_a * i];
	}

	for (auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		float sum_dab = 0.0f;
		float sum_adb = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::experimental::tf32::to_tf32(tmp_a[k]);
			const auto b_v = cutf::experimental::tf32::to_tf32(b[x * ldm_b + k]);
			sum_ab = fmaf(a_v, b_v, sum_ab);
		}
		tmp_c[i / 2] += sum_adb + sum_dab + sum_ab;
	}

	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		c[x * ldm_c + y] = tmp_c[i / 2];
	}
}
} // namespace a100_tc_cor
} // namespace mtk
#endif
