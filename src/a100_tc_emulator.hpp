#ifndef __A100_TC_EMULATOR_HPP__
#define __A100_TC_EMULATOR_HPP__
#include <cutf/debug/tf32.hpp>

// If this flag is not defined, error correction does not work.
#define A100_TC_COR

namespace mtk {
namespace a100_tc_cor {
#ifdef A100_TC_COR
__device__ inline void gemm_core16x16(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id){
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		float sum_dab = 0.0f;
		float sum_adb = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::debug::tf32::to_tf32(a[y + ldm_a * k]);
			const auto b_v = cutf::debug::tf32::to_tf32(b[x * ldm_b + k]);
			const auto a_dv = cutf::debug::tf32::to_tf32(a[y + ldm_a * k] - a_v);
			const auto b_dv = cutf::debug::tf32::to_tf32(b[x * ldm_b + k] - b_v);
			sum_ab = fmaf(a_v, b_v, sum_ab);
			sum_dab = fmaf(a_dv, b_v, sum_dab);
			sum_adb = fmaf(a_v, b_dv, sum_adb);
		}
		c[x * ldm_c + y] += sum_adb + sum_dab + sum_ab;
	}
}
#else
__device__ inline void gemm_core16x16(float* const c, const unsigned ldm_c, const float* const a, const unsigned ldm_a, const float* const b, const unsigned ldm_b, const unsigned unique_id){
	const auto lane = unique_id >> 4;
	const auto y = unique_id & 0xf;
	for(auto i = 0; i < 16; i += 2){
		const auto x = i + lane;
		float sum_ab = 0.0f;
		for(unsigned k = 0; k < 16; k += 1){
			const auto a_v = cutf::debug::tf32::to_tf32(a[y + ldm_a * k]);
			const auto b_v = cutf::debug::tf32::to_tf32(b[x * ldm_b + k]);
			sum_ab = fmaf(a_v, b_v, sum_ab);
		}
		c[x * ldm_c + y] += sum_ab;
	}
}
#endif //A100_TC_COR
}
}
#endif
