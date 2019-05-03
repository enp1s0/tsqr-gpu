#include <algorithm>
#include <cmath>
#include <cutf/memory.hpp>
#include <vector>
#include "tsqr.hpp"
#include "tcqr.hpp"

namespace{
std::size_t get_batch_size_log2(const std::size_t m){
	return (std::max(5u, static_cast<unsigned>( std::ceil( std::log2(static_cast<float>(m))))) - 5u);
}
std::size_t get_batch_size(const std::size_t m){
	return 1lu << get_batch_size_log2(m);
}
}

// 必要な作業用メモリ
std::size_t mtk::tsqr::get_working_memory_size(const std::size_t m, const std::size_t n){
	const auto batch_size = get_batch_size(m);
	const auto working_q_size = n * m + 2 * n * n * (batch_size - 1);
	const auto working_r_size_0 = n * n * batch_size;
	const auto working_r_size_1 = n * n * batch_size / 2;

	return working_q_size + working_r_size_0 + working_r_size_1;
}

void mtk::tsqr::tsqr16(
		float *const q_ptr, float *const r_ptr, 
		const float *const a_ptr, const std::size_t m, const std::size_t n, 
		float *const working_memory_ptr){
	const auto batch_size_log2 = get_batch_size_log2(m);
	const auto batch_size = 1lu << batch_size_log2;
	float* const working_r_ptr[2] = {working_memory_ptr, working_memory_ptr + n * n * batch_size};
	const auto working_q_ptr = working_r_ptr[1] + n * n * batch_size / 2;

	const auto d_sub_m_list = cutf::cuda::memory::get_device_unique_ptr<unsigned>(batch_size + 1);
	const auto h_sub_m_list = cutf::cuda::memory::get_host_unique_ptr<unsigned>(batch_size + 1);

	// 1層目はsub_mが特殊なので別途計算を行う
	h_sub_m_list.get()[0] = 0;
	for(std::size_t i = 1; i < batch_size; i++){
		h_sub_m_list.get()[i] = m * i / batch_size + h_sub_m_list.get()[i - 1];
	}
	h_sub_m_list.get()[batch_size] = batch_size;
	cutf::cuda::memory::copy(d_sub_m_list.get(), h_sub_m_list.get(), batch_size + 1);

	mtk::tcqr::qr32x16_f32tc_batched(
			working_q_ptr,
			working_r_ptr[0],
			a_ptr, m, n,
			batch_size, d_sub_m_list.get()
			);

	// 2層目からはsub matrixの大きさが 2n * n となるので，一度計算しGPUに転送しておけばOK
	for(std::size_t i = 0; i < batch_size / 2 + 1; i++){
		h_sub_m_list.get()[i] = n * i;
	}
	cutf::cuda::memory::copy(d_sub_m_list.get(), h_sub_m_list.get(), batch_size / 2 + 1);

	// 再帰的QR分解のfor展開
	for(std::size_t k = batch_size_log2 - 1; k > 1; k--){
		const auto local_batch_size = 1lu << k;	
		const auto working_q_sride = 2 * n * n * (2 * batch_size - (1lu << (k + 1)));
		mtk::tcqr::qr32x16_f32tc_batched(
				working_memory_ptr + working_q_sride,
				working_r_ptr[(batch_size_log2 - k) % 2],
				working_r_ptr[1 - (batch_size_log2 - k) % 2],
				n * local_batch_size,
				n, 
				local_batch_size, d_sub_m_list.get()
				);
	}

	// 最終層はrの保存先が異なる
	const auto working_q_sride = 2 * n * n * (2 * batch_size - 2);
	mtk::tcqr::qr32x16_f32tc_batched(
			working_q_ptr + working_q_sride,
			r_ptr,
			working_r_ptr[1 - batch_size_log2 % 2],
			2 * n,
			n,
			1, d_sub_m_list.get()
			);
}
