#include <algorithm>
#include <cmath>
#include <vector>
#include <mma.h>
#include <cuda_fp16.h>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include "tsqr.hpp"
#include "tcqr.hpp"
#include "utils.hpp"
#include "matrix_copy.cuh"

//#define DEBUG
//#define DEBUG_INPUT_MATRIX_PRINT
//#define DEBUG_Q_MATRIX_PRINT

namespace{
constexpr unsigned warp_size = 32;
template <class Func>
void debug_func(Func func){
#ifdef DEBUG
	func();
#endif
}
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

	debug_func([&m, &n](){std::printf("%s : matrix size = %lu x %lu\n", __func__, m, n);});
	debug_func([&m, &n](){std::printf("%s : working memory size = %lu\n", __func__, get_working_memory_size(m, n));});
	debug_func([&batch_size](){std::printf("%s : batch_size = %lu\n", __func__, batch_size);});
	debug_func([&working_r_ptr](){std::printf("%s : working_r_ptr[0] = 0x%x\n", __func__, working_r_ptr[0]);});
	debug_func([&working_r_ptr](){std::printf("%s : working_r_ptr[1] = 0x%x\n", __func__, working_r_ptr[1]);});
	debug_func([&working_q_ptr](){std::printf("%s : working_q_ptr    = 0x%x\n", __func__, working_q_ptr);});

	const auto d_sub_m_list = cutf::cuda::memory::get_device_unique_ptr<unsigned>(batch_size + 1);
	const auto h_sub_m_list = cutf::cuda::memory::get_host_unique_ptr<unsigned>(batch_size + 1);

	// 1層目はsub_mが特殊なので別途計算を行う
	h_sub_m_list.get()[0] = 0;
	for(std::size_t i = 1; i < batch_size; i++){
		h_sub_m_list.get()[i] = m * i / batch_size;
	}
	h_sub_m_list.get()[batch_size] = m;
	cutf::cuda::memory::copy(d_sub_m_list.get(), h_sub_m_list.get(), batch_size + 1);

	debug_func([&batch_size_log2](){std::printf("%s : %lu bQR\n", __func__, batch_size_log2);});
	debug_func([](){std::printf("%s : a -> wr[0]\n", __func__);});
	mtk::tcqr::qr32x16_f32tc_batched(
			working_q_ptr,
			working_r_ptr[0],
			a_ptr, m, n,
			batch_size, d_sub_m_list.get()
			);

	// 2層目からはsub matrixの大きさが 2n * n となるので，一度計算しGPUに転送しておけばOK
	for(std::size_t i = 0; i < batch_size / 2 + 1; i++){
		h_sub_m_list.get()[i] = 2 * n * i;
	}
	cutf::cuda::memory::copy(d_sub_m_list.get(), h_sub_m_list.get(), batch_size / 2 + 1);

	// 再帰的QR分解のfor展開
	for(std::size_t k = batch_size_log2 - 1; k > 1; k--){
		debug_func([&k](){std::printf("%s : %lu bQR\n", __func__, k);});
		const auto local_batch_size = 1lu << k;	
		const auto working_q_sride = 2 * n * n * (2 * batch_size - (1lu << (k + 1)));
		const auto working_r_index = 1lu - (batch_size_log2 - k) % 2;
		debug_func([&working_r_index, local_batch_size](){std::printf("%s : a(wr[%lu]) -> a(wr[%lu]) [l_bs : %lu]\n", __func__, working_r_index, 1-working_r_index, local_batch_size);});

#ifdef DEBUG_INPUT_MATRIX_PRINT
		{
			auto h_tmp = cutf::cuda::memory::get_host_unique_ptr<float>(2 * n * n * local_batch_size);
			cutf::cuda::memory::copy(h_tmp.get(), working_r_ptr[working_r_index], 2 * n * n * local_batch_size);
			mtk::utils::print_matrix(h_tmp.get(), 2 * n * local_batch_size, n, "input");
		}
#endif

		mtk::tcqr::qr32x16_f32tc_batched(
				working_q_ptr + working_q_sride,
				working_r_ptr[1 - working_r_index],
				working_r_ptr[working_r_index],
				2 * n * local_batch_size,
				n, 
				local_batch_size, d_sub_m_list.get()
				);

#ifdef DEBUG_Q_MATRIX_PRINT
	{
		auto h_tmp = cutf::cuda::memory::get_host_unique_ptr<float>(2 * n * n * local_batch_size);
		cutf::cuda::memory::copy(h_tmp.get(), working_q_ptr + working_q_sride, 2 * n * n * local_batch_size);
		mtk::utils::print_matrix(h_tmp.get(), 2 * n * local_batch_size, n, "Q");
	}
#endif

	}

	// 最終層はrの保存先が異なる
	debug_func([](){std::printf("%s : 1 bQR\n", __func__);});
	debug_func([&batch_size_log2](){std::printf("%s : a(wr[%lu]) -> r\n", __func__, (batch_size_log2 % 2));});
	const auto working_q_sride = 2 * n * n * (2 * batch_size - 2);
	mtk::tcqr::qr32x16_f32tc_batched(
			working_q_ptr + working_q_sride,
			r_ptr,
			working_r_ptr[batch_size_log2 % 2],
			2 * n,
			n,
			1, d_sub_m_list.get()
			);
}
