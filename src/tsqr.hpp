#ifndef __TSQR_HPP__
#define __TSQR_HPP__
#include <cstddef>
#include <exception>
#include <cuda_fp16.h>

namespace mtk {
namespace tsqr {
enum compute_mode {
	fp16_notc,
	fp16_tc_nocor,
	fp32_notc,
	fp32_tc_cor,
	fp32_tc_nocor,
	mixed_tc_cor_emu,
	tf32_tc_cor,
	tf32_tc_cor_emu,
	tf32_tc_nocor,
	tf32_tc_nocor_emu,
};
// get batch size
std::size_t get_batch_size_log2(const std::size_t m);
std::size_t get_batch_size(const std::size_t m);
// get working memory type
template <compute_mode mode>
struct get_working_q_type{using type = float;};
template<> struct get_working_q_type<mtk::tsqr::compute_mode::fp16_notc      >{using type = half ;};
template<> struct get_working_q_type<mtk::tsqr::compute_mode::fp16_tc_nocor  >{using type = half ;};
template<> struct get_working_q_type<mtk::tsqr::compute_mode::fp32_tc_nocor  >{using type = half ;};

template <compute_mode mode>
struct get_working_r_type{using type = float;};
template<> struct get_working_r_type<mtk::tsqr::compute_mode::fp16_notc      >{using type = half ;};
template<> struct get_working_r_type<mtk::tsqr::compute_mode::fp16_tc_nocor  >{using type = half ;};

template <compute_mode mode>
struct get_io_type{using type = float;};
template<> struct get_io_type<mtk::tsqr::compute_mode::fp16_notc      >{using type = half ;};
template<> struct get_io_type<mtk::tsqr::compute_mode::fp16_tc_nocor  >{using type = half ;};

// get working memory size
std::size_t get_working_q_size(const std::size_t m, const std::size_t n);
std::size_t get_working_r_size(const std::size_t m, const std::size_t n);
inline std::size_t get_working_l_size(const std::size_t m) {
	return get_batch_size(m) + 1lu;
}

// integrated buffer struct
template <mtk::tsqr::compute_mode mode>
struct buffer {
	typename get_working_q_type<mode>::type* dwq;
	typename get_working_r_type<mode>::type* dwr;
	unsigned* dl;
	unsigned* hl;

	std::size_t total_memory_size;

	// constructor
	buffer() : dwq(nullptr), dwr(nullptr), dl(nullptr), hl(nullptr), total_memory_size(0lu) {}
	// destructor
	~buffer() {
		destroy();
	}

	void allocate(const std::size_t m, const std::size_t n) {
		if (dwq != nullptr || dwr != nullptr || dl != nullptr || hl != nullptr) {
			throw std::runtime_error("The buffer has been already allocated");
		}
		const auto wq_size = sizeof(typename get_working_q_type<mode>::type) * get_working_q_size(m, n);
		const auto wr_size = sizeof(typename get_working_r_type<mode>::type) * get_working_r_size(m, n);
		const auto l_size = sizeof(unsigned) * get_working_l_size(m);
		cudaMalloc(reinterpret_cast<void**>(&dwq), wq_size);
		cudaMalloc(reinterpret_cast<void**>(&dwr), wr_size);
		cudaMalloc(reinterpret_cast<void**>(&dl), l_size);
		cudaMallocHost(reinterpret_cast<void**>(&hl), l_size);
		total_memory_size = wq_size + wr_size + l_size;
	}
	void destroy() {
		cudaFree(dwq); dwq = nullptr;
		cudaFree(dwr); dwr = nullptr;
		cudaFree(dl); dl = nullptr;
		cudaFreeHost(hl); hl = nullptr;
	}
	void allocate_host(const std::size_t m, const std::size_t n) {
		if (dwq != nullptr || dwr != nullptr || dl != nullptr || hl != nullptr) {
			throw std::runtime_error("The buffer has been already allocated");
		}
		const auto wq_size = sizeof(typename get_working_q_type<mode>::type) * get_working_q_size(m, n);
		const auto wr_size = sizeof(typename get_working_r_type<mode>::type) * get_working_r_size(m, n);
		const auto l_size = sizeof(unsigned) * get_working_l_size(m);
		cudaMallocHost(reinterpret_cast<void**>(&dwq), wq_size);
		cudaMallocHost(reinterpret_cast<void**>(&dwr), wr_size);
		cudaMallocHost(reinterpret_cast<void**>(&dl), l_size);
		cudaMallocHost(reinterpret_cast<void**>(&hl), l_size);
		total_memory_size = wq_size + wr_size + l_size;
	}
	void destroy_host() {
		cudaFreeHost(dwq); dwq = nullptr;
		cudaFreeHost(dwr); dwr = nullptr;
		cudaFreeHost(dl); dl = nullptr;
		cudaFreeHost(hl); hl = nullptr;
	}
	std::size_t get_device_memory_size() const {
		return total_memory_size;
	}
};

template <mtk::tsqr::compute_mode mode>
void tsqr16(
		typename mtk::tsqr::get_io_type<mode>::type* const q_ptr, const std::size_t ldq,
		typename mtk::tsqr::get_io_type<mode>::type* const r_ptr, const std::size_t ldr,
		const typename mtk::tsqr::get_io_type<mode>::type* const a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n,
		typename get_working_q_type<mode>::type* const working_q_ptr,
		typename get_working_r_type<mode>::type* const working_r_ptr,
		unsigned* const d_working_l_ptr,
		unsigned* const h_working_l_ptr,
		cudaStream_t const cuda_stream = nullptr);


template <mtk::tsqr::compute_mode mode>
inline void tsqr16(
		typename mtk::tsqr::get_io_type<mode>::type* const q_ptr, const std::size_t ldq,
		typename mtk::tsqr::get_io_type<mode>::type* const r_ptr, const std::size_t ldr,
		const typename mtk::tsqr::get_io_type<mode>::type* const a_ptr, const std::size_t lda,
		const std::size_t m, const std::size_t n,
		mtk::tsqr::buffer<mode>& buffer,
		cudaStream_t const cuda_stream) {
	mtk::tsqr::tsqr16<mode>(
			q_ptr, ldq,
			r_ptr, ldr,
			a_ptr, lda,
			m, n,
			buffer.dwq,
			buffer.dwr,
			buffer.dl,
			buffer.hl,
			cuda_stream);
}
} // namespace tsqr
} // namespace mtk

#endif /* end of include guard */
