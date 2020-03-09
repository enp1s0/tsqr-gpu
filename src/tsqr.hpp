#ifndef __TSQR_HPP__
#define __TSQR_HPP__
#include <cstddef>
#include <cuda_fp16.h>

namespace mtk {
namespace tsqr {
// get batch size
std::size_t get_batch_size_log2(const std::size_t m);
std::size_t get_batch_size(const std::size_t m);
// get working memory type
template <class T, bool UseTC, bool Refine>
struct get_working_q_type{using type = T;};
template <> struct get_working_q_type<float, true, false>{using type = half;};

template <class T, bool UseTC, bool Refine>
struct get_working_r_type{using type = T;};

// get working memory size
std::size_t get_working_q_size(const std::size_t m, const std::size_t n);
std::size_t get_working_r_size(const std::size_t m, const std::size_t n);
inline std::size_t get_working_l_size(const std::size_t m) {
	return get_batch_size(m) + 1lu;
}

// integrated buffer struct
template <class T, bool UseTC, bool Refine>
struct buffer {
	typename get_working_q_type<T, UseTC, Refine>::type* dwq;
	typename get_working_r_type<T, UseTC, Refine>::type* dwr;
	unsigned* dl;
	unsigned* hl;

	// constructor
	buffer() : dwq(nullptr), dwr(nullptr), dl(nullptr), hl(nullptr) {}
	// destructor
	~buffer() {
		destroy();
	}

	void allocate(const std::size_t m, const std::size_t n) {
		cudaMalloc(reinterpret_cast<void**>(&dwq), sizeof(typename get_working_q_type<T, UseTC, Refine>::type) * get_working_q_size(m, n));
		cudaMalloc(reinterpret_cast<void**>(&dwr), sizeof(typename get_working_r_type<T, UseTC, Refine>::type) * get_working_r_size(m, n));
		cudaMalloc(reinterpret_cast<void**>(&dl), sizeof(unsigned) * get_working_l_size(m));
		cudaMallocHost(reinterpret_cast<void**>(&dl), sizeof(unsigned) * get_working_l_size(m));
	}
	void destroy() {
		cudaFree(dwq); dwq = nullptr;
		cudaFree(dwr); dwr = nullptr;
		cudaFree(dl); dl = nullptr;
		cudaFreeHost(hl); hl = nullptr;
	}
};

template <bool UseTC, bool Refine, class T, class CORE_T>
void tsqr16(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const T* const a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n,
		typename get_working_q_type<T, UseTC, Refine>::type* const working_q_ptr,
		typename get_working_r_type<T, UseTC, Refine>::type* const working_r_ptr,
		unsigned* const d_working_l_ptr,
		unsigned* const h_working_l_ptr,
		cudaStream_t const cuda_stream = nullptr);

template <bool UseTC, bool Refine, class T, class CORE_T>
void tsqr16(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const T* const a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n,
		buffer<T, UseTC, Refine>& bf,
		cudaStream_t const cuda_stream = nullptr);
} // namespace tsqr
} // namespace mtk

#endif /* end of include guard */
