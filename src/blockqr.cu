#include "blockqr.hpp"
#include "tsqr.hpp"

template <class T, bool UseTC, bool Refinement>
void mtk::qr(
		T* const q_ptr, const std::size_t ldq,
		T* const r_ptr, const std::size_t ldr,
		const std::size_t m, const std::size_t n,
		const T* a_ptr, const std::size_t lda) {
	constexpr auto tsqr_colmun_size = 16;

	const auto column_block_size = (n + tsqr_colmun_size - 1) / tsqr_colmun_size;

	// QR factorization of each block
	for (std::size_t b = 0; b < column_block_size; b++) {
		// compute A'
		for (std::size_t i = 0; i < b; i++) {

		}

		//QR factorization of A'
	}
}
