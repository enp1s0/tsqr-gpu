#ifndef __TSQR_EXPERIMENTAL_HPP__
#define __TSQR_EXPERIMENTAL_HPP__

namespace mtk {
namespace experimental {
template <int min_exponent, class T>
void force_exponent(T* const ptr, const std::size_t size, cudaStream_t cuda_stream = 0);
} // namespace experimental
} // namespace mtk
#endif
