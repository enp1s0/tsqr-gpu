#ifndef __TSQR_EXPERIMENTAL_HPP__
#define __TSQR_EXPERIMENTAL_HPP__

namespace mtk {
namespace experimental {
template <class T>
void force_exponent(T* const ptr, const int min_exponent, const std::size_t size, cudaStream_t const cuda_stream = 0);
} // namespace experimental
} // namespace mtk
#endif
