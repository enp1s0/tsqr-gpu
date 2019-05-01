#ifndef __TCQR_HPP__
#define __TCQR_HPP__

namespace mtk {
namespace tcqr{
void qr32x16_f32tc(float* const q, float* const r, const float* const a, const unsigned m, const unsigned n);
template <class T, class Norm_t, bool UseTC>
void qr16x16_batched(T* const* const q, T* const * const r, const T* const* const a, const std::size_t m, const std::size_t n, const std::size_t batch_size);
}
} // namespace mtk

#endif /* end of include guard */
