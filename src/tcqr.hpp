#ifndef __TCQR_HPP__
#define __TCQR_HPP__

namespace mtk {
namespace tcqr{
template <class T, bool UseTC>
void qr32x16(T* const q, T* const r,const T* const a, const unsigned int m, const unsigned int n);
void qr32x16_f32tc_batched(float *const q, float *const r, const float *const a, const unsigned int m, const unsigned int n, const std::size_t batch_size, const unsigned* a_start_position);
}
} // namespace mtk

#endif /* end of include guard */
