#ifndef __TCQR_HPP__
#define __TCQR_HPP__

namespace mtk {
namespace tcqr{
template <bool UseTC, class Q_T, class R_T, class A_T>
void qr32x16(Q_T* const q, R_T* const r,const A_T* const a, const unsigned int m, const unsigned int n);
template <bool UseTC, class Q_T, class R_T, class A_T>
void qr32x16_batched(Q_T *const q, R_T *const r, const A_T *const a, const unsigned int m, const unsigned int n, const std::size_t batch_size, const unsigned* a_start_position);
}
} // namespace mtk

#endif /* end of include guard */
