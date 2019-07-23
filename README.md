# TSQRのGPU実装 (using TensorCore)

![TSQR](https://gitlab.momo86.net/mutsuki/tsqr-gpu/raw/master/docs/tsqr.svg)

## ビルド

```
git clone [this remote repository] --recursive
cd tsqr-gpu
make
```

## 実行環境
- 少なくともC++ 11
- CUDA 9.1 以上
- Tensorコアを使いたければTensotコアを積んでいるGPU

## 依存
- cutf : [https://github.com/enp1s0/cutf](https://github.com/enp1s0/cutf)
- gemm_core : [https://gitlab.momo86.net/mutsuki/gemm_core](https://gitlab.momo86.net/mutsuki/gemm_core)

## 実装済み
- m x n (n <= 16) な行列に対するTSQR
	- FP16 using TensorCore
	- FP32 using TensorCore
	- FP16
	- FP32

## 予定
- 幅がN以下限定の行列に対するTSQRの実装

## その後の予定
- rsvdへの応用

## 関連リポジトリ
- [TCQR](https://gitlab.momo86.net/mutsuki/tcqr)
