# 基於光流分析之向量量化變分自編碼器影像重建模型 (Optical-Flow-based VQ-VAE)

> Optical Flow Analysis + Vector Quantized Variational Autoencoder (VQ-VAE) for Image Reconstruction

本專案為 **2025 National Symposium on System Science and Engineering (國立中興大學)** 發表論文《基於光流分析之向量量化變分自編碼器之影像重建模型研究》的實作程式碼。我們將 **FlowNet** 的光流分析架構與 **VQ-VAE (Vector Quantized Variational Autoencoder)** 結合，探討光流特徵、損失函數選擇、以及深度可分離式卷積對影像重建品質與運算成本的影響。

論文全文：`NSSSE-2025_基於光流分析之向量量化變分自編碼器之影像重建模型研究.pdf`（附於 repo 中）

---

## 目錄

- [研究動機](#研究動機)
- [模型架構](#模型架構)
- [本專案（程式碼）與論文設定的差異](#本專案程式碼與論文設定的差異)
- [實驗結果](#實驗結果)
- [環境需求](#環境需求)
- [使用方式](#使用方式)
- [檔案結構](#檔案結構)
- [參考文獻](#參考文獻)

---

## 研究動機

傳統 VAE (Variational Autoencoder) 因為採用連續潛在空間，容易造成特徵過度混合 (Overlapping)，使重建影像模糊；VQ-VAE (Van Den Oord et al., 2018) 透過**離散化編碼**改善了這個問題。另一方面，FlowNet (Dosovitskiy et al., 2015) 提出了以卷積神經網路分析前後兩幀影像光流變化的架構，能有效捕捉影像間的動態資訊。

本研究將兩者結合：

- 用 FlowNet 的**下採樣 + 光流預測**模組作為 VQ-VAE 的 **Encoder**，藉此在壓縮特徵的同時保留影像間的變化資訊；
- 用 FlowNet 的**上採樣（反卷積）**模組作為 VQ-VAE 的 **Decoder**，但移除光流預測部分，專注於影像重建；
- 中間以**向量量化 (Vector Quantization)** 計算離散潛在向量 (Latent Vector)，作為 Encoder 與 Decoder 之間傳遞特徵的媒介。

並進一步比較：
- **L1 vs. L2 損失函數**對重建品質的影響
- **FlowNet 兩種模式**：Simple（兩幀疊合後下採樣）vs. Correlation（兩幀分別分析再融合）
- 加入 **深度可分離式卷積 (Depthwise Separable Convolution, DSC)** 對模型參數量與運算時間的影響

---

## 模型架構

```
輸入影像 (Image1, Image2)
        │
        ▼
┌───────────────────────┐
│   Encoder              │
│   FlowNetSConv          │
│   ├─ Depthwise Sep. Conv (conv1~conv3) 特徵萃取＋下採樣
│   ├─ CorrelationLayer   → 計算兩幀特徵的相關性 (Correlation 模式)
│   └─ conv3_1 ~ conv6_1  → 深層特徵萃取
└───────────────────────┘
        │  多尺度特徵 (out_conv3_1, out_conv4, out_conv5, out_conv6)
        ▼
┌───────────────────────┐
│  Vector Quantizer       │
│  將連續特徵對應到最近的 │
│  codebook 向量（離散化）│
│  Loss = commitment_loss  │
│        + codebook_loss  │
└───────────────────────┘
        │  量化後潛在向量
        ▼
┌───────────────────────┐
│   Decoder               │
│   FlowNetSDeconv         │
│   ├─ deconv2~deconv5    → 反卷積上採樣，逐層還原解析度
│   └─ predict_flow2~6    → 逐層預測並融合特徵（不含光流輸出）
└───────────────────────┘
        │
        ▼
      重建影像
```

### 各模組對應到程式碼中的類別

| 論文架構 | 對應程式碼 |
|---|---|
| FlowNet 特徵萃取 + 光流預測（Encoder 下採樣） | `FlowNetSConv` |
| 兩幀特徵相關性計算（Correlation 模式） | `CorrelationLayer` |
| 向量量化模組 | `VectorQuantizer` |
| FlowNet 上採樣（Decoder，不含光流預測） | `FlowNetSDeconv` |
| 深度可分離式卷積 | `conv()` 函式中的 `DepthwiseConv2D` + 1x1 `Conv2D`（Pointwise） |
| 整體 VQ-VAE 模型組裝 | `get_encoder()` / `get_decoder()` / `get_vqvae()` |
| 訓練迴圈與損失計算 | `VQVAETrainer` |

---

## 本專案（程式碼）與論文設定的差異

論文本體的實驗是使用 **FlyingChairs 擴充資料集**進行 Simple / Correlation 兩種模式、L1 / L2 兩種損失函數的完整比較（詳見論文圖 3、圖 4、表 1）。

本 repo 提供的程式碼則是以 **CIFAR-10** 資料集，實作論文中 **「Correlation 模式 + L2 損失函數」** 這一組設定，作為架構的簡化驗證版本（方便在一般硬體上快速重現與展示），對應概念上等同於論文圖 4(b) 的實驗。

若要重現論文完整的六組實驗（Simple/Correlation × L1/L2/DSC），需自行替換為 FlyingChairs 資料集並依論文表 1 的超參數（batch size=32、lr=0.00001、eps=0.000001、commitment loss lr=0.01、epoch=15）調整訓練設定。

---

## 實驗結果

以下為本 repo 程式碼在 CIFAR-10 上訓練 30 epoch 的結果（Correlation 模式 + L2 損失函數）：

![Training Result](./training_result.png)

- **Loss**：由初始約 0.11 快速收斂，30 epoch 後穩定在約 0.02 左右，訓練集與測試集損失曲線接近，沒有明顯過擬合。
- **重建相似度 (Accuracy)**：由約 47% 提升至約 71%，訓練與測試相似度曲線大致貼合。

> 相似度計算方式：`similarity = (1 - MSE / 資料變異數) × 100%`，詳見 `VQVAETrainer.calculate_similarity()`。

論文中 FlyingChairs 資料集上的完整六組比較結果（含重建影像視覺化）請參考論文圖 3、圖 4 及表 1。

---

## 環境需求

```
python >= 3.9
tensorflow >= 2.x
numpy
matplotlib
```

安裝：

```bash
pip install tensorflow numpy matplotlib
```

---

## 使用方式

```bash
python vqvae_flownet.py
```

程式會自動：
1. 下載並前處理 CIFAR-10 資料集
2. 建立 VQ-VAE + FlowNet(Correlation) 模型
3. 訓練 30 個 epoch
4. 繪製並顯示 Loss / Accuracy 曲線圖

可調整的主要超參數（於程式底部）：

```python
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=64, num_embedding=128)
history = vqvae_trainer.fit(
    x_train_scaled,
    validation_data=(x_test_scaled, x_test_scaled),
    epochs=30,
    batch_size=128
)
```

---

## 檔案結構

```
.
├── vqvae_flownet.py     # 主程式：模型定義、訓練迴圈
├── training_result.png  # 訓練結果圖 (Loss / Accuracy)
├── paper.pdf             # 論文全文
└── README.md
```

---

## 參考文獻

1. A. Van Den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," *Adv. Neural Inf. Process. Syst.*, vol. 2017-Decem, pp. 6307–6316, 2017.
2. D. P. Kingma and M. Welling, "Auto-encoding variational bayes," *2nd Int. Conf. Learn. Represent. (ICLR)*, 2014.
3. A. Dosovitskiy et al., "FlowNet: Learning optical flow with convolutional networks," *Proc. IEEE Int. Conf. Comput. Vis.*, pp. 2758–2766, 2015.
4. F. Chollet (Google), "Xception: Deep Learning with Depthwise Separable Convolutions," *CVPR*, pp. 1251–1258, 2017.

---

## 引用本研究

若本專案對你的研究有幫助，歡迎引用：

```
謝長恩、劉建宏、周仕翔、陳昀聖、楊柏遠, "基於光流分析之向量量化變分自編碼器之影像重建模型研究,"
Proceedings of 2025 National Symposium on System Science and Engineering,
National Chung Hsing University, Taichung, 16-18 May, 2025.
```
