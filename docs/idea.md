# TCR-FL：以時間一致性正則化抵禦動態場域下之聯邦學習交替式投毒攻擊（Alternating/Temporal Poisoning）

---

## 1. 研究背景與問題定義

### 1.1 背景

聯邦學習（FL）以 **FedAvg** 為代表，透過多端本地更新再由伺服器聚合來訓練全域模型。([Proceedings of Machine Learning Research][1])
然而 FL 的安全性長期受到 **Byzantine/Poisoning**（梯度或模型更新被惡意操控）影響，因此出現了大量「每一輪（per-round）基於更新集合」的 robust aggregation，例如：

* **Krum**（以距離挑選“最可信”的更新）([NeurIPS Papers][2])
* **Coordinate-wise median / Trimmed mean**（逐座標中位數/截尾平均）([Proceedings of Machine Learning Research][3])
* **Bulyan**（組合式規則，先選再做座標層級聚合）([Proceedings of Machine Learning Research][4])

### 1.2 你的 Gap 1（研究缺口）

現有主流防禦多假設資料分布或行為在時間上相對穩定，或攻擊在每輪可被「當輪的幾何/統計離群」抓到；但在 LITS/車聯網/機群控制這類場景，系統高度動態：

* 無人機移動、任務切換、感測分布非平穩（concept drift / distributed drift）([Proceedings of Machine Learning Research][5])
* 攻擊者行為具時間相關（例如：每隔幾輪才投毒、交替式 label flip），導致「當輪看起來不離群、跨輪才異常」

> 研究問題（形式化）：
> 在 **非平穩資料 + 移動/任務切換** 的 FL 系統中，如何建立「跨輪時間結構」作為第一級安全訊號，抵禦具有 temporal correlation 的 poisoning？

---

## 2. 研究貢獻（Contributions）

1. **提出 TCR-FL 防禦框架**：以「時間一致性（temporal consistency）」對 client 更新進行連續追蹤與權重調節，而非僅依賴 per-round 的 robust aggregation。
2. **建立時間序列 × FL 安全建模**：把 client 更新視為時間序列 ( {g_i^t}_t )，將異常定義為「相對於自身歷史的偏離」，而非僅是群體中的離群點。
3. **理論化“正常漂移 vs 交替投毒”可分性**：在合理的 drift 假設下，證明 honest client 的時間殘差上界；並對交替式投毒給出殘差下界，推出權重分離（weight separation）。
4. **不需預知惡意比例**：相較 Krum/Bulyan 需設定 (f) 或對攻擊比例敏感，TCR-FL 以連續性訊號自適應衰減惡意權重。
5. **可與既有 robust aggregation 相容**：TCR-FL 產生的時間權重可作為前置濾波（re-weighting），再接 median/trimmed mean/Krum 等形成可堆疊防禦。

---

## 3. 創新點（Innovations）

1. **防禦訊號從“空間（群體）”擴展到“時間（個體軌跡）”**：把時間結構當作第一級安全訊號。
2. **針對 Alternating / Temporally sparse poisoning 的天然優勢**：這類攻擊常刻意讓當輪更新“不夠離群”，但跨輪必然造成不一致。已有工作也指出「梯度軌跡」可用於辨識交替式投毒，但多偏向離線/更重的軌跡分析；TCR-FL 主打輕量、線上化的權重正則。([Nature][6])
3. **與動態環境（concept drift）共存**：透過 EMA（指數移動平均）去追蹤“合理漂移”的基線，而不是把所有變動都當攻擊。([Proceedings of Machine Learning Research][5])

---

## 4. 理論洞見（Theoretical Insights）

### 洞見 A：honest 更新的“時間殘差”是可控的

在移動/任務連續的情境，單一 client 的局部目標 (f_i^t) 隨時間緩慢變化（drift 有界），因此
[
|g_i^t - g_i^{t-1}|
]
在機率或期望上可被上界束縛（由 drift + SGD 噪聲決定）。

### 洞見 B：交替式投毒在“自身時間軌跡”上必然造成跳變

例如每 3 輪 flip label 的攻擊會導致梯度方向/幅度週期性反轉，形成顯著的 temporal residual，即使它在當輪的群體分佈中不一定離群。

### 洞見 C：時間權重能形成“乘法式”抑制，並累積效應

[
w_i^t = \exp(-\lambda \mathcal{L}_{TC}(i))
]
是乘法型抑制；當惡意行為持續造成較大殘差，權重會快速趨近 0，且不需要硬性剔除。

---

## 5. 方法論（Methodology）

### 5.1 系統與威脅模型

* FL 設定：伺服器收集 (n) 個 client 在 round (t) 的更新（梯度或模型差分） (g_i^t)，產生全域更新。FedAvg 為基線。([Proceedings of Machine Learning Research][1])
* 攻擊者：一部分 client 可進行 data/model poisoning（含 label flipping）。FLTrust 等工作也明確把 label flipping 視為代表性 poisoning。([NDSS Symposium][7])
* 動態性：client 資料分布隨時間漂移（distributed concept drift）。([Proceedings of Machine Learning Research][5])

### 5.2 TCR-FL 核心定義（你的主公式 + 完整化）

令 EMA（對每個 client 維護一個歷史基線）：
[
\bar g_i^{t-1}=\beta \bar g_i^{t-2}+(1-\beta)g_i^{t-1},\quad \beta\in(0,1)
]

時間一致性懲罰：
[
\mathcal{L}_{TC}(i,t)=|g_i^t-\bar g_i^{t-1}|_2
]

時間權重（未正規化）：
[
w_i^t=\exp(-\lambda \mathcal{L}_{TC}(i,t))
]

聚合權重正規化：
[
\alpha_i^t=\frac{w_i^t}{\sum_{j=1}^n w_j^t}
]

全域聚合（以梯度形式寫）：
[
g^t=\sum_{i=1}^n \alpha_i^t g_i^t,\quad \theta^{t+1}=\theta^t-\eta g^t
]

> **可擴展版本（建議寫在方法章）**

* 尺度不敏感：用 cosine distance 取代 L2、或對 (\mathcal{L}_{TC}) 做 layer-wise normalization。
* 漂移偵測：若多數 client 同時殘差上升，可能是任務切換而非攻擊 → 觸發“全域 reset/降懲罰”機制（避免把概念漂移誤判為惡意）。

### 5.3 與 robust aggregation 的結合（強化說服力）

TCR-FL 可作為 **pre-weighting**，再套用 trimmed mean / median / Krum：

* 先計算 (w_i^t)，對更新縮放 (g_i^t \leftarrow w_i^t g_i^t)
* 再對 ({g_i^t}) 做 coordinate-wise trimmed mean ([Proceedings of Machine Learning Research][3]) 或 Krum ([NeurIPS Papers][2])
  這能回應審稿人常見質疑：「單一機制是否足夠？」——你可以主張 TCR-FL 提供正交訊號（temporal），可與空間 robust 結合。

---

## 6. 數學理論推演與證明（Proof Plan）

> 下面提供可直接放進論文的“假設—引理—定理—證明要點”骨架（你之後可再補細節與常數）。

### 6.1 假設（Honest drift + SGD noise）

對 honest client (i\in\mathcal{H})，設
[
g_i^t=\nabla f_i^t(\theta^t)+\xi_i^t
]
其中 (\xi_i^t) 為零均值噪聲、(\mathbb{E}|\xi_i^t|^2\le \sigma^2)。
非平穩性以 drift 上界表達：
[
|\nabla f_i^t(\theta)-\nabla f_i^{t-1}(\theta)|\le \delta_t,\ \forall \theta
]
（(\delta_t) 小表示 slowly drifting；概念漂移文獻常以此類方式建模動態目標。([Proceedings of Machine Learning Research][5])）

### 6.2 引理 1：Honest client 的時間殘差上界

**Lemma 1（EMA tracking bound）**
若 (f_i^t) 的梯度漂移有界且噪聲有界，則對任意 honest client：
[
\mathbb{E},\mathcal{L}_{TC}(i,t)\le C_1\delta_t + C_2\sigma
]
其中 (C_1,C_2) 與 (\beta)（EMA 記憶長度）相關。

**證明要點**
把
(|g_i^t-\bar g_i^{t-1}|)
拆成
(|(\nabla f_i^t-\nabla f_i^{t-1}) + (\nabla f_i^{t-1}-\bar g_i^{t-1}) + (\xi_i^t-\bar\xi)|)，
再用三角不等式 + EMA 對平穩訊號的收斂性（幾何級數）得到上界。

### 6.3 引理 2：交替式投毒的時間殘差下界（以週期 label flip 為例）

對惡意 client (i\in\mathcal{M})，令其在某些輪次輸出 (g_i^t) 與 honest 梯度方向顯著不一致（例如每 3 輪 flip label 導致期望梯度近似反向）。在可分資料與線性分類等簡化條件下，可導出存在 (\Delta>0) 使得在投毒輪：
[
\mathcal{L}_{TC}(i,t)\ge \Delta
]
（直觀：EMA 是以先前 honest-like 方向為主的平均，一旦攻擊反轉方向，與 EMA 的距離會跳大。）

### 6.4 定理 1：權重分離（malicious weight suppression）

若對所有 honest client 有 (\mathcal{L}*{TC}\le \varepsilon)，且惡意 client 在無限多個輪次滿足 (\mathcal{L}*{TC}\ge \Delta>\varepsilon)，則
[
\frac{\alpha_m^t}{\alpha_h^t}\le \exp(-\lambda(\Delta-\varepsilon))
]
因此取足夠大的 (\lambda) 可讓惡意權重在投毒輪次指數級被抑制。

### 6.5 定理 2：收斂/追蹤（Convergence / Tracking）

在（強）凸、(L)-smooth 的簡化設定下，使用加權聚合得到的 (g^t) 若滿足

* 惡意貢獻被上界抑制（由定理 1）
* honest 聚合近似無偏（或偏差可控）

則可得到類似於 trimmed mean/median 類方法在強凸下的收斂型式（誤差由 drift 與殘餘惡意影響決定）。可在 Related Work 補充：median/trimmed mean 在 Byzantine setting 的收斂分析是既有理論基礎，你的部分是把“誤差項”改寫成“時間權重後的有效惡意比例”。([Proceedings of Machine Learning Research][3])

> 建議寫法：你不必承諾最強的最優收斂率；proposal 階段把「可證明：在 drift 有界 + 攻擊造成殘差下界時，惡意貢獻在加權後呈指數抑制」寫清楚，就很有說服力。

---

## 7. 預計使用資料集（Datasets）

### 7.1 Toy 合成資料（對齊你描述的設定）

你已經有完整可重現的 2D obstacle classification 合成資料規格（含 slowly drifting + 每 3 輪 label flip），可直接作為主要 toy benchmark

---

## 8. 與現有研究之區別（Positioning）

### 8.1 vs Robust aggregation（Krum/TrimmedMean/Bulyan）

* 既有方法主要利用“當輪更新集合”的幾何/統計結構：Krum ([NeurIPS Papers][2])、trimmed mean/median ([Proceedings of Machine Learning Research][3])、Bulyan ([Proceedings of Machine Learning Research][4])
* **TCR-FL** 改用“跨輪的個體軌跡一致性”作為訊號 → 能抓到當輪不離群、但跨輪跳變的攻擊。

### 8.2 vs 信任/參考資料集式防禦（FLTrust）

FLTrust 用 server 的小型乾淨資料（root dataset）來計算信任並縮放更新。([arXiv][12])
TCR-FL 不需要 root dataset，而是利用每個 client 自身的時間連續性；兩者可互補（FLTrust 解決“方向可信度”，TCR-FL 解決“時間穩定度”）。

### 8.3 vs Sybil/相似度防禦（FoolsGold）

FoolsGold 透過 client 更新相似度來抑制 sybil（彼此太像的攻擊者）。([arXiv][13])
TCR-FL 不依賴“多個攻擊者彼此相似”，而依賴“單一攻擊者跨時間不一致”，對交替式投毒更直接。

### 8.4 vs 既有“時間/軌跡”偵測研究

已有研究指出可用梯度軌跡做交替投毒辨識，但常偏向較重的軌跡分析或非線上設定。([Nature][6])
你的差異化主張：**把時間結構嵌入聚合規則本身（線上、輕量、可疊加）**。

---

## 9. Experiment 設計（含指標與消融）

### 9.1 實驗 A：Toy（你原始設定）

**任務**：2D obstacle classification
**Clients**：8 honest（資料 slowly drifting）+ 2 malicious（每 3 輪 flip label）

**比較方法（Baselines）**

* FedAvg ([Proceedings of Machine Learning Research][1])
* Trimmed Mean / Coordinate-wise median ([Proceedings of Machine Learning Research][3])
* Krum ([NeurIPS Papers][2])
* （可選）Bulyan ([Proceedings of Machine Learning Research][4])
* **TCR-FL**
* **TCR-FL + TrimmedMean**（展示可疊加性）

**評估指標**

* Global accuracy over rounds
* Attack Success Rate（ASR）：

  * 對 label flipping：可定義為「投毒輪次中，全域模型在乾淨驗證集的錯誤率增加量」或「目標類別被錯分比例」
* 惡意 client 權重曲線：(\alpha_i^t) 隨時間變化（你主張的核心可視化）
* 檢測延遲（time-to-suppress）：惡意權重降到某閾值所需輪數

### 9.2 實驗 B：動態性強化（更像 LITS）

加入兩個因素，驗證“你不是把 drift 當成攻擊”：

1. **任務切換（task switching）**：每 (S) 輪更換 obstacle map 或 label rule（合成資料可做到）
2. **漂移幅度掃描**：調大/調小 drift，觀察 TCR-FL 誤殺率（false suppression）

### 9.4 消融研究（Ablation）

* (\lambda)（懲罰強度）
* (\beta)（EMA 記憶長度）
* 距離函數：L2 vs cosine
* 有無 drift-aware reset（偵測到“全體一起變”就暫時降低懲罰）
* 與 robust aggregation 的組合：TCR-FL 前置 vs 後置 vs 不結合

---

## 10. 可能風險與可寫進 proposal 的對策（加分項）

* **Secure Aggregation 限制**：若系統採用安全聚合，伺服器可能看不到單一 client 更新序列 → 需要設計“可驗證的時間一致性分數”或在可信執行環境/跨輪匿名但可鏈結的機制下運作（proposal 可先註明為未來工作/系統假設）。
* **概念漂移誤判**：用“全體殘差同步上升”做 change-point detection，觸發 EMA reset，避免把任務切換當攻擊（這會讓方法更像是為 LITS 設計，而不是只適用 toy）。

---

如果你願意，我也可以把這份 outline 再進一步整理成「論文章節架構（Introduction/Related Work/Method/Theory/Experiments）」版本，並幫你把 **定理與證明** 寫成更接近可投稿的形式（把假設、常數界、以及在 strongly convex / non-convex 下各寫一版）。

[1]: https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf?utm_source=chatgpt.com "Communication-Efficient Learning of Deep Networks from ..."
[2]: https://papers.neurips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf?utm_source=chatgpt.com "Machine Learning with Adversaries: Byzantine Tolerant ..."
[3]: https://proceedings.mlr.press/v80/yin18a/yin18a.pdf?utm_source=chatgpt.com "Byzantine-Robust Distributed Learning: Towards Optimal ..."
[4]: https://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf?utm_source=chatgpt.com "The Hidden Vulnerability of Distributed Learning in Byzantium"
[5]: https://proceedings.mlr.press/v206/jothimurugesan23a/jothimurugesan23a.pdf?utm_source=chatgpt.com "Federated Learning under Distributed Concept Drift"
[6]: https://www.nature.com/articles/s41598-024-70375-w?utm_source=chatgpt.com "Identifying alternately poisoning attacks in federated ..."
[7]: https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf?utm_source=chatgpt.com "Byzantine-robust Federated Learning via Trust Bootstrapping"
[8]: https://arxiv.org/abs/1810.10438?utm_source=chatgpt.com "UAVid: A Semantic Segmentation Dataset for UAV Imagery"
[9]: https://www.nuscenes.org/nuscenes?utm_source=chatgpt.com "Scene planning"
[10]: https://www.cvlibs.net/datasets/kitti/?utm_source=chatgpt.com "The KITTI Vision Benchmark Suite"
[11]: https://github.com/VisDrone/VisDrone-Dataset?utm_source=chatgpt.com "VisDrone-Dataset"
[12]: https://arxiv.org/abs/2012.13995?utm_source=chatgpt.com "FLTrust: Byzantine-robust Federated Learning via Trust ..."
[13]: https://arxiv.org/abs/1808.04866?utm_source=chatgpt.com "Mitigating Sybils in Federated Learning Poisoning"
