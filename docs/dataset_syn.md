下面給你一份可以直接照著做的 **2D obstacle classification 合成資料集規格**（含具體數學公式），對齊你 toy experiment 的設定：**8 honest（分布 slowly drifting）+ 2 poisoning（每 3 round flip label）**，而且能自然呈現「時間連續性」vs「攻擊時間不一致」。

---

## 1) 全域環境與標籤函數（Obstacle map）

先定義整個 2D 空間（感測平面）：
[
\mathcal{X}=[-1,1]^2,\quad x=(x_1,x_2)\in\mathbb{R}^2
]

用 (K) 個圓形障礙物（最簡單也最好控）：

* 第 (k) 個障礙物中心：(c_k\in\mathbb{R}^2)
* 半徑：(r_k>0)

**硬標籤（binary）**：點落在任何障礙物內就算 obstacle
[
y(x)=\mathbf{1}\Big(\exists k\in{1,\dots,K}:\ |x-c_k|_2\le r_k\Big)
]

（可選）若你想讓 decision boundary 更平滑（訓練更穩），可以用 soft label / logit：
[
d(x)=\min_k\big(|x-c_k|_2-r_k\big),\quad
p(y{=}1\mid x)=\sigma!\left(-\alpha, d(x)\right)
]
其中 (\sigma(\cdot)) 是 sigmoid，(\alpha>0) 控制邊界陡峭度。最後 hard label 用 (\mathbf{1}(p>0.5)) 即可。

---

## 2) Honest client 的「移動造成分布漂移」：時間動態資料分布

你要的 LITS 味道，其實就是：**每個 client 在時間 (t) 的「觀測位置/感測區域」會慢慢移動**，所以 (p_i^t(x)) 不是固定的。

### 2.1 client 軌跡（位置隨時間變）

給每個 client (i) 一個「中心位置」(m_i^t)（代表無人機此時在平面上的感測中心）。

一個非常好用、可控的「慢漂移」模型：

**(A) 等速 + 小擾動**
[
m_i^t=m_i^{t-1}+v_i+\epsilon_i^t,\quad \epsilon_i^t\sim\mathcal{N}(0,\sigma_m^2 I)
]
其中 (|v_i|) 設很小（例如 0.005～0.02），就會是 slowly drifting。

或更「可重現」的：

**(B) 圓周/橢圓軌跡（不會飄走出界）**
[
m_i^t=
\begin{bmatrix}
a_i\cos(\omega t+\phi_i)\
b_i\sin(\omega t+\phi_i)
\end{bmatrix}
+\delta_i
]

* (\omega) 小一點（慢移動）
* (\delta_i) 是 client 的區域偏置（造成 non-IID）

### 2.2 每個 round 的取樣分布（client 的局部資料）

令 client (i) 在 round (t) 的資料分布是一個「以 (m_i^t) 為中心的高斯感測」+ 少量背景雜訊（避免太單一）：

[
x \sim (1-\rho),\mathcal{N}(m_i^t,\Sigma_i)\ +\ \rho,\mathcal{U}([-1,1]^2)
]

* (\rho\in[0,0.2])（例如 0.05）是背景 uniform 噪聲比例
* (\Sigma_i) 讓每台機器感測形狀不同（non-IID）

一個常用的 (\Sigma_i) 寫法（可旋轉的橢圓高斯）：
[
\Sigma_i = R(\theta_i)
\begin{bmatrix}
\sigma_{i,1}^2 & 0\
0 & \sigma_{i,2}^2
\end{bmatrix}
R(\theta_i)^\top,\quad
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\
\sin\theta & \cos\theta
\end{bmatrix}
]

這樣你就同時有：

* **client-to-client non-IID**（不同 (\delta_i,\Sigma_i)）
* **time drift**（(m_i^t) 隨 (t) 變）

### 2.3 每個 round 的資料集

每 round、每 client 抽 (n) 筆：
[
\mathcal{D}*i^t={(x*{i,j}^t,y_{i,j}^t)}*{j=1}^{n},\quad
y*{i,j}^t = y(x_{i,j}^t)
]

---

## 3) Poisoning client：每 3 round flip label（時間相關攻擊）

令惡意 client 集合 (\mathcal{M})（大小 2）。你描述的是「每 3 round 翻轉 labels」——可以用一個時間開關函數定義：

[
s(t)=\mathbf{1}(t \bmod 3 = 0)
]

則對惡意 client (i\in\mathcal{M})，其回報標籤是：
[
\tilde y_{i,j}^t=
\begin{cases}
1-y_{i,j}^t, & s(t)=1\
y_{i,j}^t, & s(t)=0
\end{cases}
]

而 honest client (i\notin\mathcal{M}) 就是 (\tilde y=y)。

> 這個設計會讓惡意 client 的梯度在時間軸上「週期性劇烈跳動」，非常符合你 TCR-FL 的核心假設。

---

## 4) 參數建議（你可以直接用這組）

**環境**

* (K=3)
* (c_1=(0.3,0.3), r_1=0.25)
* (c_2=(-0.4,0.2), r_2=0.20)
* (c_3=(0.0,-0.5), r_3=0.30)

**Client（10 台）**

* honest 8 台：(|v_i|\sim \text{Uniform}(0.005,0.02))（或圓周軌跡用小 (\omega)）
* malicious 2 台：資料分布生成同 honest（關鍵只在 label flipping）
* (\rho=0.05)
* (\sigma_{i,1},\sigma_{i,2}\in[0.05,0.12])
* 每 round 每 client (n=200)
* 總 rounds (T=30\sim 60)

---

## 5)（可選）把「任務切換」也加進去（更像 LITS）

如果你想要真的有 **task switching**（例如換任務區域/換障礙地圖），可以讓 obstacle map 隨時間做「分段切換」：

令任務 id：
[
q(t)=\left\lfloor \frac{t-1}{S}\right\rfloor \in {0,1,\dots}
]
每個任務 (q) 有自己的障礙物中心 ({c_k^{(q)}})。

標籤函數變成：
[
y^t(x)=\mathbf{1}\Big(\exists k:\ |x-c_k^{(q(t))}|_2\le r_k\Big)
]

這樣 honest client 仍然可能在任務切換時出現「合理的」變化（因為全世界換了任務），而惡意 client 的變化會更「不對勁」（你可以用 TCR-FL 的設計去分辨：是全域共同變 vs 單點異常跳變）。

---

## 6) 最小化生成流程（照著做就能跑 FL）

1. 固定 obstacle 參數 ({c_k,r_k})（或任務版 ({c_k^{(q)},r_k})）
2. 初始化每個 client 的 (\delta_i,\Sigma_i) 與 (m_i^0)（或軌跡參數）
3. for (t=1\to T)：

   * 更新 (m_i^t)
   * 抽樣 (x_{i,j}^t)
   * 算 (y(x))
   * 若 (i\in\mathcal{M}) 且 (t\bmod 3=0)：label flip 得 (\tilde y)
   * 輸出 (\mathcal{D}_i^t) 給該 round 的 local train

---

如果你願意，我也可以把上述公式「直接落成」一個可重現的 PyTorch dataset/loader（每 round 動態產生、seed 可控、含 honest/malicious 分組），並順便幫你把 **attack success rate** 跟 **global accuracy over rounds** 的計算定義也補齊（同樣用數學式寫清楚）。
