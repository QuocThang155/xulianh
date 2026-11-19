# Phân công & Hướng dẫn làm chi tiết — Bài tập 2 (Nhóm N)

**Môn:** Thị giác máy tính (121036) — Trường ĐH GTVT TP.HCM  
**Tài liệu gốc:** `Computer_Vision 45.pdf`  
**File notebook chung:** `bt1_nhomN.ipynb` (đã có sẵn khung + phần đã hoàn thành của Hiển)

---

## ⛔ QUY ĐỊNH BẮT BUỘC (đọc kỹ để tránh 0 điểm cả nhóm)

1. **Phần Cài đặt thuật toán (A.1, B.1):**
   - **KHÔNG dùng AI** (không ChatGPT, không Gemini, **tắt luôn Copilot gợi ý code trong VS Code**), kể cả nhờ AI sửa lỗi.
   - **KHÔNG gọi hàm dựng sẵn** bên trong hàm phải nộp: cấm `cv2.cornerHarris`, `cv2.goodFeaturesToTrack`, `skimage.feature.corner_*`, `skimage.segmentation.*`, `cv2.watershed`.
   - **Được phép dùng:** `numpy`, `matplotlib`, `math`, `time`, `scipy.ndimage.gaussian_filter`, `heapq`, `collections.deque`, `cv2.imread/cvtColor/resize`, `scipy.ndimage.label` (chỉ ở phần phân tích).
2. **Phần Thí nghiệm (TN1, TN2, TN3) & Kiểm chứng:**
   - Được dùng AI để sinh ảnh, dựng khung đo đạc, vẽ biểu đồ — nhưng **ý tưởng và kết luận phải là của bạn**.
   - **Bắt buộc khai báo prompt:** ngay phía trên mỗi cell code có dùng AI phải có 1 cell markdown ghi `**Prompt đã dùng (nguyên văn):** "..."`. Dùng AI mà không khai báo ⇒ **0 điểm cả bài**.
   - Ở phần thí nghiệm, được gọi hàm thư viện (OpenCV, scikit-image) để đối chiếu.
3. **Nộp bài:** 1 notebook duy nhất đã chạy **Restart & Run All** không lỗi, lưu kèm output + thư mục `anh/`.

---

## Tổng quan phân công 5 thành viên (~20% mỗi người)

| MSSV | Thành viên | Trạng thái | Phụ trách |
|---|---|---|---|
| 051205011311 | **Nguyễn Minh Hiển** | ✅ Đã xong | **A (1/2):** Harris cổ điển (5 hàm), A2 (`top_n`, `anms`), `grid_occupancy`, Kiểm chứng A1/A4/A5, TN2 |
| 079205042179 | **Nguyễn Đình Huy** | ⏳ Đang làm | **A (2/2):** A1 (`scale_pyramid`, `log_normalized`, `harris_laplace`), `repeatability`, Kiểm chứng A2/A3, TN1 |
| 079205010059 | **Nguyễn Quốc Hưng** | ⏳ Đang làm | **B (1/3):** SRG lõi (`srg` với `lazy`/`frozen` và phá hòa `tie`), Kiểm chứng B1/B2/B3 |
| 066205015306 | **Nguyễn Kha Rin** | ⏳ Đang làm | **B (2/3):** Region Growing cổ điển (`region_growing_classic`, `stack`/`queue`), Kiểm chứng B4, TN3 (ngưỡng T) |
| 060205000948 | **Lương Quốc Thắng** | ⏳ Đang làm | **B (3/3):** `srg_step_frames`, `seed_sensitivity`, Kiểm chứng B5, TN3 (phần SRG + giải thích), **Ghép notebook & Run All** |

**Thứ tự phụ thuộc:**
- **Huy** lấy 5 hàm Harris cổ điển trong `bt1_nhomN.ipynb` của Hiển để làm tiếp ngay.
- **Hưng** và **Rin** làm song song ngay từ bây giờ (**Hưng** nên xong `srg` sớm để **Rin** so sánh ở Kiểm chứng B4 và **Thắng** chạy TN3).
- Trong lúc chờ `srg`, **Thắng** viết sẵn `srg_step_frames`, `seed_sensitivity` và chuẩn bị ảnh thử trong `anh/`.

---

## Bước 0 — Cài môi trường & Mở notebook (cho bạn nào chưa cài)

1. Mở **PowerShell** tại thư mục bài tập, chạy lần lượt từng dòng:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install numpy matplotlib scipy opencv-python scikit-image jupyter ipykernel
   ```
   *(Nếu dòng 2 báo lỗi "running scripts is disabled", chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, nhập `Y`, rồi chạy lại dòng 2).*
2. Mở thư mục bài tập bằng **VS Code** → bấm **Yes, I trust the authors** (tắt Restricted Mode).
3. Cài 2 extension của Microsoft: **Python** và **Jupyter**.
4. **Tắt Copilot gợi ý code** trong phần cài đặt: bấm `Ctrl+Shift+P` → gõ `Copilot: Disable Completions`.
5. Tạo notebook riêng theo tên mình (vd: `A2_Huy.ipynb`, `B1_Hung.ipynb`, `B2_Rin.ipynb`, `B3_Thang.ipynb`), bấm **Select Kernel → Python Environments → `.venv`**.

**Quy ước dữ liệu chung (cả nhóm tuân thủ để ghép không lỗi):**
- Ảnh xám đầu vào: mảng 2D `np.float64`, giá trị trong `[0, 1]`. Ảnh dùng chung lưu trong thư mục `anh/`.
- Phần B: `seeds` là `list` các `(row, col)`; hạt giống thứ `i` mang nhãn `i` (`0, 1, 2, ...`); nhãn đầu ra là mảng `np.int32` cùng kích thước ảnh (`-1` = chưa thuộc vùng nào).

---

## 1. Hướng dẫn chi tiết cho Nguyễn Đình Huy — Phần A (2/2)

**Mục tiêu:** Cài biến thể A1 (Harris–Laplace) + `repeatability`, làm Kiểm chứng A2, A3 và TN1.  
**Tài liệu:** PDF mục 3.2, 3.3, 3.6, 3.7 (trang 3–5) và mục 5 (trang 8).

### Bước 1: Lấy nền từ Hiển
Mở `bt1_nhomN.ipynb` của Hiển, copy 5 hàm ở ô **A.1.1** (`gaussian_blur`, `structure_tensor`, `harris_response`, `nms_2d`, `harris_classic`) sang notebook của bạn để dùng lại.

### Bước 2: Tự cài 4 hàm (KHÔNG dùng AI)

1. **`scale_pyramid(sigma0=1.2, xi=1.2, n_levels=10)`**
   - Công thức: $\sigma_n = \xi^n \cdot \sigma_0$ với $n = 0, 1, \dots, n_{\text{levels}}-1$.
   - Trả về mảng 1D numpy độ dài `n_levels`.

2. **`log_normalized(img, sigma)`**
   - Công thức (3): $|\text{LoG}(x; \sigma)| = \sigma^2 \cdot |L_{xx}(x; \sigma) + L_{yy}(x; \sigma)|$.
   - Cách tính: làm mờ `img` với `sigma` bằng `gaussian_blur` → lấy đạo hàm bậc 1 `(Ly, Lx)` bằng `np.gradient` → lấy đạo hàm bậc 2 `Lyy` (đạo hàm theo trục 0 của `Ly`) và `Lxx` (đạo hàm theo trục 1 của `Lx`) bằng `np.gradient` → trả về `(sigma ** 2) * np.abs(Lxx + Lyy)`.

3. **`harris_laplace(img, sigmas, k=0.04, thresh_rel=0.01, log_thresh=0.0)`**
   - **Giai đoạn 1 (tính trước bản đồ LoG cho mọi tầng):** với mỗi `sigma` trong `sigmas`, tính `log_normalized(img, sigma)` lưu thành mảng 3D `logs` có kích thước `(len(sigmas), H, W)`.
   - **Giai đoạn 2 (duyệt từng tầng `n` từ `1` đến `len(sigmas) - 2`):** *(bỏ tầng đầu `0` và tầng cuối vì cần so sánh với cả `n-1` và `n+1`)*.
     - Đặt `sigma_i = sigmas[n]`, `sigma_d = 0.7 * sigmas[n]`.
     - Tính `R = harris_response(img, sigma_i, sigma_d, k)`, lấy cực đại 2D bằng `nms_2d(R, window=3)` và lọc `R > thresh_rel * R.max()`.
     - **Lọc theo trục tỉ lệ (cực kỳ quan trọng — KHÔNG dùng `argmax R`!):** với mỗi điểm ứng viên `(r, c)`, chỉ giữ lại khi thỏa CẢ 3 điều kiện:
       - `logs[n, r, c] > logs[n - 1, r, c]`
       - `logs[n, r, c] > logs[n + 1, r, c]`
       - `logs[n, r, c] > log_thresh`
   - Trả về mảng `(N, 4)` gồm các cột `(row, col, sigma, response)`.

4. **`repeatability(pts_a, pts_b, transform, img_shape, eps=3.0)`**
   - `transform` là hàm nhận `(row, col)` ở ảnh A trả về `(row_b, col_b)` tương ứng ở ảnh B.
   - Bước 1: Chiếu từng điểm của `pts_a` sang ảnh B qua `transform`, **chỉ giữ các điểm mà tọa độ chiếu còn nằm trong biên `img_shape`** (`0 <= row_b < H` và `0 <= col_b < W`). Gọi số điểm hợp lệ này là `M`. Nếu `M == 0` trả về `0.0`.
   - Bước 2: Với mỗi điểm chiếu hợp lệ, tính khoảng cách Euclid nhỏ nhất tới tập `pts_b[:, :2]`. Nếu khoảng cách nhỏ nhất `<= eps` thì tính là tìm lại được (`match`).
   - Trả về `số_điểm_match / M` (số thực trong `[0, 1]`).

### Bước 3: Chạy 2 kiểm chứng (A.2 mục 2 & 3)
- **KC A2 (Quay 30°):** quay ảnh bàn cờ (hoặc ảnh thật) một góc 30° quanh tâm (dùng `cv2.getRotationMatrix2D` và `cv2.warpAffine`), viết hàm `transform` tương ứng với ma trận quay, chạy `harris_classic` trên ảnh gốc & ảnh quay rồi tính `repeatability`.
- **KC A3 (Phóng to 3×):** phóng to ảnh lên 3× bằng `cv2.resize`, hàm `transform(r, c) -> (3*r, 3*c)`. Tính và báo cáo `repeatability` của cả **`harris_classic`** và **`harris_laplace`** (sẽ thấy `harris_laplace` cao hơn rõ rệt).

### Bước 4: Làm TN1 — Harris–Laplace vs Harris *(được dùng AI, nhớ ghi prompt nguyên văn)*
1. **Bằng chứng khác biệt:** chọn/tạo ảnh chứa cấu trúc ở nhiều tỉ lệ (hoặc cặp ảnh gốc vs phóng to 2×–3×), vẽ hình kèm vòng tròn bán kính tỉ lệ với `sigma` và báo cáo `repeatability` của 2 phương pháp.
2. **Quét tham số không tầm thường (chọn 1–2 trục):**
   - Quét tỉ số `xi` (vd `1.1 → 1.6`) hoặc số tầng `n_levels` → vẽ đường cong `repeatability` để tìm điểm bão hòa.
   - Hoặc quét `log_thresh` → vẽ % ứng viên bị loại và `repeatability` theo `log_thresh`.
3. **Giải thích 3 câu:** đại lượng nào tính khác đi (`|LoG|` chuẩn hóa theo tỉ lệ $\sigma^2|L_{xx}+L_{yy}|$ và chuẩn hóa $\sigma_D^2$ trong $M$), ở bước nào (bước lọc theo trục tỉ lệ), vì sao bước **loại điểm không có cực trị LoG** mới tạo ra tính bất biến tỉ lệ.

---

## 2. Hướng dẫn chi tiết cho Nguyễn Quốc Hưng — Phần B (1/3)

**Mục tiêu:** Cài thuật toán lõi `srg` (nặng nhất Phần B nên không phải làm thí nghiệm) và chạy 3 kiểm chứng B1, B2, B3.  
**Tài liệu:** PDF mục 4.2, 4.3, 4.4, 4.5 (trang 6–8).

### Bước 1: Tự cài `srg(img, seeds, connectivity=4, stale="lazy", tie="insertion")` (KHÔNG dùng AI)

**Cấu trúc dữ liệu cần dùng:**
- `labels = np.full((H, W), -1, dtype=np.int32)` — bản đồ nhãn (`-1` là chưa gán).
- `boundary_mask = np.zeros((H, W), dtype=bool)` — đánh dấu pixel biên.
- `region_sum` và `region_cnt` (mảng độ dài `len(seeds)`): lưu **tổng cường độ** và **số pixel** của mỗi vùng để tính trung bình `mean_i = region_sum[i] / region_cnt[i]` trong $O(1)$.
- `seed_val`: lưu giá trị `img[r, c]` ban đầu của từng hạt giống (dùng cho `stale="frozen"`).
- Hàng đợi ưu tiên `heap = []` dùng `heapq.heappush` / `heapq.heappop`.
- Bộ đếm `counter = 0` (cho `tie="insertion"`) và `rng = np.random.default_rng(0)` (cho `tie="random"`).

**Hàm tính khóa phá hòa `tie_key(r, c)`:**
- Nếu `tie == "insertion"`: tăng `counter += 1`, trả về `counter`.
- Nếu `tie == "raster"`: trả về `r * W + c`.
- Nếu `tie == "random"`: trả về `rng.random()`.

**Hàm tính `delta(val, label_id)`:**
- Nếu `stale == "frozen"`: `abs(val - seed_val[label_id])`.
- Nếu `stale == "lazy"`: `abs(val - region_sum[label_id] / region_cnt[label_id])`.

**Các bước chạy thuật toán:**
1. **Khởi tạo:** với mỗi hạt giống `i, (r, c)` trong `enumerate(seeds)`:
   - Gán `labels[r, c] = i`, `region_sum[i] = img[r, c]`, `region_cnt[i] = 1`.
2. **Đưa lân cận ban đầu của các hạt giống vào `heap`:**
   - Lân cận 4: `[(-1,0), (1,0), (0,-1), (0,1)]`; lân cận 8: thêm `[(-1,-1), (-1,1), (1,-1), (1,1)]`.
   - Với mỗi pixel `(nr, nc)` chưa gán nhãn kề với hạt giống `i`, đẩy tuple `(delta, tie_key, nr, nc, i)` vào `heap`.
3. **Vòng lặp `while heap:`:**
   - Lấy phần tử nhỏ nhất: `d, _, r, c, lbl = heapq.heappop(heap)`.
   - Nếu `labels[r, c] != -1`: bỏ qua (`continue`) — pixel này đã được gán nhãn trước đó rồi.
   - **Xử lý `stale == "lazy"`:** tính lại `d_now = delta(img[r, c], lbl)`. Nếu `abs(d_now - d) > 1e-12`: đẩy lại `(d_now, tie_key(r, c), r, c, lbl)` vào `heap` rồi `continue`! *(Đây là chi tiết được chấm điểm ở mục 4.3.1).*
   - **Kiểm tra pixel biên:** xem các lân cận đã gán nhãn quanh `(r, c)`; nếu có từ **2 nhãn phân biệt trở lên** ⇒ gán `boundary_mask[r, c] = True` (và với `stale == "lazy"`, chọn `lbl` trong các vùng kề làm `delta` nhỏ nhất).
   - **Kết nạp pixel `(r, c)` vào vùng `lbl`:**
     - `labels[r, c] = lbl`
     - `region_sum[lbl] += img[r, c]`; `region_cnt[lbl] += 1`
   - **Đẩy lân cận mới:** với mọi lân cận `(nr, nc)` của `(r, c)` còn trong biên ảnh và `labels[nr, nc] == -1`, tính `d_nb = delta(img[nr, nc], lbl)` rồi đẩy `(d_nb, tie_key(nr, nc), nr, nc, lbl)` vào `heap`.
4. Trả về `(labels, boundary_mask)`.

*(Mẹo kiểm tra tốc độ: trên ảnh 256×256, `srg` chạy khoảng 0.5–2 giây là đạt).*

### Bước 2: Chạy 3 kiểm chứng (B.2 mục 1, 2, 3)
- **KC B1 (Phân hoạch đầy đủ):** chạy `labels, bmask = srg(img, seeds)` và khẳng định `assert (labels >= 0).all()`.
- **KC B2 (Bảo toàn số vùng):** khẳng định `len(np.unique(labels)) == len(seeds)`.
- **KC B3 (Bất biến thứ tự hạt giống):** chạy `srg` với `seeds` (được `labels1`) và với `seeds[::-1]` (được `labels2`). Đổi lại nhãn `labels2` theo ánh xạ đảo ngược (`len(seeds) - 1 - labels2`) rồi so sánh tỉ lệ pixel khớp với `labels1` (dùng ảnh có nhiễu nhẹ để tránh đồng hạng tuyệt đối, và kiểm tra `tie="raster"` cho kết quả trùng khít 100%).

---

## 3. Hướng dẫn chi tiết cho Nguyễn Kha Rin — Phần B (2/3)

**Mục tiêu:** Cài Region Growing cổ điển, làm Kiểm chứng B4 và TN3 (phần quét ngưỡng `T`).  
**Tài liệu:** PDF mục 4.1, 4.4, 4.5 (trang 6–8) và mục 5.1 (trang 8).

### Bước 1: Tự cài `region_growing_classic(img, seeds, T, order="stack")` (KHÔNG dùng AI)
- Import thêm `from collections import deque`.
- Khởi tạo `labels = np.full(img.shape[:2], -1, dtype=np.int32)`.
- Duyệt lần lượt từng hạt giống `i, (sr, sc)` trong `enumerate(seeds)`:
  - Nếu `labels[sr, sc] != -1`: bỏ qua.
  - Khởi tạo: `labels[sr, sc] = i`, `mu = float(img[sr, sc])`, `n = 1`.
  - Khởi tạo danh sách chờ `frontier = deque([(sr, sc)])`.
  - `while frontier:`
    - Nếu `order == "stack"`: lấy phần tử cuối `r, c = frontier.pop()` (LIFO).
    - Nếu `order == "queue"`: lấy phần tử đầu `r, c = frontier.popleft()` (FIFO).
    - Duyệt 4 lân cận `(nr, nc)` nằm trong ảnh:
      - Nếu `labels[nr, nc] == -1` và `abs(img[nr, nc] - mu) <= T`:
        - `labels[nr, nc] = i`
        - Cập nhật trung bình vùng: `mu = (mu * n + img[nr, nc]) / (n + 1)`; `n += 1`
        - `frontier.append((nr, nc))`
- Trả về `labels`.

### Bước 2: Làm Kiểm chứng B4 (Đối chiếu stack vs queue vs SRG)
- Trên cùng 1 ảnh (vd ảnh có 2–3 vùng và vùng chuyển tiếp mượt hoặc nhiễu nhẹ) và cùng `seeds`:
  - Chạy `lab_stack = region_growing_classic(img, seeds, T, order="stack")`
  - Chạy `lab_queue = region_growing_classic(img, seeds, T, order="queue")`
  - Chạy `lab_srg, _ = srg(img, seeds)` *(lấy hàm `srg` từ Hưng)*
- In bảng báo cáo **số pixel khác nhãn** của từng cặp: `(stack vs queue)`, `(stack vs srg)`, `(queue vs srg)`, và vẽ 3 ảnh nhãn cạnh nhau.

### Bước 3: Làm TN3 — Quét tham số ngưỡng `T` *(được dùng AI, nhớ ghi prompt nguyên văn)*
- Tạo (hoặc chọn) ảnh có 2 vùng tương phản rõ (vd vật thể mức sáng `0.3` trên nền `0.6`, độ tương phản $\Delta I = 0.3$, thêm nhiễu nhẹ) và thử thêm 1 ảnh độ tương phản thấp hơn (vd $\Delta I = 0.15$).
- Quét ngưỡng `T` từ `0.01` đến `0.50` (vd 25–30 mốc). Tại mỗi `T`, đo **diện tích vùng lớn nhất** (số pixel của vùng hạt giống) và **số pixel chưa gán nhãn**.
- Vẽ đường cong diện tích theo `T` → chỉ ra **điểm chuyển pha (ngưỡng tới hạn $T_{\text{crit}}$)** nơi đường cong **nhảy vọt đột ngột** (vùng bị **rò rỉ / leaking** tràn sang nền!), và cho thấy $T_{\text{crit}}$ tỉ lệ trực tiếp với độ tương phản $\Delta I$ của ảnh.
- Gửi phần này cho **Thắng** ghép vào TN3.

---

## 4. Hướng dẫn chi tiết cho Lương Quốc Thắng — Phần B (3/3) + Ghép bài

**Mục tiêu:** Cài 2 hàm công cụ phân tích, làm Kiểm chứng B5, hoàn thiện TN3 (phần SRG + giải thích), và ghép toàn bộ notebook cuối cùng.  
**Tài liệu:** PDF mục 4.3, 4.4, 4.5 (trang 6–8) và mục 5 (trang 8–9).

### Bước 1: Tự cài 2 hàm phân tích (KHÔNG dùng AI)

1. **`srg_step_frames(img, seeds, n_frames=8, **kwargs)`**
   - Ý tưởng: chụp lại ảnh `labels.copy()` tại `n_frames` mốc tiến trình đều nhau (ví dụ khi số pixel đã gán đạt $\frac{1}{8}, \frac{2}{8}, \dots, \frac{8}{8}$ tổng số pixel $H \times W$).
   - Cách dễ nhất: thêm tùy chọn ghi lại snapshot bên trong vòng lặp của `srg` (hoặc copy vòng lặp `srg` vào `srg_step_frames`, đếm biến `assigned += 1` mỗi khi gán nhãn 1 pixel; khi `assigned` chạm mốc `int(k * H * W / n_frames)` thì `frames.append(labels.copy())`).
   - Trả về `list` gồm `n_frames` mảng nhãn.

2. **`seed_sensitivity(img, seeds, segment_fn, n_trials=20, jitter=3)`**
   - Mục tiêu: đo xem nếu dời mỗi hạt giống ngẫu nhiên `±jitter` pixel thì mỗi pixel trên ảnh **đổi nhãn bao nhiêu lần** (hoặc tỉ lệ đổi nhãn) so với lần chạy gốc.
   - Cách làm:
     - Chạy lần gốc: `base = segment_fn(img, seeds)` (nếu `segment_fn` trả về tuple `(labels, bmask)` như `srg` thì lấy `base = base[0]`).
     - Tạo mảng đếm `changes = np.zeros(img.shape[:2], dtype=np.int32)` và `rng = np.random.default_rng(0)`.
     - Lặp `n_trials` lần: với mỗi `(r, c)` trong `seeds`, cộng độ lệch nguyên ngẫu nhiên trong `[-jitter, jitter]` (dùng `np.clip` để không tràn ra ngoài biên ảnh `0..H-1`, `0..W-1`) → chạy `res = segment_fn(img, jittered_seeds)` (lấy `res[0]` nếu là tuple) → cộng dồn `changes += (res != base)`.
     - Trả về `changes` (hoặc bản đồ tần suất).

### Bước 2: Chạy Kiểm chứng B5 (Ảnh phẳng)
- Tạo ảnh hằng số (vd `np.full((128, 128), 0.5)`) với 2 hạt giống (vd `(64, 32)` và `(64, 96)`).
- Chạy `srg` với cả 3 quy tắc `tie="insertion"`, `tie="raster"`, `tie="random"`.
- Vẽ 3 hình cạnh nhau và **mô tả hình dạng ranh giới thu được** (vì mọi $\delta = 0$, kết quả phản ánh trực tiếp quy tắc phá hòa: `insertion` cho đường trung trực/Voronoi theo thứ tự BFS, `raster` lệch theo thứ tự hàng-cột từ trên xuống dưới, `random` cho biên răng cưa ngẫu nhiên!).

### Bước 3: Hoàn thiện TN3 — SRG vs Region Growing *(được dùng AI, nhớ ghi prompt nguyên văn)*
1. **Bằng chứng khác biệt SRG vs Region Growing:**
   - Chọn ảnh có **chuyển màu mượt (gradient)** hoặc 2 vùng tương phản thấp + nhiễu (để Region Growing hoặc bị dừng sớm để lại nhiều `-1`, hoặc rò rỉ tràn vùng; còn SRG phân hoạch sạch đẹp cả ảnh).
   - Kèm bảng số đo: số pixel chưa gán (`-1`), số pixel khác nhãn, độ chính xác/số vùng, thời gian chạy, và minh họa tiến trình bằng `srg_step_frames`.
2. **So sánh bắt buộc mục 4.3 (chấm điểm trực tiếp!):**
   - **`stale="lazy"` vs `"frozen"`** trên **ảnh có gradient mượt** → báo cáo số pixel khác nhãn và giải thích vì sao `frozen` không thích nghi được khi màu vùng chuyển dần.
   - **Phá hòa (`tie`):** chỉ ra bằng số liệu (số pixel khác nhãn > 0) rằng đổi giữa `insertion`, `raster`, `random` làm đổi kết quả trên ảnh phẳng hoặc ảnh 8-bit lượng tử hóa.
3. **Quét tham số SRG:**
   - Ghép phần quét ngưỡng `T` của **Rin** vào đây.
   - Dùng `seed_sensitivity` khảo sát độ nhạy theo vị trí/số lượng hạt giống (vẽ heatmap các pixel hay đổi nhãn — sẽ tập trung dọc biên mờ!).
   - So sánh `connectivity=4` vs `connectivity=8` (số pixel khác nhãn, hình dạng biên).
4. **Giải thích 3 câu (mục 5.1):** đại lượng nào tính khác đi ($\delta(x) = |I(x) - \bar{I}_{A_i}|$ được sắp toàn cục bằng hàng đợi ưu tiên thay vì chỉ kiểm tra `<= T` theo stack/queue), ở bước nào (bước chọn pixel tiếp theo để kết nạp), và vì sao dẫn tới việc bỏ được ngưỡng `T`, pixel chắc chắn kết nạp trước, pixel biên mơ hồ xử lý sau cùng.

### Bước 4: Checklist ghép bài & nộp bài cuối cùng
1. Mở `bt1_nhomN.ipynb` (đã có sẵn Phần A1/2 của Hiển), ghép lần lượt:
   - Phần của **Huy** vào `A.1.2`, `A.1.4`, `A.2 (mục 2, 3)`, `TN1`.
   - Phần của **Rin** vào `B.1.1`, `B.2 (mục 4)`, `TN3 (ngưỡng T)`.
   - Phần của **Hưng** vào `B.1.2`, `B.2 (mục 1, 2, 3)`.
   - Phần của **Thắng** vào `B.1.3`, `B.2 (mục 5)`, `TN3 (phần còn lại)` và mục `Kết luận`.
2. Kiểm tra hàm `grid_occupancy` ở mục `A.1.5` xem có khớp hàm thầy phát trên LMS không.
3. Rà soát **toàn bộ cell AI ở phần thí nghiệm**: mọi cell dùng AI đều phải có dòng `**Prompt đã dùng (nguyên văn):**` ngay phía trên.
4. Rà soát **toàn bộ cell cài đặt (A.1, B.1)**: không gọi bất kỳ hàm bị cấm nào.
5. Bấm **Restart** → **Run All** trong VS Code, chờ chạy sạch từ cell đầu tới cell cuối **không 1 lỗi đỏ nào**, kiểm tra mọi biểu đồ đều hiển thị đầy đủ → `Ctrl + S`.
6. Nén/nộp đúng yêu cầu: notebook `bt1_nhomN.ipynb` (nhớ hỏi thầy tên file là `bt1_` hay `bt2_`) kèm thư mục `anh/`.
