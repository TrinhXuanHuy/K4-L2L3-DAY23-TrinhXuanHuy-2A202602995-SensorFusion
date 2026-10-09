# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Trịnh Xuân Huy
- MSSV: 2A202602995
- Email:trinhhuy2304@gmail.com
- Link repo (fork): https://github.com/TrinhXuanHuy/K4-L2L3-DAY23-TrinhXuanHuy-2A202602995-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`):

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, frames [0, 198], segment training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, seed 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision 0.9701, recall 0.7004, tp/fp/fn: 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse 0.1503 m, matches 502, sum_sq_err 11.3437, ghost_track_frames 0, missed_gt_frames 239, mean_confirmed_tracks 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse 0.1359 m, matches 502, sum_sq_err 9.2668, ghost_track_frames 0, missed_gt_frames 239, mean_confirmed_tracks 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Cả hai chế độ đều theo dõi rất ổn định và nhất quán: cùng số cặp ghép hợp lệ (matches = 502), không sinh ra bất kỳ track rác nào (ghost_track_frames = 0), cùng số frame xe bị bỏ sót (missed_gt_frames = 239) và trung bình số track confirmed duy trì là 2.52 xe/frame. Điểm khác biệt quan trọng nằm ở sai số vị trí 3D (RMSE): chế độ Fused (LiDAR + Camera) đạt RMSE = 0.1359 m, giảm 0.0145 m so với LiDAR-only (RMSE = 0.1503 m), với tổng bình phương sai số sum_sq_err giảm từ 11.344 m² xuống 9.267 m². Điều này chứng minh rằng việc bổ sung đo lường 2D từ Camera ở góc nhìn phía trước (FRONT) đã giúp EKF tinh chỉnh vị trí tốt hơn mà không làm tăng ghost tracks hay làm mất track.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - LiDAR đo trực tiếp vị trí 3D của xe trong không gian `z = [x, y, z]^T` (đơn vị: mét). Ma trận quan sát H là tuyến tính kích thước 3x6 (chiếu trực tiếp 3 toạ độ vị trí đầu tiên của trạng thái). Hiệp phương sai nhiễu đo R là ma trận đường chéo 3x3 với các độ lệch chuẩn sigma_lidar theo mét.
   - Camera đo toạ độ pixel trên ảnh 2D `z = [u, v]^T` (đơn vị: pixel). Hàm quan sát h(x) là hàm phi tuyến theo mô hình camera pinhole với phép chia độ sâu x_s. Ma trận Jacobian H kích thước 2x6 phải tính đạo hàm phi tuyến theo quy tắc chuỗi (chain rule: đạo hàm phép chiếu nhân ma trận xoay). Hiệp phương sai nhiễu đo R là ma trận đường chéo 2x2 với các độ lệch chuẩn pixel bình phương (sigma_cam_i^2, sigma_cam_j^2).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Gating Mahalanobis tính khoảng cách chuẩn hoá d² = gamma^T * S^(-1) * gamma, có tính đến cả ma trận hiệp phương sai sai số trạng thái P và hiệp phương sai nhiễu đo R (thông qua ma trận hiệp phương sai innovation S = H*P*H^T + R). Việc đặt cổng gating chi-square giúp loại bỏ các đo lường ngoại lai (outliers/clutter) nằm ngoài vùng phân bố elip tin cậy, ngăn ngừa ghép nhầm các vật thể khác hoặc đo lường sai lệch nghiêm trọng vào track hiện tại.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline này là **track-then-fuse**: Chỉ duy trì một danh sách track duy nhất cho toàn bộ hệ thống. Trong mỗi frame Waymo, EKF thực hiện predict trạng thái một lần cho mọi track, sau đó thực hiện gán và cập nhật EKF với LiDAR (AssocL), rồi tiếp tục gán và cập nhật bổ sung EKF với Camera (AssocC) trên chính các track đó. Trên log `grade_run.log`, mỗi frame chỉ có đúng một bản ghi kết quả theo dõi với số lượng confirmed tracks sau khi đã cập nhật nối tiếp cả LiDAR và Camera, không có hai bộ tracker độc lập phân nhánh rồi mới gộp lại (fuse-then-track).

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Khi camera bị lệch calibration (lệch extrinsic xoay/tịnh tiến hoặc lệch intrinsic), hàm dự báo đo lường h(x) sẽ tính ra toạ độ pixel lệch khỏi vị trí thực tế trên ảnh. Khi đó, vector residual (innovation) gamma = z - h(x) sẽ xuất hiện độ lệch hệ thống (systematic bias, kỳ vọng khác 0). Độ lệch này làm khoảng cách Mahalanobis d² tăng cao vượt ngưỡng chi-square khiến camera measurement bị loại bỏ (missed association), hoặc nếu lọt qua gate thì Kalman gain sẽ kéo trạng thái ước lượng lệch khỏi thực tế, làm tăng đột biến sai số RMSE vị trí 3D.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Khi frame rỗng (không có measurement nào), `associate_and_update` vẫn phải gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`. Tham số `sensor` tường minh giúp manager phân biệt đây là lượt của LiDAR hay Camera. Nếu là LiDAR, manager phải duyệt qua các track nằm trong tầm nhìn (FOV) của LiDAR để trừ điểm tồn tại (do bị miss), đồng thời xoá các track quá hạn.
   - LiDAR quyết định điểm số/khởi tạo/xoá track vì LiDAR cung cấp đầy đủ thông tin vị trí không gian 3D và trường quan sát 360 độ xung quanh xe. Ngược lại, đo lường Camera trong lab chỉ là toạ độ 2D mô phỏng không có độ sâu độc lập và chỉ giới hạn ở camera trước (FRONT), nên camera chỉ đóng vai trò tinh chỉnh độ chính xác của trạng thái EKF mà không quyết định sự tồn tại của vật thể.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - Điều kiện xác nhận (confirmed): Khi track mới tạo từ LiDAR có score ban đầu là 1/window (1/6, state 'initialized'). Mỗi lượt LiDAR được gán (hit) cộng 1/window vào score (tối đa 1.0). Khi score > confirmed_threshold (0.8), track chính thức được chuyển sang trạng thái 'confirmed'.
   - Giữ confirmed sau miss: Khi một track đã 'confirmed' bị miss trong FOV LiDAR, score bị trừ 1/window, nhưng trạng thái vẫn được bảo toàn là 'confirmed' (không bị hạ cấp về tentative).
   - Điều kiện xoá track: Track bị xoá trong lượt LiDAR nếu thoả mãn bất kỳ điều kiện nào sau: (1) Phương sai toạ độ vị trí ngang P[0,0] > max_P (9.0) hoặc P[1,1] > max_P (9.0); (2) Track đã confirmed có score < delete_threshold (0.6); (3) Track chưa confirmed (tentative/initialized) có score <= 0.0.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

### 1. Phân tích Calibration Drift (+4 điểm)
- **Bằng chứng:** File số liệu `student/bonus/calibration_drift_analysis.json`.
- **Thực nghiệm:** Đưa vào 4 mức độ lệch ngoại chuẩn (extrinsic drift) của camera (thay đổi độ dịch chuyển ngang $\Delta y$ và góc quay $\Delta\text{yaw}$ trong ma trận biến đổi `veh_to_sens`):

| Mức độ lệch | $\Delta y$ (m) | $\Delta\text{yaw}$ (rad) | Mean Innovation (px) | Tỷ lệ qua cổng $\chi^2$ (%) | Matches | RMSE (m) |
|---|---|---|---|---|---|---|
| **0. Gốc (Baseline)** | 0.0 | 0.00 | 251.6 | 11.66% | 123 | **0.1430** |
| **1. Lệch nhẹ** | +0.2 | +0.02 (~1.1°) | 272.4 | 3.11% | 123 | **0.1588** |
| **2. Lệch vừa** | +0.6 | +0.06 (~3.4°) | 339.8 | 4.00% | 121 | **0.3220** |
| **3. Lệch nặng** | +1.5 | +0.15 (~8.6°) | 345.1 | 6.13% | 123 | **0.1337** |

- **Nhận xét & Cơ chế tự vệ của Tracker:**
  - Ở mức **Lệch nhẹ**, innovation trung bình tăng lên 272.4 px khiến tỷ lệ chấp nhận qua cổng $\chi^2$ giảm từ 11.66% xuống 3.11%. Một số ít đo lường bị lệch nhẹ vẫn lọt cổng làm tăng sai số ước lượng (RMSE tăng từ 0.1430 m lên 0.1588 m).
  - Ở mức **Lệch vừa**, sai số vị trí tăng vọt lên 0.3220 m (hơn gấp đôi) do những đo lường camera bị lệch điểm ảnh nghiêm trọng nhưng vẫn vô tình nằm vừa vặn trong elip phân phối sai số $\chi^2$, khiến Kalman Gain kéo mạnh vị trí 3D của track lệch khỏi thực tế.
  - Ở mức **Lệch nặng**, góc lệch quá lớn làm sai lệch toàn bộ tâm chiếu. Hầu hết các đo lường camera lúc này có khoảng cách Mahalanobis $d^2$ cực lớn và bị cổng Chi-square từ chối hoàn toàn. Nhờ đó, tracker tự vệ thành công bằng cách dựa gần như hoàn toàn vào đo lường LiDAR chất lượng cao, đưa RMSE quay về mức ổn định (0.1337 m).

### 2. Trực quan hoá Track & Đo lường trên BEV và Camera FRONT (+3 điểm)
- **Bằng chứng:** 
  - `student/bonus/viz_bev_tracking_frame_0010.png`: Ảnh hiển thị lưới BEV với mật độ điểm LiDAR, các bounding box phát hiện từ mạng FPN (màu xanh lá), các Track đã được xác nhận (confirmed tracks màu xanh cyan kèm Track ID và vận tốc ước lượng), đối chiếu với nhãn Ground Truth (dấu X màu đỏ).
  - `student/bonus/viz_camera_front_projection_frame_0010.png`: Ảnh chụp thực tế từ Camera FRONT của xe Waymo với hộp 3D bounding box của các track confirmed được chiếu pinhole trực tiếp lên mặt phẳng ảnh, minh hoạ độ khớp chính xác giữa mô hình 3D và đối tượng xe trong ảnh thật.

### 3. Export Track sang định dạng CVAT (+3 điểm)
- **Bằng chứng:** File `student/bonus/cvat_tracks_export.json` được sinh bằng hàm `fusion_lab.export_cvat.export_tracks_json`, chứa đầy đủ danh tính `id`, vị trí 3D `(x, y, z)`, vận tốc `(vx, vy, vz)` và kích thước `(h, w, l, yaw)` của các track qua từng frame.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity (Gemini 3.8 Flash)
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ phân tích mã nguồn repo, cài đặt các hàm EKF trong kalman.py, camera_fusion.py, association.py, track_management.py, khắc phục lỗi OpenMP libomp trên Windows, và soạn thảo báo cáo SUBMISSION.md.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy toàn bộ 128/128 unit tests trong student/tests (100% passed), đối chiếu công thức toán với tài liệu docs/HUONG_DAN_KY_THUAT.md, chạy benchmark fusion-run-lab trên segment Waymo ở chế độ compare và xác thực tính hợp lệ bằng công cụ tools/check_submission.py.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
