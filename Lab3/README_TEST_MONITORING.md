# Monitoring Lab3

## 1. Giới thiệu

Lab3 cung cấp hệ thống FastAPI và monitoring stack (Prometheus, Grafana, Alertmanager, Node Exporter) để giám sát và kiểm thử ứng dụng máy học. Tài liệu này hướng dẫn chi tiết cách kiểm thử API và giám sát hệ thống.
Demo hệ thống có thể được xem tại: [Demo Lab3](https://drive.google.com/file/d/1p661pxz8_dA9Bz7O1YiCinlMcPwBPEfH/view?usp=sharing).
---

## 2. Khởi động hệ thống

### a. Chạy toàn bộ hệ thống (không dùng Docker cho app)

1. Mở Terminal (PowerShell).
2. Di chuyển vào thư mục `Lab3`:
   ```powershell
   cd Lab3
   ```
3. Chạy script khởi động:
   ```powershell
   ./run_app_without_docker.ps1
   ```
   - Script sẽ tự động cài đặt các package Python cần thiết, khởi động monitoring stack bằng Docker Compose và chạy FastAPI app ở chế độ reload.

### b. Dừng hệ thống

- Để dừng monitoring stack:
  ```powershell
  ./stop_without_docker.ps1
  ```
- Để dừng FastAPI app: Nhấn `Ctrl+C` trong terminal đang chạy app.

---

## 3. Testing API

### a. Test thủ công qua Swagger UI
- Truy cập: [http://localhost:5050/docs](http://localhost:5050/docs)
- Thử các endpoint trực tiếp trên giao diện web.

### b. Test tự động bằng script
- Chạy script test:
  ```powershell
  python .\test_api.py
  ```
- Kết quả test sẽ hiển thị trên terminal.

---

## 4. Monitoring

### a. Truy cập các dịch vụ giám sát
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **Grafana:** [http://localhost:3000](http://localhost:3000) (Tài khoản mặc định: admin/admin)

### b. Theo dõi metrics
- Vào Grafana, chọn dashboard đã cấu hình sẵn để xem các chỉ số hệ thống và ứng dụng.
- Có thể tạo alert hoặc dashboard mới nếu cần.

---

## 5. Mô phỏng lỗi để kiểm tra alert
- Chạy script mô phỏng lỗi:
  ```powershell
  ./simulate_errors.ps1
  ```
- Kiểm tra alert trên Grafana hoặc Alertmanager.

---

## 6. Lưu ý
- Đảm bảo Docker đã được cài đặt và đang chạy.
- Nếu gặp lỗi port hoặc package, kiểm tra lại các bước cài đặt và log trên terminal.

---

