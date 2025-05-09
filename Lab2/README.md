# Lab2: Triển khai Mô hình với Docker và Flask

Thư mục này chứa triển khai của một API phục vụ mô hình dự đoán thời tiết Úc được phát triển trong Lab1.

## Demo
[Link Demo](https://drive.google.com/file/d/1wWhVJByaO0OpxF4zHl9UfbqDNgF96dZk/view?usp=sharing)

## Tổng quan

Dự án này triển khai một dịch vụ REST API để phục vụ mô hình dự đoán thời tiết đã được huấn luyện trong Lab1. Dịch vụ được đóng gói bằng Docker và cung cấp các endpoint để kiểm tra trạng thái và thực hiện dự đoán.

## Yêu cầu

- Docker
- Docker Compose
- Tệp mô hình đã huấn luyện (`model.pkl`) từ Lab1

## Cấu trúc Dự án

Dự án bao gồm các thành phần sau:

1. **Flask API** (`app.py`) - Ứng dụng chính phục vụ mô hình
2. **Cấu hình Docker**
   - `Dockerfile` - Định nghĩa container
   - `docker-compose.yml` - Điều phối dịch vụ
3. **Mô hình** (`model.pkl`) - Mô hình đã huấn luyện từ Lab1
4. **Dependencies** (`requirements.txt`) - Các gói Python cần thiết

## API Endpoints

### Kiểm tra Trạng thái
```bash
curl http://localhost:5000/health
```

### Dự đoán
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MinTemp": 20.0,
    "MaxTemp": 25.0,
    "Rainfall": 0.0,
    "WindGustDir": "N",
    "WindGustSpeed": 30.0,
    "WindDir9am": "N",
    "WindDir3pm": "N",
    "WindSpeed9am": 10.0,
    "WindSpeed3pm": 15.0,
    "Humidity9am": 60.0,
    "Humidity3pm": 55.0,
    "Pressure9am": 1015.0,
    "Pressure3pm": 1013.0,
    "Temp9am": 22.0,
    "Temp3pm": 24.0,
    "RainToday": "No"
  }'
```

## Ví dụ Dữ liệu Đầu vào

### Ví dụ 1: Dữ liệu từ trạm Albury (Ngày nắng)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MinTemp": 13.4,
    "MaxTemp": 22.9,
    "Rainfall": 0.6,
    "WindGustDir": "W",
    "WindGustSpeed": 44.0,
    "WindDir9am": "W",
    "WindDir3pm": "WNW",
    "WindSpeed9am": 20.0,
    "WindSpeed3pm": 24.0,
    "Humidity9am": 71.0,
    "Humidity3pm": 22.0,
    "Pressure9am": 1007.7,
    "Pressure3pm": 1007.1,
    "Temp9am": 16.9,
    "Temp3pm": 21.8,
    "RainToday": "No"
  }'
```

### Ví dụ 2: Dữ liệu từ trạm Albury (Ngày mưa)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MinTemp": 15.2,
    "MaxTemp": 19.8,
    "Rainfall": 12.4,
    "WindGustDir": "NW",
    "WindGustSpeed": 52.0,
    "WindDir9am": "NW",
    "WindDir3pm": "W",
    "WindSpeed9am": 28.0,
    "WindSpeed3pm": 35.0,
    "Humidity9am": 89.0,
    "Humidity3pm": 82.0,
    "Pressure9am": 1002.3,
    "Pressure3pm": 1000.5,
    "Temp9am": 16.8,
    "Temp3pm": 18.5,
    "RainToday": 1
  }'
```

Lưu ý:
- Các giá trị này được lấy từ dữ liệu thực tế của trạm thời tiết Albury
- RainToday được chuyển đổi từ "No" thành 0 và "Yes" thành 1
- Tất cả các giá trị đều là số thực, không có giá trị NaN
- Các giá trị nằm trong khoảng hợp lệ cho mỗi trường
- Ví dụ 1 là ngày nắng (RainToday = 0)
- Ví dụ 2 là ngày mưa (RainToday = 1)

## Định dạng Phản hồi

Endpoint dự đoán trả về phản hồi JSON với:
- `prediction`: giá trị boolean cho biết ngày mai có mưa hay không
- `probability`: giá trị float cho biết xác suất mưa

Ví dụ phản hồi:
```json
{
  "prediction": true,
  "probability": 0.85
}
```

## Chạy Dịch vụ

### Phát triển Cục bộ

1. Đảm bảo bạn có tệp mô hình đã huấn luyện (`model.pkl`) trong thư mục Lab2
2. Khởi động dịch vụ:
```bash
docker-compose up --build
```

API sẽ có sẵn tại `http://localhost:5000`

### Triển khai lên Docker Hub

Để triển khai dịch vụ lên Docker Hub:

1. Build image:
```bash
docker build -t yourusername/weather-prediction-api:latest .
```

2. Đẩy lên Docker Hub:
```bash
docker push yourusername/weather-prediction-api:latest
```

## Chi tiết Kỹ thuật

### Các Đặc trưng Đầu vào của Mô hình

API chấp nhận các đặc trưng sau cho việc dự đoán:
- Đo nhiệt độ (MinTemp, MaxTemp, Temp9am, Temp3pm)
- Mức độ ẩm (Humidity9am, Humidity3pm)
- Thông tin gió (WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm)
- Chỉ số áp suất (Pressure9am, Pressure3pm)
- Trạng thái mưa hiện tại (RainToday_coded)

### Cấu hình Môi trường

Dịch vụ có thể được cấu hình bằng các biến môi trường trong `docker-compose.yml`:
- `PORT`: Cổng dịch vụ API (mặc định: 5000)
- `MODEL_PATH`: Đường dẫn đến tệp mô hình (mặc định: model.pkl)

## Lưu ý

- API yêu cầu mô hình phải có sẵn dưới dạng `model.pkl` trong cùng thư mục
- Dịch vụ sử dụng cổng 5000 theo mặc định
- Các biến môi trường có thể được sửa đổi trong tệp docker-compose.yml

## Xử lý Lỗi và Yêu cầu Dữ liệu

### Lỗi Giá trị NaN

Khi gửi dữ liệu đến API, cần đảm bảo rằng:
- Không có giá trị NaN trong dữ liệu đầu vào
- Tất cả các trường đều phải có giá trị hợp lệ
- Các giá trị số phải nằm trong khoảng hợp lý

Nếu gặp lỗi "Input X contains NaN", bạn cần:
1. Kiểm tra dữ liệu đầu vào để đảm bảo không có giá trị thiếu
2. Tiền xử lý dữ liệu trước khi gửi đến API
3. Đảm bảo tất cả các trường trong JSON request đều được điền đầy đủ

### Các Giá trị Hợp lệ

- Nhiệt độ: giá trị số thực
- Độ ẩm: giá trị số từ 0 đến 100
- Áp suất: giá trị số dương
- Tốc độ gió: giá trị số không âm
- Hướng gió: một trong các giá trị hợp lệ (N, NE, E, SE, S, SW, W, NW)
- RainToday: 0 (không mưa) hoặc 1 (có mưa)  
