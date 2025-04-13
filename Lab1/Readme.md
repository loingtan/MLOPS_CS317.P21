# Lab1: Experiment Tracking and Model Registry with Metaflow and MLflow

Thư mục này chứa triển khai của một pipeline MLOps hoàn chỉnh để dự đoán lượng mưa ở Úc sử dụng Bộ dữ liệu Thời tiết Kaggle.


## Demo
[Link Demo](https://drive.google.com/file/d/1B2tRoEtk104akju5LS1ywv-PArg7546M/view?usp=sharing)

## Bộ dữ liệu

### Bộ dữ liệu Thời tiết Úc

Bộ dữ liệu được sử dụng trong dự án này là Bộ dữ liệu Thời tiết Úc từ Kaggle, bao gồm các quan sát thời tiết hàng ngày từ nhiều trạm thời tiết ở Úc. Nhiệm vụ dự đoán cụ thể là xác định liệu ngày mai có mưa hay không (cột "RainTomorrow") dựa trên các điều kiện thời tiết hiện tại.

Các đặc điểm chính của bộ dữ liệu bao gồm:
- Các phép đo nhiệt độ (MinTemp, MaxTemp, Temp9am, Temp3pm)
- Độ ẩm (Humidity9am, Humidity3pm)
- Thông tin gió (WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm)
- Chỉ số áp suất (Pressure9am, Pressure3pm)
- Thông tin mưa hiện tại (RainToday)
- Biến mục tiêu: RainTomorrow (Yes/No)

#### Nguồn dữ liệu
Bộ dữ liệu có sẵn trên Kaggle: [Bộ dữ liệu Thời tiết Kaggle (Úc)](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

#### Cách lấy dữ liệu

1. Tải tệp `weatherAUS.csv` từ Kaggle
2. Đặt nó vào thư mục `Lab1/dataset/`
3. Pipeline được cấu hình để sử dụng đường dẫn này theo mặc định

## Tổng quan về Framework

### Các thành phần MLOps

Triển khai này tích hợp một số công cụ và framework MLOps quan trọng:

1. **Metaflow** - Dùng cho điều phối quy trình và quản lý pipeline
2. **MLflow** - Dùng cho theo dõi thực nghiệm, đăng ký mô hình và lưu trữ các artifacts
3. **Optuna** - Dùng cho tối ưu hóa hyperparameter
4. **Scikit-learn** - Dùng cho tiền xử lý và mô hình hóa ML

### Kiến trúc Pipeline

Pipeline bao gồm các bước sau:

1. **Data Loading** - Tải bộ dữ liệu thô và thực hiện tiền xử lý ban đầu
2. **Data Splitting** - Chia dữ liệu thành các tập huấn luyện, kiểm chứng và kiểm tra
3. **Data Preprocessing** - Xử lý các giá trị thiếu, đặc tính phân loại và thực hiện chuẩn hóa
4. **Hyperparameter Optimization** - Sử dụng Optuna để tìm hyperparameter tối ưu
5. **Final Model Training** - Huấn luyện mô hình tốt nhất với cross-validation
6. **Model Evaluation** - Đánh giá hiệu suất trên tập kiểm tra
7. **Model Registration** - Đăng ký mô hình trong Model Registry của MLflow

## Chạy Pipeline

### Thực thi cơ bản

Để chạy toàn bộ pipeline với các tham số mặc định:
```bash
python main.py run
```

### Tham số dòng lệnh

Pipeline chấp nhận một số tham số dòng lệnh:

```bash
python main.py run \
  --test_split 0.2 \
  --val_split 0.2 \
  --n_trials 20 \
  --experiment_name "AUS_Weather_Prediction_Experiment" \
  --model_name "WeatherClassifierAUS"
```

Các tham số:
- `test_split`: Tỷ lệ dữ liệu cho tập kiểm tra (mặc định: 0.2)
- `val_split`: Tỷ lệ dữ liệu cho tập kiểm chứng (mặc định: 0.2)
- `n_trials`: Số lần thử Optuna cho tối ưu hóa hyperparameter (mặc định: 20)
- `experiment_name`: Tên cho thực nghiệm MLflow (mặc định: "AUS_Weather_Prediction_Experiment")
- `model_name`: Tên cho mô hình đăng ký (mặc định: "WeatherClassifierAUS")

### Xem kết quả

#### Metaflow Card

Sau khi chạy pipeline, bạn có thể xem Metaflow card với:
```bash
python main.py card view <run_id>
```

Thay thế `<run_id>` bằng ID lần chạy được hiển thị sau khi thực thi.

#### MLflow UI

Để xem thông tin theo dõi thực nghiệm chi tiết:
```bash
mlflow ui
```

Sau đó truy cập http://localhost:5000 trong trình duyệt của bạn.

## Chi tiết mô hình

Pipeline huấn luyện một bộ phân loại LogisticRegression với các hyperparameter được tối ưu hóa thông qua Optuna:

- Các hyperparameter được điều chỉnh:
  - Regularization strength (C)
  - Solver (liblinear or saga)
  - Maximum iterations (max_iter)
  - Penalty type (l1 or l2)
  
- Các chỉ số đánh giá:
  - Accuracy
  - Log Loss
  - Confusion Matrix
