# CS317.P21 - Phát triển và vận hành hệ thống máy học

![Trường Đại học Công nghệ Thông tin | University of Information Technology](https://i.imgur.com/WmMnSRt.png)



## Giới thiệu môn
- Tên môn học: Phát triển và vận hành hệ thống máy học
- Mã môn học: CS317.P21
## Tổng quan
Repo này là tổng hợp các bài lab trong quá trình học môn học Phát triển và vận hành hệ thống máy học (CS317.P21) tại Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM. 

## Tải về
1. Clone the repository:

   ```bash
   git clone https://github.com/loingtan/MLOPS_CS317.P21.git
   cd MLOPS_CS317.P21
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ``
## Sử dụng
Để hiểu rõ hơn về cách sử dụng repo này, bạn có thể tham khảo các bài lab. Mỗi bài lab sẽ có hướng dẫn chi tiết về cách thực hiện và các yêu cầu cần thiết.

## Lab 3: Monitoring and Logging

Lab 3 implements comprehensive monitoring and logging for the Weather Prediction API built in Lab 2. The implementation includes:

### Key Components:
- **Prometheus**: For metrics collection
- **Grafana**: For metrics visualization
- **AlertManager**: For alert configuration and notification
- **Fluent Bit**: For log collection and aggregation
- **Node Exporter**: For system metrics collection

### Features:
- System resource monitoring (CPU, RAM, disk, network)
- API performance monitoring (requests/second, error rate, latency)
- Model monitoring (inference speed, confidence score)
- Advanced logging with multiple sources
- Configurable alerts with notification capabilities
- Comprehensive dashboards

To use Lab 3:

```bash
cd Lab3
./setup.sh          # Set up the environment
docker-compose up -d # Start the monitoring stack
./run_app.sh        # Run the Flask app
./test.sh           # Run tests and generate metrics
```

For more details, refer to the Lab3/README.md file.