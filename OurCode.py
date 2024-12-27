import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
data = pd.read_csv('CarPrice_Assignment.csv')

#------------BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH MÔ TẢ------------
# Chọn chỉ các cột có kiểu số để tính toán các thống kê
numeric_data = data.select_dtypes(include=[np.number])

# Tóm lược dữ liệu với các thống kê mô tả cơ bản
summary = numeric_data.describe()

# In ra các chỉ số cơ bản
print("Tóm lược dữ liệu:")
print(summary)

# Tính toán thêm một số chỉ số mô tả khác cho các cột số
mean = numeric_data.mean()  # Trung bình
median = numeric_data.median()  # Trung vị
mode = numeric_data.mode().iloc[0]  # Mode (giá trị xuất hiện nhiều nhất)
std_dev = numeric_data.std()  # Độ lệch chuẩn
variance = numeric_data.var()  # Phương sai
range_values = numeric_data.max() - numeric_data.min()  # Biên độ (range)

# In các chỉ số tóm lược thêm
print("\nMức độ tập trung:")
print(f"Mean:\n{mean}\n")
print(f"Median:\n{median}\n")
print(f"Mode:\n{mode}\n")

print("\nMức độ phân tán:")
print(f"Standard Deviation:\n{std_dev}\n")
print(f"Variance:\n{variance}\n")
print(f"Range:\n{range_values}\n")

# Kiểm tra và in ra số lượng các giá trị thiếu trong mỗi cột
missing_values = numeric_data.isnull().sum()
print("Số lượng giá trị thiếu trong các cột:")
print(missing_values)
# Tỷ lệ thiếu cho mỗi cột
missing_percentage = (missing_values / len(numeric_data)) * 100
print("\nTỷ lệ giá trị thiếu trong các cột:")
print(missing_percentage)
# Điền giá trị trung bình cho các cột có giá trị thiếu
numeric_data_filled = numeric_data.fillna(numeric_data.mean())
# Kiểm tra sau khi điền giá trị thiếu
missing_values_after = numeric_data_filled.isnull().sum()
print("\nSố lượng giá trị thiếu sau khi điền:")
print(missing_values_after)
# Tóm lược dữ liệu sau khi điền giá trị thiếu
summary_after_processing = numeric_data_filled.describe()
# In tóm lược dữ liệu sau khi điền
print("\nTóm lược dữ liệu sau khi điền giá trị thiếu:")
print(summary_after_processing)


# Kiểm tra số lượng dữ liệu bị trùng (duplicate rows)
duplicate_rows = numeric_data_filled.duplicated().sum()
print(f"\nSố lượng dòng trùng lặp: {duplicate_rows}")
# Xóa các dòng trùng lặp nếu có
numeric_data_filled = numeric_data_filled.drop_duplicates()
# Tóm lược dữ liệu sau khi xử lý
summary_after_processing = numeric_data_filled.describe()
# In tóm lược dữ liệu sau khi xử lý
print("\nTóm lược dữ liệu sau khi xử lý:")
print(summary_after_processing)

# Đếm số lượng giá trị riêng biệt trong mỗi cột
unique_values_count = numeric_data.nunique()
# In ra kết quả
print("\nSố lượng giá trị riêng biệt trong các cột:")
print(unique_values_count)

#------------KẾT THÚC QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH MÔ TẢ------------

#------------BẮT ĐẦU QUÁ TRÌNH PHÂN TÍCH MÔ TẢ------------
# Biểu đồ 1: Phân phối giá xe
# Mô tả: Biểu đồ histogram thể hiện phân phối của giá xe (price).
#        Kết hợp với đường KDE (Kernel Density Estimate) để mô tả mật độ xác suất.
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='price', bins=20, kde=True, color='blue')
plt.title('Phân phối giá xe (Price)', fontsize=16)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Biểu đồ 2: Phân phối công suất động cơ
# Mô tả: Biểu đồ boxplot hiển thị phân phối công suất động cơ (horsepower), giúp phát hiện
#        các giá trị ngoại lệ (outliers) và đánh giá tập trung của dữ liệu.
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='horsepower', color='green')
plt.title('Phân phối công suất động cơ (Horsepower)', fontsize=16)
plt.xlabel('Horsepower')
plt.show()

# Biểu đồ 3: Scatterplot giữa Horsepower và Price
# Mô tả: Biểu đồ scatterplot thể hiện mối quan hệ giữa công suất động cơ (horsepower) và giá xe (price).
#        Sử dụng màu sắc để phân loại loại nhiên liệu (fueltype) và kiểu nạp khí (aspiration).
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='horsepower', y='price', hue='fueltype', style='aspiration')
plt.title('Mối quan hệ giữa Horsepower và Price', fontsize=16)
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.legend(title='Loại nhiên liệu và Aspiration')
plt.show()

# Biểu đồ 4: Heatmap ma trận tương quan
# Mô tả: Heatmap biểu diễn ma trận tương quan giữa các cột số như enginesize, horsepower, price,...
#        Giá trị tương quan cao (gần 1 hoặc -1) cho thấy mối quan hệ mạnh giữa các thuộc tính.
numeric_columns = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower', 'citympg', 'highwaympg', 'price']
correlation_matrix = data[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap: Ma trận tương quan giữa các thuộc tính', fontsize=16)
plt.show()

# Biểu đồ 5: Số lượng xe theo loại nhiên liệu
# Mô tả: Biểu đồ cột hiển thị số lượng xe theo từng loại nhiên liệu (fueltype).
#        Giúp đánh giá tỷ lệ xe sử dụng gas và diesel trong tập dữ liệu.
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='fueltype', palette='Set2')
plt.title('Số lượng xe theo loại nhiên liệu', fontsize=16)
plt.xlabel('Loại nhiên liệu')
plt.ylabel('Số lượng xe')
plt.show()

# Biểu đồ 6: Phân phối giá xe theo số lượng xi-lanh
# Mô tả: Biểu đồ boxplot hiển thị phân phối giá xe (price) dựa trên số lượng xi-lanh (cylindernumber).
#        Giúp so sánh sự khác biệt về giá giữa các loại xe có cấu hình xi-lanh khác nhau.
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='cylindernumber', y='price', palette='coolwarm')
plt.title('Phân phối giá xe theo số lượng xi-lanh', fontsize=16)
plt.xlabel('Số lượng xi-lanh')
plt.ylabel('Giá xe')
plt.show()

#------------KẾT THÚC QUÁ TRÌNH PHÂN TÍCH MÔ TẢ------------

#------------BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------
# **Bước 2: Tiền xử lý dữ liệu**
# Chọn các cột quan trọng làm đặc trưng cho mô hình
features = ['horsepower', 'curbweight', 'enginesize', 'carwidth']  # Các cột đầu vào
X = data[features]  # X là tập đặc trưng
Y = data['price']  # Y là cột mục tiêu

# **Bước 3: Chia dữ liệu train-test**
# Dùng train-test split để chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#------------KÉT THÚC QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------


#------------BẮT ĐẦU PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------
# **Bước 4: Xây dựng mô hình hồi quy tuyến tính**
model = LinearRegression()  # Khởi tạo mô hình Linear Regression
model.fit(X_train, Y_train)  # Huấn luyện mô hình với tập train

# **Bước 5: Đánh giá mô hình**
# Dự đoán trên tập kiểm tra
Y_pred = model.predict(X_test)
# Tính các chỉ số đánh giá:
r2 = r2_score(Y_test, Y_pred)  # Hệ số xác định R²
mse = mean_squared_error(Y_test, Y_pred)  # Mean Squared Error
print("R² (Hệ số xác định):", r2)
print("MSE (Sai số bình phương trung bình):", mse)

# **Bước 6: Trực quan hóa kết quả**
# So sánh giá trị thực tế và dự đoán
plt.figure(figsize=(8, 5))
plt.scatter(Y_test, Y_pred, alpha=0.7, color="blue")  # Biểu đồ scatter
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color="red")  # Đường y = x
plt.title('So sánh giá trị thực tế và dự đoán')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.show()

# **Bước 7: Nhập dữ liệu từ người dùng và dự đoán giá xe
# Nhập dữ liệu từ người dùng và dự đoán giá xe
horsepower_input = float(input("Nhập giá trị Horsepower: "))
curbweight_input = float(input("Nhập giá trị Curbweight: "))
enginesize_input = float(input("Nhập giá trị Enginesize: "))
carwidth_input = float(input("Nhập giá trị Carwidth: "))

user_input = pd.DataFrame({
    'horsepower': [horsepower_input],
    'curbweight': [curbweight_input],
    'enginesize': [enginesize_input],
    'carwidth': [carwidth_input]
})

predicted_price = model.predict(user_input)
print(f"\nGiá xe dự đoán: ${predicted_price[0]:.2f}")
#------------KẾT THÚC PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------


