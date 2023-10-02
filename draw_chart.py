import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('./output.csv')

# Lấy giá trị PSNR và độ lệch chuẩn (std PSNR)
psnr_values = data['SSIM']
std_psnr_values = data['std SSIM']

# Lấy giá trị epoch từ 1000 đến 50000 với bước nhảy là 1000
epochs = list(range(1000, 51000, 1000))

# Vẽ biểu đồ error bar
plt.errorbar(epochs, psnr_values, yerr=std_psnr_values, fmt='o-', color='b', ecolor='r', capsize=5)
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('Giá trị SSIM và std SSIM của mô hình NeRF+MRE(6) qua từng vòng lặp')
plt.grid(True)
plt.show()
