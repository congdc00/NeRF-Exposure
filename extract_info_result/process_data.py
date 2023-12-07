import re
import csv

# Đọc nội dung từ file .txt
with open("./data/enerf_bf_lambda_09", "r") as file:
    content = file.readlines()
i = 1
file_name = "output.csv"
type_model = "NeRF_MRE"  # Mở một file mới để lưu kết quả
with open(file_name, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Lặp qua từng dòng trong nội dung đọc được
    if type_model == "NeRF_MRE":
        writer.writerow(
            ["PSNR", "std PSNR", "SSIM", "std SSIM", "Exposure", "PE", "std PE"]
        )
        for line in content:
            # Sử dụng regular expression để tìm các giá trị --Exposure, --PE, --Std PE trong dòng
            std_psnr_match = re.search(r"-- std PSNR: ([\d.]+)", line)
            ssim_match = re.search(r"-- SSIM ([\d.]+)", line)
            std_ssim_match = re.search(r"-- std SSIM: ([\d.]+)", line)
            exposure_match = re.search(r"--Exposure ([\d.]+)", line)
            pe_match = re.search(r"--PE ([\d.]+)", line)
            std_pe_match = re.search(r"--Std PE ([\d.]+)", line)

            loss_match = re.search(r"loss=([\d.]+)", line)
            train_match = re.search(r"train/num_rays=([\d.]+)", line)
            psnr_match = re.search(r"val/psnr=([\d.]+)", line)

            # Nếu các giá trị được tìm thấy, ghi chúng vào tệp tin output.txt
            if (
                std_psnr_match
                and std_psnr_match
                and std_ssim_match
                and exposure_match
                and pe_match
                and std_pe_match
            ):
                std_psnr_value = std_psnr_match.group(1)
                ssim_value = ssim_match.group(1)
                std_ssim_value = std_ssim_match.group(1)
                exposure_value = exposure_match.group(1)
                pe_value = pe_match.group(1)
                std_pe_value = std_pe_match.group(1)
                i += 1
            elif loss_match and train_match and psnr_match:
                i += 1
                psnr_value = psnr_match.group(1)
                if i == 3:
                    writer.writerow(
                        [
                            psnr_value,
                            std_psnr_value,
                            ssim_value,
                            std_ssim_value,
                            exposure_value,
                            pe_value,
                            std_pe_value,
                        ]
                    )

            else:
                i = 0
    else:
        # Lặp qua từng dòng trong nội dung đọc được
        writer.writerow(["PSNR", "std PSNR", "SSIM", "std SSIM"])
        for line in content:
            # Sử dụng regular expression để tìm các giá trị --Exposure, --PE, --Std PE trong dòng
            std_psnr_match = re.search(r"-- std PSNR: ([\d.]+)", line)
            ssim_match = re.search(r"-- SSIM ([\d.]+)", line)
            std_ssim_match = re.search(r"-- std SSIM: ([\d.]+)", line)
            loss_match = re.search(r"loss=([\d.]+)", line)
            train_match = re.search(r"train/num_rays=([\d.]+)", line)
            psnr_match = re.search(r"val/psnr=([\d.]+)", line)

            # Nếu các giá trị được tìm thấy, ghi chúng vào tệp tin output.txt
            if std_psnr_match and std_psnr_match and std_ssim_match:
                std_psnr_value = std_psnr_match.group(1)
                ssim_value = ssim_match.group(1)
                std_ssim_value = std_ssim_match.group(1)
                i += 1
            elif loss_match and train_match and psnr_match:
                i += 1
                psnr_value = psnr_match.group(1)
                if i == 3:
                    writer.writerow(
                        [psnr_value, std_psnr_value, ssim_value, std_ssim_value]
                    )

            else:
                i = 1
                

