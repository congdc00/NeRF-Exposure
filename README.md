# Hướng dẫn cài đặt

# Yêu cầu

Hệ điều hành: Linux

GPU: Nvidia
Cuda: 11.3

# Chuẩn bị môi trường
`git clone https://github.com/congdinhchi/SSNeRF.git`<br>
`%cd SSNeRF`<br>
`pip3 install torch torchvision torchaudio` <br>
`pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`<br>
`pip install -r requirements.txt`<br>

# Chuẩn bị dữ liệu
Từ ./SSNeRF  
`mkdir load`

Dữ liệu có thể được tải về từ [NeRF_Data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1?usp=share_link) và đặt vào thư mục load  


# Chạy

với `$NAME_DATA` là tên của tập dữ liệu đã tải về

Chạy demo [Train NeRF](https://colab.research.google.com/drive/1Vv-3qg_7sOZrH1k_r9CG9fekyQPYRpS3?usp=share_link)

Phương pháp NeRF + MRE
`python launch.py --config configs/ssnerf1-blender.yaml --gpu 0 --train dataset.scene=$NAME_DATA tag=example`

Phương pháp NeuS + MRE
`python launch.py --config configs/eneus-blender.yaml --gpu 0 --train dataset.scene=$NAME_DATA tag=example system.loss.lambda_mask=0.0`
