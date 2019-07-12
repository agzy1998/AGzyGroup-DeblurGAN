# What is this repo ?

这是校企实训项目《DeblurGAN实现运动图像去模糊化》的仓库

![info1](<https://raw.githubusercontent.com/agzy1998/MarkDownPhotos/master/info1.png>)


# Installation

```
conda create -n envName python=3.6
conda activate envName
pip install -r requirements
pip install -e .
```

# Dataset

Get the [GOPRO dataset](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing), and extract it in the `deblur-gan` directory. The directory name should be `GOPRO_Large`.

Use:
```
python scripts/organize_gopro_dataset.py --dir_in=GOPRO_Large --dir_out=images
```


# Training

```
python scripts/train.py --n_images=512 --batch_size=16 --log_dir /path/to/log/dir
```

Use `python scripts/train.py --help` for all options

# Testing

```
python scripts/test.py
```

Use `python scripts/test.py --help` for all options

# Deblur your own image

```
python scripts/deblur_image.py --image_path=path/to/image
```
or  run the welcome.py to Deblur you own image in the GUI.

![info2](<https://raw.githubusercontent.com/agzy1998/MarkDownPhotos/master/info2.png>)
