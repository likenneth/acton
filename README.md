## Video Tokneization

Codebase for video tokenization, based on our paper [Towards Tokenized Human Dynamics Representation](https://arxiv.org/pdf/2111.11433.pdf).

![](show.gif)

### Prerequisites (tested under Python 3.8 and CUDA 11.1):
 
apt-get install ffmpeg  
pip install torch==1.8  
pip install torchvision  
pip install pytorch-lightning  
pip install pytorch-lightning-bolts  
pip install aniposelib wandb gym test-tube ffmpeg-python matplotlib easydict scikit-learn   

If you are using python >= 3.8, you might want to use this once `ln -s ~/anaconda3/lib/libopenh264.so ~/anaconda3/envs/<env_name>/lib/libopenh264.so.5`

### Data Preparation

1. Make a directory besides this repo and name it `aistplusplus`
2. Download from [AIST++](https://google.github.io/aistplusplus_dataset/download.html) until it looks like
```angular2html
├── annotations
│   ├── cameras
│   ├── ignore_list.txt
│   ├── keypoints2d
│   ├── keypoints3d
│   ├── motions
│   └── splits
└── video_list.txt
```

### How to run:

1. Write one configuration file, e.g., `configs/tan.yaml`  
   
2. Run `python pretrain.py --cfg configs/tan.yaml` with GPU, which will create a folder under `logs` for this run. Folder name specified by the `NAME` in configuration file. 
   
3. Then use `python cluster.py --cfg configs/tan.yaml` and check results in `demo.ipynb`
