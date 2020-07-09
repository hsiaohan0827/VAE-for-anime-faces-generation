# VAE-for-anime-faces-generation
implement a VAE structure to generate anime faces



## Setup
1. 創建一個新環境
```
python3 -m venv env_name
```
2. activate environment
```
source env_name/bin/activate
```
3. 安裝requirement.txt中的套件
```
pip3 install -r requirements.txt
```


## Download Data
1. Using Anime Faces Dataset (https://www.kaggle.com/soumikrakshit/anime-faces) consisting of 21551 anime faces.

2. place images under data/img/ i.e. data/img/1.jpg, data/img/2.jpg ...
   
   
## Training
1.  修改config/config.py中的config
```python
# CONFIG
class VAEConfig:
    h_dim = 400
    z_dim = 32
class TrainConfig:
    isTrain = True
    img_size = 32
    KL_weight = 1
```

### configuration
- **h_dim** - hidden layer dimension.
- **z_dim** - latent vector dimension.
- **isTrain** - if the model do train or test
- **img_size** - input image size.

3.  run main.py
```
python3 main.py
```

### tensorboardX
可以使用tensorboard觀察loss及accuracy變化
```
tensorboard --logdir logs
```

## Testing
1. 修改isTrain及ckpt
```python
class TrainConfig:
    isTrain = True
    ckpt = 'VAE|bs-64|Adam-lr-0.001|img-32|KL-1|h-512|z-64'
    epoch = 1000
    """ Testing """
    img1_num = 323
    img2_num = 267
    interpolate_num = 8
```
2. run main.py, get interpolate result for two real images.
```
python3 main.py
```
![interpolate image](https://github.com/hsiaohan0827/VAE-for-anime-faces-generation/blob/master/interpolate-323-267-KL-1.png)
