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

2. Download images and labels, transforming .xml to a .csv file, with header row 'filename', 'label', 'xmax', 'xmin', 'ymax', 'ymin'.
   For example:
   | filename | label | xmax | xmin | ymax | ymin |
   | -------- | :---: | :--: | :--: | :--: | :--: |
   |c1\_1844849.jpg|good|1246|127|1312|227|
   |c1\_1844849.jpg|none|745|889|862|999|
   
