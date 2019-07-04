## DCGAN代码

首先获取代码：
```
git clone https://github.com/fancyerii/DCGAN-tensorflow.git
cd DCGAN-tensorflow
```

接着下载数据：
```
python download.py mnist
```

下载celebA数据:
```
python download.py celebA
```

如果因为各种原因下载不了，celebA的[官网](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)提供了百度网盘下载，去[百度网盘](https://pan.baidu.com/s/1eSNpdRG#list/path=%2F) 下载img_align_celeba.zip到当前目录的data子目录下，然后解压后重命名为celebA。

然后开始训练：
```
$ python main.py --dataset mnist --input_height=28 --output_height=28 --train --epoch=100
$ python main.py --dataset celebA --input_height=108 --train --crop
```

celebA的数据较大，如果没有GPU建议尝试mnist数据集。

训练完了模型之后可以用下面的命令来生成图片：
```
$ python main.py --dataset mnist --input_height=28 --output_height=28
$ python main.py --dataset celebA --input_height=108 --crop
```


