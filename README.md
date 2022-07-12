## 进度

### ./datasets
已完工。
出现GeneratorDataset创建迭代器失败的问题。

### ./modules
2d已写完，已测试，出现问题。
运行时出现莫名其妙的变量未定义（not defined）问题。

3d已写完，测试中。

### ./utils
差三线性插值，暂时使用torch代替。

### ./
extract已完工。

train_image.py, train_video.py已完工。测试中。

## 运行

训练

```shell
python train_image.py --image-path data/imgs/air_balloons.jpg --vae-levels 3 --checkname myimagetest
```