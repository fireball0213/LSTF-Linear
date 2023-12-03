### 导入所需数据文件

```
./dataset/ETT/ETTh1.csv
./dataset/illness/national_illness.csv
```



### 依赖项安装

```
pip install -r requirements.txt
```



## 复现结果指令

ETT+自回归+标准化

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --target OT --model Autoregression --transform Standardization
```

illness+指数平滑+归一化+α=0.9

```
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --target OT --model ExponentialMovingAverage --transform Normalization --alpha 0.9
```

illness+DES+标准化+α=0.8+β=0.2

```
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --target OT --model DoubleExponentialSmoothing --transform Standardization --alpha 0.8 --beta 0.2
```

illness+TsfKNN+标准化+STL分解

```
python main.py --data_path ./dataset/illness/national_illness.csv --dataset Custom --target OT --model TsfKNN --n_neighbors 5 --msas MIMO --distance euclidean --decompose True
```

