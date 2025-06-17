# 时间序列异常检测

这个项目是从[Time-Series-Library](https://github.com/thuml/Time-Series-Library)中提取的异常检测部分，专注于时间序列异常检测任务。

## 支持的模型

- TimesNet
- Transformer
- Autoformer
- Informer

## 支持的数据集

- PSM
- MSL
- SMAP
- SMD
- SWAT

## 使用方法

1. 准备数据集。你可以从[Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)或[Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)下载预处理好的数据集，然后将它们放在`./dataset`文件夹下。

2. 训练和评估模型。我们在`./scripts`文件夹下提供了实验脚本。你可以按照以下方式复现实验结果：

```bash
# 异常检测
bash ./scripts/SMD/TimesNet.sh
```

## 项目结构

- `models/`: 包含所有模型的实现
- `exp/`: 包含实验类的实现
- `data_provider/`: 包含数据加载和预处理的代码
- `utils/`: 包含工具函数
- `scripts/`: 包含运行实验的脚本
- `layers/`: 包含模型层的实现