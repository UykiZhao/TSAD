#!/bin/bash

# PSM数据集
echo "运行PSM数据集的实验..."
bash scripts/PSM/Autoformer.sh
bash scripts/PSM/Informer.sh
bash scripts/PSM/Transformer.sh
bash scripts/PSM/TimesNet.sh

# SMD数据集
echo "运行SMD数据集的实验..."
bash scripts/SMD/Autoformer.sh
bash scripts/SMD/Informer.sh
bash scripts/SMD/Transformer.sh
bash scripts/SMD/TimesNet.sh

# MSL数据集
echo "运行MSL数据集的实验..."
bash scripts/MSL/Autoformer.sh
bash scripts/MSL/Informer.sh
bash scripts/MSL/Transformer.sh
bash scripts/MSL/TimesNet.sh

# SMAP数据集
echo "运行SMAP数据集的实验..."
bash scripts/SMAP/Autoformer.sh
bash scripts/SMAP/Informer.sh
bash scripts/SMAP/Transformer.sh
bash scripts/SMAP/TimesNet.sh

# SWAT数据集
echo "运行SWAT数据集的实验..."
bash scripts/SWAT/Autoformer.sh
bash scripts/SWAT/Informer.sh
bash scripts/SWAT/Transformer.sh
bash scripts/SWAT/TimesNet.sh

echo "所有实验完成!" 