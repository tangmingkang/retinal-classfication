# Retinal Image Classfication
### 使用教程
clone源码并安装依赖包，所有图像放到datasets/image路径下，使用generate_csv.py可以生成相应的标签文件label.csv
```bash
git clone https://github.com/tangmingkang/retinal-classfication.git
cd retinal-classfication
pip install -r requirements.txt
```
### 运行说明
训练模型
```bash
chmod a+x ./scripts/train.sh
./scripts/train.sh
```
测试模型，将结果输出到subs文件夹
```bash
chmod a+x ./scripts/predict.sh
./scripts/train.sh
```
使用grad-cam可视化模型，结果输出到visualization
```bash
chmod a+x ./scripts/visual.sh
./scripts/train.sh
```
**注：kernel_type指定参数文件与日志文件名称，未设置时使用默认命名**
