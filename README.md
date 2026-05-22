# Bill Chen 的小世界

这是一个前后端都有的 React + Node 网站，用摄像头做两件事：

- 手势数字识别：采集手势关键点，训练本地深度学习模型，并用 softmax 输出数字 1 到 10 的概率。
- YOLO 物品检测：采集物品图片样本，训练本地 YOLO 模型，再实时检测手机、杯子、iPad、鼠标、纸、纸巾。

## 目录

- `frontened/`：React 前端。
- `backened/`：Node 后端和 Python 训练脚本。
- `backened/data/gesture_training.sqlite`：手势训练样本数据库。
- `backened/data/gesture_model.pt`：PyTorch 手势模型权重。
- `backened/data/gesture_model.json`：前端可直接推理的手势模型。
- `backened/data/object_dataset/`：YOLO 物品训练图片和标签。
- `backened/data/object_runs/detect_objects/weights/best.pt`：训练后的 YOLO 物品模型。

## 启动

```powershell
npm install
npm run dev
```

然后打开 `http://localhost:5173`。如果 Vite 绑定到 IPv6，也可以打开 `http://[::1]:5173`。

## 手势训练

1. 打开摄像头。
2. 进入“手势训练”。
3. 选择数字，摆出对应手势。
4. 保存样本，或者连续保存到 100。
5. 点击“训练深度模型”。

## YOLO 物品训练

1. 打开摄像头。
2. 进入“物品检测”。
3. 选择物品类别，把物品完整放进蓝色采集框。
4. 多保存一些不同角度、不同距离、不同光照的样本。
5. 点击“训练 YOLO 模型”。
6. 训练完成后点击“检测一次”或“开始实时检测”。

YOLO 依赖 `ultralytics`。如果训练按钮提示没有安装，可以运行：

```powershell
python -m pip install ultralytics
```
