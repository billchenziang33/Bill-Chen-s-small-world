# Bill Chen 的小世界

这是一个前后端分离的 React 手势识别网站。

## 功能

- 在浏览器里开启摄像头
- 实时检测手部 21 个关键点
- 在视频上绘制更贴合手指位置的蓝色骨架
- 识别数字 `1` 到 `10` 的手势
- 支持你为每个数字采集 100 个样本，保存到本地 SQLite 数据库
- 支持用本地 PyTorch 训练一个小型深度学习模型，并导出给前端实时推理
- 通过后端 API 保存最近识别记录

## 目录

- `frontened/`：React 前端
- `backened/`：Node 后端和 API
- `backened/data/gesture_training.sqlite`：本地训练样本数据库
- `backened/data/gesture_model.pt`：PyTorch 训练后的模型权重
- `backened/data/gesture_model.json`：前端可直接推理的模型权重
- `scripts/`：本地开发启动脚本
- `package.json`：项目脚本和依赖

## 启动

第一次运行先安装依赖：

```powershell
npm install
```

然后启动开发模式：

```powershell
npm run dev
```

打开 `http://localhost:5173`。

如果你的电脑把 Vite 绑定到了 IPv6，也可以打开 `http://[::1]:5173`。

## 训练自己的手势

1. 先开启摄像头。
2. 在“个人训练模型”里选择一个数字。
3. 摆出这个数字对应的手势。
4. 点击“采集 1 个”手动采集，或者点击“连续采集到 100”自动采集。
5. 每个数字采满 100 个样本后，点击“训练深度模型”。
6. 训练完成后，识别会融合深度学习模型、样本相似度模型和原本规则模型。

训练数据会保存在项目本地 SQLite 数据库里，点击“清空当前数字”或“清空全部”可以重新训练。

你不需要手动打开 PyTorch。PyTorch 是 Python 代码调用的库，只要当前 Python 环境里安装了 `torch`，点击页面里的“训练深度模型”即可。

## 端口

- 前端开发服务：`5173`
- 开发模式后端 API：`3001`
- 生产模式后端：`3000`
