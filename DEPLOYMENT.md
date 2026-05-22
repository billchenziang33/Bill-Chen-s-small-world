# 云端部署说明

这个项目建议拆成两部分部署：

1. 前端部署到 Vercel，负责页面、摄像头、手势/物品识别界面。
2. 后端部署到一台能长期运行 Node + Python 的云服务器，负责 SQLite、PyTorch、Ultralytics YOLO、训练数据和模型文件。

原因是 Vercel 更适合静态前端和短时间 Serverless API，不适合长时间 YOLO 训练、持续写入本地文件、保存 `.pt` 模型和 SQLite 数据库。

## Vercel 前端

项目根目录已经有 `vercel.json`，Vercel 会执行：

```powershell
npm install
npm run build
```

输出目录是：

```text
frontened/dist
```

本地开发时可以继续运行：

```powershell
npm run dev
```

## 前端连接云端后端

上线到 Vercel 后，需要在 Vercel Project Settings 里添加环境变量：

```text
VITE_API_BASE_URL=https://你的后端域名
```

比如你的后端部署到了 Render：

```text
VITE_API_BASE_URL=https://billchen-small-world-api.onrender.com
```

这样前端里的 `/api/training-samples`、`/api/train-object-model`、`/api/object-detect` 等请求都会自动发到云端后端。

## 后端云服务器要求

后端服务器需要支持：

```text
Node.js 22+
Python 3.11/3.12/3.13
pip install ultralytics torch torchvision opencv-python
可持久保存 backened/data
```

后端启动命令：

```powershell
npm install
python -m pip install ultralytics torch torchvision opencv-python
npm run build
npm start
```

如果云平台有环境变量 `PORT`，当前后端会自动使用它。

## Render 后端部署

项目已经准备好 Docker 部署文件：

```text
Dockerfile
render.yaml
backened/requirements.txt
```

推荐操作：

1. 把当前项目上传到 GitHub。
2. 打开 Render，选择 New Blueprint 或 New Web Service。
3. 连接这个 GitHub 仓库。
4. 如果使用 Blueprint，Render 会读取 `render.yaml`。
5. 如果手动创建 Web Service，Runtime/Language 选择 Docker。
6. Health Check Path 设置为 `/api/health`。
7. 添加 Persistent Disk，挂载路径设置为 `/app/backened/data`，用于保存 SQLite、训练样本和模型文件。
8. 部署完成后，你会得到类似 `https://bill-chen-small-world-api.onrender.com` 的后端地址。
9. 回到 Vercel 项目设置，添加环境变量：

```text
VITE_API_BASE_URL=https://你的-render-后端地址
```

10. 在 Vercel 里重新部署前端。

Render 的持久化磁盘很重要。没有磁盘的话，云端采集的样本、SQLite 数据库、YOLO 训练出来的模型，在服务重启或重新部署后可能会丢失。

## 很重要

如果只把前端放到 Vercel，而后端还留在你电脑本地，别人打开网站时无法使用训练、保存样本、YOLO 检测这些功能。要让别人也能完整使用，后端也必须部署到公网 HTTPS 地址。
