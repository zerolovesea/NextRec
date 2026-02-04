# NextRec Studio

NextRec Studio是一个基于Vue3的前端项目，旨在辅助用户在浏览器中，通过可视化的方式，配置支持NextRec CLI的训练、特征、模型与推理 YAML 文件。

## 功能简介

- 训练/特征/模型/推理四个页面独立配置，避免配置冗余复杂的配置文件
- 各类模型的分析和参数介绍，帮助用户快速了解模型能做什么

[](/assets/nextrec_studio_ui_zh.png)

[](/assets/nextrec_studio_model_config_zh.png)

## 启动方式

我们提供了三种启动方式：脚本 / Dockerfile / Docker Compose。

### 脚本启动

```bash
cd nextrec_studio
./start_nextrec_studio.sh
```

浏览器访问：`http://localhost:15173`

### Dockerfile 启动

```bash
docker build -t nextrec-studio .
docker run --rm -p 15173:15173 nextrec-studio
```

浏览器访问：`http://localhost:15173`

### Docker Compose 启动

```bash
docker compose up --build
```

浏览器访问：`http://localhost:15173`

## 使用方式

在页面配置完训练所需要的参数后，将配置文件下载至本地，命令行执行`nextrec --mode train --train_config train_config.yaml`和`nextrec --mode predict --predict_config predict_config.yaml`进行训练和推理。
