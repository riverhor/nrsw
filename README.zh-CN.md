# NRSW 文件伪造检测

正常参照谱加权（NRSW）用于检测已知版式文档上的篡改。训练时需要未篡改的参考文件，以及少量已确认的伪造文件。每份训练用伪造文件只需标出可能被修改的字段，例如姓名或证件号码。检测新文件时，只需提供图像和版式。

[English](README.md) · [繁體中文](README.zh-TW.md) · [训练指南](docs/training.md) · [系统架构](docs/architecture.md) · [数据格式](docs/data.md) · [数据集下载与引用](docs/datasets.md) · [复现论文结果](docs/reproduction.md)

## 在线演示

打开[简体中文演示](https://riverhor.github.io/nrsw/?lang=zh-CN#upload)，即可试用 200 张 [IDSpace](docs/datasets.md) 斯洛伐克身份证合成图像。可抽选未篡改样本，或按字段与手法挑选篡改样本。每张图像都能查看字段权重、伪造分数和复核判定。[测试结果](https://riverhor.github.io/nrsw/?lang=zh-CN#evaluation)另列 NRSW 与对照方法在 4,106 张留出图像上的表现。也可上传同版式图像，执行实时检测。

要在本地打开演示，请从项目目录运行以下命令。建议使用 Python 3.11 或更新版本的虚拟环境。

```console
python -m pip install .
nrsw serve
```

在浏览器中打开 [localhost:8000](http://127.0.0.1:8000/?lang=zh-CN)。也可以运行 `nrsw demo --output runs/example`，用合成特征训练一个小型示例模型。

要在本地演示中实时检测，请按照[上传指南](docs/upload.md)加载已登记的检测器。

## 用自己的图像训练

按照[训练指南](docs/training.md)标出候选篡改字段、定义版式坐标，并将训练和校准数据分到独立的来源组。以下命令使用 CUDA。请先安装[支持 CUDA 的 PyTorch 和 torchvision](https://pytorch.org/get-started/locally/)，并准备好 `data/roles.jsonl` 和 `data/layout.json`。

```console
python -m pip install ".[vision]"
nrsw fit-images --manifest data/roles.jsonl --layout data/layout.json --output runs/card --device cuda --allow-download --fpr 0.05
nrsw predict-image --image data/submission.png --layout data/layout.json --model runs/card/model --calibration runs/card/cuts.json --device cuda --fpr 0.05
```

首次运行会下载指定版本的 Stable Diffusion 1.5 和 MobileNetV3-Small 权重。使用 CPU 时，将两条命令中的设备选项改为 `--device cpu`。离线运行时，在两条命令中传入 `--sd-snapshot` 和 `--mobile-weights`。训练会保存图像特征、模型和校准阈值。清单包含留出的测试数据时，也会输出评估结果。分数高于阈值的文件会被标记为待复核。

NRSW 参照真实文件各字段的自然变化，衡量某张文件的字段有多异常，并用这些证据分配字段权重及保留最强的证据分数。加上 `--field-classifiers` 可以[单独对字段评分](docs/field-classifiers.md)。可在目标误报率下，用验证数据比较两种检测器。

[Python API 示例](README.md#use-the-python-api)演示如何从图像特征训练和检测。[多版式指南](docs/layouts.md)介绍如何让多种版式共用图像编码器，并为每种版式保存自己的字段设置、模型和阈值。

## 复现论文结果

[复现指南](docs/reproduction.md)提供五项 CPU 分数重播流程，使用另行取得的研究数据包，也包含 Figure 10 与补充 Figure S6 以加入字段分类器的 NRSW 为基准的比较。[原始图像复现指南](docs/raw-reproduction.md)说明如何使用配套源码和数据，下载图像、提取特征、训练并评估主要实验。

## 许可与引用

代码采用 [MIT 许可证](LICENSE)。引用信息见 [CITATION.cff](CITATION.cff)，数据和权重的许可见[第三方声明](THIRD_PARTY_NOTICES.md)。
