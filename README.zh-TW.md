# NRSW 文件偽造偵測

正常參照譜加權（NRSW）用真件和少量已確認的偽件，為已知版型訓練偵測器。每張訓練偽件只需標出候選竄改欄位，例如姓名或證件號碼。判定新文件時，只需提供影像和版型。

[English](README.md) · [简体中文](README.zh-CN.md) · [訓練指南](docs/training.md) · [系統架構](docs/architecture.md) · [資料格式](docs/data.md) · [資料集下載與引用](docs/datasets.md) · [復現論文結果](docs/reproduction.md)

## 試用示範

開啟[線上示範](https://riverhor.github.io/nrsw/#upload)，即可試用 200 張 [IDSpace](docs/datasets.md) 斯洛伐克身分證合成影像。可抽選未竄改樣本，或依欄位與手法挑選竄改樣本。每張影像都能查看欄位權重、偽造分數和複核判斷。[測試結果](https://riverhor.github.io/nrsw/#evaluation)另列 NRSW 與對照方法在 4,106 張留出影像上的表現。也可上傳同版型影像，即時執行偵測。

若要在本機開啟示範，請在 Python 3.11 以上的虛擬環境中，從專案目錄執行下列指令。

```console
python -m pip install .
nrsw serve
```

在瀏覽器開啟 [localhost:8000](http://127.0.0.1:8000)。也可用 `nrsw demo --output runs/example` 在本機用合成特徵訓練一個小型範例模型。

若要讓本機示範也能即時偵測，請依[上傳指南](docs/upload.md)載入已登錄的偵測器。

## 復現論文結果

依照[原始影像復現指南](docs/raw-reproduction.md)下載資料、擷取特徵、訓練及評估偵測器。[復現指南](docs/reproduction.md)列出各項實驗所需的檔案，以及如何用 CPU 和已保存的預測分數核對論文數字。

## 用自己的影像訓練

依照[訓練指南](docs/training.md)標註候選竄改欄位、定義版型座標，並將訓練與校準資料分成獨立群組。以下指令使用 CUDA。請先安裝[支援 CUDA 的 PyTorch 與 torchvision](https://pytorch.org/get-started/locally/)，並準備好 `data/roles.jsonl` 和 `data/layout.json`。

```console
python -m pip install ".[vision]"
nrsw fit-images --manifest data/roles.jsonl --layout data/layout.json --output runs/card --device cuda --allow-download --fpr 0.05
nrsw predict-image --image data/submission.png --layout data/layout.json --model runs/card/model --calibration runs/card/cuts.json --device cuda --fpr 0.05
```

首次執行會下載指定版本的 Stable Diffusion 1.5 與 MobileNetV3-Small 權重。使用 CPU 時，兩個指令都改成 `--device cpu`。離線使用時，請在兩個指令提供 `--sd-snapshot` 和 `--mobile-weights`。訓練會保存抽取後的特徵、模型及校準檔。若清單含留出的測試資料，也會輸出評估結果。偽造分數高於門檻時，文件會被標記為待複核。

NRSW 以真件的自然變異為參照，衡量各欄位的變化有多反常。文件分類器透過這項證據分配欄位權重，並保留最強的證據分數。加入 `--field-classifiers` 可[同時對個別欄位評分](docs/field-classifiers.md)。可用驗證資料比較兩種偵測器在目標誤報率下的表現。

[Python API 範例](README.md#use-the-python-api)示範如何從特徵訓練和推論。[多版型指南](docs/layouts.md)說明如何用同一組影像編碼器服務多種版型，各版型都有自己的欄位設定、模型與門檻。

## 授權與引用

NRSW 採用 [MIT 授權](LICENSE)。引用資訊見 [CITATION.cff](CITATION.cff)，資料與權重授權見[第三方聲明](THIRD_PARTY_NOTICES.md)。
