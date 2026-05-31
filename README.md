# NovelCLF

NovelCLF 是一个中文小说类型分类项目，当前支持四个类别：**都市、科幻、历史、玄幻**。

这个项目的定位不是只追求一个高分结果，而是把一个机器学习项目从数据、特征、模型、评估、预测到 Web 展示完整走一遍。它适合作为机器学习入门后的第一个可展示作品。

## 项目结构

```
NovelCLF/
├── src/                         # 核心模块
│   ├── config.py                #   集中配置（路径、模型名、常量）
│   ├── preprocess.py            #   数据加载与文本分块
│   ├── features.py              #   特征提取（TF-IDF、BERT 嵌入）
│   ├── train.py                 #   模型训练与推理
│   ├── evaluate.py              #   评估指标与分类报告
│   ├── predict.py               #   预测（单段/文件/整书投票）
│   ├── explain.py               #   特征重要性分析
│   ├── visualize.py             #   词云、停用词加载
│   ├── persistence.py           #   模型持久化
│   └── simple_nb.py             #   纯 Python 实现的 Naive Bayes（免 scikit-learn 回退）
├── scripts/                     # 命令行入口
│   ├── build_dataset.py         #   构建段落级数据集
│   ├── train_model.py           #   训练并评估模型
│   ├── predict_text.py          #   预测文本或文件（支持 --voting 整书投票）
│   ├── cross_validate.py        #   Leave-One-Book-Out 交叉验证
│   └── plot_learning_curve.py   #   学习曲线绘制
├── app.py                       # Streamlit 交互式看板（预测/对比/数据/洞察）
├── models/
│   ├── bert-base-chinese/       #   预训练 BERT 模型
│   ├── nb_tfidf.joblib          #   已训练的 Naive Bayes 模型
│   ├── nb_word.joblib           #   已训练的分词版 Naive Bayes
│   ├── svm_tfidf.joblib         #   已训练的 SVM 模型
│   └── svm_tfidf_word.joblib    #   已训练的分词版 SVM
├── data/
│   ├── raw/                     #   原始小说（四个分类目录）
│   │   ├── dushi/
│   │   ├── kehuan/
│   │   ├── lishi/
│   │   └── xuanhuan/
│   ├── processed/               #   处理后数据
│   │   ├── novel_chunks.csv     #     段落级数据集（已生成）
│   │   └── split_report.json    #     分块报告
│   └── reports/                 #   报告（预留）
├── outputs/
│   ├── metrics/                 #   实验指标 JSON（16 个文件）
│   ├── figures/                 #   混淆矩阵图（8 张）
│   └── wordclouds/              #   词云图（预留）
├── archive/                     # 旧版原型（main.py / nb.py）
├── requirements.txt
├── 停用词表.txt
├── CHANGELOG.md
└── TUTORIAL.md                  # 入门教程
```

## 当前能力

- 将整本小说切分为段落级样本（`build_dataset.py`）
- 生成可复现的 `novel_chunks.csv` 数据集
- 训练 **5 种模型**：TF-IDF（字符/分词）+ Naive Bayes、TF-IDF（字符/分词）+ Linear SVM、BERT + SVM
- 在未安装 scikit-learn 时自动使用纯 Python 实现的 Naive Bayes 回退
- 支持 **chunk_random**（快速验证）和 **book_holdout**（按书隔离，防止数据泄漏）两种评估策略
- 支持 **Leave-One-Book-Out 交叉验证**（`cross_validate.py`）
- 支持 **学习曲线** 诊断过拟合/欠拟合（`plot_learning_curve.py`）
- 支持 **整本书多数投票预测**（`--voting` 参数）
- 输出指标 JSON 和混淆矩阵图片
- 提供 Streamlit 交互式看板（`app.py`），含预测、模型对比、数据集探索、实验洞察四面板
- 提取特征重要性，分析模型依赖的关键词

## 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 构建数据集

```bash
python scripts/build_dataset.py --chunk-size 1000 --max-chunks-per-book 400
```

输出：
- `data/processed/novel_chunks.csv`
- `data/processed/split_report.json`

## 训练模型

先训练最快的 baseline：

```bash
python scripts/train_model.py --model nb --split chunk_random
```

再做更严格的按书切分评估：

```bash
python scripts/train_model.py --model svm_tfidf --split book_holdout
```

可选模型：`nb`, `nb_word`, `svm_tfidf`, `svm_tfidf_word`, `bert_svm`

## 交叉验证与学习曲线

```bash
# Leave-One-Book-Out 交叉验证
python scripts/cross_validate.py --model nb

# 绘制学习曲线
python scripts/plot_learning_curve.py --model nb --split book_holdout --steps 5
```

## 预测文本

```bash
python scripts/predict_text.py --model nb --text "星舰穿过虫洞，远方殖民地传来最后一段求救信号。"

# 整本投票预测
python scripts/predict_text.py --model nb --file novel.txt --voting

# 单文件预测
python scripts/predict_text.py --model nb --file example.txt
```

## 启动 Web 应用

```bash
streamlit run app.py
```

## 推荐学习路线

1. 先运行 `build_dataset.py`，理解为什么要把小说分块。
2. 训练 `nb`，把它当作第一个基线。
3. 训练 `svm_tfidf`，观察传统模型提升。
4. 对比 `chunk_random` 和 `book_holdout` 的指标差异，理解**数据泄漏**的危害。
5. 运行 `cross_validate.py` 做严谨的 LOBO 评估。
6. 运行 `plot_learning_curve.py` 诊断过拟合。
7. 再尝试 `bert_svm`，理解预训练模型如何作为特征提取器。
8. 最后启动 `app.py`，把项目整理成可展示作品。

## 注意事项

- 原始小说文件（`data/raw/`）较大，不建议提交到公开仓库。
- 生成的数据集、模型文件和输出图表默认被 `.gitignore` 忽略。
- 旧版原型代码位于 `archive/` 目录，仅供参考。
- 如果 Windows 控制台中中文显示成乱码，但 JSON/CSV 文件本身正常，可以在 PowerShell 中先执行：
  ```powershell
  $env:PYTHONUTF8=1
  ```
