# NovelCLF

NovelCLF 是一个中文小说类型分类学习项目，当前支持四个类别：都市、科幻、历史、玄幻。

这个项目的定位不是只追求一个高分结果，而是把一个机器学习项目从数据、特征、模型、评估、预测到 Web 展示完整走一遍。它适合作为机器学习入门后的第一个可展示作品。

## 当前能力

- 将整本小说切分为段落级样本。
- 生成可复现的 `novel_chunks.csv` 数据集。
- 训练 TF-IDF + Naive Bayes 基线模型。
- 在未安装 scikit-learn 时，`nb` 会自动使用一个标准库实现的字符 n-gram Naive Bayes，方便先跑通全流程。
- 训练 TF-IDF + Linear SVM 传统进阶模型。
- 预留 BERT + SVM 语义特征模型。
- 支持命令行预测文本或 `.txt` 文件。
- 输出指标 JSON 和混淆矩阵图片。
- 提供 Streamlit 交互式演示雏形。

## 安装依赖

建议使用虚拟环境：

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

说明：

- `chunk_random` 会随机切分段落，适合快速检查流程是否跑通。
- `book_holdout` 会把整本书作为测试来源，更适合观察真实泛化能力。

## 预测文本

```bash
python scripts/predict_text.py --model nb --text "星舰穿过虫洞，远方殖民地传来最后一段求救信号。"
```

预测文件：

```bash
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
4. 对比 `chunk_random` 和 `book_holdout` 的指标差异。
5. 再尝试 `bert_svm`，理解预训练模型如何作为特征提取器。
6. 最后完善 `app.py` 和 `TUTORIAL.md`，把项目整理成作品。

## 注意事项

当前原始小说文件较大，不建议直接提交到公开仓库。生成的数据集、模型文件和输出图表也默认被 `.gitignore` 忽略。

如果 Windows 控制台中中文显示成乱码，但 JSON/CSV 文件本身正常，可以在 PowerShell 中先执行：

```powershell
$env:PYTHONUTF8=1
```
