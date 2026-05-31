# TODO · NovelCLF 下一步开发计划

## 🟢 近期（核心能力增强）

### 1. ✅ BERT 端到端微调
- ~~当前 `bert_svm` 仅将 BERT 作为固定特征提取器 + SVM 分类器~~
- 已实现 `src/bert_finetune.py`：完整的 BertForSequenceClassification 微调流水线
- 支持 `--model bert_finetune` 参数，包含冻结/解冻策略、AdamW + warmup 调度器
- 已集成至 `train.py`、`predict.py`、`app.py`、CLI

### 2. ✅ BERT 特征缓存持久化
- ~~`data/processed/bert_embeddings.npy` + `bert_embedding_meta.csv` 已在 config.py 定义路径但未实现持久化逻辑~~
- 已实现 `train.py` 中自动缓存（训练时检测并加载 `.npy`）
- 已创建 `scripts/extract_embeddings.py` 独立特征提取入口（含 tqdm 进度条、`--force` 重建）

### 3. 更多预训练模型支持
- RoBERTa-wwm-ext（哈工大讯飞）
- MacBERT
- 通过 `src/config.py` 统一管理模型名与路径映射
- `--bert-model` 参数支持选择不同预训练模型

### 4. ✅ 超参数自动调优
- 已创建 `scripts/tune_hyperparams.py`
- TF-IDF 参数：`max_features`, `ngram_range`, `min_df` + SVM: `C`
- 使用 `GridSearchCV` + 书级 `GroupKFold`（防数据泄漏）
- 输出最佳参数组合到 `outputs/metrics/tuning_{model}.json`

---

## 🟡 中期（工程化与部署）

### 5. ✅ RESTful API
- 已创建 `api.py`：完整 FastAPI 应用，含 Pydantic 模型、CORS、OpenAPI 自动文档
- `POST /predict` — 文本预测（JSON body）
- `POST /predict/file` — 上传 .txt 文件预测（支持 `voting=true` 多数投票）
- `GET /models` — 列出可用模型及磁盘状态
- 运行：`uvicorn api:app --reload`

### 6. ✅ Docker 容器化
- 已创建 `Dockerfile`（多阶段构建，CPU-only torch，非 root 用户）
- 已创建 `Dockerfile.api`（轻量 API 专用镜像）
- 已创建 `docker-compose.yml`（编排 Streamlit:8501 + API:8000，共享模型卷）
- 已创建 `.dockerignore`（排除 .git、raw 数据、大文件）

### 7. ✅ 实验追踪
- 已创建 `src/experiment.py`：轻量 JSONL 文件追踪器（无需 MLflow 外部依赖）
- 记录每次训练的：参数、指标、模型路径、UUID、时间戳
- 已创建 `scripts/run_experiment.py`：批量实验入口（`--models all --splits all`）
- 已创建 `scripts/show_experiments.py`：查看历史记录（`--best` 显示最佳模型）

### 8. ✅ CI/CD 流水线
- 已创建 `.github/workflows/ci.yml`：
  - lint：ruff 代码检查
  - test：全模块 py_compile 编译检查 + 导入冒烟测试
- 已创建 `requirements-ci.txt`（轻量 CI 依赖，不含 torch）

---

## 🔵 长期（模型改进与数据扩展）

### 9. 长文本模型
- 当前 BERT 分块策略丢失段落间上下文
- 尝试 Longformer / BigBird / 滑动窗口融合
- 适用于整本书级别预测场景

### 10. 更多深度学习模型
- TextCNN（适合短文本分类基线）
- BiLSTM + Attention
- FastText（超轻量，适合快速实验）
- 统一评估框架，直接对比所有模型

### 11. 数据增强
- EDA（随机替换、插入、交换、删除）
- 回译增强（中→英→中）
- 基于 MLM 的上下文替换（BERT 掩码填充）
- 注意：仅对训练集增强，不污染测试集

### 12. 类别扩展与数据扩充
- 新增类别：仙侠、武侠、恐怖、悬疑、游戏、体育
- 每个类别目标 10+ 本来源书
- 来源书质量筛选（去除乱码/广告/重复内容）

---

## ⚪ 优化与完善

### 13. 类别不均衡处理
- 当前各类别样本数可能不均衡（受小说长度和分块数量影响）
- 新增选项：`--balance` 支持 SMOTE / 类别权重 / 欠采样

### 14. ✅ 嵌入空间可视化
- 已创建 `scripts/visualize_embeddings.py`
- 支持 t-SNE / PCA 降维，按真实类别着色
- 支持 BERT 缓存嵌入和 TF-IDF 特征两种模式
- 输出到 `outputs/figures/embeddings_tsne.png` / `embeddings_pca.png`

### 15. 推理加速
- ONNX 导出 BERT 模型
- 批量推理支持
- 模型量化（FP16 / INT8）

### 16. Web 部署
- 部署到 HuggingFace Spaces
- 部署到 Vercel / Railway
- `streamlit` + `FastAPI` 双模式切换

---

## 执行优先级建议

```
第一优先级: 1 (BERT微调) → 2 (缓存) → 4 (调优)
第二优先级: 5 (API) → 6 (Docker) → 7 (实验追踪)
第三优先级: 9 (长文本) → 10 (更多模型) → 11 (数据增强)
持续进行: 12 (扩充数据) + 13-16 (优化与工程)
```
