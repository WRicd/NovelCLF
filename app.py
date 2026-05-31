from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import DATASET_PATH, MODEL_PATHS, STOPWORDS_PATH
from src.predict import predict_text
from src.preprocess import summarize_dataset
from src.visualize import load_stopwords, top_words


import json
import pandas as pd

st.set_page_config(page_title="NovelCLF Dashboard", layout="wide", page_icon="📚")

st.title("📚 NovelCLF 中文小说分类器")
st.caption("从基础 TF-IDF 到深度学习，探索文本分类背后的数据泄漏与泛化能力。")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ 预测模型设置")
    model_name = st.selectbox(
        "选择预训练模型",
        options=["nb", "nb_word", "svm_tfidf", "svm_tfidf_word", "bert_svm", "bert_finetune"],
        format_func=lambda value: {
            "nb": "TF-IDF (字符) + Naive Bayes",
            "nb_word": "TF-IDF (分词) + Naive Bayes",
            "svm_tfidf": "TF-IDF (字符) + Linear SVM",
            "svm_tfidf_word": "TF-IDF (分词) + Linear SVM",
            "bert_svm": "BERT + SVM",
            "bert_finetune": "BERT 端到端微调",
        }[value],
    )
    st.markdown("---")
    st.info(f"📁 **当前模型路径:**\n`{MODEL_PATHS[model_name].name}`")

# --- Tabs ---
tab_predict, tab_compare, tab_data, tab_insight = st.tabs([
    "🔮 实时预测舱", 
    "📊 多维模型对比", 
    "📚 数据集探索",
    "🧠 实验洞察"
])

# --- 1. 实时预测舱 ---
with tab_predict:
    st.markdown("### 📝 输入段落或上传全本小说")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("上传 .txt 小说文件（支持全本多数投票预测）", type=["txt"])
        default_text = "星舰穿过虫洞，远方殖民地传来最后一段求救信号。他猛地推开舱门，手中的脉冲枪已经充能完毕。"
        text = st.text_area("或者直接输入片段", value=default_text, height=200)
        
        mode = st.radio("预测模式", ["单段落预测", "整本切块投票预测 (仅限上传文件)"], horizontal=True)

    with col2:
        st.markdown("### 🎯 预测结果")
        predict_placeholder = st.empty()
        chart_placeholder = st.empty()

    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        st.success(f"成功读取文件: {uploaded.name} (共 {len(text)} 字符)")

    if st.button("🚀 启动分析引擎", type="primary"):
        if not text.strip():
            st.warning("请输入文本或上传文件。")
        else:
            try:
                from src.predict import predict_book_voting
                import tempfile
                
                with st.spinner(f"正在使用 {model_name} 分析中..."):
                    if uploaded and mode == "整本切块投票预测 (仅限上传文件)":
                        # Write to temp file for predict_book_voting
                        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as tmp:
                            tmp.write(text)
                            tmp_path = Path(tmp.name)
                        
                        result = predict_book_voting(tmp_path, model_name=model_name)
                        tmp_path.unlink() # cleanup
                        
                        predict_placeholder.metric("🔥 最终投票预测", result["label"], f"置信度: {result['confidence']:.1%}")
                        chart_placeholder.bar_chart(pd.Series(result["vote_distribution"], name="投票数"))
                        st.info(f"总计切分并预测了 {result['total_chunks_predicted']} 个段落。")
                        
                    else:
                        result = predict_text(text, model_name=model_name)
                        predict_placeholder.metric("🔥 预测类别", result["label"])
                        scores = result.get("scores")
                        if scores:
                            chart_placeholder.bar_chart(pd.Series(scores, name="置信度分数"))

                # 提取高频词
                if STOPWORDS_PATH.exists() and mode == "单段落预测":
                    words = top_words(text, load_stopwords(STOPWORDS_PATH), top_n=15)
                    if words:
                        st.markdown("---")
                        st.subheader("💡 文本高频词汇")
                        st.dataframe(pd.DataFrame(words), hide_index=True, use_container_width=True)
            except Exception as exc:
                st.error(f"推理失败: {exc}")

# --- 2. 多维模型对比 ---
with tab_compare:
    st.markdown("### 🏆 实验效果对比台")
    st.caption("读取 `outputs/metrics/` 下的实验日志进行性能对比。")
    
    from src.config import METRICS_DIR, FIGURES_DIR
    
    if METRICS_DIR.exists():
        metrics_files = list(METRICS_DIR.glob("*.json"))
        records = []
        for f in metrics_files:
            # Skip feature files
            if "top_features" in f.name:
                continue
            try:
                data = json.loads(f.read_text("utf-8"))
                records.append({
                    "Model": data.get("model", f.stem),
                    "Split": data.get("split", "unknown"),
                    "Accuracy": data.get("accuracy", 0),
                    "Macro-F1": data.get("macro_f1", 0),
                    "Train Time (s)": data.get("train_seconds", 0)
                })
            except:
                pass
                
        if records:
            df = pd.DataFrame(records)
            
            # Pivot table for accuracy comparison
            if "Split" in df.columns and "Model" in df.columns:
                pivot = df.pivot_table(index="Model", columns="Split", values="Accuracy")
                st.markdown("#### 模型泛化能力落差 (Accuracy)")
                st.bar_chart(pivot)
                
            st.markdown("#### 详细评测指标")
            st.dataframe(
                df.style.format({"Accuracy": "{:.2%}", "Macro-F1": "{:.2%}", "Train Time (s)": "{:.1f}"}),
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("#### 📉 混淆矩阵探微")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**chunk_random 混淆矩阵 (过拟合)**")
                cm_random = FIGURES_DIR / f"{model_name}_chunk_random_confusion_matrix.png"
                if cm_random.exists():
                    st.image(str(cm_random), use_column_width=True)
                else:
                    st.info("该模型的随机切分混淆矩阵不存在。")
            with col2:
                st.write("**book_holdout 混淆矩阵 (真实泛化)**")
                cm_holdout = FIGURES_DIR / f"{model_name}_book_holdout_confusion_matrix.png"
                if cm_holdout.exists():
                    st.image(str(cm_holdout), use_column_width=True)
                else:
                    st.info("该模型的跨书泛化混淆矩阵不存在。")
        else:
            st.info("未找到实验日志，请先运行 `scripts/train_model.py`。")

# --- 3. 数据集探索 ---
with tab_data:
    if DATASET_PATH.exists():
        summary = summarize_dataset(DATASET_PATH)
        c1, c2, c3 = st.columns(3)
        c1.metric("📚 来源书籍数", len(summary["book_counts"]))
        c2.metric("✂️ 段落样本数", summary["total_chunks"])
        c3.metric("🏷️ 类别总数", len(summary["label_counts"]))
        
        st.markdown("---")
        st.subheader("📊 类别样本分布")
        st.bar_chart(pd.Series(summary["label_counts"]))
        
        with st.expander("查看每本书的具体切块数"):
            st.dataframe(pd.Series(summary["book_counts"], name="段落数"), use_container_width=True)
    else:
        st.info("还没有生成数据集。请先运行：`python scripts/build_dataset.py`")

# --- 4. 实验洞察 ---
with tab_insight:
    st.markdown("""
    ## 🧐 我们从实验中学到了什么？
    
    ### 1. “完美”的 100% 准确率背后的陷阱（数据泄漏）
    在使用 `chunk_random` 划分数据集时，朴素贝叶斯和 SVM 模型几乎都能达到 **99% - 100%** 的准确率。但当我们使用 `book_holdout`（按书隔离测试集）时，准确率断崖式下跌到 **14% - 40%**。
    
    **为什么？** 通过分析 `svm_tfidf_word` 的特征重要性，我们发现：
    - 都市类决定性特征：`林逸`、`苏玥`、`秦天`
    - 玄幻类决定性特征：`张若尘`、`龙皓晨`、`林动`
    
    模型根本没有学习什么是“科幻”或“都市”，它仅仅是**背下了每本书的主角名字**！当测试集中出现了没见过的新书（新主角），模型就彻底失效了。
    
    ### 2. 字符 n-gram vs 中文分词
    我们实验了 `char` 级别（字）和 `word` 级别（jieba分词）的 TF-IDF。
    有趣的是，在小说这种高度口语化、文学化且充满自创词（人名、功法）的文本中，**基于字符 n-gram 的泛化能力往往比死板的分词更好**，而且省去了分词带来的性能开销。
    
    ### 3. 跨越局限：投票机制与深度学习
    为了解决单段落信息太少导致泛化差的问题，我们引入了**整本书多数投票（Majority Voting）**机制。即使个别段落被误判，只要全书几百个段落的“群智”是对的，就能纠正偏差。
    同时，我们也预留了 `BERT` 接口，以期利用强大的上下文理解能力来真正捕捉文本的“语义氛围”，而不仅仅是字词统计。
    """)

