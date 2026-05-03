# 词嵌入向量化方法、谷歌BERT预训练模型与支持向量分类器SVC
import os
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import  precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel

import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. 配置部分 ---

# 定义类别和对应的文件夹路径
categories = {
    '玄幻': 'xuanhuan',
    '历史': 'lishi',
    '都市': 'dushi',
    '科幻': 'kehuan'
}

# BERT模型路径
model_path = "C:/Users/bb287/Desktop/novel_clf/models/bert-base-chinese"


# 模型保存路径，默认保存在桌面
def get_desktop_path():
    """获取桌面路径"""
    return os.path.join(os.path.expanduser("~"), "Desktop")


classifier_save_path = os.path.join(get_desktop_path(), "text_clf_model.pkl")


# --- 2. 核心功能函数 ---

def load_data(data_folders):
    """从文件夹加载文本数据"""
    print("开始加载数据...")
    data = []
    for category, folder in data_folders.items():
        if not os.path.isdir(folder):
            print(f"警告: 文件夹 '{folder}' 不存在，跳过类别 '{category}'。")
            continue
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        text = file.read()
                        data.append({'text': text, 'label': category})
                except Exception as e:
                    print(f"读取文件 {filepath} 出错: {e}")
    if not data:
        raise FileNotFoundError("未加载到任何数据，请检查文件夹路径和文件内容。")
    print(f"数据加载完成，共 {len(data)} 条。")
    return pd.DataFrame(data)


def extract_bert_features(texts, tokenizer, model, device, max_length=512):
    """
    使用BERT模型提取文本特征，自动处理长文本。
    对于长文本，采用分块、分别提取特征、再取平均值的方法。

    参数:
        texts (pd.Series or list): 待处理的文本列表。
        tokenizer: BERT分词器。
        model: BERT模型。
        device: 'cuda' 或 'cpu'。
        max_length: BERT模型最大处理长度，上限为512。

    返回:
        np.array: 特征向量数组。
    """
    model.eval()
    all_features = []

    # 确保 texts 是一个可迭代的列表
    if isinstance(texts, pd.Series):
        text_list = texts.tolist()
    else:
        text_list = texts

    for text in tqdm(text_list, desc="提取特征"):
        # 1. 对整个文本进行分词
        # 使用 add_special_tokens=False，因为我们将手动为每个块添加特殊token
        input_ids = tokenizer.encode(text, add_special_tokens=False)

        # 2. 将token IDs分割成块
        # 每个块的长度为 max_length - 2，为 [CLS] 和 [SEP] 留出位置
        chunk_size = max_length - 2
        chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        chunks = chunks[:50]  # 块数上限为50，上调此数应该能提升分类正确率

        if not chunks:  # 处理空文本的情况
            # 使用一个零向量作为其特征
            chunk_vectors = [np.zeros((1, model.config.hidden_size))]
        else:
            chunk_vectors = []
            with torch.no_grad():
                for chunk in chunks:
                    # 3. 为每个块添加 [CLS] 和 [SEP]
                    token_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]

                    # 转换为模型输入
                    inputs = torch.tensor([token_ids]).to(device)
                    attention_mask = torch.ones_like(inputs)  # 创建注意力掩码

                    # 4. 提取 [CLS] 向量
                    outputs = model(input_ids=inputs, attention_mask=attention_mask)
                    cls_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    chunk_vectors.append(cls_vector)

        # 5. 对所有块的 [CLS] 向量取平均值，作为整个文本的特征
        # np.squeeze移除不必要的维度，然后对第一个轴（块的数量）求平均
        avg_vector = np.mean(np.squeeze(np.array(chunk_vectors), axis=1), axis=0)
        all_features.append(avg_vector)

    return np.array(all_features)


def train_and_save_model(classifier_path):
    """
    执行完整的模型训练流程，并保存训练好的分类器。
    """
    print("--- 开始训练新模型 ---")

    # 加载数据
    df = load_data(categories)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # 加载BERT模型和分词器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained(model_path).to(device)

    # 提取训练集和测试集的BERT特征
    print("正在提取训练集BERT特征...")
    X_train_bert = extract_bert_features(X_train, tokenizer, bert_model, device)

    print("\n正在提取测试集BERT特征...")
    X_test_bert = extract_bert_features(X_test, tokenizer, bert_model, device)

    # 训练SVM分类器
    print("\n正在训练SVM分类器...")
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train_bert, y_train)
    print("分类器训练完成。")

    # 在测试集上评估模型
    print("\n--- 模型评估报告 ---")
    y_pred = clf.predict(X_test_bert)

    # 计算精确率、召回率和F1分数，设置zero_division=1以避免未定义指标警告
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    print(f"精确率（Precision）: {precision:.4f}")
    print(f"召回率（Recall）: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 保存训练好的SVM分类器
    try:
        joblib.dump(clf, classifier_path)
        print(f"\n分类器已成功保存到: {classifier_path}")
    except Exception as e:
        print(f"保存分类器失败: {e}")


def classify_text(text, tokenizer, bert_model, svm_classifier, device):
    """
    使用加载好的模型对单条新文本进行分类 。
    """
    print(f"\n正在对文本进行分类: '{text[:50]}...'")

    # 提取特征，该函数现在内部处理长文本
    text_vector = extract_bert_features(
        [text],  # 需要以列表形式传入
        tokenizer,
        bert_model,
        device
    )

    # 进行预测
    probabilities = svm_classifier.predict_proba(text_vector)

    # 获取置信度最高的类别
    pred_index = np.argmax(probabilities)
    pred_class = svm_classifier.classes_[pred_index]
    pred_prob = probabilities[0, pred_index]

    print(f"分类结果: {pred_class} (置信度: {pred_prob:.2f})")
    return pred_class


def classify_file(filepath, tokenizer, bert_model, svm_classifier, device):
    """
    读取一个txt文件并对其内容进行分类。
    """
    print(f"\n--- 正在读取文件进行分类: {filepath} ---")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text_to_classify = f.read()
        if not text_to_classify.strip():
            print("错误: 文件为空或只包含空白字符。")
            return None
        # 调用核心分类函数
        return classify_text(text_to_classify, tokenizer, bert_model, svm_classifier, device)
    except FileNotFoundError:
        print(f"错误: 文件未找到于路径 '{filepath}'")
        return None
    except Exception as e:
        print(f"读取或处理文件时出错: {e}")
        return None


# --- 3. 主程序入口 ---
if __name__ == "__main__":
    # 检查模型文件是否存在
    if not os.path.exists(classifier_save_path):
        print(f"未找到.pkl格式的分类器模型 '{classifier_save_path}'。")
        train_and_save_model(classifier_save_path)
    else:
        print(f"已找到模型: {classifier_save_path}，将跳过训练，直接加载模型进行分类。")

    # --- 后续分类任务 ---
    if os.path.exists(classifier_save_path):
        print("\n--- 进入文本分类模式 ---")

        print("正在加载BERT模型和已保存的分类器...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bert_tokenizer = BertTokenizer.from_pretrained(model_path)
            bert_model_for_inference = BertModel.from_pretrained(model_path).to(device)
            svm_classifier_loaded = joblib.load(classifier_save_path)
            print("所有模型组件加载成功！")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("程序将退出。请检查BERT模型路径或删除损坏的.pkl文件后重试。")
            exit()

        # 预测一个长文本文件
        long_text_file_path = os.path.join(get_desktop_path(), "novel.txt")
        if not os.path.exists(long_text_file_path):
            print(f"\n将在桌面创建测试文件 'novel.txt'，请将待分类小说写入该文件。")
            with open(long_text_file_path, 'w', encoding='utf-8') as file:
                file.write("请在这里写入小说的内容。\n")

        # 调用函数对novel.txt进行分类，并保存分类结果
        pred_class = classify_file(long_text_file_path, bert_tokenizer, bert_model_for_inference, svm_classifier_loaded, device)

        # 词云可视化部分
        # 1. 读取桌面上的novel.txt文件
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        novel_path = os.path.join(desktop_path, 'novel.txt')

        with open(novel_path, 'r', encoding='utf-8') as f:
            novel_text = f.read()

        # 2. 使用jieba进行中文分词并统计词频
        words = jieba.lcut(novel_text)  # 精确模式分词
        word_counts = Counter(words)  # 统计词频

        # 3. 读取同目录下的停用词表
        stopwords_path = 'Stop.txt'
        with open(stopwords_path, 'r', encoding='gbk') as f:
            stopwords = set([line.strip() for line in f.readlines()])

        # 4. 过滤停用词并筛选长度大于1的词语
        filtered_words = {
            word: count for word, count in word_counts.items()
            if word not in stopwords and len(word) > 1  # 移除单字和停用词
        }

        # 5. 确保分类词是词云中最大的词
        if pred_class:
            # 获取当前最大词频
            current_max = max(filtered_words.values()) if filtered_words else 0

            # 设置分类词的词频为当前最大值加一
            # 即使分类词不在原始文本中，也会强制添加到词云
            filtered_words[pred_class] = current_max + 1

            print(f"已将分类词 '{pred_class}' 设置为最大词（词频: {filtered_words[pred_class]}）")

        # 6. 生成词云图
        wc = WordCloud(
            font_path='simhei.ttf',  # 使用黑体
            background_color='white',  # 白色背景
            max_words=200,  # 最多显示200个词
            width=1000, height=800,  # 图像尺寸
            # prefer_horizontal=1,  # 优先水平显示
            relative_scaling=0.3  # 非高频词缩放比例（突出最大词）
        )

        # 生成词云
        wc.generate_from_frequencies(filtered_words)

        # 显示和保存词云图
        plt.figure(figsize=(12, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

        # 保存到当前目录
        wc.to_file(f'wordcloud_{pred_class}.png')  # 文件名包含分类名
        print(f"词云图已保存为 'wordcloud_{pred_class}.png'")

    else:
        print("\n错误：训练失败或模型未能保存，无法进行分类。")
