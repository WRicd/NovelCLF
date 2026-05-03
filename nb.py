# TF-IDF向量化方法与朴素贝叶斯分类器
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  precision_score, recall_score, f1_score

# 定义类别和对应的文件夹
categories = {
    '玄幻': 'xuanhuan',
    '历史': 'lishi',
    '都市': 'dushi',
    '科幻': 'kehuan'
}

# 加载数据
data = []
for category, folder in categories.items():
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                data.append({'text': text, 'label': category})

# 转换为DataFrame
df = pd.DataFrame(data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 训练朴素贝叶斯分类器
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test_tfidf)

# 计算精确率、召回率和F1分数，设置zero_division=1以避免未定义指标警告
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

# 对新文本进行分类
def classify_text(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

# 分类测试
new_text = "这是待分类的文本。"
print("分类结果：", classify_text(new_text))