import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 将输入的文本分割成词语列表
def jieba_tokenizer(text):
    tokens = list(jieba.cut(text))
    #print(f"Tokens: {tokens}")
    return tokens

def load_dataset(file):
    # 加载数据集
    docs_df = pd.read_csv(file, sep="\t", names=["doc_id", "url", "title", "content"])
    docs_df['content'] = docs_df['content'].fillna("") 
    docs_df['title'] = docs_df['title'].fillna("")
    # 创建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer(stop_words=None, tokenizer=jieba_tokenizer)
    docs_df['combined_text'] = docs_df['title'] + " " + docs_df['content']
    tfidf_matrix = vectorizer.fit_transform(docs_df['combined_text'])
    return docs_df, vectorizer, tfidf_matrix