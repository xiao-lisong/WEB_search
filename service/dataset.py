import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
from scipy.sparse import vstack
import re
# 将输入的文本分割成词语列表
def jieba_tokenizer(text):
    text = re.sub(r'[_\s]+', ' ', text)
    text = re.sub(r'[-\s]+', ' ', text)
    tokens = list(jieba.cut(text))
    #print(f"Tokens: {tokens}")
    return tokens

def load_dataset(file):
    # 加载数据集
    #docs_df = pd.read_csv(file, sep="\t", names=["doc_id", "url", "title", "content"])

    chunksize = 1000  # 每次读取的行数
    total_lines = sum(1 for _ in open(file))  # 计算文件总行数（用于进度条）

    # 进度条显示
    with tqdm(total=total_lines, desc="Loading File") as pbar:
        chunks = []
        for chunk in pd.read_csv(file, sep="\t", names=["doc_id", "url", "title", "content"], chunksize=chunksize, on_bad_lines='skip'):
            chunks.append(chunk)
            pbar.update(len(chunk))  # 更新进度条

    # 合并所有读取的块
    docs_df = pd.concat(chunks, ignore_index=True)
    print(f"File loaded successfully with {len(docs_df)} rows.")
    docs_df['url'] = docs_df['url'].fillna("") 
    docs_df['content'] = docs_df['content'].fillna("") 
    docs_df['title'] = docs_df['title'].fillna("")
    # 创建 TF-IDF 向量化器
    docs_df['combined_text'] = docs_df['url'] + " " + docs_df['title'] + " " + docs_df['content']
    # print(docs_df['combined_text'])

    vectorizer = TfidfVectorizer(stop_words=None, tokenizer=jieba_tokenizer)
    # 将所有文档合并为一个列表，避免在每次迭代中单独转换
    documents = docs_df['combined_text'].tolist()

    # 使用 tqdm 显示进度
    tfidf_matrix = []  # 存储每个批次的结果
    batch_size = 1000
    vectorizer.fit(documents)
    with tqdm(total=len(documents), desc="Vectorizing Documents") as pbar:
        for i in range(0, len(documents), batch_size):
            # 提取当前批次
            batch = documents[i:i + batch_size]
            # 向量化当前批次
            batch_matrix = vectorizer.transform(batch)
            # 将批次结果存储到列表
            tfidf_matrix.append(batch_matrix)
            # 每处理一个批次更新进度条
            pbar.update(len(batch))
    
    # 将所有批次合并成一个稀疏矩阵
    tfidf_matrix = vstack(tfidf_matrix)

    # 最终的 TF-IDF 矩阵
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    return docs_df, vectorizer, tfidf_matrix