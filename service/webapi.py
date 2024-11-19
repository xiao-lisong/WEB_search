from flask import Blueprint, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import math
# 创建蓝图对象
service = Blueprint("service", __name__)

vectorizer = None
tfidf_matrix = None
docs_df = None

def update_env(__docs_df, __vectorizer, __tfidf_matrix):
    global docs_df, vectorizer, tfidf_matrix
    docs_df = __docs_df
    vectorizer = __vectorizer
    tfidf_matrix = __tfidf_matrix

def replace_nan_with_null(data):
    """递归替换字典或列表中的 NaN 和 Infinity 值"""
    if isinstance(data, dict):
        return {k: replace_nan_with_null(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_null(v) for v in data]
    elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None  # 将 NaN 或 Infinity 替换为 None
    return data

@service.route("/")
def home():
    return render_template("index.html")

@service.route('/search', methods=['POST'])
def search():
    # 获取用户的查询
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "查询不能为空"}), 400

    page = data.get('page', 1)
    per_page = 10  # 每页显示10条结果

    # 计算偏移量
    start = (page - 1) * per_page
    end = start + per_page
    print(f'quert : [{query}]')
    # print(vectorizer)
    # 将查询转化为 TF-IDF 向量
    query_tfidf = vectorizer.transform([query])

    # 计算相似度
    cos_similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    similarity_scores = cos_similarities[0]

    # 获取前 100 个相关文档
    top_indices = similarity_scores.argsort()[::-1][:100]
    results = []
    # print(docs_df)
    for idx in top_indices:
        results.append({
            "doc_id": docs_df.iloc[idx]["doc_id"],
            "url": docs_df.iloc[idx]["url"],
            "title": docs_df.iloc[idx]["title"],
            "content": docs_df.iloc[idx]["content"][:500],  # 截断内容展示
            'score': f"{similarity_scores[idx]:.3f}"
        })

    results1 = results[start:end]
    results1 = replace_nan_with_null(results1)

    response = jsonify({
        "results": results1,
        "page": page,
        "total": len(top_indices),
        "total_pages": (len(top_indices) + per_page - 1) // per_page
    })

    return response
