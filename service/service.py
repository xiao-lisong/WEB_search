from dataset import load_dataset
from webapi import service, update_env
from flask import Flask
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

if __name__ == "__main__":
    # 初始化 Flask 应用
    app = Flask(__name__)

    # 加载数据到全局
    docs_df, vectorizer, tfidf_matrix = load_dataset("MSMARCO/subset100000_msmarco-docs.tsv")

    # 传递到蓝图
    update_env(docs_df, vectorizer, tfidf_matrix)

    # 注册蓝图
    app.register_blueprint(service)
    # print(docs_df)
    app.run(debug=True)
