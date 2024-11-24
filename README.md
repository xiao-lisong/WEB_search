## 本仓库为信息检索作业 Web检索系统代码

如需自己部署，请按照以下流程运行

1.修改数据集路径，service/service.py，将all_dataset.tsv修改为该路径下已有的数据集文件
docs_df, vectorizer, tfidf_matrix = load_dataset("MSMARCO/all_dataset.tsv")

2.运行服务，python service/service.py

3.运行成功后进行访问，本地可以通过127.0.0.1:5000访问