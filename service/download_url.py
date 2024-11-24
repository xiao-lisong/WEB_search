import requests
from bs4 import BeautifulSoup

# 设置 User-Agent 头部模拟浏览器
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 读取 URL 列表文件
with open("MSMARCO/top_url.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f.readlines()]

# 保存输出文件的路径
output_file = "output_data.txt"

# 定义请求函数
def fetch_data(idx, url):
    try:
        # 发送请求获取页面内容，设置超时防止卡住
        response = requests.get(f"http://{url}", headers=headers, timeout=10)
        response.raise_for_status()  # 确保请求成功

        # 使用正确的编码
        response.encoding = response.apparent_encoding

        # 使用 BeautifulSoup 解析页面
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取标题，避免 title 为 None 的情况
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        
        # 提取页面内容并清理换行符
        content = soup.get_text().strip().replace("\n", " ").replace("\r", " ")[:1000]  # 取前1000字符，并清理换行符

        # 返回处理后的数据
        return f"{idx}\t{url}\t{title}\t{content}\n"
    except requests.exceptions.Timeout:
        return f"{idx}\t{url}\tTimeout\tThe request timed out.\n"
    except requests.exceptions.RequestException as e:
        return f"{idx}\t{url}\tError\t{str(e)}\n"

# 打开输出文件准备写入，指定 utf-8 编码
with open(output_file, "a", encoding="utf-8") as out_f:
    for idx, url in enumerate(urls):
        result = fetch_data(idx, url)
        out_f.write(result)

print(f"数据已保存到 {output_file}")
