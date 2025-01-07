# import requests
# url = 'https://fanyi.baidu.com/sug'
# data = {'kw': 'apple'} # 你只需要改kw对应的值
# res = requests.post(url, data=data).json()
# print(res['data'][0]['v'])
#
import requests
import time
import hashlib
import uuid
youdao_url = 'https://openapi.youdao.com/api'  # 有道api地址
app_key = "uxrIpJUaxprcn4tvQBUFMSlMXRbzOfnI"  # 应用密钥
app_id = "746c5a9d6c19a99d"  # 应用id

def translate_youdao(translate_text):
    # 翻译文本生成sign前进行:的处理
    input_text = ""

    # 当文本长度小于等于20时，取文本
    if (len(translate_text) <= 20):
        input_text = translate_text

    # 当文本长度大于20时，进行特殊处理
    elif (len(translate_text) > 20):
        input_text = translate_text[:10] + str(len(translate_text)) + translate_text[-10:]

    time_curtime = int(time.time())  # 秒级时间戳获取

    uu_id = uuid.uuid4()  # 随机生成的uuid数，为了每次都生成一个不重复的数。
    sign = hashlib.sha256(
        (app_id + input_text + str(uu_id) + str(time_curtime) + app_key).encode('utf-8')).hexdigest()  # sign生成

    data = {
        'q': translate_text,  # 翻译文本
        'from': "en",  # 源语言
        'to': "zh-CHS",  # 翻译语言
        'appKey': app_id,  # 应用id
        'salt': uu_id,  # 随机生产的uuid码
        'sign': sign,  # 签名
        'signType': "v3",  # 签名类型，固定值
        'curtime': time_curtime,  # 秒级时间戳
    }

    r = requests.get(youdao_url, params=data).json()  # 获取返回的json()内容
    return r["translation"][0]
import requests
import json

# 全局变量配置
API_URL = "http://www.trans-home.com/api/index/translate"
TOKEN = "szjsaFIeoBUgoJ1yslzq" # 请替换为实际的 API token

def translate(keywords, target_language):
    """
    调用第三方翻译 API 进行文本翻译

    :param keywords: 需要翻译的文本
    :param target_language: 目标语言代码，如 'en' (英语), 'de' (德语) 等
    :return: 翻译后的文本或错误信息
    """
    # 构建请求的完整 URL
    url = f"{API_URL}?token={TOKEN}"

    # 构建请求数据
    payload = json.dumps({
        "keywords": keywords,          # 需要翻译的文本
        "targetLanguage": target_language  # 目标语言
    })

    # 请求头设置
    headers = {
        'Content-Type': 'application/json'  # 设置请求数据格式为 JSON
    }

    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=payload)

    # 如果请求成功，处理返回的 JSON 数据
    if response.status_code == 200:
        result = response.json()  # 解析返回的 JSON 数据

        # 判断翻译是否成功
        if result.get("code") == 1:
            # 翻译成功，返回翻译后的文本
            return result["data"]["text"]
        else:
            # 翻译失败，返回错误信息
            return f"Error: {result.get('info')}"
    else:
        # 请求失败，返回 HTTP 错误状态码
        return f"Request failed with status code: {response.status_code}"

# 示例使用
if __name__ == "__main__":
    # translated_text = translate("cat", "zh-CHS")
    # print(translated_text)  # 打印翻译结果
    # print({})
    translate_youdao("cat")