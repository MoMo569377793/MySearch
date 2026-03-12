#!/usr/bin/env python3
# test_pdf_api.py

import base64, requests, json, sys

API_URL = ""
API_KEY = ""
MODEL   = "gpt-5.4"
PDF_FILE = "artifacts/pdfs/2504.16214v3.pdf"

with open(PDF_FILE, "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode()

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

tests = {
    "file字段": {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "test.pdf", "file_data": f"data:application/pdf;base64,{pdf_b64}"}},
            {"type": "text", "text": "用一句话总结PDF"}
        ]}]
    },
    "image_url字段": {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_b64}"}},
            {"type": "text", "text": "用一句话总结PDF"}
        ]}]
    },
    "document字段": {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
            {"type": "text", "text": "用一句话总结PDF"}
        ]}]
    },
}

for name, payload in tests.items():
    print(f"\n{'='*50}")
    print(f"📋 测试: {name}")
    try:
        r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        print(f"状态码: {r.status_code}")
        print(f"原始响应: {r.text[:500]}")
        data = r.json()
        if r.status_code == 200 and "choices" in data:
            content = data["choices"][0]["message"]["content"]
            print(f"✅ 成功！响应: {content[:200]}")
        else:
            error = data.get("error", {}).get("message", json.dumps(data, ensure_ascii=False)[:200])
            print(f"❌ 失败 [{r.status_code}]: {error}")
    except Exception as e:
        print(f"❌ 异常: {e}")