手动补充 PDF 的推荐目录。

建议把你自己下载的论文 PDF 放在这里，例如：

- `manual_pdfs/triton.pdf`
- `manual_pdfs/roofline.pdf`

然后用：

```bash
python3 main.py --config config/config-poe.json paper-set-pdf \
  --paper-id 123 \
  --pdf-url "file:///home/momo/git_ws/search/manual_pdfs/your_paper.pdf"
```

再执行：

```bash
python3 main.py --config config/config-ikun.json enrich --paper-id 123 --with-llm --workers 1 --force
```

说明：

- 这里存的是你手动维护的原始 PDF。
- 程序在 `enrich` 时会把 PDF 复制到自己的缓存目录 `artifacts/pdfs/`。
- 不建议直接手动往 `artifacts/pdfs/` 里塞文件。
