"""提取 PixelDP 论文全文到 txt."""
import PyPDF2

src = r"D:\文献资料\对抗攻击相关\Certified Robustness to Adversarial Examples with Differential Privacy.pdf"
dst = r"D:\py\_scratch\research\pixeldp_fulltext.txt"

reader = PyPDF2.PdfReader(src)
print("页数:", len(reader.pages))
out = []
for i, page in enumerate(reader.pages):
    try:
        t = page.extract_text() or ""
    except Exception as e:
        t = f"[extract error: {e}]"
    out.append(f"\n----- PAGE {i+1} -----\n{t}")
text = "".join(out)
with open(dst, "w", encoding="utf-8") as f:
    f.write(text)
print("写入:", dst, len(text), "字符")
print(text[:1500])
