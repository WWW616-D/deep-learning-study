"""提取 Decision-Based Adversarial Attacks (Boundary Attack) 论文全文."""
import PyPDF2

src = r"D:\文献资料\对抗攻击相关\Decision_Based Adversarial Attacks_Reliable Attacks Against Black_Box Machine Learning Models.pdf"
dst = r"D:\py\_scratch\research\boundary_attack_fulltext.txt"

reader = PyPDF2.PdfReader(src)
print("页数:", len(reader.pages))
out = []
for i, page in enumerate(reader.pages):
    try:
        t = page.extract_text() or ""
    except Exception as e:
        t = f"[error: {e}]"
    out.append(f"\n----- PAGE {i+1} -----\n{t}")
text = "".join(out)
with open(dst, "w", encoding="utf-8") as f:
    f.write(text)
print("写入:", dst, len(text), "字符")
