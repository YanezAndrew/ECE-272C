"""Convert Report.md to Report.pdf using weasyprint."""
import os
import re
import base64
import markdown

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH    = os.path.join(REPORT_DIR, "Report.md")
PDF_PATH   = os.path.join(REPORT_DIR, "Report.pdf")


def embed_images(html: str, base_dir: str) -> str:
    """Replace relative img src paths with base64 data URIs."""
    def replacer(m):
        src = m.group(1)
        if src.startswith("http"):
            return m.group(0)
        img_path = os.path.join(base_dir, src)
        if not os.path.exists(img_path):
            return m.group(0)
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(src)[1].lstrip(".").lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f'src="data:{mime};base64,{data}"'
    return re.sub(r'src="([^"]+)"', replacer, html)


def render_mermaid_as_text(md: str) -> str:
    """Replace mermaid code blocks with a styled pre block (weasyprint can't render mermaid)."""
    def replacer(m):
        code = m.group(1).strip()
        return f'<div class="mermaid-block"><pre>{code}</pre></div>'
    return re.sub(r"```mermaid\n(.*?)```", replacer, md, flags=re.DOTALL)


CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm;
}
body {
    font-family: -apple-system, "Segoe UI", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1a1a1a;
    max-width: 100%;
}
h1 { font-size: 20pt; border-bottom: 2px solid #6366f1; padding-bottom: 6px; color: #111; }
h2 { font-size: 15pt; border-bottom: 1px solid #d1d5db; padding-bottom: 4px; color: #1e3a5f; margin-top: 28px; }
h3 { font-size: 12pt; color: #374151; margin-top: 18px; }
h4 { font-size: 11pt; color: #4b5563; }
code {
    background: #f3f4f6;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 9pt;
    font-family: "SF Mono", "Fira Code", monospace;
}
pre {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.5;
    font-family: "SF Mono", "Fira Code", monospace;
}
pre code { background: none; padding: 0; }
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 9.5pt;
    margin: 12px 0;
}
th {
    background: #6366f1;
    color: white;
    padding: 7px 10px;
    text-align: left;
}
td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; }
tr:nth-child(even) td { background: #f9fafb; }
img {
    max-width: 100%;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin: 10px 0;
}
blockquote {
    border-left: 4px solid #6366f1;
    margin: 0;
    padding: 6px 16px;
    color: #374151;
    background: #f5f3ff;
}
a { color: #6366f1; }
.mermaid-block {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    padding: 12px;
    margin: 12px 0;
}
.mermaid-block pre {
    background: none;
    border: none;
    margin: 0;
    padding: 0;
    font-size: 8pt;
}
hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }
"""


def build():
    with open(MD_PATH, "r") as f:
        md_text = f.read()

    # pre-process mermaid blocks
    md_text = render_mermaid_as_text(md_text)

    # convert markdown → html
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "attr_list"],
    )

    # embed images
    html_body = embed_images(html_body, REPORT_DIR)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    from weasyprint import HTML
    HTML(string=full_html, base_url=REPORT_DIR).write_pdf(PDF_PATH)
    size_kb = os.path.getsize(PDF_PATH) // 1024
    print(f"Saved: {PDF_PATH}  ({size_kb} KB)")


if __name__ == "__main__":
    build()
