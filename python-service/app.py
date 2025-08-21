from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, re, fitz
from bs4 import BeautifulSoup
import html as html_escape

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_style(style_str: str):
    style_dict = {}
    for part in style_str.split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            style_dict[key.strip().lower()] = value.strip()
    return style_dict

def html_to_json_from_string(html_str: str):
    soup = BeautifulSoup(html_str, "lxml")
    elements = []
    for span in soup.find_all("span"):
        style = parse_style(span.get("style", ""))
        def px_to_float(v, default=0):
            try:
                return float(re.sub("px", "", v))
            except Exception:
                return default
        x = px_to_float(style.get("left", "0"))
        y = px_to_float(style.get("top", "0"))
        size = px_to_float(style.get("font-size", "12"), 12)
        font = (style.get("font-family", "") or "").replace('"', "")
        content = (span.get_text() or "").strip()
        if not content:
            continue
        elements.append({
            "type": "text", "content": content, "font": font, "size": size,
            "x": x, "y": y, "color": style.get("color"), "background": style.get("background-color"),
        })
    for img in soup.find_all("img"):
        style = parse_style(img.get("style", ""))
        def px(v):
            try:
                return float(re.sub("px", "", v))
            except Exception:
                return 0.0
        elements.append({
            "type": "image", "src": img.get("src", ""),
            "width": px(style.get("width", "0")), "height": px(style.get("height", "0")),
            "x": px(style.get("left", "0")), "y": px(style.get("top", "0")),
        })
    return elements

def pdf_to_html_with_elements(pdf_path):
    doc = fitz.open(pdf_path)
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'><title>PDF to HTML</title></head><body>",
        "<div style='position:relative; width:100%; margin:auto; background:#fff;'>"
    ]
    all_elements = []
    for page_num, page in enumerate(doc, start=1):
        w, h = page.rect.width, page.rect.height
        html_parts.append(f"<div style='position:relative; width:{w}px; height:{h}px; border-bottom:1px solid #ccc;'>")
        html_parts.append(f"<!-- Page {page_num} -->")
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if "lines" not in block: 
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    element = {
                        "text": text,
                        "x": x0, "y": y0,
                        "width": x1 - x0, "height": y1 - y0,
                        "font_size": span.get("size", 12),
                        "font_family": span.get("font", ""),
                        "page": page_num,
                    }
                    all_elements.append(element)
                    html_parts.append(
                        f"<span style='position:absolute; left:{element['x']}px; top:{element['y']}px; "
                        f"font-size:{element['font_size']}px; font-family:\"{element['font_family']}\";'>"
                        f"{html_escape.escape(text)}</span>"
                    )
        html_parts.append("</div>")
    html_parts.append("</div></body></html>")
    doc.close()
    return "\n".join(html_parts), all_elements

@app.get('/health')
def health():
    return jsonify({"status": "ok", "service": "python"})

@app.post('/convert/pdf-to-html')
def convert_pdf_to_html():
    if 'file' not in request.files:
        return jsonify({"error": "missing file field 'file'"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename or 'upload.pdf')
    upload_id = request.form.get('uploadId') or 'unknown'
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)
    try:
        html_str, elements = pdf_to_html_with_elements(file_path)
        return jsonify({
            "filename": filename,
            "uploadId": upload_id,
            "html": html_str,
            "elements": elements  # top-level array
        })
    except Exception as e:
        return jsonify({"error": f"failed to convert: {e}"}), 500

@app.post('/generate/html-to-pdf')
def generate_html_to_pdf():
    return jsonify({
        "error": "PDF generation moved to PHP layer. Use PHP MPDF for PDF generation.",
        "suggestion": "Use the PHP controller's MPDF integration instead"
    }), 501

@app.post('/convert/html-to-json')
def convert_html_to_json():
    payload = request.get_json(silent=True) or {}
    html_str = payload.get('html')
    if not html_str and 'html' in request.files:
        html_str = request.files['html'].read().decode('utf-8', errors='ignore')
    if not html_str:
        return jsonify({"error": "missing 'html' content"}), 400
    return jsonify({"elements": html_to_json_from_string(html_str)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    app.run(host='0.0.0.0', port=port)