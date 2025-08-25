from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from collections import OrderedDict, Counter
import os, re, statistics, fitz
from bs4 import BeautifulSoup
import html as html_escape

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
    expose_headers=["Content-Type"]
)

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

def _int_color_to_rgb(color_int: int):
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return [int(r), int(g), int(b)]

def _css_color_to_rgb(css: str):
    if not css:
        return None
    css = css.strip()
    m = re.match(r"rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", css, re.I)
    if m:
        r, g, b = map(int, m.groups())
        return [max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))]
    m = re.match(r"#([0-9a-f]{6})$", css, re.I)
    if m:
        hx = m.group(1)
        return [int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)]
    m = re.match(r"#([0-9a-f]{3})$", css, re.I)
    if m:
        h = m.group(1)
        return [int(h[0]*2, 16), int(h[1]*2, 16), int(h[2]*2, 16)]
    return None

def _infer_bold_from_font(font_name: str) -> bool:
    if not font_name:
        return False
    f = font_name.lower()
    return any(k in f for k in ["bold", "black", "heavy", "demi", "semibold"])

def _infer_italic_from_font(font_name: str) -> bool:
    if not font_name:
        return False
    f = font_name.lower()
    return ("italic" in f) or ("oblique" in f)

def _css_flags(style: dict):
    fw = (style.get("font-weight") or "").strip().lower()
    fs = (style.get("font-style") or "").strip().lower()
    td = (style.get("text-decoration") or "").strip().lower()
    bold = fw in ("bold", "bolder", "600", "700", "800", "900")
    italic = fs in ("italic", "oblique")
    underline = "underline" in td
    strike = ("line-through" in td) or ("strikethrough" in td)
    return bold, italic, underline, strike

def html_to_pages_from_string(html_str: str):
    soup = BeautifulSoup(html_str, "lxml")
    elements = []
    next_id = 0

    for span in soup.find_all("span"):
        style = parse_style(span.get("style", ""))
        def px_to_float(v, default=0.0):
            try:
                return float(re.sub("px", "", v))
            except Exception:
                return default
        x = px_to_float(style.get("left", "0"))
        y = px_to_float(style.get("top", "0"))
        w = px_to_float(style.get("width", "0"))
        h = px_to_float(style.get("height", "0"))
        size = px_to_float(style.get("font-size", "12"), 12.0)
        font = (style.get("font-family", "") or "").replace('"', "")
        color_rgb = _css_color_to_rgb(style.get("color"))
        bold_css, italic_css, underline_css, strike_css = _css_flags(style)

        content = (span.get_text() or "").strip()
        if not content:
            continue

        bbox = [x, y, x + w, y + h] if (w and h) else [x, y, x, y]
        elements.append(OrderedDict([
            ("id", f"t:1:{next_id}"),
            ("type", "text"),
            ("content", content),
            ("font", font),
            ("fontSize", float(size)),
            ("bbox", [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]),
            ("color", color_rgb),
            ("bold", bool(bold_css)),
            ("italic", bool(italic_css)),
            ("underline", bool(underline_css)),
            ("strike", bool(strike_css)),
        ]))
        next_id += 1

    return [OrderedDict([("page", 1), ("elements", elements)])]

def _build_lines_from_spans(page_dict):
    lines = []
    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans = line.get("spans", [])
            if not spans:
                continue
            text_parts = []
            x0s, y0s, x1s, y1s = [], [], [], []
            fonts, sizes, colors = [], [], []
            for sp in spans:
                t = sp.get("text", "")
                if t:
                    text_parts.append(t)
                bx = sp.get("bbox", [0, 0, 0, 0])
                x0s.append(bx[0]); y0s.append(bx[1]); x1s.append(bx[2]); y1s.append(bx[3])
                fonts.append(sp.get("font", ""))
                sizes.append(float(sp.get("size", 0)))
                colors.append(int(sp.get("color", 0)))
            line_text = "".join(text_parts).strip()
            if not line_text:
                continue
            line_obj = {
                "text": line_text,
                "left": float(min(x0s)), "top": float(min(y0s)),
                "right": float(max(x1s)), "bottom": float(max(y1s)),
                "height": float(max(y1s) - min(y0s)),
                "fonts": fonts, "sizes": sizes, "colors": colors
            }
            lines.append(line_obj)
    lines.sort(key=lambda l: (l["top"], l["left"]))
    return lines

def _mean(values, default=0.0):
    try:
        return statistics.mean(values) if values else default
    except statistics.StatisticsError:
        return default

def _dominant(items):
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]

def _style_equal(a, b, size_tol=0.5):
    if (a.get("font","").lower() != b.get("font","").lower()): return False
    if abs(float(a.get("fontSize",0)) - float(b.get("fontSize",0))) > size_tol: return False
    if int(a.get("colors_dominant", a.get("color",0))) != int(b.get("colors_dominant", b.get("color",0))): return False
    # optional flags if present on lines
    if bool(a.get("bold", False)) != bool(b.get("bold", False)): return False
    if bool(a.get("italic", False)) != bool(b.get("italic", False)): return False
    return True

def _group_by_avg_gap_and_style(lines):
    if not lines:
        return [], 0.0
    # set line-level dominant style for grouping
    for l in lines:
        l["font"] = _dominant(l["fonts"]) or ""
        sizes_pos = [s for s in l["sizes"] if s > 0]
        l["fontSize"] = float(statistics.median(sizes_pos)) if sizes_pos else 12.0
        l["colors_dominant"] = _dominant(l["colors"]) if l["colors"] else 0
        l["bold"] = _infer_bold_from_font(l["font"])
        l["italic"] = _infer_italic_from_font(l["font"])

    gaps = []
    for i in range(1, len(lines)):
        gaps.append(max(0.0, lines[i]["top"] - lines[i-1]["bottom"]))
    avg_gap = _mean(gaps, default=(statistics.median([l["height"] for l in lines]) if lines else 12.0))

    groups = []
    current = [lines[0]] if lines else []
    for i in range(1, len(lines)):
        prev, cur = lines[i-1], lines[i]
        gap = max(0.0, cur["top"] - prev["bottom"])
        if gap <= avg_gap and _style_equal(prev, cur):
            current.append(cur)
        else:
            groups.append(current)
            current = [cur]
    if current:
        groups.append(current)
    return groups, avg_gap

def pdf_to_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")

        # text: lines -> groups
        lines = _build_lines_from_spans(page_dict)
        groups, avg_gap = _group_by_avg_gap_and_style(lines)

        page_elements = []
        next_id = 0

        # paragraph elements
        for glines in groups:
            if not glines:
                continue
            left = min(l["left"] for l in glines)
            top = min(l["top"] for l in glines)
            right = max(l["right"] for l in glines)
            bottom = max(l["bottom"] for l in glines)

            content = "\n".join(l["text"] for l in glines)

            s0 = glines[0]
            dom_font = s0["font"] or ""
            dom_size = float(s0["fontSize"])
            color_rgb = _int_color_to_rgb(int(s0["colors_dominant"])) if s0.get("colors_dominant") is not None else None
            bold = bool(s0.get("bold", False))
            italic = bool(s0.get("italic", False))

            line_items = []
            for l in glines:
                line_items.append(OrderedDict([
                    ("text", l["text"]),
                    ("bbox", [float(l["left"]), float(l["top"]), float(l["right"]), float(l["bottom"])])
                ]))

            page_elements.append(OrderedDict([
                ("id", f"t:{page_num}:{next_id}"),
                ("type", "text"),
                ("content", content),
                ("font", dom_font),
                ("fontSize", dom_size),
                ("bbox", [float(left), float(top), float(right), float(bottom)]),
                ("color", color_rgb),
                ("bold", bold),
                ("italic", italic),
                ("underline", False),
                ("strike", False),
                ("lines", line_items)
            ]))
            next_id += 1

        # images: PyMuPDF xref + bbox (recommended core)
        for z_idx, img in enumerate(page.get_images(full=True)):
            # tuple: (xref, smask, width, height, bpc, colorspace, ...) varies by version
            xref = int(img[0])
            smask = img[1]
            width = int(img[2]) if len(img) > 2 else None
            height = int(img[3]) if len(img) > 3 else None
            bpc = int(img[4]) if len(img) > 4 else None
            cs = img[5] if len(img) > 5 else None  # may be int or str depending on version

            rect = page.get_image_bbox(img)
            bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
            has_alpha = (smask not in (0, -1, None))

            page_elements.append(OrderedDict([
                ("id", f"i:{page_num}:{next_id}"),
                ("type", "image"),
                ("page", page_num),
                ("bbox", bbox),
                ("xref", xref),
                ("naturalSize", OrderedDict([("width", width), ("height", height)])),
                ("bitsPerComponent", bpc),
                ("colorSpace", cs),
                ("hasAlpha", bool(has_alpha)),
                ("zIndex", z_idx)
            ]))
            next_id += 1

        pages.append(OrderedDict([
            ("page", page_num),
            ("elements", page_elements),
        ]))

    doc.close()
    return pages

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
        pages = pdf_to_pages(file_path)
        return jsonify(OrderedDict([
            ("filename", filename),
            ("uploadId", upload_id),
            ("pages", pages),
        ]))
    except Exception as e:
        return jsonify({"error": f"failed to convert: {e}"}), 500
    finally:
        try:
            os.unlink(file_path)
        except Exception:
            pass

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
    return jsonify({"pages": html_to_pages_from_string(html_str)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    app.run(host='0.0.0.0', port=port)