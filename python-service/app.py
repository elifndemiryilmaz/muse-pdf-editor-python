from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from collections import OrderedDict, Counter
import os, re, statistics, base64, math, fitz
from bs4 import BeautifulSoup
import html as html_escape

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "ngrok-skip-browser-warning"], expose_headers=["Content-Type"])

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PT_TO_MM = 25.4 / 72.0  # 1 pt = 1/72 inch

# ---------- Helpers ----------
def parse_style(style_str: str):
    style_dict = {}
    for part in style_str.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            style_dict[k.strip().lower()] = v.strip()
    return style_dict

def _int_color_to_rgb(color_int: int):
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return [int(r), int(g), int(b)]

def _css_color_to_rgb(css: str):
    if not css: return None
    css = css.strip()
    m = re.match(r"rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", css, re.I)
    if m:
        r, g, b = map(int, m.groups())
        return [max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255))]
    m = re.match(r"#([0-9a-f]{6})$", css, re.I)
    if m:
        hx = m.group(1); return [int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)]
    m = re.match(r"#([0-9a-f]{3})$", css, re.I)
    if m:
        h = m.group(1); return [int(h[0]*2, 16), int(h[1]*2, 16), int(h[2]*2, 16)]
    return None

def _infer_bold_from_font(font_name: str) -> bool:
    if not font_name: return False
    f = font_name.lower()
    return any(k in f for k in ["bold", "black", "heavy", "demi", "semibold"])

def _infer_italic_from_font(font_name: str) -> bool:
    if not font_name: return False
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

def _mean(values, default=0.0):
    try: return statistics.mean(values) if values else default
    except statistics.StatisticsError: return default

def _median(values, default=0.0):
    try: return statistics.median(values) if values else default
    except statistics.StatisticsError: return default

def _dominant(items):
    if not items: return None
    return Counter(items).most_common(1)[0][0]

def _style_equal(a, b, size_tol=0.5):
    if (a.get("font","").lower() != b.get("font","").lower()): return False
    if abs(float(a.get("fontSize",0)) - float(b.get("fontSize",0))) > size_tol: return False
    if int(a.get("colors_dominant", a.get("color",0))) != int(b.get("colors_dominant", b.get("color",0))): return False
    if bool(a.get("bold", False)) != bool(b.get("bold", False)): return False
    if bool(a.get("italic", False)) != bool(b.get("italic", False)): return False
    return True

def _rgb_from_tuple(t):
    if t is None: return None
    vals = list(t)
    if not vals: return None
    if max(vals) <= 1.0: vals = [int(round(v*255)) for v in vals]
    return [int(v) for v in vals[:3]]

# ---------- HTML path (utility) ----------
def html_to_pages_from_string(html_str: str):
    soup = BeautifulSoup(html_str, "lxml")
    elements, next_id = [], 0
    for span in soup.find_all("span"):
        style = parse_style(span.get("style", ""))
        def px(v, d=0.0):
            try: return float(re.sub("px", "", v))
            except Exception: return d
        x, y = px(style.get("left", "0")), px(style.get("top", "0"))
        w, h = px(style.get("width", "0")), px(style.get("height", "0"))
        size = px(style.get("font-size", "12"), 12.0)
        font = (style.get("font-family", "") or "").replace('"', "")
        color_rgb = _css_color_to_rgb(style.get("color"))
        b,i,u,s = _css_flags(style)
        content_text = (span.get_text() or "").strip()
        if not content_text: continue
        bbox = [x, y, x + w, y + h] if (w and h) else [x, y, x, y]
        line_height = float(h) if h > 0 else float(size) * 1.2
        rgb = color_rgb or [0,0,0]
        weight = "bold" if b else "normal"
        style_i = "italic" if i else "normal"
        deco = "underline" if u else ("line-through" if s else "none")
        safe = html_escape.escape(content_text)
        style_attr = f"font-family:{font}; font-size:{size}pt; line-height:{line_height}pt; color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); font-weight:{weight}; font-style:{style_i}; text-decoration:{deco}; margin:0;"
        content_html = f'<p style="{style_attr}">{safe}</p>'
        elements.append(OrderedDict([
            ("id", f"t:1:{next_id}"), ("type", "text"),
            ("content", content_html),
            ("font", font), ("fontSize", float(size)), ("lineHeight", float(line_height)),
            ("rotation", 0.0),
            ("bbox", [float(b) for b in bbox]),
            ("color", color_rgb), ("bold", bool(b)), ("italic", bool(i)),
            ("underline", bool(u)), ("strike", bool(s))
        ]))
        next_id += 1
    return [OrderedDict([("page", 1), ("elements", elements)])]

# ---------- Text extraction ----------
def _build_lines_from_spans(page_dict):
    lines = []
    for block in page_dict.get("blocks", []):
        if "lines" not in block: continue
        for line in block["lines"]:
            spans = line.get("spans", [])
            if not spans: continue
            text_parts, x0s, y0s, x1s, y1s, fonts, sizes, colors = [], [], [], [], [], [], [], []
            for sp in spans:
                t = sp.get("text", "") or ""
                bx = sp.get("bbox", [0,0,0,0])
                if t: text_parts.append(t)
                x0s.append(bx[0]); y0s.append(bx[1]); x1s.append(bx[2]); y1s.append(bx[3])
                fonts.append(sp.get("font", "") or ""); sizes.append(float(sp.get("size", 0))); colors.append(int(sp.get("color", 0)))
            line_text = "".join(text_parts).strip()
            if not line_text: continue
            # rotation from line direction vector if present (dir: [dx, dy])
            angle_deg = 0.0
            try:
                dir_vec = line.get("dir")
                if dir_vec and len(dir_vec) == 2:
                    angle_deg = math.degrees(math.atan2(float(dir_vec[1]), float(dir_vec[0])))
            except Exception:
                angle_deg = 0.0
            lines.append({
                "text": line_text,
                "left": float(min(x0s)), "top": float(min(y0s)),
                "right": float(max(x1s)), "bottom": float(max(y1s)),
                "height": float(max(y1s) - min(y0s)),
                "fonts": fonts, "sizes": sizes, "colors": colors,
                "angle": float(angle_deg)
            })
    lines.sort(key=lambda l: (l["top"], l["left"]))
    return lines

def _group_by_avg_gap_and_style(lines):
    if not lines: return ([], 0.0)
    for l in lines:
        l["font"] = _dominant(l["fonts"]) or ""
        pos = [s for s in l["sizes"] if s > 0]
        l["fontSize"] = float(_median(pos, 12.0))
        l["colors_dominant"] = _dominant(l["colors"]) if l["colors"] else 0
        l["bold"] = _infer_bold_from_font(l["font"])
        l["italic"] = _infer_italic_from_font(l["font"])
    gaps = [max(0.0, lines[i]["top"] - lines[i-1]["bottom"]) for i in range(1, len(lines))]
    avg_gap = _mean(gaps, default=_median([l["height"] for l in lines], 12.0))
    groups, cur = [], [lines[0]]
    for i in range(1, len(lines)):
        prev, curline = lines[i-1], lines[i]
        gap = max(0.0, curline["top"] - prev["bottom"])
        if gap <= avg_gap and _style_equal(prev, curline): cur.append(curline)
        else: groups.append(cur); cur = [curline]
    if cur: groups.append(cur)
    return groups, avg_gap

# ---------- Shapes (vector) ----------
def _shapes_from_drawings(page):
    out = []
    drawings = page.get_drawings()
    for idx, d in enumerate(drawings):
        rect = d.get("rect", None)
        stroke = _rgb_from_tuple(d.get("color"))
        fill = _rgb_from_tuple(d.get("fill"))
        lw = float(d.get("width") or d.get("linewidth") or 1.0)
        dashes = d.get("dashes")
        lineCap = d.get("lineCap")
        lineJoin = d.get("lineJoin")
        miter = d.get("miterLimit")
        opacity = d.get("opacity")
        blendmode = d.get("blendmode")
        if rect is not None:
            bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        else:
            xmins, ymins, xmaxs, ymaxs = [], [], [], []
            for it in d.get("items", []) or d.get("paths", []) or []:
                try:
                    for p in it:
                        xmins.append(float(p[0])); ymins.append(float(p[1]))
                        xmaxs.append(float(p[0])); ymaxs.append(float(p[1]))
                except Exception:
                    pass
            if xmins:
                bbox = [min(xmins), min(ymins), max(xmaxs), max(ymaxs)]
            else:
                bbox = None
        if not bbox: continue
        out.append(OrderedDict([
            ("id", None),
            ("type", "shape"),
            ("page", page.number+1),
            ("bbox", bbox),
            ("strokeColor", stroke),
            ("fillColor", fill),
            ("lineWidth", lw),
            ("strokeDash", dashes),
            ("strokeCap", lineCap),
            ("strokeJoin", lineJoin),
            ("miterLimit", miter),
            ("opacity", opacity),
            ("blendMode", blendmode),
            ("zIndex", idx),
            ("rotation", 0.0)  # rotation of vector shapes not trivially available; default 0
        ]))
    return out

# ---------- Annotations ----------
def _annots_from_page(page):
    out, i = [], 0
    try:
        a = page.first_annot
    except Exception:
        a = None
    while a:
        typ = None
        try: typ = a.type[1]
        except Exception: pass
        r = a.rect
        info, colors = {}, {}
        try: info = a.info or {}
        except Exception: pass
        try: colors = a.colors or {}
        except Exception: pass
        subtype = info.get("name") or typ
        contents = info.get("content")
        title = info.get("title")
        out.append(OrderedDict([
            ("id", None),
            ("type", "annotation"),
            ("page", page.number+1),
            ("subtype", subtype),
            ("bbox", [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]),
            ("title", title),
            ("contents", contents),
            ("colors", colors)
        ]))
        i += 1
        a = a.next
    return out

# ---------- ISO sizes ----------
_KNOWN_SIZES_MM = OrderedDict([
    ("A0", (841, 1189)),
    ("A1", (594, 841)),
    ("A2", (420, 594)),
    ("A3", (297, 420)),
    ("A4", (210, 297)),
    ("A5", (148, 210)),
    ("A6", (105, 148)),
    ("Letter", (216, 279)),
    ("Legal", (216, 356)),
])

def _classify_page_iso(width_pt: float, height_pt: float):
    w_mm = float(width_pt) * PT_TO_MM
    h_mm = float(height_pt) * PT_TO_MM
    s_min, s_max = sorted([w_mm, h_mm])
    best_name, best_score = None, 1e9
    for name, (mm_min, mm_max) in _KNOWN_SIZES_MM.items():
        score = abs(s_min - mm_min) + abs(s_max - mm_max)
        if score < best_score:
            best_name, best_score = name, score
    if best_score <= 20.0:
        return best_name
    return "Unknown"

# ---------- Background raster ----------
def _page_background_data_uri(page, dpi: int = 150):
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    data = pix.tobytes("png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"

# ---------- Main extraction ----------
def pdf_to_pages(pdf_path, bg_enable=True, bg_dpi=150):
    doc = fitz.open(pdf_path)
    pages = []
    for page_idx, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")

        lines = _build_lines_from_spans(page_dict)
        groups, avg_gap = _group_by_avg_gap_and_style(lines)

        elements, next_id = [], 0
        for glines in groups:
            left = min(l["left"] for l in glines); top = min(l["top"] for l in glines)
            right = max(l["right"] for l in glines); bottom = max(l["bottom"] for l in glines)
            # style / metrics
            s0 = glines[0]
            dom_font = s0["font"] or ""
            dom_size = float(s0["fontSize"])
            color_rgb = _int_color_to_rgb(int(s0["colors_dominant"])) if s0.get("colors_dominant") is not None else None
            bold = bool(s0.get("bold", False)); italic = bool(s0.get("italic", False))
            line_height = float(_median([l["height"] for l in glines], dom_size * 1.2))
            # rotation: median of line angles in group
            angles = [float(l.get("angle", 0.0)) for l in glines]
            rotation = float(_median(angles, 0.0))
            # HTML content: lines joined with <br/>
            safe_lines = [html_escape.escape(l["text"]) for l in glines]
            rgb = color_rgb or [0, 0, 0]
            weight = "bold" if bold else "normal"
            style_i = "italic" if italic else "normal"
            style_attr = (
                f"font-family:{dom_font}; font-size:{dom_size}pt; line-height:{line_height}pt; "
                f"color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); font-weight:{weight}; font-style:{style_i}; margin:0;"
            )
            content_html = f'<p style="{style_attr}">' + "<br/>".join(safe_lines) + "</p>"
            elements.append(OrderedDict([
                ("id", f"t:{page_idx}:{next_id}"),
                ("type", "text"),
                ("content", content_html),
                ("font", dom_font),
                ("fontSize", dom_size),
                ("lineHeight", line_height),
                ("rotation", rotation),
                ("bbox", [float(left), float(top), float(right), float(bottom)]),
                ("color", color_rgb),
                ("bold", bold),
                ("italic", italic),
                ("underline", False),
                ("strike", False)
            ]))
            next_id += 1

        for z_idx, img in enumerate(page.get_images(full=True)):
            xref = int(img[0]); smask = img[1]
            width = int(img[2]) if len(img) > 2 else None
            height = int(img[3]) if len(img) > 3 else None
            bpc = int(img[4]) if len(img) > 4 else None
            cs = img[5] if len(img) > 5 else None
            rect = page.get_image_bbox(img)
            bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
            has_alpha = (smask not in (0, -1, None))
            elements.append(OrderedDict([
                ("id", f"i:{page_idx}:{next_id}"), ("type", "image"), ("page", page_idx),
                ("bbox", bbox), ("xref", xref),
                ("naturalSize", OrderedDict([("width", width), ("height", height)])),
                ("bitsPerComponent", bpc), ("colorSpace", cs),
                ("hasAlpha", bool(has_alpha)), ("zIndex", z_idx),
                ("rotation", 0.0)
            ]))
            next_id += 1

        shapes = _shapes_from_drawings(page)
        for s in shapes:
            s["id"] = f"v:{page_idx}:{next_id}"
            elements.append(s); next_id += 1

        annots = _annots_from_page(page)
        for a in annots:
            a["id"] = f"a:{page_idx}:{next_id}"
            elements.append(a); next_id += 1

        width_pt = float(page.rect.width)
        height_pt = float(page.rect.height)
        orientation = "portrait" if height_pt >= width_pt else "landscape"
        iso = _classify_page_iso(width_pt, height_pt)

        background = None
        if bg_enable:
            try:
                background = OrderedDict([
                    ("type", "image"),
                    ("dpi", int(bg_dpi)),
                    ("src", _page_background_data_uri(page, dpi=int(bg_dpi)))
                ])
            except Exception:
                background = None

        bg_ids = []
        page_area = width_pt * height_pt if (width_pt and height_pt) else 1.0
        for el in elements:
            if el["type"] in ("image", "shape"):
                x1,y1,x2,y2 = el["bbox"]
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                area_ratio = area / page_area if page_area else 0.0
                z = el.get("zIndex", 0)
                if area_ratio >= 0.7 and z <= 2:
                    bg_ids.append(el["id"])

        pages.append(OrderedDict([
            ("page", page_idx),
            ("rotation", int(page.rotation or 0)),
            ("size", OrderedDict([
                ("width", width_pt),
                ("height", height_pt),
                ("unit", "pt"),
                ("iso", iso),
                ("orientation", orientation)
            ])),
            ("background", background),
            ("backgroundElementIds", bg_ids),
            ("elements", elements),
        ]))

    doc.close()
    return pages

# ---------- Routes ----------
@app.get('/health')
def health():
    return jsonify({"status": "ok", "service": "python"})

@app.post('/convert/pdf-to-html')
def convert_pdf_to_html():
    if 'file' not in request.files:
        return jsonify({"error": "missing file field 'file'"}), 400
    f = request.files['file']
    filename = secure_filename(f.filename or 'upload.pdf')
    upload_id = request.form.get('uploadId') or 'unknown'
    bg_flag = request.form.get('bg', request.args.get('bg', '1'))
    bg_dpi = int(request.form.get('bgDpi', request.args.get('bgDpi', '150')))
    bg_enable = str(bg_flag).lower() not in ('0','false','no')

    file_path = os.path.join(UPLOAD_DIR, filename)
    f.save(file_path)
    try:
        pages = pdf_to_pages(file_path, bg_enable=bg_enable, bg_dpi=bg_dpi)
        return jsonify(OrderedDict([
            ("filename", filename),
            ("uploadId", upload_id),
            ("pages", pages),
        ]))
    except Exception as e:
        return jsonify({"error": f"{e}"}), 500
    finally:
        try: os.unlink(file_path)
        except Exception: pass

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