from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from collections import OrderedDict, Counter
import os, re, statistics, base64, math, fitz
from bs4 import BeautifulSoup  # kept for html_to_pages utility
from PIL import Image, ImageDraw

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "ngrok-skip-browser-warning"], expose_headers=["Content-Type"])

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PT_TO_MM = 25.4 / 72.0  # 1 pt = 1/72 inch

# ============ Helpers ============

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

def _draw_img_to_base64(pil_image: Image.Image) -> str:
    from io import BytesIO
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def convert_transform_pdf_to_html(transform, scale=1.0):
    # PDF transform (a,b,c,d,e,f) -> CSS matrix(a,b,c,d,e,f), with e,f scaled to px
    if not transform or len(transform) < 6:
        return "none"
    a,b,c,d,e,f = transform[:6]
    return f"matrix({a},{b},{c},{d},{e*scale}px,{f*scale}px)"

def rect_intersects(r1, r2):
    return not (r2.x1 <= r1.x0 or r2.x0 >= r1.x1 or r2.y1 <= r1.y0 or r2.y0 >= r1.y1)

def get_clip_list_for_drawings(drawings):
    clips = []
    for d in drawings:
        if d.get("type") == "clip":
            clips.append(d)
    return clips

def find_active_clip_for_bbox(clips, bbox):
    # bbox is [x0,y0,x1,y1]
    if not clips or not bbox: return None
    r = fitz.Rect(bbox)
    # pick first clip whose scissor rect intersects the image rect
    for c in clips:
        sc = c.get("scissor")
        if isinstance(sc, fitz.Rect) and rect_intersects(r, sc):
            return {"scissor": sc}
    return None

# ============ HTML utility (returns text-only elements) ============
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
        elements.append(OrderedDict([
            ("id", f"t:1:{next_id}"), ("type", "text"),
            ("content", content_text),  # plain text
            ("font", font), ("fontSize", float(size)), ("lineHeight", float(line_height)),
            ("rotation", 0.0),
            ("bbox", [float(b) for b in bbox]),
            ("color", color_rgb), ("bold", bool(b)), ("italic", bool(i)),
            ("underline", bool(u)), ("strike", bool(s))
        ]))
        next_id += 1
    return [OrderedDict([("page", 1), ("elements", elements)])]

# ============ Text extraction (paragraph grouping) ============
def _build_lines_from_spans(page_dict):
    lines = []
    for block in page_dict.get("blocks", []):
        if "lines" not in block: continue
        for line in block["lines"]:
            spans = line.get("spans", [])
            if not spans: continue
            text_parts, x0s, y0s, x1s, y1s, fonts, sizes, colors = [], [], [], [], [], [], [], []
            spans_data = []
            for sp in spans:
                t = sp.get("text", "") or ""
                bx = sp.get("bbox", [0,0,0,0])
                if t: text_parts.append(t)
                x0s.append(bx[0]); y0s.append(bx[1]); x1s.append(bx[2]); y1s.append(bx[3])
                fonts.append(sp.get("font", "") or ""); sizes.append(float(sp.get("size", 0))); colors.append(int(sp.get("color", 0)))
                if t:
                    spans_data.append({
                        "text": t,
                        "bbox": [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])],
                        "font": sp.get("font", "") or "",
                        "size": float(sp.get("size", 0)),
                        "color": int(sp.get("color", 0))
                    })
            line_text = "".join(text_parts).strip()
            if not line_text: continue
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
                "angle": float(angle_deg),
                "spans_data": spans_data
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

# Normalize path points to local coords
def _as_xy(val1, val2=None):
    if val2 is not None:
        # Two separate args (numbers or Points)
        try:
            if isinstance(val1, fitz.Point):
                x = float(val1.x)
            else:
                x = float(val1)
            if isinstance(val2, fitz.Point):
                y = float(val2.y)
            else:
                y = float(val2)
            return x, y
        except Exception:
            return None, None
    # Single composite arg
    v = val1
    if isinstance(v, fitz.Point):
        return float(v.x), float(v.y)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return None, None
    return None, None


# ============ Vector drawings → raster images ============
def _vector_drawings_to_images(page, scale=1.0):
    images = []
    drawings = page.get_drawings()
    if not drawings:
        return images
    next_local_id = 0
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        bbox = fitz.Rect(rect).round()
        width_px = max(1, int(round(bbox.width * scale)))
        height_px = max(1, int(round(bbox.height * scale)))

        color = d.get("color")
        fill_color = d.get("fill")
        stroke_rgb = tuple(int(c*255) if c <= 1 else int(c) for c in list(color)[:3]) if color else (0,0,0)
        fill_rgb = tuple(int(c*255) if c <= 1 else int(c) for c in list(fill_color)[:3]) if fill_color else None

        # vector linewidth (not bbox.width!)
        lw = 1
        try:
            line_width = d.get("width") or d.get("linewidth") or 1
            lw = max(1, int(round(float(line_width) * scale)))
        except Exception:
            pass

        # Canvas for this drawing
        img = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        

        path = []
        items = d.get("items") or d.get("paths") or []
        for item in items:
            if not item:
                continue
            op = item[0]
            if op == "m":
                # Flush previous path
                if len(path) >= 2:
                    draw.line(path, fill=stroke_rgb, width=lw)
                    path = []
                # Move to
                x = y = None
                if len(item) >= 3:
                    x, y = _as_xy(item[1], item[2])
                elif len(item) >= 2:
                    x, y = _as_xy(item[1])
                if x is not None and y is not None:
                    path.append(((x - bbox.x0)*scale, (y - bbox.y0)*scale))
            elif op == "l":
                x = y = None
                if len(item) >= 3:
                    x, y = _as_xy(item[1], item[2])
                elif len(item) >= 2:
                    x, y = _as_xy(item[1])
                if x is not None and y is not None:
                    path.append(((x - bbox.x0)*scale, (y - bbox.y0)*scale))
            elif op == "re":
                r = None
                if len(item) >= 2:
                    rect_item = item[1]
                    try:
                        if isinstance(rect_item, fitz.Rect):
                            r = rect_item
                        elif isinstance(rect_item, (list, tuple)) and len(rect_item) >= 4:
                            r = fitz.Rect(float(rect_item[0]), float(rect_item[1]), float(rect_item[2]), float(rect_item[3]))
                        else:
                            r = fitz.Rect(rect_item)
                    except Exception:
                        r = None
                if r is not None:
                    x0 = (r.x0 - bbox.x0)*scale; y0 = (r.y0 - bbox.y0)*scale
                    x1 = (r.x1 - bbox.x0)*scale; y1 = (r.y1 - bbox.y0)*scale
                    if fill_rgb is not None:
                        draw.rectangle([x0, y0, x1, y1], fill=fill_rgb, outline=stroke_rgb, width=lw)
                    else:
                        draw.rectangle([x0, y0, x1, y1], outline=stroke_rgb, width=lw)
            # Other ops (curves) ignored here

        if len(path) >= 2:
            draw.line(path, fill=stroke_rgb, width=lw)

        b64 = _draw_img_to_base64(img)
        images.append(OrderedDict([
            ("id", None),
            ("type", "image"),
            ("page", page.number + 1),
            ("bbox", [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]),
            ("src", f"data:image/png;base64,{b64}"),
            ("zIndex", d.get("sequence", next_local_id)),
            ("rotation", 0.0)
        ]))
        next_local_id += 1
    return images

# ============ ISO sizes ============
_KNOWN_SIZES_MM = OrderedDict([
    ("A0", (841, 1189)), ("A1", (594, 841)), ("A2", (420, 594)),
    ("A3", (297, 420)), ("A4", (210, 297)), ("A5", (148, 210)), ("A6", (105, 148)),
    ("Letter", (216, 279)), ("Legal", (216, 356)),
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

# ============ Image extraction → JSON elements ============
def _extract_images_as_elements(doc, page, scale=1.0, enable_clip=True, enable_transform=True):
    elements = []
    drawings_ext = page.get_drawings(extended=True)  # for clips
    clips = get_clip_list_for_drawings(drawings_ext) if enable_clip else []

    for img in page.get_images(full=True):
        xref = img[0]; smask = img[1]
        infos = page.get_image_info(xref)
        if not infos:
            continue
        info = infos[0]
        bbox = info.get("bbox")
        if not bbox:
            continue

        # Extract image bytes (fallback to Pixmap if needed)
        src = None
        try:
            image_dict = doc.extract_image(xref)
            raw_bytes = image_dict.get("image")
            if raw_bytes:
                src = f"data:image/{image_dict.get('ext','png')};base64," + base64.b64encode(raw_bytes).decode("ascii")
        except Exception:
            try:
                pix = fitz.Pixmap(doc, xref)
                src = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode("ascii")
            except Exception:
                src = None

        # Optional transform and clip-path
        transform = convert_transform_pdf_to_html(info.get("transform"), scale) if enable_transform else "none"
        active_clip = find_active_clip_for_bbox(clips, bbox) if enable_clip else None
        clip_path = None
        if active_clip:
            sc = active_clip["scissor"]
            # Convert to polygon relative to image bbox origin
            left, top = bbox[0], bbox[1]
            x0 = (sc.x0 - left)*scale; y0 = (sc.y0 - top)*scale
            x1 = (sc.x1 - left)*scale; y1 = (sc.y1 - top)*scale
            clip_path = f"polygon({x0}px {y0}px, {x1}px {y0}px, {x1}px {y1}px, {x0}px {y1}px)"

        elements.append(OrderedDict([
            ("id", f"i:{page.number+1}:{len(elements)}"),
            ("type", "image"),
            ("page", page.number+1),
            ("bbox", [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]),
            ("src", src),
            ("transform", transform),
            ("clipPath", clip_path),
            ("rotation", 0.0)
        ]))
    return elements

# ============ Main extraction ============
def pdf_to_pages(pdf_path, scale=1.0):
    doc = fitz.open(pdf_path)
    pages_out = []

    for page_idx, page in enumerate(doc, start=1):
        pd = page.get_text("dict")
        lines = _build_lines_from_spans(pd)
        groups, _ = _group_by_avg_gap_and_style(lines)

        elements, next_id = [], 0
        # Text elements: cluster into visual rows, split by long spaces and geometric gaps on the same row
        for glines in groups:
            s0 = glines[0]
            dom_font = s0["font"] or ""
            dom_size = float(s0["fontSize"])
            color_rgb = _int_color_to_rgb(int(s0["colors_dominant"])) if s0.get("colors_dominant") is not None else None
            line_height = float(_median([l["height"] for l in glines], dom_size * 1.2))
            angles = [float(l.get("angle", 0.0)) for l in glines]
            rotation = float(_median(angles, 0.0))

            splitter = re.compile(r"\s{5,}")

            # Cluster lines into rows by similar top
            row_tol = 1.0
            rows, cur_row, cur_top = [], [], None
            for ln in glines:
                if cur_top is None or abs(ln["top"] - cur_top) <= row_tol:
                    cur_row.append(ln)
                    if cur_top is None:
                        cur_top = ln["top"]
                else:
                    rows.append(cur_row)
                    cur_row = [ln]
                    cur_top = ln["top"]
            if cur_row:
                rows.append(cur_row)

            # Aggregate rows without splits, emit components for rows with splits
            agg_left = agg_top = agg_right = agg_bottom = None
            agg_text_lines = []
            agg_fonts, agg_sizes, agg_colors = [], [], []

            for row in rows:
                # Flatten spans in this row and sort left-to-right
                row_spans = []
                for ln in row:
                    row_spans.extend(ln.get("spans_data", []))
                row_spans = sorted(row_spans, key=lambda sp: sp["bbox"][0])

                row_left = min(l["left"] for l in row); row_top = min(l["top"] for l in row)
                row_right = max(l["right"] for l in row); row_bottom = max(l["bottom"] for l in row)
                row_text = " ".join(l["text"] for l in row)

                has_long_space = any(splitter.search(sp.get("text", "")) for sp in row_spans)
                row_sizes = [float(sp.get("size", dom_size)) for sp in row_spans] or [dom_size]
                row_size_dom = float(_dominant(row_sizes)) if row_sizes else dom_size
                X_GAP_TOL = max(1.0, row_size_dom * 1.5)
                has_geom_gap = any(
                    (row_spans[i+1]["bbox"][0] - row_spans[i]["bbox"][2]) > X_GAP_TOL
                    for i in range(len(row_spans) - 1)
                )

                if not (has_long_space or has_geom_gap):
                    # Accumulate row into aggregated paragraph
                    agg_text_lines.append(row_text)
                    if agg_left is None:
                        agg_left, agg_top, agg_right, agg_bottom = row_left, row_top, row_right, row_bottom
                    else:
                        agg_left = min(agg_left, row_left)
                        agg_top = min(agg_top, row_top)
                        agg_right = max(agg_right, row_right)
                        agg_bottom = max(agg_bottom, row_bottom)
                    # Accumulate styles from this row's spans
                    for sp in row_spans:
                        agg_fonts.append(sp.get("font", "") or "")
                        try:
                            agg_sizes.append(float(sp.get("size", dom_size)))
                        except Exception:
                            pass
                        try:
                            agg_colors.append(int(sp.get("color", 0)))
                        except Exception:
                            pass
                    continue

                # Split this row into components using long spaces and geometric gaps
                cur_text = ""
                cur_left = cur_top = cur_right = cur_bottom = None
                comp_fonts, comp_sizes, comp_colors = [], [], []

                def flush_component():
                    nonlocal cur_text, cur_left, cur_top, cur_right, cur_bottom, comp_fonts, comp_sizes, comp_colors, next_id
                    txt = cur_text.strip()
                    if txt and cur_left is not None:
                        comp_font = _dominant(comp_fonts) or dom_font
                        comp_size = float(_dominant(comp_sizes) if comp_sizes else dom_size)
                        comp_color_int = _dominant(comp_colors) if comp_colors else None
                        comp_color_rgb = _int_color_to_rgb(int(comp_color_int)) if comp_color_int is not None else color_rgb
                        comp_bold = _infer_bold_from_font(comp_font)
                        comp_italic = _infer_italic_from_font(comp_font)
                        elements.append(OrderedDict([
                            ("id", f"t:{page_idx}:{next_id}"),
                            ("type", "text"),
                            ("content", txt),
                            ("font", comp_font),
                            ("fontSize", comp_size),
                            ("lineHeight", line_height),
                            ("rotation", rotation),
                            ("bbox", [float(cur_left), float(cur_top), float(cur_right), float(cur_bottom)]),
                            ("color", comp_color_rgb),
                            ("bold", comp_bold),
                            ("italic", comp_italic),
                            ("underline", False),
                            ("strike", False)
                        ]))
                        next_id += 1
                    cur_text = ""
                    cur_left = cur_top = cur_right = cur_bottom = None
                    comp_fonts, comp_sizes, comp_colors = [], [], []

                prev_right = None
                for sp in row_spans:
                    t = sp["text"]
                    bx = sp["bbox"]
                    # Geometric split between spans
                    if prev_right is not None and (bx[0] - prev_right) > X_GAP_TOL:
                        flush_component()
                    prev_right = bx[2]

                    w = max(0.0, bx[2] - bx[0])
                    n = max(1, len(t))
                    parts = []
                    last = 0
                    for m in splitter.finditer(t):
                        parts.append((t[last:m.start()], last, m.start()))
                        last = m.end()
                    parts.append((t[last:], last, len(t)))

                    for idx, (ptxt, a, b) in enumerate(parts):
                        x0p = bx[0] + (w * (a / n))
                        x1p = bx[0] + (w * (b / n))
                        slice_bbox = [x0p, bx[1], x1p, bx[3]]

                        content_piece = (ptxt or "").strip()
                        if content_piece:
                            if cur_left is None:
                                cur_left, cur_top, cur_right, cur_bottom = slice_bbox
                                cur_text = content_piece
                            else:
                                cur_left = min(cur_left, slice_bbox[0])
                                cur_top = min(cur_top, slice_bbox[1])
                                cur_right = max(cur_right, slice_bbox[2])
                                cur_bottom = max(cur_bottom, slice_bbox[3])
                                cur_text = (cur_text + " " + content_piece).strip()
                            comp_fonts.append(sp.get("font", "") or "")
                            try:
                                comp_sizes.append(float(sp.get("size", dom_size)))
                            except Exception:
                                pass
                            try:
                                comp_colors.append(int(sp.get("color", 0)))
                            except Exception:
                                pass

                        if idx < len(parts) - 1:
                            flush_component()

                flush_component()

            # Emit one aggregated paragraph element if we accumulated any rows
            if agg_text_lines:
                comp_font = _dominant(agg_fonts) or dom_font
                comp_size = float(_dominant(agg_sizes) if agg_sizes else dom_size)
                comp_color_int = _dominant(agg_colors) if agg_colors else None
                comp_color_rgb = _int_color_to_rgb(int(comp_color_int)) if comp_color_int is not None else color_rgb
                comp_bold = _infer_bold_from_font(comp_font)
                comp_italic = _infer_italic_from_font(comp_font)
                content_text = "\n".join(agg_text_lines)
                elements.append(OrderedDict([
                    ("id", f"t:{page_idx}:{next_id}"),
                    ("type", "text"),
                    ("content", content_text),
                    ("font", comp_font),
                    ("fontSize", comp_size),
                    ("lineHeight", line_height),
                    ("rotation", rotation),
                    ("bbox", [float(agg_left), float(agg_top), float(agg_right), float(agg_bottom)]),
                    ("color", comp_color_rgb),
                    ("bold", comp_bold),
                    ("italic", comp_italic),
                    ("underline", False),
                    ("strike", False)
                ]))
                next_id += 1

        # Bitmap images
        elements.extend(_extract_images_as_elements(doc, page, scale=scale))

        # Vector drawings → raster images
        vimgs = _vector_drawings_to_images(page, scale=scale)
        for vi in vimgs:
            vi["id"] = f"i:{page_idx}:{next_id}"
            elements.append(vi); next_id += 1

        width_pt = float(page.rect.width)
        height_pt = float(page.rect.height)
        orientation = "portrait" if height_pt >= width_pt else "landscape"
        iso = _classify_page_iso(width_pt, height_pt)

        pages_out.append(OrderedDict([
            ("page", page_idx),
            ("rotation", int(page.rotation or 0)),
            ("size", OrderedDict([
                ("width", width_pt),
                ("height", height_pt),
                ("unit", "pt"),
                ("iso", iso),
                ("orientation", orientation)
            ])),
            ("elements", elements),
        ]))

    doc.close()
    return pages_out

# ============ Routes ============

@app.post('/convert/pdf-to-html')
def convert_pdf_to_html():
    if 'file' not in request.files:
        return jsonify({"error": "missing file field 'file'"}), 400
    f = request.files['file']
    filename = secure_filename(f.filename or 'upload.pdf')
    upload_id = request.form.get('uploadId') or 'unknown'
    # You can expose a scale param if you want px scaling; default 1.0 here
    scale = float(request.form.get('scale', request.args.get('scale', '1.0')))

    file_path = os.path.join(UPLOAD_DIR, filename)
    f.save(file_path)
    try:
        pages = pdf_to_pages(file_path, scale=scale)
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