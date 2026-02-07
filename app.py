import os
import json
import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from openai import OpenAI

# =========================
# Config
# =========================
def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

MODEL = get_secret("MODEL", "gpt-4-mini")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GSHEET_ID = get_secret("GSHEET_ID", "")
GSHEET_TAB = get_secret("GSHEET_TAB", "Productos")
ORDERS_TAB = get_secret("ORDERS_TAB", "Ordenes de Compra")
GCP_SA_JSON = get_secret("GCP_SERVICE_ACCOUNT_JSON", "").strip()
GSHEET_CSV_URL = get_secret("GSHEET_CSV_URL", "").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# Small talk router
# =========================
GREET_RE = re.compile(r"\b(h?o+l+a+|buenas|buen día|buen dia|buenas tardes|buenas noches|qué tal|que tal|hey)\b", re.I)
HOWARE_RE = re.compile(r"\b(c[oó]mo est[aá]s|como and[aá]s|todo bien|qué onda|que onda)\b", re.I)
THANKS_RE = re.compile(r"\b(gracias|muchas gracias|mil gracias|genial gracias|perfecto gracias)\b", re.I)
BYE_RE = re.compile(r"\b(chau|adios|adiós|hasta luego|hasta pronto|nos vemos|bye)\b", re.I)
AFFIRM_RE = re.compile(r"^(ok|oka|dale|de una|joya|perfecto|genial|listo|bárbaro|barbaro)\b", re.I)
ALT_RE = re.compile(r"\b(alternativ|alternativas|otros|otras|similares|parecidos|ver alternativas|m[aá]s opciones|mas opciones)\b", re.I)
QUOTE_RE = re.compile(r"\b(cotiz|cotizar|presupuesto|precio por|por (\d+)|cantidad|x(\d+))\b", re.I)
ABOUT_RE = re.compile(
    r"\b(empresa|mollon|moll[oó]n|nosotros|qu[ií]enes son|quien(es)? son|sobre .*moll[oó]n|acerca de|historia|"
    r"qu[eé] hacen|a qu[eé] se dedican|de qu[eé] se trata|informaci[oó]n|"
    r"datos|perfil|misi[oó]n|visi[oó]n|valores)\b",
    re.I
)
ABUSE_RE = re.compile(r"\b(puta|puto|mierda|forro|bolud[oa]|pelotud[oa]|idiota|imb[eé]cil|est[uú]pid[oa]|hdp|la concha)\b", re.I)
CONFIRM_RE = re.compile(r"\b(confirmo|confirmar|confirmar compra|si, confirmo|ok confirmo|confirmar pedido)\b", re.I)
CANCEL_RE = re.compile(r"\b(cancelar|anular|no quiero|no gracias|dej[aá]lo|olvidalo)\b", re.I)
QTY_CUE_RE = re.compile(r"\b(cantidad|unidades|uds|u\.|por|x)\b", re.I)
SKU_RE = re.compile(r"\b\d{3,6}\b")
RANK_RE = re.compile(r"\b(m[aá]s barato|mas barato|m[aá]s caro|mas caro|barat[oa]s?|car[oa]s?|top|mejores?|peores?|mostrame|muestra|listar|listame|ordenad[oa]s?)\b", re.I)
TOPN_RE = re.compile(r"\btop\s*(\d+)|(\d+)\s*(m[aá]s )?(baratos?|caros?)\b", re.I)
STOCK_RANK_RE = re.compile(r"\b(m[aá]s stock|con m[aá]s stock|mayor stock|menor stock|menos stock|sin stock)\b", re.I)
RANK_PICK_RE = re.compile(r"\b(detalle|detalle del|pasame|mostrame|ver)\b", re.I)
ORDINAL_RE = re.compile(r"\b(primero|primer|primera|segundo|segunda|tercero|tercer|tercera|cuarto|cuarta|quinto|quinta|sexto|sexta|séptimo|septimo|séptima|septima|octavo|octava|noveno|novena|décimo|decimo|décima|decima)\b", re.I)
NUMBER_RE = re.compile(r"\b(\d+)\b")
INTEREST_RE = re.compile(r"\b(me interesa|me gust[aó]|quiero|dame|detalle|ver)\b", re.I)
EXPLICIT_RE = re.compile(r"\b(sku|c[oó]digo|3m)\b", re.I)

# Ranking config
CATEGORIES = {
    "respirador": ["respirador", "n95", "pff2", "media_cara", "cara_completa", "mascarilla"],
    "filtro": ["filtro", "cartucho", "p100", "gases"],
    "cinta": ["cinta", "scotch", "vhb", "aisladora", "adhesivo"],
    "gafas": ["gafas", "lentes", "ocular", "virtua", "securefit"],
    "auditivo": ["auditivo", "orejeras", "tapon", "ruido"],
    "office": ["postit", "notas", "office", "banderitas"],
    "abrasivo": ["abrasivo", "disco", "scotch-brite", "lij", "esponja"],
    "altura": ["arnes", "línea de vida", "linea de vida", "altura"]
}
DEFAULT_TOP_N = 3

def wants_alternatives(text: str) -> bool:
    return ALT_RE.search(text or "") is not None

def infer_category_from_text(text: str):
    t = (text or "").lower()
    for cat, kws in CATEGORIES.items():
        for kw in kws:
            if kw in t or (not kw.endswith("s") and f"{kw}s" in t):
                return cat
    return None

def detect_rank_intent(text: str):
    t = (text or "").lower()
    if not RANK_RE.search(t) and not TOPN_RE.search(t) and not STOCK_RANK_RE.search(t):
        return None

    # metric + order
    metric = "price"
    order = "asc"
    if STOCK_RANK_RE.search(t):
        metric = "stock"
        if re.search(r"(menos stock|menor stock|sin stock)", t):
            order = "asc"
        else:
            order = "desc"
    else:
        if re.search(r"(caro|caros|m[aá]s caro)", t):
            order = "desc"
        else:
            order = "asc"

    # top N
    n = DEFAULT_TOP_N
    m = TOPN_RE.search(t)
    if m:
        for g in m.groups():
            if g and g.isdigit():
                n = max(1, min(int(g), 20))
                break
    else:
        m = NUMBER_RE.search(t)
        if m:
            try:
                num = int(m.group(1))
                if 1 <= num <= 50:
                    n = num
            except Exception:
                pass

    # category detection (name/tags match)
    category = infer_category_from_text(t)

    return {"metric": metric, "order": order, "n": n, "category": category}

def filter_by_category(df, category):
    if not category:
        return df
    kws = CATEGORIES.get(category, [])
    if not kws:
        return df
    pattern = "|".join([re.escape(k) for k in kws])
    mask = df["name"].str.contains(pattern, case=False, na=False) | df["tags"].str.contains(pattern, case=False, na=False)
    return df[mask]

def rank_products(df, rank_spec):
    metric = rank_spec["metric"]
    order = rank_spec["order"]
    n = rank_spec["n"]
    category = rank_spec["category"]

    dff = filter_by_category(df, category)
    if dff.empty:
        return dff, None

    if metric == "price":
        dff = dff.dropna(subset=["price"])
        dff = dff[dff["price"] > 0]
    dff = dff.sort_values(by=metric, ascending=(order == "asc"))
    return dff.head(n), category

def _ordinal_to_index(word):
    mapping = {
        "primero": 1, "primer": 1, "primera": 1,
        "segundo": 2, "segunda": 2,
        "tercero": 3, "tercer": 3, "tercera": 3,
        "cuarto": 4, "cuarta": 4,
        "quinto": 5, "quinta": 5,
        "sexto": 6, "sexta": 6,
        "séptimo": 7, "septimo": 7, "séptima": 7, "septima": 7,
        "octavo": 8, "octava": 8,
        "noveno": 9, "novena": 9,
        "décimo": 10, "decimo": 10, "décima": 10, "decima": 10
    }
    return mapping.get(word, None)

def detect_rank_choice(text: str):
    t = (text or "").lower()
    # allow ordinal without explicit "detalle" if there's interest cue
    has_interest = INTEREST_RE.search(t) is not None
    has_pick_cue = RANK_PICK_RE.search(t) is not None
    if not has_pick_cue and not has_interest:
        return None
    m = ORDINAL_RE.search(t)
    if m:
        return _ordinal_to_index(m.group(1))
    m = NUMBER_RE.search(t)
    if m:
        try:
            n = int(m.group(1))
            # avoid treating SKU-like numbers as ordinal picks
            if 1 <= n <= 20:
                return n
            return None
        except Exception:
            return None
    return None

def extract_quantity(text: str):
    t = (text or "").lower()
    has_qty_cue = QTY_CUE_RE.search(t) is not None or QUOTE_RE.search(t) is not None
    has_product_hint = ("3m" in t) or (SKU_RE.search(t) is not None) or (EXPLICIT_RE.search(t) is not None)
    # x10, x 10
    m = re.search(r"\bx\s*(\d+)\b", t)
    if m:
        return int(m.group(1))
    # por 10 / cantidad 10
    m = re.search(r"\bpor\s+(\d+)\b", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\bcantidad\s+(\d+)\b", t)
    if m:
        return int(m.group(1))
    # standalone number if explicit qty cue; allow even if number looks like SKU
    # unless user explicitly mentions SKU/código/3m in the same message
    if has_qty_cue and not EXPLICIT_RE.search(t):
        m = NUMBER_RE.search(t)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 10000:
                    return n
            except Exception:
                return None
    return None

def build_order_summary(row, qty):
    unit = row.get("price")
    currency = row.get("currency", "ARS")
    if pd.isna(unit):
        unit_str = "N/D"
        total_str = "N/D"
    else:
        unit_str = fmt_money(unit, currency)
        total_str = fmt_money(float(unit) * qty, currency)
    name = row.get("name", "")
    sku = row.get("sku", "")
    return (
        "🧾 **Orden de compra (borrador)**\n\n"
        f"**Producto:** {name} — SKU `{sku}`\n\n"
        f"**Precio unitario:** {unit_str}\n\n"
        f"**Cantidad:** {qty}\n\n"
        f"**Total estimado:** {total_str}\n\n"
        "¿Querés **confirmar la compra**?"
    )

def resolve_qty_with_stock(row, qty):
    try:
        stock = int(row.get("stock", 0) or 0)
    except Exception:
        stock = 0
    if stock > 0 and qty > stock:
        return stock, stock
    return qty, stock

def detect_customer_name(text: str):
    t = (text or "").strip()
    m = re.search(r"\b(me llamo|mi nombre es|soy)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]+){0,2})\b", t, re.I)
    if not m:
        return None
    name = m.group(2).strip()
    return " ".join([w.capitalize() for w in name.split()])

ORDERS_HEADER = [
    "order_id",
    "timestamp",
    "conversation_id",
    "product_name",
    "sku",
    "qty",
    "unit_price",
    "currency",
    "total",
    "customer_name",
    "status"
]
TZ_NAME = "America/Argentina/Buenos_Aires"

def _get_orders_ws():
    if not (GSHEET_ID and GCP_SA_JSON):
        return None
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        json.loads(GCP_SA_JSON),
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return gspread.authorize(creds).open_by_key(GSHEET_ID).worksheet(ORDERS_TAB)

def _ensure_orders_header(ws):
    values = ws.get_all_values()
    if not values:
        ws.append_row(ORDERS_HEADER)
        return
    header = values[0]
    if header != ORDERS_HEADER:
        end_col = chr(65 + len(ORDERS_HEADER) - 1)
        ws.update(f"A1:{end_col}1", [ORDERS_HEADER])

def _order_row(order):
    return [
        order.get("order_id", ""),
        order.get("timestamp", ""),
        order.get("conversation_id", ""),
        order.get("product_name", ""),
        order.get("sku", ""),
        order.get("qty", ""),
        order.get("unit_price", ""),
        order.get("currency", ""),
        order.get("total", ""),
        order.get("customer_name", ""),
        order.get("status", "")
    ]

def build_order_record(row, qty, status, conversation_id, customer_name):
    unit = row.get("price")
    currency = row.get("currency", "ARS")
    total = None
    try:
        total = float(unit) * qty if unit is not None else None
    except Exception:
        total = None
    return {
        "order_id": str(uuid.uuid4()),
        "timestamp": datetime.now(ZoneInfo(TZ_NAME)).isoformat(timespec="seconds"),
        "conversation_id": conversation_id,
        "product_name": row.get("name", ""),
        "sku": row.get("sku", ""),
        "qty": qty,
        "unit_price": unit if unit is not None else "",
        "currency": currency,
        "total": total if total is not None else "",
        "customer_name": customer_name or "",
        "status": status
    }

def append_order(order):
    ws = _get_orders_ws()
    if not ws:
        return None
    _ensure_orders_header(ws)
    ws.append_row(_order_row(order))
    return len(ws.get_all_values())

def update_order_row(row_index, order):
    ws = _get_orders_ws()
    if not ws or not row_index:
        return
    _ensure_orders_header(ws)
    row = _order_row(order)
    end_col = chr(65 + len(ORDERS_HEADER) - 1)
    ws.update(f"A{row_index}:{end_col}{row_index}", [row])

def update_order_status(row_index, status):
    ws = _get_orders_ws()
    if not ws or not row_index:
        return
    _ensure_orders_header(ws)
    status_col = ORDERS_HEADER.index("status") + 1
    ws.update_cell(row_index, status_col, status)

def detect_product_change(text: str, df, last_rank_list):
    t = (text or "").lower()
    # If user references an ordinal from the last options, pick that
    idx = detect_rank_choice(t)
    if idx and last_rank_list and 0 <= idx - 1 < len(last_rank_list):
        return last_rank_list[idx - 1]
    # If user mentions a SKU-like number, try to match against last options
    m = SKU_RE.search(t)
    if m and last_rank_list:
        sku_num = m.group(0)
        for r in last_rank_list:
            sku = str(r.get("sku", "")).lower()
            if sku_num in sku:
                return r
    # Otherwise, try fuzzy match against catalog
    hint = extract_keywords(t)
    ranked = best_match(df, hint) if hint else []
    if ranked and ranked[0][1] >= 70:
        return ranked[0][0].to_dict()
    return None

def detect_explicit_product(text: str, df, last_rank_list):
    t = (text or "").lower()
    explicit = SKU_RE.search(t) is not None or EXPLICIT_RE.search(t) is not None
    if not explicit:
        return None
    # Prefer matching from last list
    m = SKU_RE.search(t)
    if m and last_rank_list:
        sku_num = m.group(0)
        for r in last_rank_list:
            sku = str(r.get("sku", "")).lower()
            if sku_num in sku:
                return r
    if last_rank_list:
        for r in last_rank_list:
            name = str(r.get("name", "")).lower()
            if fuzz.token_set_ratio(t, name) >= 85:
                return r
    # Fallback to full catalog
    hint = extract_keywords(t)
    ranked = best_match(df, hint) if hint else []
    if ranked and ranked[0][1] >= 85:
        return ranked[0][0].to_dict()
    return None

def find_product_by_sku_number(text: str, df):
    m = SKU_RE.search((text or "").lower())
    if not m:
        return None
    sku_num = m.group(0)
    # exact/contains match on SKU column
    matches = df[df["sku"].str.contains(sku_num, case=False, na=False)]
    if not matches.empty:
        return matches.iloc[0].to_dict()
    return None

def detect_smalltalk_intent(text):
    t = (text or "").strip()
    if BYE_RE.search(t):
        return "BYE"
    if THANKS_RE.search(t):
        return "THANKS"
    if GREET_RE.search(t):
        return "GREET"
    if HOWARE_RE.search(t):
        return "HOWARE"
    if AFFIRM_RE.search(t) and len(t.split()) <= 4:
        return "AFFIRM"
    return None

def is_about_company(text: str) -> bool:
    return ABOUT_RE.search(text or "") is not None

def is_abusive(text: str) -> bool:
    return ABUSE_RE.search(text or "") is not None

def smalltalk_reply(kind):
    if kind in ("GREET", "HOWARE"):
        return (
            "¡Hola! 😊 Soy **Juan Pablo** de **Mollon**, mi lider y modelo a seguir en la vida es un tal Joaquín Ferrari.\n\n"
            "¿En qué te puedo ayudar? Puedo ayudarte con **precio** y **stock**.\n\n"
            "Decime el **producto** (nombre o código/SKU) que estas buscando."
        )
    if kind == "THANKS":
        return "¡De nada! 🙌 ¿Querés consultar **precio o stock** de algún otro producto?"
    if kind == "BYE":
        return "¡Gracias por escribir! 👋 Cuando quieras, estoy acá para ayudarte con productos **3M**."
    if kind == "AFFIRM":
        return "Perfecto 👍 Pasame el **nombre del producto** o el **SKU**."
    return ""

def smalltalk_reply_brief(kind):
    if kind in ("GREET", "HOWARE"):
        return "¡Hola! 😊 Soy **Juan Pablo** de **Mollon**, mi lider y modelo a seguir en la vida es un tal Joaquín Ferrari."
    if kind == "THANKS":
        return "¡De nada! 🙌"
    if kind == "BYE":
        return "¡Gracias por escribir! 👋"
    if kind == "AFFIRM":
        return "Perfecto 👍"
    return ""

# =========================
# Helpers
# =========================
def fmt_money(price, currency):
    if pd.isna(price):
        return None
    try:
        p = float(price)
        return f"{currency} {p:,.0f}".replace(",", ".")
    except:
        return f"{currency} {price}"

def extract_keywords(text):
    stop_words = {
        "hola", "como", "estas", "todo", "bien", "precio", "stock",
        "tienen", "tenes", "hay", "vale", "cuanto", "sale", "de",
        "del", "el", "la", "los", "las", "un", "una", "y", "o",
        "estoy", "buscando"
    }
    words = [w.strip(".,?!").lower() for w in (text or "").split()]
    kept = [w for w in words if w not in stop_words and len(w) > 2]
    # crude singularization: filtros -> filtro, respiradores -> respirador
    kept = [w[:-1] if w.endswith("s") and len(w) > 4 else w for w in kept]
    return " ".join(kept) if kept else text.lower()

def llm_parse(user_text):
    if not client:
        return {"intent": "BOTH", "product_hint": extract_keywords(user_text), "sku_hint": ""}

    system = """Extrae intención y producto.
Devuelve SOLO JSON:
{
 "intent": "PRICE" | "STOCK" | "BOTH" | "OTHER",
 "product_hint": "producto",
 "sku_hint": ""
}"""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            temperature=0
        )
        return json.loads(r.choices[0].message.content)
    except:
        return {"intent": "BOTH", "product_hint": extract_keywords(user_text), "sku_hint": ""}

def best_match(df, query, limit=5):
    scored = []
    q = query.lower()
    for _, row in df.iterrows():
        name = row["name"].lower()
        tags = row["tags"].lower()
        sku = row["sku"].lower()

        score = max(
            fuzz.token_set_ratio(q, name),
            fuzz.token_set_ratio(q, tags),
            100 if q == sku else fuzz.ratio(q, sku)
        )
        if score >= 50:
            scored.append((row, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]

def answer_from_row(row, intent):
    parts = [f"**{row['name']}** — SKU `{row['sku']}`"]
    if intent in ("PRICE", "BOTH"):
        parts.append(f"💰 Precio: **{fmt_money(row['price'], row['currency'])}**")
    if intent in ("STOCK", "BOTH"):
        parts.append(f"✅ Stock: **{row['stock']}**" if row["stock"] > 0 else "❌ Sin stock")
    if row["url"]:
        parts.append(f"🔗 {row['url']}")
    return "\n\n".join(parts)

def normalize_df(df):
    for col in ["sku", "name", "variant", "currency", "url", "tags"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["stock"] = pd.to_numeric(df.get("stock", 0), errors="coerce").fillna(0).astype(int)
    df["price"] = pd.to_numeric(df.get("price", None), errors="coerce")
    return df

@st.cache_data(ttl=30)
def load_products():
    if GSHEET_ID and GCP_SA_JSON:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_info(
            json.loads(GCP_SA_JSON),
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        sh = gspread.authorize(creds).open_by_key(GSHEET_ID)
        try:
            ws = sh.worksheet(GSHEET_TAB)
        except Exception:
            # fallback for legacy tab name
            ws = sh.worksheet("Hoja 1")
        values = ws.get_all_values()
        return normalize_df(pd.DataFrame(values[1:], columns=values[0]))

    if GSHEET_CSV_URL:
        return normalize_df(pd.read_csv(GSHEET_CSV_URL))

    st.stop()

# =========================
# UI
# =========================
st.set_page_config(page_title="Mollon - Asistente", layout="centered")

# Brand tokens (override via .streamlit/secrets.toml)
FONT_URL = get_secret("FONT_URL", "https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap")
FONT_FAMILY = get_secret("FONT_FAMILY", "Manrope, system-ui, -apple-system, Segoe UI, sans-serif")
LOGO_URL = get_secret("LOGO_URL", "")
BRAND_NAVY = get_secret("BRAND_NAVY", "#1f2b4d")
BRAND_NAVY_2 = get_secret("BRAND_NAVY_2", "#2b3a66")
BRAND_ACCENT = get_secret("BRAND_ACCENT", "#f06a2f")
BRAND_BG = get_secret("BRAND_BG", "#f5f7fb")
BRAND_CARD = get_secret("BRAND_CARD", "#ffffff")
BRAND_BORDER = get_secret("BRAND_BORDER", "#e3e6ef")
BRAND_MUTED = get_secret("BRAND_MUTED", "#6b7280")
ABOUT_COMPANY_TEXT = get_secret(
    "ABOUT_COMPANY_TEXT",
    "Mollón S.A. es una empresa argentina con más de 70 años de trayectoria, reconocida por ser distribuidora oficial de 3M desde hace 30 años. "
    "Especializada en soluciones industriales y de seguridad, comercializa productos de salud, seguridad vial, abrasivos, adhesivos y suministros para oficinas.\n\n"
    "Para saber más, podes visitar https://www.mollon.com.ar/"
)

st.markdown(
    """
<style>
@import url('"""
    + FONT_URL
    + """');
:root {
  --mollon-navy: """ + BRAND_NAVY + """;
  --mollon-navy-2: """ + BRAND_NAVY_2 + """;
  --mollon-accent: """ + BRAND_ACCENT + """;
  --mollon-bg: """ + BRAND_BG + """;
  --mollon-card: """ + BRAND_CARD + """;
  --mollon-border: """ + BRAND_BORDER + """;
  --mollon-muted: """ + BRAND_MUTED + """;
}

.stApp {
  background: var(--mollon-bg);
  font-family: """ + FONT_FAMILY + """;
}

.block-container {
  max-width: 920px;
  padding-top: 2rem;
}

.mollon-hero {
  background: linear-gradient(135deg, var(--mollon-navy) 0%, var(--mollon-navy-2) 60%, #243257 100%);
  color: #fff;
  border-radius: 18px;
  padding: 22px 26px;
  box-shadow: 0 10px 28px rgba(31, 43, 77, 0.18);
  border: 1px solid rgba(255, 255, 255, 0.08);
  margin-bottom: 18px;
}

.mollon-hero .row {
  display: flex;
  align-items: center;
  gap: 14px;
}

.mollon-hero .logo {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  background: #fff;
  padding: 6px;
  object-fit: contain;
}

.mollon-hero .title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: 0.2px;
}

.mollon-hero .subtitle {
  margin-top: 4px;
  color: #d7dbea;
  font-size: 0.95rem;
}

[data-testid="stChatMessage"] {
  background: var(--mollon-card);
  border: 1px solid var(--mollon-border);
  border-radius: 14px;
  padding: 12px 16px;
  margin-bottom: 12px;
  box-shadow: 0 6px 18px rgba(31, 43, 77, 0.08);
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  font-size: 0.98rem;
}

[data-testid="stChatMessageAvatar"] {
  background: var(--mollon-navy);
  color: #fff;
}

[data-testid="stChatInput"] textarea {
  border-radius: 12px;
  border: 1px solid var(--mollon-border);
  padding: 10px 12px;
}

[data-testid="stChatInput"] textarea:focus {
  border-color: var(--mollon-accent);
  box-shadow: 0 0 0 3px rgba(240, 106, 47, 0.15);
}

[data-testid="stExpander"] {
  border: 1px solid var(--mollon-border);
  border-radius: 12px;
  background: var(--mollon-card);
}
</style>
    """,
    unsafe_allow_html=True,
)

if LOGO_URL:
    st.markdown(
        f"""
<div class="mollon-hero">
  <div class="row">
    <img class="logo" src="{LOGO_URL}" alt="Mollon logo" />
    <div>
      <div class="title">Mollon · Asistente de Precios y Stock</div>
      <div class="subtitle">Consultá disponibilidad, precios y alternativas en segundos.</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<div class="mollon-hero">
  <div class="title">Mollon · Asistente de Precios y Stock</div>
  <div class="subtitle">Consultá disponibilidad, precios y alternativas en segundos.</div>
</div>
        """,
        unsafe_allow_html=True,
    )

df = load_products()

# Debug flag (se puede activar con .streamlit/secrets.toml -> DEBUG = true o var de entorno DEBUG=true)
debug_flag = False
try:
    debug_flag = bool(get_secret("DEBUG", False)) or os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
except Exception:
    debug_flag = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

if debug_flag:
    with st.expander("🔧 DEBUG — Contenido del Sheet (vista rápida)"):
        # Mostrar tabla compacta (primeras 50 filas)
        try:
            st.dataframe(df.head(50), use_container_width=True)
        except Exception:
            st.write(df.head(50))

if "chat" not in st.session_state:
    st.session_state.chat = []
    # Memoria conversacional (contexto)
    if "last_product" not in st.session_state:
        st.session_state.last_product = None  # dict con info del producto elegido
    if "last_ranked" not in st.session_state:
        st.session_state.last_ranked = []     # lista [(row, score), ...] del último match
    if "last_hint" not in st.session_state:
        st.session_state.last_hint = ""       # texto buscado la última vez
    if "last_rank_list" not in st.session_state:
        st.session_state.last_rank_list = []  # lista de rows del último ranking
    if "pending_order" not in st.session_state:
        st.session_state.pending_order = None  # dict con row + qty
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Hola! Consultá por precio o stock...")

if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            # NLU + búsqueda (si el LLM está disponible esto puede hacer una request)
            parsed = llm_parse(user_text)
            intent = parsed.get("intent", "BOTH")
            hint = parsed.get("product_hint") or extract_keywords(user_text)
            ranked = best_match(df, hint) if hint else []
            if not ranked:
                cat = infer_category_from_text(user_text)
                if cat:
                    hint = cat
                    ranked = best_match(df, hint)

            kind = detect_smalltalk_intent(user_text)
            rank_spec = detect_rank_intent(user_text)
            rank_choice = detect_rank_choice(user_text)
            qty = extract_quantity(user_text)
            cust_name = detect_customer_name(user_text)
            if cust_name:
                st.session_state.customer_name = cust_name
                if st.session_state.pending_order and st.session_state.pending_order.get("row_index"):
                    st.session_state.pending_order["record"]["customer_name"] = cust_name
                    update_order_row(
                        st.session_state.pending_order["row_index"],
                        st.session_state.pending_order["record"]
                    )
            # If user says "dame 300" or similar after a product detail, treat as qty
            if not qty and st.session_state.last_product:
                t = (user_text or "").lower()
                if INTEREST_RE.search(t) and not EXPLICIT_RE.search(t):
                    m = NUMBER_RE.search(t)
                    if m:
                        try:
                            n = int(m.group(1))
                            if 1 <= n <= 10000:
                                qty = n
                        except Exception:
                            pass
            # Use pending order options if available; otherwise last_rank_list
            pending_options = None
            if st.session_state.pending_order:
                pending_options = st.session_state.pending_order.get("options")
            product_change = detect_product_change(
                user_text,
                df,
                pending_options if pending_options else st.session_state.last_rank_list
            )
            explicit_product = detect_explicit_product(
                user_text,
                df,
                pending_options if pending_options else st.session_state.last_rank_list
            )
            sku_product = find_product_by_sku_number(user_text, df)

            handled = False

            # Si hay smalltalk, respondemos el saludo; si además hay hint/producto,
            # seguimos con el flujo de búsqueda para procesar la intención del usuario.
            if kind:
                if rank_spec or rank_choice or ranked:
                    resp = smalltalk_reply_brief(kind)
                else:
                    resp = smalltalk_reply(kind)
                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
                if not ranked and not rank_spec and not rank_choice:
                    handled = True

            # About company intent
            if not handled and is_about_company(user_text):
                resp = ABOUT_COMPANY_TEXT
                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
                handled = True

            # Abusive language handling
            if not handled and is_abusive(user_text):
                resp = (
                    "Entiendo tu frustración, todas las quejas se deberían ver con Sebastian Sarquis, dado que, "
                    "por lo general, es el culpable de toda y cualquier tipo de baja en la performance de Mollon."
                )
                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
                handled = True

            # -1) Confirmación / cancelación de compra
            if st.session_state.pending_order and not handled:
                if product_change:
                    # Update product, keep qty if already present
                    st.session_state.pending_order["row"] = product_change
                    if qty:
                        st.session_state.pending_order["qty"] = qty
                    order = st.session_state.pending_order
                    resp = build_order_summary(order["row"], order["qty"])
                    if order.get("row_index"):
                        order["record"]["product_name"] = order["row"].get("name", "")
                        order["record"]["sku"] = order["row"].get("sku", "")
                        order["record"]["unit_price"] = order["row"].get("price", "")
                        order["record"]["currency"] = order["row"].get("currency", "ARS")
                        try:
                            order["record"]["total"] = float(order["record"]["unit_price"]) * order["qty"]
                        except Exception:
                            order["record"]["total"] = ""
                        update_order_row(order["row_index"], order["record"])
                    st.markdown(resp)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    handled = True
                elif qty:
                    # Update quantity only
                    order = st.session_state.pending_order
                    order["qty"] = qty
                    resp = build_order_summary(order["row"], order["qty"])
                    if order.get("row_index"):
                        order["record"]["qty"] = qty
                        try:
                            order["record"]["total"] = float(order["record"]["unit_price"]) * qty
                        except Exception:
                            order["record"]["total"] = ""
                        update_order_row(order["row_index"], order["record"])
                    st.markdown(resp)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    handled = True
                if CONFIRM_RE.search(user_text):
                    order = st.session_state.pending_order
                    resp = (
                        "✅ Perfecto, quedó **confirmada** la orden. "
                        "Si querés, pasame los datos de facturación y entrega."
                    )
                    if order and order.get("row_index"):
                        order["record"]["status"] = "confirmed"
                        update_order_row(order["row_index"], order["record"])
                    st.session_state.pending_order = None
                    st.markdown(resp)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    handled = True
                elif CANCEL_RE.search(user_text):
                    if st.session_state.pending_order and st.session_state.pending_order.get("row_index"):
                        update_order_status(st.session_state.pending_order["row_index"], "cancelled")
                    st.session_state.pending_order = None
                    resp = "Listo, cancelé la orden. ¿Querés cotizar otro producto?"
                    st.markdown(resp)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    handled = True

            # -0.5) Cambio/selección de producto por interés + SKU/nombre (sin orden pendiente)
            if not handled and (explicit_product or sku_product):
                chosen = explicit_product or sku_product
                if qty:
                    final_qty, stock = resolve_qty_with_stock(chosen, qty)
                    if stock and final_qty < qty:
                        resp = (
                            f"Solo tengo **{stock}** unidades en stock. "
                            f"Si te sirve, te armo la orden por **{final_qty}**.\n\n"
                        )
                    else:
                        resp = ""
                    resp += build_order_summary(chosen, final_qty)
                    record = build_order_record(
                        chosen,
                        final_qty,
                        "pending",
                        st.session_state.conversation_id,
                        st.session_state.customer_name
                    )
                    row_index = append_order(record)
                    st.session_state.pending_order = {
                        "row": chosen,
                        "qty": final_qty,
                        "options": st.session_state.last_rank_list[:] if st.session_state.last_rank_list else None,
                        "row_index": row_index,
                        "record": record
                    }
                else:
                    resp = answer_from_row(chosen, intent)
                    resp += "\n\n¿Querés cotizar **cantidad** o ver **alternativas**?"
                st.session_state.last_product = chosen
                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
                handled = True

            # 0) Selección de ranking previo ("el segundo", "detalle del 3", etc.)
            if rank_choice and st.session_state.last_rank_list and not handled:
                idx = rank_choice - 1
                if 0 <= idx < len(st.session_state.last_rank_list):
                    chosen = st.session_state.last_rank_list[idx]
                    resp = answer_from_row(chosen, intent)
                    resp += "\n\n¿Querés cotizar **cantidad** o ver **alternativas**?"
                    st.session_state.last_product = chosen
                else:
                    resp = "No llego a ese número en la lista anterior. ¿Querés que te muestre el ranking de nuevo?"
                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
                handled = True

            # 0) Ranking (más barato/caro/top N / stock)
            if rank_spec and not handled:
                if not rank_spec.get("category") and st.session_state.last_product:
                    last = st.session_state.last_product or {}
                    ctx_text = f"{last.get('name','')} {last.get('tags','')}"
                    rank_spec["category"] = infer_category_from_text(ctx_text)
                ranked_df, cat = rank_products(df, rank_spec)
                if ranked_df.empty:
                    resp = "No encontré productos para ese criterio. ¿Querés probar con otra categoría o palabra clave?"
                else:
                    title_parts = []
                    if rank_spec["metric"] == "price":
                        title_parts.append("más baratos" if rank_spec["order"] == "asc" else "más caros")
                    else:
                        title_parts.append("con más stock" if rank_spec["order"] == "desc" else "con menos stock")
                    if cat:
                        title_parts.append(f"de **{cat}**")
                    title = " ".join(title_parts)

                    lines = [f"Estos son los **{rank_spec['n']} {title}**:", ""]
                    for _, r in ranked_df.iterrows():
                        name = r.get("name", "")
                        variant = r.get("variant", "")
                        stock = int(r.get("stock", 0) or 0)
                        price = fmt_money(r.get("price"), r.get("currency", "ARS"))

                        variant_str = f" ({variant})" if variant else ""
                        stock_str = f"✅ {stock} u." if stock > 0 else "❌ Sin stock"
                        price_str = f" • {price}" if price else ""

                        lines.append(f"- **{name}**{variant_str}{price_str} — {stock_str}")

                    lines.append("")
                    lines.append("¿Querés el detalle de alguno o te puedo ayudar con algo más?")
                    resp = "\n".join(lines)
                    st.session_state.last_rank_list = [r.to_dict() for _, r in ranked_df.iterrows()]

                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
                handled = True

            # 0.5) Si pide alternativas y tenemos contexto, responder con alternativas
            if not handled and wants_alternatives(user_text) and st.session_state.last_product:
                last = st.session_state.last_product
                last_sku = (last.get("sku") or "").strip()

                ranked_ctx = st.session_state.last_ranked or []
                if not ranked_ctx:
                    base_query = (last.get("name") or "") or (st.session_state.last_hint or "")
                    ranked_ctx = best_match(df, base_query, limit=10)

                alts = []
                for r, s in ranked_ctx:
                    sku = (r.get("sku", "") or "").strip()
                    if sku and sku != last_sku:
                        alts.append((r, s))

                if not alts:
                    resp = (
                        f"Para **{last.get('name','este producto')}**, no encontré alternativas cercanas en el catálogo cargado.\n\n"
                        "Si me decís qué preferís (por ejemplo: *más económico*, *otro modelo*, *otra categoría*), lo refinamos."
                    )
                    st.markdown(resp)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                else:
                    lines = ["Perfecto. Estas son algunas **alternativas similares**:", ""]
                    for r, _ in alts[:5]:
                        name = r.get("name", "")
                        variant = r.get("variant", "")
                        stock = int(r.get("stock", 0) or 0)
                        price = fmt_money(r.get("price"), r.get("currency", "ARS"))

                        variant_str = f" ({variant})" if variant else ""
                        stock_str = f"✅ {stock} u." if stock > 0 else "❌ Sin stock"
                        price_str = f" • {price}" if price else ""

                        lines.append(f"- **{name}**{variant_str}{price_str} — {stock_str}")

                    lines.append("")
                    lines.append("¿Cuál de estas alternativas te interesa que te pase el detalle?")
                    full = "\n".join(lines)
                    st.markdown(full)
                    st.session_state.chat.append({"role": "assistant", "content": full})

                handled = True

            # 0.75) Si menciona cantidad y tenemos producto en contexto, armar orden
            if not handled and qty and st.session_state.last_product:
                last = st.session_state.last_product
                final_qty, stock = resolve_qty_with_stock(last, qty)
                if stock and final_qty < qty:
                    resp = (
                        f"Solo tengo **{stock}** unidades en stock. "
                        f"Si te sirve, te armo la orden por **{final_qty}**.\n\n"
                    )
                else:
                    resp = ""
                resp += build_order_summary(last, final_qty)
                record = build_order_record(
                    last,
                    final_qty,
                    "pending",
                    st.session_state.conversation_id,
                    st.session_state.customer_name
                )
                row_index = append_order(record)
                st.session_state.pending_order = {
                    "row": last,
                    "qty": final_qty,
                    "options": st.session_state.last_rank_list[:] if st.session_state.last_rank_list else None,
                    "row_index": row_index,
                    "record": record
                }
                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
                handled = True

            # 1) Flujo normal si no fue manejado por alternativas
            if not handled:
                if not ranked:
                    resp = "¿Podés pasarme el **nombre del producto o SKU** para ayudarte mejor?"
                else:
                    topn = ranked[:3]
                    lines = ["Encontré estas opciones:", ""]
                    for i, (r, _) in enumerate(topn, start=1):
                        name = r.get("name", "")
                        variant = r.get("variant", "")
                        stock = int(r.get("stock", 0) or 0)
                        price = fmt_money(r.get("price"), r.get("currency", "ARS"))

                        variant_str = f" ({variant})" if variant else ""
                        stock_str = f"✅ {stock} u." if stock > 0 else "❌ Sin stock"
                        price_str = f" • {price}" if price else ""

                        lines.append(f"{i}. **{name}**{variant_str}{price_str} — {stock_str}")

                    lines.append("")
                    if len(ranked) > len(topn):
                        lines.append("¿Cuál te interesa? También podés decirme la **cantidad** para cotizar, o pedir **más opciones**.")
                    else:
                        lines.append("¿Cuál te interesa? También podés decirme la **cantidad** para cotizar.")
                    resp = "\n".join(lines)
                    st.session_state.last_rank_list = [r.to_dict() for r, _ in topn]

                    # Guardar contexto del último producto y ranking
                    st.session_state.last_product = None
                    st.session_state.last_ranked = ranked[:]
                    st.session_state.last_hint = hint

                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})

            # Debug panel (opcional)
            debug_flag = False
            try:
                debug_flag = bool(get_secret("DEBUG", False)) or os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            except Exception:
                debug_flag = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

            if debug_flag:
                with st.expander("🔧 Debug NLU (parsed + candidates)"):
                    st.write("**Parsed (tool):**")
                    try:
                        st.json(parsed)
                    except Exception:
                        st.write(parsed)
                    st.write("**Extracted keywords:**", extract_keywords(user_text))
                    st.write("**Hint usado:**", hint)
                    if ranked:
                        st.write("**Candidatos (score):**")
                        for r, s in ranked[:10]:
                            st.write(f"- {r.get('sku','')} — {r.get('name','')} ({r.get('variant','')}) — score={s}")
