import os
import json
import re
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
GSHEET_TAB = get_secret("GSHEET_TAB", "products")
GCP_SA_JSON = get_secret("GCP_SERVICE_ACCOUNT_JSON", "").strip()
GSHEET_CSV_URL = get_secret("GSHEET_CSV_URL", "").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# Small talk router
# =========================
GREET_RE = re.compile(r"\b(hola|buenas|buen día|buen dia|buenas tardes|buenas noches|qué tal|que tal|hey)\b", re.I)
HOWARE_RE = re.compile(r"\b(c[oó]mo est[aá]s|como and[aá]s|todo bien|qué onda|que onda)\b", re.I)
THANKS_RE = re.compile(r"\b(gracias|muchas gracias|mil gracias|genial gracias|perfecto gracias)\b", re.I)
BYE_RE = re.compile(r"\b(chau|adios|adiós|hasta luego|hasta pronto|nos vemos|bye)\b", re.I)
AFFIRM_RE = re.compile(r"^(ok|oka|dale|de una|joya|perfecto|genial|listo|bárbaro|barbaro)\b", re.I)
ALT_RE = re.compile(r"\b(alternativ|alternativas|otros|otras|similares|parecidos|ver alternativas)\b", re.I)
QUOTE_RE = re.compile(r"\b(cotiz|cotizar|presupuesto|precio por|por (\d+)|cantidad|x(\d+))\b", re.I)
RANK_RE = re.compile(r"\b(m[aá]s barato|mas barato|m[aá]s caro|mas caro|barat[oa]s?|car[oa]s?|top|mejores?|peores?|mostrame|muestra|listar|listame|ordenad[oa]s?)\b", re.I)
TOPN_RE = re.compile(r"\btop\s*(\d+)|(\d+)\s*(m[aá]s )?(baratos?|caros?)\b", re.I)
STOCK_RANK_RE = re.compile(r"\b(m[aá]s stock|con m[aá]s stock|mayor stock|menor stock|menos stock|sin stock)\b", re.I)
RANK_PICK_RE = re.compile(r"\b(detalle|detalle del|pasame|mostrame|ver)\b", re.I)
ORDINAL_RE = re.compile(r"\b(primero|segundo|tercero|cuarto|quinto|sexto|séptimo|septimo|octavo|noveno|décimo|decimo)\b", re.I)
NUMBER_RE = re.compile(r"\b(\d+)\b")

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
            if kw in t:
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
        "primero": 1, "segundo": 2, "tercero": 3, "cuarto": 4, "quinto": 5,
        "sexto": 6, "séptimo": 7, "septimo": 7, "octavo": 8, "noveno": 9,
        "décimo": 10, "decimo": 10
    }
    return mapping.get(word, None)

def detect_rank_choice(text: str):
    t = (text or "").lower()
    if not RANK_PICK_RE.search(t):
        return None
    m = ORDINAL_RE.search(t)
    if m:
        return _ordinal_to_index(m.group(1))
    m = NUMBER_RE.search(t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
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

def smalltalk_reply(kind):
    if kind in ("GREET", "HOWARE"):
        return (
            "¡Hola! 😊 Soy **Juan Pablo** de **Mollon**, mi lider y modelo a seguir en la vida es un tal Joaquín Ferrari.\n\n"
            "¿En qué te puedo ayudar? Puedo ayudarte con **precio** y **stock**.\n\n"
            "Decime el **producto** (nombre o código/SKU) y, si aplica, la **variante**."
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
        "del", "el", "la", "los", "las", "un", "una", "y", "o"
    }
    words = [w.strip(".,?!").lower() for w in (text or "").split()]
    kept = [w for w in words if w not in stop_words and len(w) > 2]
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
        ws = gspread.authorize(creds).open_by_key(GSHEET_ID).worksheet(GSHEET_TAB)
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

            kind = detect_smalltalk_intent(user_text)
            rank_spec = detect_rank_intent(user_text)
            rank_choice = detect_rank_choice(user_text)

            # Si hay smalltalk, respondemos el saludo; si además hay hint/producto,
            # seguimos con el flujo de búsqueda para procesar la intención del usuario.
            if kind:
                if rank_spec or rank_choice:
                    resp = smalltalk_reply_brief(kind)
                else:
                    resp = smalltalk_reply(kind)
                st.markdown(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})

            handled = False

            # 0) Selección de ranking previo ("el segundo", "detalle del 3", etc.)
            if rank_choice and st.session_state.last_rank_list:
                idx = rank_choice - 1
                if 0 <= idx < len(st.session_state.last_rank_list):
                    chosen = st.session_state.last_rank_list[idx]
                    resp = answer_from_row(chosen, intent)
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

            # 1) Flujo normal si no fue manejado por alternativas
            if not handled:
                if not ranked:
                    resp = "¿Podés pasarme el **nombre del producto o SKU** para ayudarte mejor?"
                else:
                    best_row = ranked[0][0]
                    resp = answer_from_row(best_row, intent)
                    resp += "\n\n¿Querés cotizar **cantidad** o ver **alternativas**?"

                    # Guardar contexto del último producto y ranking
                    st.session_state.last_product = best_row.to_dict()
                    st.session_state.last_ranked = ranked[:]
                    st.session_state.last_hint = hint
                    st.session_state.last_rank_list = []

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
