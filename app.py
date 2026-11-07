import inspect
import json
import math
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from html import escape as html_escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3.util.connection as urllib3_conn

# Bazı ağlarda IPv6 (AAAA) DNS sorgusu ~10 sn takılıyor; HTTP isteklerinde yalnızca IPv4 kullan
urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

# set_page_config ilk Streamlit komutu olmalı
st.set_page_config(
    page_title="SPACE GRAVE | SP4C3 GR4V3 - NASA Space Apps JPL Command Center",
    page_icon="🛰️",
    layout="wide"
)

try:
    from sgp4.api import Satrec, SatrecArray, jday
    from scipy.spatial import cKDTree
except ImportError:
    st.error("Eksik paketler var. Terminalde çalıştır: pip install streamlit sgp4 scipy numpy pandas plotly requests")
    st.stop()

RE = 6378.137  # km, Earth equatorial radius
WGS84_F = 1 / 298.257223563  # Dünya basıklığı
CREWED_PREFIXES = ("ISS", "CSS", "TIANGONG")


# Streamlit sürümleri arası uyumluluk:
# yeni sürümlerde use_container_width kaldırıldı -> width="stretch"
def stretch(fn):
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


# ANTI-BAN HEADERS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

# FALLBACK TLE VERİSİ
FALLBACK_TLE = """ISS (ZARYA)
1 25544U 98067A   23304.53761765  .00016717  00000-0  30043-3 0  9997
2 25544  51.6416 270.8359 0001402 121.3653 286.0718 15.50150914422687
HST (HUBBLE)
1 20580U 90037B   23304.53580556  .00001099  00000-0  48316-4 0  9991
2 20580  28.4695 244.4019 0002271 270.6277 159.2064 15.08779951465225
TIANGONG
1 48274U 21035A   23304.51528935  .00011500  00000-0  17529-3 0  9997
2 48274  41.4746 308.8687 0002213 133.0232 236.4385 15.59751992143493
"""

def _lang_from_url():
    # Navigasyondaki bağlantı sayfayı yenilediği için seçili dil ?lang= ile taşınır
    try:
        return "en" if str(st.query_params.get("lang", "")).lower() == "en" else "tr"
    except Exception:
        return "tr"


# --- SESSION STATE ---
for key, default in {
    "intro_passed": False,
    "lang": _lang_from_url(),
    "active_page": "📊 General Status Overview",
    "defense_active": False,
    "conj_df": None,
    "conj_meta": None,
    "reentry": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "bot_open" not in st.session_state:
    st.session_state.bot_open = False
if "bot_log" not in st.session_state:
    # Kayıtlar dilden bağımsız tutulur (soru kimliği); gösterim anında seçili dile göre üretilir
    st.session_state.bot_log = [{"r": "bot", "g": True}]


# --- DİL (TÜR / ENG) ---
def T(tr, en):
    return tr if st.session_state.lang == "tr" else en


def _keyed(key):
    # st.container(key=...) eski Streamlit sürümlerinde yok -> normal kutu olarak çalışır
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def _bot_ph():
    return T("💬 Başka bir soru seç…", "💬 Pick another question…")


def set_lang(lang):
    st.session_state.lang = lang
    st.session_state["sg_bot_pick"] = _bot_ph()


def _i18n_html(html, pairs):
    # İngilizce yazılmış gömülü (iframe) sayfaları Türkçe moduna çevirir
    if st.session_state.lang != "tr":
        return html
    for a, b in pairs:
        html = html.replace(a, b)
    return html
def render_live_threat_panel():
    st.markdown("""
    <div style="border: 1px solid rgba(252,61,33,0.5); background: linear-gradient(90deg, rgba(252,61,33,0.15) 0%, rgba(252,61,33,0.02) 100%); padding: 14px 20px; border-radius: 12px; display: flex; align-items: center; margin-bottom: 24px;">
        <div style="width: 10px; height: 10px; background: #fc3d21; border-radius: 50%; box-shadow: 0 0 14px #fc3d21; margin-right: 16px;"></div>
        <div>
            <div style="color: #fc3d21; font-weight: 800; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;">Canlı Tehdit Radarı Aktif</div>
            <div style="color: #e2e8f0; font-size: 14px; margin-top: 3px;">50 km güvenlik küresi devrede. Olası yakınlaşmalar izleniyor.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
def render_lang_switch(right, bottom):
    """Sağ altta sabit ENG / TÜR kutucukları (giriş ve canlı izleme sayfasında ortak)."""
    active = "sg_lang_en" if st.session_state.lang == "en" else "sg_lang_tr"
    with _keyed("sg_lang"):
        st.button("ENG", key="sg_lang_en", on_click=set_lang, args=("en",))
        st.button("TÜR", key="sg_lang_tr", on_click=set_lang, args=("tr",))
    st.markdown(f"""
    <style>
    .st-key-sg_lang {{ position:fixed; right:{right}; bottom:{bottom}; z-index:1000000; display:flex;
      flex-direction:row !important; width:auto !important; gap:6px !important; padding:5px; border-radius:12px;
      background:rgba(6,12,30,.84); border:1px solid rgba(56,189,248,.38); backdrop-filter:blur(12px);
      -webkit-backdrop-filter:blur(12px); box-shadow:0 8px 28px rgba(0,0,0,.55); }}
    .st-key-sg_lang > div {{ width:auto !important; }}
    .st-key-sg_lang .stButton > button {{ min-height:30px; height:30px; padding:0 14px; border-radius:8px;
      background:transparent; border:1px solid transparent; box-shadow:none; }}
    .st-key-sg_lang .stButton > button p {{ font-size:12px; font-weight:700; letter-spacing:.12em; color:#94a3b8; margin:0; }}
    .st-key-sg_lang .stButton > button:hover {{ border-color:#38bdf8; }}
    .st-key-{active} .stButton > button {{ background:rgba(56,189,248,.22); border-color:#38bdf8; }}
    .st-key-{active} .stButton > button p {{ color:#fff; }}
    </style>
    """, unsafe_allow_html=True)


# --- STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0b1329 0%, #040817 50%, #000000 100%);
        color: #f3f4f6;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stSidebar"] { display: none; }
    .nasa-glass-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(7, 11, 25, 0.99) 100%);
        backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(0, 51, 102, 0.8);
        padding: 22px; border-radius: 20px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(11, 75, 148, 0.15);
        text-align: center; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden; margin-bottom: 10px;
    }
    .fin-card {
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(16, 185, 129, 0.1) !important;
    }
    .metric-title { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; }
    .metric-value { color: #38bdf8; font-size: 26px; font-weight: 900; margin-top: 6px; text-shadow: 0 0 20px rgba(56,189,248,0.7); }
    .metric-value-fin { color: #10b981; font-size: 26px; font-weight: 900; margin-top: 6px; text-shadow: 0 0 20px rgba(16,185,129,0.7); }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #fc3d21 0%, #990000 100%);
        color: white; border: 1px solid rgba(252,61,33,0.8); border-radius: 12px; font-weight: 900; padding: 0.85rem 1.8rem;
        box-shadow: 0 0 30px rgba(252, 61, 33, 0.7);
    }
    </style>
""", unsafe_allow_html=True)


# --- VERİ KATMANI ---
def utc_now():
    return datetime.now(timezone.utc)


def to_jd(t):
    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)


def to_jd_arrays(times):
    # SatrecArray.sgp4 bitişik (C-contiguous) diziler ister; jf[:, 0] gibi sütun dilimleri hata verir
    jd, fr = zip(*(to_jd(t) for t in times))
    return np.array(jd), np.array(fr)


def embed_html(html, height):
    # st.components.v1.html yeni Streamlit sürümlerinde kaldırıldı -> st.iframe
    if hasattr(st, "iframe"):
        st.iframe(html.strip(), height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def geodetic_alt(r):
    n = np.linalg.norm(r, axis=-1)
    return n - RE * (1.0 - (r[..., 2] / n) ** 2 * WGS84_F)


def norad_id(l1):
    return l1[2:7].strip()


def parse_tles(text):
    # 3 satırlık blokları okur; bozuk/eksik bir blok gelirse kaymadan devam eder
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    out, i = [], 0
    while i + 2 < len(lines):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return out


def elements_now(tles):
    if not tles:
        return np.array([]), np.array([])
    sats = [Satrec.twoline2rv(a, b) for _, a, b in tles]
    jd, fr = to_jd(utc_now())
    e, r, _ = SatrecArray(sats).sgp4(np.array([jd]), np.array([fr]))
    ok = e[:, 0] == 0
    return geodetic_alt(r[:, 0, :])[ok], np.degrees([s.inclo for s in sats])[ok]


GROUPS = ["active", "visual"]
# Gerçek enkaz grupları (CelesTrak). Hiçbiri alınamazsa eski davranışa (visual) dönülür.
DEBRIS_GROUPS = ["cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris", "cosmos-1408-debris"]


# İndirilen TLE'ler diske yazılır: CelesTrak aynı grubu 2 saat içinde tekrar indirene 403 döndürür,
# ayrıca sunucu yeniden başlatıldığında her şeyi baştan indirmek gerekmez.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tle_cache")
CACHE_TTL_S = 2 * 3600


def _fetch_group(g):
    """Gerçek tarayıcı taklidi yapan, hata toleranslı ve önbellek destekli veri çekme motoru"""
    path = os.path.join(CACHE_DIR, f"{g}.tle")
    has_cache = os.path.exists(path)
    
    # Önbellek taze ise diskten oku
    if has_cache and time.time() - os.path.getmtime(path) < CACHE_TTL_S:
        with open(path, encoding="utf-8") as f:
            tles = parse_tles(f.read())
            if tles:
                return tles, "cache"

    # CelesTrak IP engeline karşı gerçekçi tarayıcı başlıkları (Headers)
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    }

    session = requests.Session()
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle"

    for attempt in range(3):
        try:
            r = session.get(url, timeout=(10, 30), headers=browser_headers)
            if r.status_code == 200 and r.text and "1 " in r.text:
                tles = parse_tles(r.text)
                if tles:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(r.text)
                    return tles, "live"
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1)

    # Canlı çekilemediyse ancak eskimiş (stale) cache varsa sisteme çökmemek için onu ver
    if has_cache:
        with open(path, encoding="utf-8") as f:
            tles = parse_tles(f.read())
            if tles:
                return tles, "stale"

    return [], "fail"


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_catalog():
    # Tüm gruplar paralel indirilir (sırayla indirmek her biri için ayrı bekleme demekti)
    all_groups = GROUPS + DEBRIS_GROUPS
    with ThreadPoolExecutor(max_workers=len(all_groups)) as ex:
        fetched = dict(zip(all_groups, ex.map(_fetch_group, all_groups)))

    sources = {g: src for g, (_, src) in fetched.items()}
    is_offline = all(src == "fail" for src in sources.values())

    # Alınamayan ana grup için önce diğer ana gruba, o da yoksa yedek TLE'lere düş
    data = {g: fetched[g][0] for g in GROUPS}
    for g in GROUPS:
        if not data[g]:
            data[g] = next((data[o] for o in GROUPS if data[o]), None) or parse_tles(FALLBACK_TLE)

    # Enkaz: gerçek debris grupları; olmazsa 'visual' grubuna düş
    debris, seen = [], set()
    for g in DEBRIS_GROUPS:
        for t in fetched[g][0]:
            nid = norad_id(t[1])
            if nid not in seen:
                seen.add(nid)
                debris.append(t)
    if not debris:
        debris = data.get("visual", [])

    a_alt, a_inc = elements_now(data.get("active", []))
    d_alt, d_inc = elements_now(debris)
    if is_offline:
        status_label = "OFFLINE 🟠 (Yedek Veri)"
    elif "fail" in sources.values():
        status_label = "PARTIAL 🟡 (Bazı gruplar alınamadı)"
    elif "stale" in sources.values():
        status_label = "CACHED 🟡 (Önbellek)"
    else:
        status_label = "LIVE 🟢"
    return {"groups": data, "debris": debris, "a_alt": a_alt, "a_inc": a_inc,
            "d_alt": d_alt, "d_inc": d_inc,
            "fetched": utc_now().strftime("%Y-%m-%d %H:%M:%S UTC"), "status_label": status_label}


def build_table(tles):
    if not tles:
        return pd.DataFrame()
    sats = [Satrec.twoline2rv(a, b) for _, a, b in tles]
    jd, fr = to_jd(utc_now())
    e, r, v = SatrecArray(sats).sgp4(np.array([jd]), np.array([fr]))
    rows = []
    for i, (name, l1, _) in enumerate(tles):
        if e[i, 0] != 0:
            continue
        s = sats[i]
        epoch = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=(s.jdsatepoch + s.jdsatepochF - 2451545.0))
        alt = float(geodetic_alt(r[i, 0]))
        if name.upper().startswith(CREWED_PREFIXES):
            status = "CREWED 🟢"
        elif alt < 300:
            status = "DECAYING ⚠️"
        else:
            status = "NOMINAL"
        rows.append({
            "Satellite Name": name, "NORAD ID": norad_id(l1), "Epoch Time": epoch.strftime("%Y-%m-%d %H:%M:%S"),
            "Altitude (km)": round(alt), "Inclination (°)": round(math.degrees(s.inclo), 2),
            "Period (min)": round(2 * math.pi / s.no_kozai, 1),
            "Speed (km/s)": round(float(np.linalg.norm(v[i, 0])), 2),
            "Status": status,
        })
    return pd.DataFrame(rows)


with st.spinner(T("🚀 Hızlı veri yükleniyor...", "🚀 Loading data...")):
    CAT = load_catalog()
df_api = build_table(CAT["groups"]["visual"])


def _feed_label(s):
    if st.session_state.lang == "tr":
        return s
    return (s.replace("(Yedek Veri)", "(Fallback Data)")
             .replace("(Bazı gruplar alınamadı)", "(Some groups unavailable)")
             .replace("(Önbellek)", "(Cache)"))


FEED_LABEL = _feed_label(CAT["status_label"])

# Tablo başlıkları / değerleri Türkçe modunda yalnızca gösterimde çevrilir (veri aynı kalır)
_COLS_TR = {
    "Satellite Name": "Uydu Adı", "NORAD ID": "NORAD ID", "Epoch Time": "Epok Zamanı",
    "Altitude (km)": "İrtifa (km)", "Inclination (°)": "Eğim (°)", "Period (min)": "Periyot (dk)",
    "Speed (km/s)": "Hız (km/s)", "Status": "Durum",
    "Object A": "Nesne A", "Object B": "Nesne B", "Miss Distance (km)": "Geçiş Mesafesi (km)",
    "Rel. Velocity (km/s)": "Göreli Hız (km/s)", "TCA (UTC)": "TCA (UTC)", "Risk": "Risk",
}
_VALS_TR = {
    "CREWED 🟢": "MÜRETTEBATLI 🟢", "DECAYING ⚠️": "ALÇALIYOR ⚠️", "NOMINAL": "NORMAL",
    "🔴 CRITICAL": "🔴 KRİTİK", "🟠 HIGH": "🟠 YÜKSEK", "🟡 WATCH": "🟡 İZLEME", "⚪ SCREENED": "⚪ ELENDİ",
}


def _disp(df):
    if st.session_state.lang != "tr" or df is None or df.empty:
        return df
    d = df.copy()
    for c in ("Status", "Risk"):
        if c in d.columns:
            d[c] = d[c].replace(_VALS_TR)
    return d.rename(columns=_COLS_TR)

# --- SPLASH EKRANI ---
# Giriş ekranı: arka planda gerçek yörünge verisiyle dönen 3D Dünya + bilgilendirme bölümleri.
# Ana uygulamaya etkisi yoktur: CSS yalnızca bu çalıştırmada eklenir ve sonunda st.stop() çağrılır.

def _enter_app():
    st.session_state.intro_passed = True


def _launch_from_url():
    # Navigasyondaki "Canlı İzlemeyi Başlat" bağlantısı ?launch=1 ile gelir
    try:
        return str(st.query_params.get("launch", "")) == "1"
    except Exception:
        return False


def _container_has_key():
    try:
        return "key" in inspect.signature(st.container).parameters
    except (TypeError, ValueError):
        return False


def _flat(html):
    # st.markdown girintili / boş satırlı HTML'i kod bloğu sanabilir; satırları tek satıra indir
    return " ".join(l.strip() for l in html.strip().splitlines() if l.strip())


def _fmt_int(n):
    return f"{n:,}".replace(",", ".")


def _sun_unit_three(jd_ut):
    """Güneşin yönü (Three.js eksenlerinde birim vektör). Düşük hassasiyetli güneş efemerisi (~0.01°)."""
    n = jd_ut - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    x, y, z = math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)
    return [round(x, 5), round(z, 5), round(-y, 5)]  # TEME (x, y, z) -> three.js (x, z, -y)


@st.cache_data(ttl=600, show_spinner=False)
def splash_globe_payload(n_active=1500, n_debris=1300):
    """Giriş ekranındaki küre için: gerçek TLE'lerden şu anki konum/hız (tek SGP4 adımı) + gerçek güneş yönü."""
    jd0, fr0 = to_jd(utc_now())
    out = {"kind": [], "p": [], "v": [], "sun": _sun_unit_three(jd0 + fr0),
           "gmst0": math.radians((280.46061837 + 360.98564736629 * (jd0 + fr0 - 2451545.0)) % 360),
           "omega": 7.2921159e-5}
    try:
        cat = load_catalog()
        rng = np.random.default_rng(21)

        def pick(lst, n):
            if not lst or n <= 0:
                return []
            return [lst[i] for i in rng.choice(len(lst), min(n, len(lst)), replace=False)]

        # İstasyonlar (ISS / CSS / Tiangong): her önekten bir tane, kendi yörünge halkasıyla çizilir
        crew, used = [], set()
        for t in cat["groups"]["visual"]:
            for p in CREWED_PREFIXES:
                if t[0].upper().startswith(p) and p not in used:
                    used.add(p)
                    crew.append(t)
        sel = ([(2, t) for t in crew] + [(0, t) for t in pick(cat["groups"]["active"], n_active)]
               + [(1, t) for t in pick(cat["debris"], n_debris)])
        seen, kinds, sats = set(), [], []
        for k, (name, l1, l2) in sel:
            nid = norad_id(l1)
            if nid in seen:
                continue
            seen.add(nid)
            kinds.append(k)
            sats.append(Satrec.twoline2rv(l1, l2))
        if not sats:
            return out
        e, r, v = SatrecArray(sats).sgp4(np.array([jd0]), np.array([fr0]))
        r, v = r[:, 0, :], v[:, 0, :]
        rn, vn = np.linalg.norm(r, axis=1), np.linalg.norm(v, axis=1)
        # Yalnızca LEO bandı ve o anda yaklaşık dairesel olanlar (animasyon dairesel yörünge yaklaşımı kullanır)
        ok = ((e[:, 0] == 0) & (rn > RE + 150) & (rn < RE + 2500)
              & (np.abs(np.einsum("ij,ij->i", r, v)) / (rn * vn) < 0.15))
        to3 = lambda a: np.stack([a[:, 0], a[:, 2], -a[:, 1]], axis=1)
        k_list = np.array(kinds)[ok].astype(int).tolist()
        p_list = np.round(to3(r[ok]) / RE, 4).ravel().tolist()
        v_list = np.round(to3(v[ok]), 3).ravel().tolist()
        out["kind"], out["p"], out["v"] = k_list, p_list, v_list
    except Exception:
        pass
    return out


INTRO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --sg-cyan:#38bdf8; --sg-red:#fc3d21; --sg-green:#10b981; --sg-mute:#94a3b8; --sg-line:rgba(56,189,248,.16); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp { background:#040817; }
[data-testid="stMain"], section.main, .main { scroll-behavior:smooth; }
.block-container, [data-testid="stMainBlockContainer"] { max-width:1240px !important; padding:0 clamp(18px,3vw,40px) 40px !important; }

/* Arka plan: tüm ekranı kaplayan sabit 3D Dünya */
.st-key-sg_bg { position:fixed !important; inset:0; width:100vw; height:100vh; z-index:0; pointer-events:none; margin:0; padding:0; }
.st-key-sg_bg > div, .st-key-sg_bg .element-container, .st-key-sg_bg [data-testid="stElementContainer"],
.st-key-sg_bg [data-testid="stIFrame"], .st-key-sg_bg iframe { width:100vw !important; height:100vh !important; max-width:none !important; margin:0 !important; border:0 !important; }
.st-key-sg_fg { position:relative; z-index:3; gap:.5rem; font-family:'Inter','Segoe UI',-apple-system,sans-serif; }

/* Sabit navigasyon */
.sg-nav { position:fixed; top:0; left:0; right:0; height:64px; z-index:50; display:flex; align-items:center; justify-content:space-between;
  padding:0 clamp(16px,3vw,40px); background:linear-gradient(180deg,rgba(4,8,23,.9),rgba(4,8,23,.55));
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border-bottom:1px solid var(--sg-line); font-family:'Inter','Segoe UI',sans-serif; }
.sg-logo { display:flex; align-items:center; gap:11px; color:#fff !important; text-decoration:none !important; font-family:'Space Grotesk','Inter',sans-serif; font-weight:700; letter-spacing:.16em; font-size:15px; }
.sg-logo-mark { width:22px; height:22px; border-radius:50%; border:1.5px solid var(--sg-cyan); position:relative; box-shadow:0 0 12px rgba(56,189,248,.4); }
.sg-logo-mark::after { content:""; position:absolute; width:5px; height:5px; border-radius:50%; background:var(--sg-red); top:-3px; right:1px; }
.sg-links { display:flex; gap:clamp(14px,2.2vw,34px); }
.sg-links a { color:var(--sg-mute) !important; text-decoration:none !important; font-size:13.5px; font-weight:500; transition:color .2s; }
.sg-links a:hover { color:#fff !important; }
.sg-cta { color:#fff !important; text-decoration:none !important; font-size:13px; font-weight:600; padding:9px 18px; border-radius:999px; white-space:nowrap;
  border:1px solid rgba(56,189,248,.55); background:rgba(56,189,248,.10); transition:all .2s; }
.sg-cta:hover { background:rgba(56,189,248,.22); border-color:var(--sg-cyan); }
.sg-veil { position:fixed; inset:0; z-index:-1; pointer-events:none;
  background:linear-gradient(90deg,rgba(4,8,23,.88) 0%,rgba(4,8,23,.60) 34%,rgba(4,8,23,.08) 62%,rgba(4,8,23,0) 100%),linear-gradient(0deg,rgba(4,8,23,.55),transparent 28%); }

/* Hero */
.sg-hero { padding:clamp(96px,17vh,170px) 0 22px; max-width:640px; animation:sgUp .9s ease both; font-family:'Inter','Segoe UI',sans-serif; }
.sg-badge { display:inline-flex; align-items:center; gap:9px; padding:7px 14px; border-radius:999px; border:1px solid var(--sg-line); background:rgba(8,16,38,.65);
  color:#bae6fd; font-size:11.5px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; }
.sg-dot { width:7px; height:7px; border-radius:50%; background:var(--sg-green); animation:sgPulse 2.2s infinite; }
.sg-dot.warn { background:#f59e0b; animation:none; }
.sg-h1 { font-family:'Space Grotesk','Inter',sans-serif; font-size:clamp(32px,4.1vw,58px); line-height:1.08; font-weight:700; letter-spacing:-.02em; color:#fff; margin:22px 0 18px; }
.sg-h1 em { font-style:normal; background:linear-gradient(90deg,#38bdf8,#e0f2fe); -webkit-background-clip:text; background-clip:text; color:transparent; }
.sg-lead { color:#cbd5e1; font-size:clamp(15px,1.25vw,17.5px); line-height:1.7; max-width:560px; margin:0; }
[class*="st-key-sg_launch"] button { height:54px; border-radius:999px !important; border:1px solid rgba(252,61,33,.9) !important;
  background:linear-gradient(135deg,#fc3d21 0%,#c42a12 100%) !important; padding:0 26px !important;
  box-shadow:0 10px 34px rgba(252,61,33,.32), inset 0 1px 0 rgba(255,255,255,.18) !important; transition:transform .2s ease, box-shadow .2s ease !important; }
[class*="st-key-sg_launch"] button:hover { transform:translateY(-2px); box-shadow:0 16px 44px rgba(252,61,33,.48), inset 0 1px 0 rgba(255,255,255,.22) !important; }
[class*="st-key-sg_launch"] button p { color:#fff !important; font-weight:700 !important; font-size:15px !important; letter-spacing:.01em; }
.sg-ghost { display:flex; align-items:center; justify-content:center; height:54px; border-radius:999px; border:1px solid rgba(148,163,184,.35);
  color:#e2e8f0 !important; text-decoration:none !important; font-weight:600; font-size:14.5px; transition:all .2s; font-family:'Inter','Segoe UI',sans-serif; }
.sg-ghost:hover { border-color:var(--sg-cyan); color:#fff !important; background:rgba(56,189,248,.08); }
.sg-metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin-top:34px; max-width:640px; border:1px solid var(--sg-line);
  border-radius:16px; overflow:hidden; background:var(--sg-line); font-family:'Inter','Segoe UI',sans-serif; }
.sg-metric { background:rgba(6,12,30,.80); padding:14px 16px; }
.sg-metric b { display:block; font-family:'Space Grotesk','Inter',sans-serif; font-size:21px; color:#fff; font-weight:700; }
.sg-metric b.ok { color:var(--sg-green); } .sg-metric b.warn { color:#f59e0b; }
.sg-metric span { font-size:10px; letter-spacing:.13em; text-transform:uppercase; color:var(--sg-mute); }
.sg-src { margin-top:16px; color:#64748b; font-size:11px; letter-spacing:.12em; text-transform:uppercase; font-family:'Inter','Segoe UI',sans-serif; }
.sg-src i { font-style:normal; color:#cbd5e1; margin-left:12px; padding:3px 10px; border:1px solid var(--sg-line); border-radius:999px; letter-spacing:.04em; text-transform:none; font-size:11.5px; }

/* Bölümler */
.sg-sec { margin-top:clamp(80px,12vh,140px); scroll-margin-top:90px; font-family:'Inter','Segoe UI',sans-serif; }
.sg-kicker { color:var(--sg-cyan); font-size:12px; letter-spacing:.22em; font-weight:700; text-transform:uppercase; }
.sg-h2 { font-family:'Space Grotesk','Inter',sans-serif; font-size:clamp(26px,3vw,40px); color:#fff; font-weight:700; letter-spacing:-.015em; line-height:1.15; margin:10px 0 14px; max-width:780px; }
.sg-p { color:#a9b8cc; font-size:15.5px; line-height:1.75; max-width:780px; margin:0 0 8px; }
.sg-p b { color:#fff; }
.sg-muted { color:#7f90a8; font-size:14px; }
.sg-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:16px; margin-top:30px; }
.sg-card { padding:24px; border-radius:18px; background:linear-gradient(160deg,rgba(14,24,48,.84),rgba(6,11,26,.92)); border:1px solid var(--sg-line);
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); transition:transform .25s ease, border-color .25s ease; }
.sg-card:hover { transform:translateY(-3px); border-color:rgba(56,189,248,.45); }
.sg-stat { font-family:'Space Grotesk','Inter',sans-serif; font-size:34px; font-weight:700; color:var(--sg-cyan); letter-spacing:-.02em; }
.sg-huge { font-size:clamp(44px,6vw,76px); }
.sg-ct { color:#fff; font-size:16px; font-weight:600; margin:8px 0 8px; }
.sg-cp { color:#94a3b8; font-size:14px; line-height:1.65; margin:0; }
.sg-step-n { font-family:'Space Grotesk','Inter',sans-serif; color:rgba(56,189,248,.6); font-size:13px; letter-spacing:.2em; font-weight:700; }
.sg-price { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px; margin-top:30px; }
.sg-tier { position:relative; display:flex; flex-direction:column; gap:16px; border-top:2px solid var(--c,#38bdf8); }
.sg-tier.pop { border-color:rgba(56,189,248,.55); background:linear-gradient(160deg,rgba(10,36,58,.78),rgba(6,11,26,.94)); box-shadow:0 20px 60px rgba(6,182,212,.12); }
.sg-tag { position:absolute; top:18px; right:18px; background:#06b6d4; color:#04121f; font-size:10px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; padding:4px 10px; border-radius:999px; }
.sg-tier-name { font-family:'Space Grotesk','Inter',sans-serif; font-size:21px; color:#fff; font-weight:700; }
.sg-lbl { font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--c,#38bdf8); font-weight:700; margin-bottom:5px; }
.sg-val { color:#cbd5e1; font-size:14px; line-height:1.65; }
.sg-faq { margin-top:60px; max-width:900px; }
.sg-faq details { border:1px solid var(--sg-line); border-radius:14px; background:rgba(8,14,32,.80); margin-bottom:10px; padding:0 22px; transition:border-color .2s; }
.sg-faq details[open] { border-color:rgba(56,189,248,.4); }
.sg-faq summary { cursor:pointer; padding:18px 0; color:#fff; font-weight:600; font-size:15.5px; list-style:none; display:flex; justify-content:space-between; gap:16px; }
.sg-faq summary::-webkit-details-marker { display:none; }
.sg-faq summary::after { content:"+"; color:var(--sg-cyan); font-size:22px; line-height:1; }
.sg-faq details[open] summary::after { content:"–"; }
.sg-faq details p { color:#a9b8cc; font-size:14.5px; line-height:1.75; padding:0 0 18px; margin:0; }
.sg-about { display:grid; grid-template-columns:1.3fr 1fr; gap:36px; align-items:center; }
.sg-final { text-align:center; }
.sg-final .sg-h2, .sg-final .sg-p { margin-left:auto; margin-right:auto; }
.sg-foot { margin-top:70px; padding-top:20px; border-top:1px solid var(--sg-line); color:#64748b; font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; text-align:center; font-family:'Inter','Segoe UI',sans-serif; }
.sg-top { position:fixed; right:24px; bottom:54px; z-index:60; display:flex; align-items:center; gap:8px; padding:11px 18px; border-radius:999px;
  background:rgba(6,12,30,.82); border:1px solid rgba(56,189,248,.45); color:#e0f2fe !important; text-decoration:none !important;
  font-family:'Inter','Segoe UI',sans-serif; font-size:13px; font-weight:600; backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  box-shadow:0 8px 28px rgba(0,0,0,.5); transition:all .2s ease; }
.sg-top:hover { border-color:var(--sg-cyan); background:rgba(56,189,248,.18); transform:translateY(-2px); }

@keyframes sgUp { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:none; } }
@keyframes sgPulse { 0% { box-shadow:0 0 0 0 rgba(16,185,129,.55); } 70% { box-shadow:0 0 0 9px rgba(16,185,129,0); } 100% { box-shadow:0 0 0 0 rgba(16,185,129,0); } }
@media (max-width:900px) {
  .sg-links { display:none; }
  .sg-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .sg-about { grid-template-columns:1fr; }
  .sg-veil { background:rgba(4,8,23,.62); }
}
@media (prefers-reduced-motion:reduce) { .sg-hero, .sg-dot { animation:none; } }
</style>
"""

# Arka plan küresi (Three.js). Yer işareti: __DATA__ -> splash_globe_payload() çıktısı (JSON)
INTRO_GLOBE_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html, body { margin:0; height:100%; overflow:hidden; background:radial-gradient(ellipse at 72% 45%, #0b1329 0%, #040817 55%, #000000 100%); }
  #c { position:fixed; inset:0; opacity:0; transition:opacity 1.8s ease; }
  body.ready #c { opacity:1; }
  #cap { position:fixed; right:22px; bottom:16px; font:500 10.5px/1.4 ui-monospace,Menlo,Consolas,monospace; letter-spacing:.14em; text-transform:uppercase; color:rgba(148,163,184,.7); }
  #cap b { color:#38bdf8; font-weight:600; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body>
<div id="c"></div><div id="cap"></div>
<script>
setTimeout(function () { document.body.classList.add('ready'); }, 3000);
(function () {
  try {
    const D = __DATA__;
    const RE = 6378.137, TAU = Math.PI * 2;
    const RATE = (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) ? 12 : 80;   // simülasyon hızı (gerçek zamanın katı)
    const BASE = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/';
    const PR = Math.min(window.devicePixelRatio || 1, 2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(PR);
    renderer.setClearColor(0x000000, 0);
    document.getElementById('c').appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 600);
    // Gerçek güneş yönü (inertial); kamera güneşten ~54° yana yerleşir -> aydınlık yarım küre + gün/gece sınırı görünür
    const sunDir = new THREE.Vector3(D.sun[0], D.sun[1], D.sun[2]).normalize();
    const camDir = sunDir.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.95);
    camDir.y = 0.30; camDir.normalize();

    function flat(r, g, b) {
      const t = new THREE.DataTexture(new Uint8Array([r, g, b, 255]), 1, 1, THREE.RGBAFormat);
      t.needsUpdate = true; return t;
    }
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin('anonymous');
    const aniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    function load(name, done) { loader.load(BASE + name, function (t) { t.anisotropy = aniso; done(t); }, undefined, function () {}); }

    // ---- Dünya: gün / gece (şehir ışıkları) / okyanus parıltısı / atmosfer kenarı ----
    const earthMat = new THREE.ShaderMaterial({
      uniforms: { dayMap: { value: flat(8, 38, 92) }, nightMap: { value: flat(0, 0, 0) }, sunDir: { value: sunDir } },
      vertexShader: `
        varying vec2 vUv; varying vec3 vN; varying vec3 vV;
        void main() {
          vUv = uv;
          vN = normalize(mat3(modelMatrix) * normal);
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vV = normalize(cameraPosition - wp.xyz);
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fragmentShader: `
        uniform sampler2D dayMap; uniform sampler2D nightMap; uniform vec3 sunDir;
        varying vec2 vUv; varying vec3 vN; varying vec3 vV;
        void main() {
          vec3 N = normalize(vN); vec3 V = normalize(vV);
          float d = dot(N, sunDir);
          float lit = smoothstep(-0.10, 0.24, d);
          vec3 day = texture2D(dayMap, vUv).rgb;
          vec3 city = texture2D(nightMap, vUv).rgb;
          vec3 dayCol = day * (0.16 + 0.96 * clamp(d, 0.0, 1.0));
          vec3 nightCol = day * 0.035 + city * vec3(1.0, 0.80, 0.52) * 1.5;
          vec3 col = mix(nightCol, dayCol, lit);
          float ocean = clamp((day.b - day.r - 0.04) * 4.0, 0.0, 1.0);
          vec3 R = reflect(-sunDir, N);
          col += vec3(0.75, 0.88, 1.0) * pow(max(dot(R, V), 0.0), 60.0) * ocean * lit * 0.35;
          float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);
          col += vec3(0.20, 0.52, 1.0) * rim * (0.20 + 0.80 * lit) * 0.75;
          float tw = exp(-pow((d - 0.03) / 0.10, 2.0));
          col += vec3(1.0, 0.45, 0.18) * tw * rim * 0.9;
          gl_FragColor = vec4(col, 1.0);
        }`
    });
    const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), earthMat);

    const cloudMat = new THREE.ShaderMaterial({
      uniforms: { map: { value: flat(0, 0, 0) }, sunDir: { value: sunDir } },
      transparent: true, depthWrite: false,
      vertexShader: `
        varying vec2 vUv; varying vec3 vN;
        void main() { vUv = uv; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        uniform sampler2D map; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN;
        void main() {
          vec4 t = texture2D(map, vUv);
          float c = t.r * t.a;
          float lit = smoothstep(-0.12, 0.30, dot(normalize(vN), sunDir));
          gl_FragColor = vec4(vec3(0.94, 0.97, 1.0) * (0.10 + 0.90 * lit), c * 0.5 * (0.2 + 0.8 * lit));
        }`
    });
    const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.007, 96, 64), cloudMat);

    const atmoMat = new THREE.ShaderMaterial({
      uniforms: { sunDir: { value: sunDir } },
      transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
      vertexShader: `
        varying vec3 vN; varying vec3 vWN;
        void main() { vN = normalize(normalMatrix * normal); vWN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        uniform vec3 sunDir; varying vec3 vN; varying vec3 vWN;
        void main() {
          float d = clamp(-dot(normalize(vN), vec3(0.0, 0.0, 1.0)), 0.0, 1.0);
          float i = pow(clamp(d / 0.5, 0.0, 1.0), 2.4);
          float k = 0.45 + 0.55 * smoothstep(-0.40, 0.45, dot(normalize(vWN), sunDir));
          gl_FragColor = vec4(vec3(0.22, 0.58, 1.0) * i * k * 0.95, 1.0);
        }`
    });
    const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.16, 96, 64), atmoMat);
    scene.add(earth, clouds, atmo);

    let ready = false;
    function markReady() { if (!ready) { ready = true; document.body.classList.add('ready'); } }
    load('earth_atmos_2048.jpg', function (t) { earthMat.uniforms.dayMap.value = t; markReady(); });
    load('earth_lights_2048.png', function (t) { earthMat.uniforms.nightMap.value = t; });
    load('earth_clouds_1024.png', function (t) { cloudMat.uniforms.map.value = t; });

    // ---- Yıldızlar ----
    const cv = document.createElement('canvas'); cv.width = cv.height = 64;
    const cx = cv.getContext('2d'), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.35, 'rgba(255,255,255,.85)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
    const DOT = new THREE.CanvasTexture(cv);

    function stars(n, size, opacity) {
      const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const u = Math.random() * 2 - 1, th = Math.random() * TAU, s = Math.sqrt(1 - u * u), R = 200;
        pos[3*i] = R * s * Math.cos(th); pos[3*i+1] = R * u; pos[3*i+2] = R * s * Math.sin(th);
        const k = Math.random(), c = k < 0.15 ? [1, 0.85, 0.7] : (k < 0.4 ? [0.7, 0.82, 1] : [0.9, 0.93, 1]), b = 0.45 + 0.55 * Math.random();
        col[3*i] = c[0] * b; col[3*i+1] = c[1] * b; col[3*i+2] = c[2] * b;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const p = new THREE.Points(g, new THREE.PointsMaterial({ size: size * PR, sizeAttenuation: false, map: DOT, vertexColors: true, transparent: true, opacity: opacity, depthWrite: false }));
      p.frustumCulled = false; return p;
    }
    const starGroup = new THREE.Group();
    starGroup.add(stars(2400, 2.6, 0.85), stars(140, 5, 1.0));
    scene.add(starGroup);

    // ---- Gerçek yörünge nesneleri: şu anki gerçek konum/hızdan dairesel yörünge yaklaşımı ----
    const N = D.kind.length;
    const P0 = new Float32Array(N * 3), Q0 = new Float32Array(N * 3), W = new Float32Array(N);
    const mainIdx = [], crewIdx = [];
    for (let i = 0; i < N; i++) {
      const x = D.p[3*i], y = D.p[3*i+1], z = D.p[3*i+2];
      const vx = D.v[3*i] / RE, vy = D.v[3*i+1] / RE, vz = D.v[3*i+2] / RE;
      const rr = Math.hypot(x, y, z), vv = Math.hypot(vx, vy, vz) || 1e-9, w = vv / rr;
      W[i] = w;
      P0[3*i] = x; P0[3*i+1] = y; P0[3*i+2] = z;
      Q0[3*i] = vx / w; Q0[3*i+1] = vy / w; Q0[3*i+2] = vz / w;
      (D.kind[i] === 2 ? crewIdx : mainIdx).push(i);
    }
    function makePoints(idx, hex, size, opacityVal, perKind) {
      const g = new THREE.BufferGeometry();
      const pos = new Float32Array(idx.length * 3), col = new Float32Array(idx.length * 3);
      const cA = new THREE.Color(hex), cD = new THREE.Color(0xfc3d21);
      idx.forEach(function (i, j) {
        const c = (perKind && D.kind[i] === 1) ? cD : cA;
        col[3*j] = c.r; col[3*j+1] = c.g; col[3*j+2] = c.b;
      });
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const p = new THREE.Points(g, new THREE.PointsMaterial({ size: size * PR, sizeAttenuation: false, map: DOT, vertexColors: true, transparent: true,
        opacity: opacityVal, depthWrite: false, blending: THREE.AdditiveBlending }));
      p.frustumCulled = false; scene.add(p);
      return { pts: p, pos: pos, geo: g, idx: idx };
    }
    const mainSet = makePoints(mainIdx, 0x38bdf8, 3.4, 0.95, true);     // mavi: aktif uydu, kırmızı: enkaz
    const crewSet = makePoints(crewIdx, 0x10b981, 8, 1.0, false);        // yeşil: insanlı istasyonlar
    crewIdx.forEach(function (i) {
      const pts = [];
      for (let k = 0; k < 160; k++) {
        const a = TAU * k / 160, c = Math.cos(a), s = Math.sin(a);
        pts.push(new THREE.Vector3(P0[3*i] * c + Q0[3*i] * s, P0[3*i+1] * c + Q0[3*i+1] * s, P0[3*i+2] * c + Q0[3*i+2] * s));
      }
      scene.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.35 })));
    });
    function place(set, t) {
      const idx = set.idx, pos = set.pos;
      for (let j = 0; j < idx.length; j++) {
        const i = idx[j], a = W[i] * t, c = Math.cos(a), s = Math.sin(a);
        pos[3*j]   = P0[3*i]   * c + Q0[3*i]   * s;
        pos[3*j+1] = P0[3*i+1] * c + Q0[3*i+1] * s;
        pos[3*j+2] = P0[3*i+2] * c + Q0[3*i+2] * s;
      }
      set.geo.attributes.position.needsUpdate = true;
    }

    // ---- Boyut / kamera: geniş ekranda küre sağa kayar, metin solda kalır ----
    function resize() {
      const w = Math.max(window.innerWidth, 1), h = Math.max(window.innerHeight, 1), asp = w / h;
      renderer.setSize(w, h);
      camera.aspect = asp;
      const half = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const wide = asp > 1.15;
      const dist = wide ? 3.8 : Math.max(3.8, 1.35 / (half * asp));
      camera.position.copy(camDir).multiplyScalar(dist);
      camera.lookAt(0, 0, 0);
      if (wide) camera.setViewOffset(w, h, -w * 0.20, 0, w, h); else camera.clearViewOffset();
      camera.updateProjectionMatrix();
    }
    resize();
    window.addEventListener('resize', resize);
    setTimeout(resize, 400);

    if (N > 0) {
      document.getElementById('cap').innerHTML = '<b>Gerçek TLE verisi</b> · SGP4 · ' + N + ' nesne · ' + RATE + '× zaman';
    }

    let t = 0, last = performance.now();
    function frame(now) {
      requestAnimationFrame(frame);
      const dt = Math.min((now - last) / 1000, 0.1); last = now;
      t += dt * RATE;
      earth.rotation.y = D.gmst0 + D.omega * t;          // Greenwich meridyeni gerçek GMST'ye göre
      clouds.rotation.y = earth.rotation.y + t * 0.00003;
      starGroup.rotation.y = t * 0.00003;
      place(mainSet, t); place(crewSet, t);
      renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
  } catch (err) { console.error(err); }
})();
</script></body></html>
"""
def render_intro():
    n_active = len(CAT["groups"]["active"])
    n_total = n_active + len(CAT["debris"])
    feed_word = FEED_LABEL.split(" ")[0]
    keyed = _container_has_key()

    # Küre: tüm ekranı kaplayan sabit arka plan (container(key=...) desteklenmiyorsa atlanır)
    if keyed:
        with st.container(key="sg_bg"):
            embed_html(INTRO_GLOBE_HTML.replace("__DATA__", json.dumps(splash_globe_payload())), height=800)

    nav_html = """
    <div class="sg-nav">
      <a class="sg-logo" href="#top"><span class="sg-logo-mark"></span>SPACE GRAVE</a>
      <div class="sg-links"><a href="#features">Özellikler</a><a href="#pricing">B2B Fiyatlandırma</a><a href="#docs">Şartname / Dokümantasyon</a><a href="#about">Hakkımızda</a></div>
      <a class="sg-cta" href="?launch=1&amp;lang=tr" target="_self">Canlı İzlemeyi Başlat</a>
    </div>
    <div class="sg-veil"></div>
    <a class="sg-top" href="#top" aria-label="Ana ekrana dön">↑ Ana ekran</a>
    """

    hero_html = f"""
    <div id="top" class="sg-hero">
      <div class="sg-badge"><span class="sg-dot{'' if feed_word == 'LIVE' else ' warn'}"></span>B2B Uzay Karar Destek Platformu</div>
      <div class="sg-h1">Space Grave ile Alçak Dünya Yörüngesi (LEO) için <em>akıllı çarpışma önleme</em> ve karar destek platformu</div>
      <p class="sg-lead">Space Grave karar destek sistemimiz ile 50 km güvenlik küresi içindeki uydu filolarınızı uzay çöplerinden koruyun ve katastrofik yıkım risklerini gerçekleşmeden öngörün.</p>
    </div>
    """

    metrics_html = f"""
    <div class="sg-metrics">
      <div class="sg-metric"><b>{_fmt_int(n_total)}</b><span>Takip edilen nesne</span></div>
      <div class="sg-metric"><b>50 km</b><span>Güvenlik küresi</span></div>
      <div class="sg-metric"><b>40 J/g</b><span>NASA parçalanma eşiği</span></div>
      <div class="sg-metric"><b class="{'ok' if feed_word == 'LIVE' else 'warn'}">{feed_word}</b><span>CelesTrak veri akışı</span></div>
    </div>
    <div class="sg-src">Güvenilir veri kaynakları <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
    """

    sections_html = """
    <div id="features" class="sg-sec">
      <div class="sg-kicker">Özellikler</div>
      <div class="sg-h2">Uzay kirliliği artık yalnızca teknik değil, finansal bir risk</div>
      <p class="sg-p">Uzay kirliliği, Alçak Dünya Yörüngesi'nde (LEO) çalışan uydu filoları için sadece teknik bir tehdit değil, aynı zamanda <b>&#36;42.3B</b> dolarlık "gizli bir finansal yüktür." Space Grave, karmaşık yörünge verilerini işleyerek uydu operatörleri ve uzay sigortacıları için anlık risk öngörüsü ve maliyet optimizasyonu sunar.</p>
      <div class="sg-grid">
        <div class="sg-card"><div class="sg-stat">50 km</div><div class="sg-ct">Güvenlik Küresi</div><p class="sg-cp">Yüksek riskli yakınlaşmalarda otomatik Kırmızı Alarm tetikler.</p></div>
        <div class="sg-card"><div class="sg-stat">SGP4</div><div class="sg-ct">Astrodinamik Motor ve 3D Görselleştirici</div><p class="sg-cp">Canlı TLE verilerinden hesaplanan koordinatlar 3D canlı Dünya görselleştiricisinde izlenir; mesafe tarayıcı yakınlaşmaları tespit eder.</p></div>
        <div class="sg-card"><div class="sg-stat">40 J/g</div><div class="sg-ct">NASA Parçalanma Standardı</div><p class="sg-cp">Kinetik enerji hesaplamalarıyla yıkım potansiyelini öngörür.</p></div>
        <div class="sg-card"><div class="sg-stat">0–100</div><div class="sg-ct">Sigorta Risk Skoru</div><p class="sg-cp">Roket gövdeleri ve yüksek yoğunluklu çöpler için dinamik poliçe analizi sunar.</p></div>
      </div>
      <p class="sg-p sg-muted" style="margin-top:28px">&#36;42.3B'lık uzay çöpü tehdidine karşı canlı TLE verilerini SGP4 motoru ve NASA parçalanma standartlarıyla işleyerek 50 km güvenlik küresi içindeki uzay çöplerini tarar; 0-100 arası dinamik sigorta risk skorunu ve kaçınma maliyetlerini anlık hesaplayarak manevra yakıt maliyetlerinizi optimize eder, hizmet kesintilerini ve zararlarınızı minimuma indiririz.</p>
    </div>

    <div id="docs" class="sg-sec">
      <div class="sg-kicker">Şartname / Dokümantasyon</div>
      <div class="sg-h2">Modüler ve ölçeklenebilir sistem mimarisi</div>
      <p class="sg-p">Platformumuz, canlı verinin çekilmesinden 3D arayüzde sunulmasına kadar 4 ana katmandan oluşan yüksek performanslı bir altyapıya sahiptir:</p>
      <div class="sg-grid">
        <div class="sg-card"><div class="sg-step-n">01</div><div class="sg-ct">Veri Toplama Katmanı</div><p class="sg-cp"><b style="color:#cbd5e1">Data Ingestion</b> — CelesTrak, Space-Track ve ESA DISCOS API'lerinden canlı TLE yörünge verileri ve uzay çöplerinin kütle/boyut parametreleri anlık çekilir.</p></div>
        <div class="sg-card"><div class="sg-step-n">02</div><div class="sg-ct">Önbellek ve Veri İşleme</div><p class="sg-cp"><b style="color:#cbd5e1">Data &amp; Cache</b> — API limitlerini aşmamak için veriler JSON/Redis önbelleğinde saklanır, veri birleştirici modülümüz ile temizlenerek hesaplama motoruna beslenir.</p></div>
        <div class="sg-card"><div class="sg-step-n">03</div><div class="sg-ct">Astrodinamik ve Finansal Risk Motoru</div><p class="sg-cp"><b style="color:#cbd5e1">Calculations</b> — SGP4 motoru ile koordinatlar hesaplanır, mesafe tarayıcı 50 km yakınlaşmaları tespit eder ve Finansal Risk Motoru anlık risk skorunu üretir.</p></div>
        <div class="sg-card"><div class="sg-step-n">04</div><div class="sg-ct">Sunum ve Alarm Paneli</div><p class="sg-cp"><b style="color:#cbd5e1">Presentation</b> — 3D Canlı Dünya Görselleştirici, Çarpışma Alarm Paneli ve B2B SaaS Fiyat Tabloları ile veriler karar vericilere sunulur.</p></div>
      </div>
      <div class="sg-src" style="margin-top:22px">API kaynakları <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
    </div>

    <div id="pricing" class="sg-sec">
      <div class="sg-kicker">B2B Fiyatlandırma</div>
      <div class="sg-h2">İhtiyacınıza uygun B2B SaaS paketleri</div>
      <p class="sg-p">Farklı paydaş gruplarının ihtiyaçlarına özel kurgulanmış <b>3 kademeli (Tiered)</b> B2B fiyatlandırma modeli:</p>
      <div class="sg-price">
        <div class="sg-card sg-tier" style="--c:#3b82f6">
          <div class="sg-tier-name">Basic / Standart</div>
          <div><div class="sg-lbl">Hedef kitle</div><div class="sg-val">Küçük takımyıldız operatörleri<br>Üniversite / araştırma</div></div>
          <div><div class="sg-lbl">Kapsam ve modellenen veri</div><div class="sg-val">Manevra maliyeti ve yakıt optimize edici<br>Kataloglanmış enkaz takibi</div></div>
          <div><div class="sg-lbl">Fiyatlandırma mantığı</div><div class="sg-val">Flat-rate: uydu / takımyıldız başına aylık veya yıllık abonelik</div></div>
        </div>
        <div class="sg-card sg-tier pop" style="--c:#06b6d4">
          <span class="sg-tag">Popüler</span>
          <div class="sg-tier-name">Pro / Enterprise</div>
          <div><div class="sg-lbl">Hedef kitle</div><div class="sg-val">Ticari uydu operatörleri<br>Filo yöneticileri</div></div>
          <div><div class="sg-lbl">Kapsam ve modellenen veri</div><div class="sg-val">Anomali tespit ve risk tahminleme<br>Hizmet kesintisi süre / maliyet analizi</div></div>
          <div><div class="sg-lbl">Fiyatlandırma mantığı</div><div class="sg-val">İzlenen aktif uydu sayısı ve gerçek zamanlı API çağrıları</div></div>
        </div>
        <div class="sg-card sg-tier" style="--c:#a855f7">
          <div class="sg-tier-name">Institutional / InsurTech</div>
          <div><div class="sg-lbl">Hedef kitle</div><div class="sg-val">Savunma ve uzay ajansları<br>Sigorta şirketleri</div></div>
          <div><div class="sg-lbl">Kapsam ve modellenen veri</div><div class="sg-val">Kümülatif finansal risk ve senaryo modeli<br>Özel risk skorlama ve poliçe fiyatlandırma</div></div>
          <div><div class="sg-lbl">Fiyatlandırma mantığı</div><div class="sg-val">Özel yıllık kontrat + kurulum + danışmanlık ücreti</div></div>
        </div>
      </div>
      <div class="sg-faq">
        <div class="sg-kicker" style="margin-bottom:16px">Sık sorulanlar</div>
        <details><summary>1. Space Grave yörünge verilerini ne kadar sıklıkla güncelliyor?</summary><p>Platformumuz CelesTrak, Space-Track ve ESA DISCOS API'leri üzerinden canlı TLE (Two-Line Element) verilerini düzenli olarak çeker. API sorgu limitlerini korumak ve maksimum performans sağlamak amacıyla veriler JSON/Redis önbelleğimizde saklanır ve otomatik senkronize edilir.</p></details>
        <details><summary>2. 50 km Güvenlik Küresi (Safety Sphere) ne anlama geliyor?</summary><p>Uydunuzun etrafında sanal olarak tanımlanan 50 km yarıçapındaki koruma hacmidir. SGP4 astrodinamik motorumuz bir uzay çöpünün bu küreye yaklaşacağını öngördüğü anda sistem "Kırmızı Alarm" durumuna geçer ve anlık kaçınma analizi başlatır.</p></details>
        <details><summary>3. Dinamik Sigorta Risk Skoru (0-100) nasıl hesaplanıyor?</summary><p>Risk skoru; uydunun bağıl hızı, çarpışma olasılığı ve kinetik enerjisi kullanılarak kurgulanır. NASA'nın 40 J/g parçalanma eşiği baz alınarak; roket gövdeleri veya yüksek yoğunluklu enkaz parçaları için kütle çarpanları uygulanır ve 0-100 arası finansal poliçe skoruna dönüştürülür.</p></details>
        <details><summary>4. Kendi özel uydu filomuzun verilerini sisteme entegre edebilir miyiz?</summary><p>Evet. Basic/Standart paketimizde kataloglanmış genel yörünge takibi sunulurken; Pro/Enterprise ve Institutional/InsurTech paketlerimizde kendi aktif uydu filolarınızın özel Ephemeris ve TLE verilerini güvenli API çağrılarıyla sisteme aktarabilirsiniz.</p></details>
        <details><summary>5. Platform yanlış alarmları (False Positive) ve gereksiz yakıt harcamasını nasıl önlüyor?</summary><p>Önbellek ve veri temizleme katmanımız (Data Pipeline), gürültülü verileri ve sapmaları filtreler. Yalnızca kütle, hız ve mesafe açısından NASA parçalanma standartlarını aşan gerçek tehlikeleri alarm olarak bildirdiği için operatörlerin gereksiz manevra yakıtı harcamasını engeller.</p></details>
      </div>
    </div>

    <div id="about" class="sg-sec">
      <div class="sg-about">
        <div>
          <div class="sg-kicker">Hakkımızda</div>
          <div class="sg-h2">Karmaşık yörünge verisini karar verilebilir riske dönüştürüyoruz</div>
          <p class="sg-p">Space Grave, NASA Space Apps Challenge kapsamında geliştirilen bir B2B karar destek platformudur. Amacımız; uydu operatörlerinin ve uzay sigortacılarının uzay çöpü kaynaklı riski gerçekleşmeden görmesini, gereksiz manevra maliyetinden kaçınmasını ve zararlarını minimuma indirmesini sağlamaktır.</p>
        </div>
        <div class="sg-card"><div class="sg-stat sg-huge">&#36;42.3B</div><div class="sg-ct">Gizli finansal yük</div><p class="sg-cp">Uzay çöpünün LEO operatörlerine yüklediği tahmini ekonomik etki — WEF risk verisi.</p></div>
      </div>
    </div>

    <div class="sg-sec sg-final">
      <div class="sg-h2">Canlı yörünge haritasına geçin</div>
      <p class="sg-p">Gerçek TLE verisiyle 3D Dünya, yakınlaşma taraması ve atmosfere yeniden giriş simülasyonu tek komuta merkezinde.</p>
    </div>
    """

    foot_html = f"""
    <div class="sg-foot">NASA Space Apps Challenge · JPL Command Center · SP4C3 GR4V3 &nbsp;|&nbsp; Veri: CelesTrak · Son güncelleme {CAT['fetched']}</div>
    """

    if st.session_state.lang == "en":
        nav_html = """
        <div class="sg-nav">
          <a class="sg-logo" href="#top"><span class="sg-logo-mark"></span>SPACE GRAVE</a>
          <div class="sg-links"><a href="#features">Features</a><a href="#pricing">B2B Pricing</a><a href="#docs">Specs / Documentation</a><a href="#about">About Us</a></div>
          <a class="sg-cta" href="?launch=1&amp;lang=en" target="_self">Start Live Tracking</a>
        </div>
        <div class="sg-veil"></div>
        <a class="sg-top" href="#top" aria-label="Back to top">↑ Top</a>
        """
        hero_html = f"""
        <div id="top" class="sg-hero">
          <div class="sg-badge"><span class="sg-dot{'' if feed_word == 'LIVE' else ' warn'}"></span>B2B Space Decision Support Platform</div>
          <div class="sg-h1">Space Grave: <em>smart collision avoidance</em> and a decision support platform for Low Earth Orbit (LEO)</div>
          <p class="sg-lead">With the Space Grave decision support system, protect your satellite fleets inside a 50 km safety sphere from space debris and foresee catastrophic risk before it happens.</p>
        </div>
        """
        metrics_html = f"""
        <div class="sg-metrics">
          <div class="sg-metric"><b>{_fmt_int(n_total)}</b><span>Tracked objects</span></div>
          <div class="sg-metric"><b>50 km</b><span>Safety sphere</span></div>
          <div class="sg-metric"><b>40 J/g</b><span>NASA fragmentation threshold</span></div>
          <div class="sg-metric"><b class="{'ok' if feed_word == 'LIVE' else 'warn'}">{feed_word}</b><span>CelesTrak data feed</span></div>
        </div>
        <div class="sg-src">Trusted data sources <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
        """
        sections_html = """
        <div id="features" class="sg-sec">
          <div class="sg-kicker">Features</div>
          <div class="sg-h2">Space debris is no longer just a technical risk, it is a financial one</div>
          <p class="sg-p">Space debris is not only a technical threat for satellite fleets operating in Low Earth Orbit (LEO); it is also a "hidden financial burden" worth <b>&#36;42.3B</b>. Space Grave processes complex orbital data to give satellite operators and space insurers instant risk foresight and cost optimization.</p>
          <div class="sg-grid">
            <div class="sg-card"><div class="sg-stat">50 km</div><div class="sg-ct">Safety Sphere</div><p class="sg-cp">Automatically triggers a Red Alert on high-risk close approaches.</p></div>
            <div class="sg-card"><div class="sg-stat">SGP4</div><div class="sg-ct">Astrodynamics Engine and 3D Visualizer</div><p class="sg-cp">Coordinates computed from live TLE data are tracked on a live 3D Earth visualizer; the distance scanner detects close approaches.</p></div>
            <div class="sg-card"><div class="sg-stat">40 J/g</div><div class="sg-ct">NASA Fragmentation Standard</div><p class="sg-cp">Predicts destruction potential with kinetic energy calculations.</p></div>
            <div class="sg-card"><div class="sg-stat">0–100</div><div class="sg-ct">Insurance Risk Score</div><p class="sg-cp">Provides dynamic policy analysis for rocket bodies and high-density debris.</p></div>
          </div>
          <p class="sg-p sg-muted" style="margin-top:28px">Against the &#36;42.3B space debris threat, we process live TLE data with the SGP4 engine and NASA fragmentation standards to scan for debris inside the 50 km safety sphere; by instantly calculating a dynamic 0-100 insurance risk score and avoidance costs, we help you optimize maneuver fuel costs and minimize service interruptions and losses.</p>
        </div>

        <div id="docs" class="sg-sec">
          <div class="sg-kicker">Specs / Documentation</div>
          <div class="sg-h2">Modular and scalable system architecture</div>
          <p class="sg-p">Our platform has a high-performance infrastructure made up of 4 main layers, from pulling live data to presenting it in the 3D interface:</p>
          <div class="sg-grid">
            <div class="sg-card"><div class="sg-step-n">01</div><div class="sg-ct">Data Collection Layer</div><p class="sg-cp"><b style="color:#cbd5e1">Data Ingestion</b> — Live TLE orbit data and debris mass/size parameters are pulled instantly from the CelesTrak, Space-Track and ESA DISCOS APIs.</p></div>
            <div class="sg-card"><div class="sg-step-n">02</div><div class="sg-ct">Cache and Data Processing</div><p class="sg-cp"><b style="color:#cbd5e1">Data &amp; Cache</b> — To stay within API limits, data is stored in a JSON/Redis cache, cleaned by our data merger module and fed to the calculation engine.</p></div>
            <div class="sg-card"><div class="sg-step-n">03</div><div class="sg-ct">Astrodynamics and Financial Risk Engine</div><p class="sg-cp"><b style="color:#cbd5e1">Calculations</b> — The SGP4 engine computes coordinates, the distance scanner detects 50 km close approaches and the Financial Risk Engine produces an instant risk score.</p></div>
            <div class="sg-card"><div class="sg-step-n">04</div><div class="sg-ct">Presentation and Alert Panel</div><p class="sg-cp"><b style="color:#cbd5e1">Presentation</b> — Data reaches decision makers through the 3D Live Earth Visualizer, the Collision Alert Panel and B2B SaaS pricing tables.</p></div>
          </div>
          <div class="sg-src" style="margin-top:22px">API sources <i>CelesTrak</i><i>Space-Track</i><i>ESA DISCOS</i></div>
        </div>

        <div id="pricing" class="sg-sec">
          <div class="sg-kicker">B2B Pricing</div>
          <div class="sg-h2">B2B SaaS plans that fit your needs</div>
          <p class="sg-p">A <b>3-tier</b> B2B pricing model designed for the needs of different stakeholder groups:</p>
          <div class="sg-price">
            <div class="sg-card sg-tier" style="--c:#3b82f6">
              <div class="sg-tier-name">Basic / Standard</div>
              <div><div class="sg-lbl">Target audience</div><div class="sg-val">Small constellation operators<br>Universities / research</div></div>
              <div><div class="sg-lbl">Scope and modeled data</div><div class="sg-val">Maneuver cost and fuel optimizer<br>Cataloged debris tracking</div></div>
              <div><div class="sg-lbl">Pricing logic</div><div class="sg-val">Flat-rate: monthly or annual subscription per satellite / constellation</div></div>
            </div>
            <div class="sg-card sg-tier pop" style="--c:#06b6d4">
              <span class="sg-tag">Popular</span>
              <div class="sg-tier-name">Pro / Enterprise</div>
              <div><div class="sg-lbl">Target audience</div><div class="sg-val">Commercial satellite operators<br>Fleet managers</div></div>
              <div><div class="sg-lbl">Scope and modeled data</div><div class="sg-val">Anomaly detection and risk forecasting<br>Outage duration / cost analysis</div></div>
              <div><div class="sg-lbl">Pricing logic</div><div class="sg-val">Number of active satellites monitored and real-time API calls</div></div>
            </div>
            <div class="sg-card sg-tier" style="--c:#a855f7">
              <div class="sg-tier-name">Institutional / InsurTech</div>
              <div><div class="sg-lbl">Target audience</div><div class="sg-val">Defense and space agencies<br>Insurance companies</div></div>
              <div><div class="sg-lbl">Scope and modeled data</div><div class="sg-val">Cumulative financial risk and scenario model<br>Custom risk scoring and policy pricing</div></div>
              <div><div class="sg-lbl">Pricing logic</div><div class="sg-val">Custom annual contract + setup + consulting fee</div></div>
            </div>
          </div>
          <div class="sg-faq">
            <div class="sg-kicker" style="margin-bottom:16px">FAQ</div>
            <details><summary>1. How often does Space Grave update orbit data?</summary><p>Our platform regularly pulls live TLE (Two-Line Element) data through the CelesTrak, Space-Track and ESA DISCOS APIs. To respect API query limits and ensure maximum performance, data is stored in our JSON/Redis cache and synced automatically.</p></details>
            <details><summary>2. What does the 50 km Safety Sphere mean?</summary><p>It is a protection volume with a 50 km radius virtually defined around your satellite. The moment our SGP4 astrodynamics engine predicts that a piece of debris will approach this sphere, the system switches to "Red Alert" and starts an instant avoidance analysis.</p></details>
            <details><summary>3. How is the Dynamic Insurance Risk Score (0-100) calculated?</summary><p>The risk score is built from the satellite's relative velocity, collision probability and kinetic energy. Based on NASA's 40 J/g fragmentation threshold, mass multipliers are applied for rocket bodies and high-density debris, and the result is converted into a 0-100 financial policy score.</p></details>
            <details><summary>4. Can we integrate our own private satellite fleet data into the system?</summary><p>Yes. The Basic/Standard plan offers cataloged general orbit tracking; with the Pro/Enterprise and Institutional/InsurTech plans you can send your own active fleets' private Ephemeris and TLE data to the system through secure API calls.</p></details>
            <details><summary>5. How does the platform prevent false positives and unnecessary fuel use?</summary><p>Our cache and data cleaning layer (Data Pipeline) filters noisy data and deviations. Because only real hazards that exceed NASA fragmentation standards in mass, velocity and distance are raised as alerts, operators avoid spending maneuver fuel unnecessarily.</p></details>
          </div>
        </div>

        <div id="about" class="sg-sec">
          <div class="sg-about">
            <div>
              <div class="sg-kicker">About Us</div>
              <div class="sg-h2">We turn complex orbital data into actionable risk</div>
              <p class="sg-p">Space Grave is a B2B decision support platform developed for the NASA Space Apps Challenge. Our goal is to let satellite operators and space insurers see debris-related risk before it happens, avoid unnecessary maneuver costs and minimize their losses.</p>
            </div>
            <div class="sg-card"><div class="sg-stat sg-huge">&#36;42.3B</div><div class="sg-ct">Hidden financial burden</div><p class="sg-cp">Estimated economic impact of space debris on LEO operators — WEF risk data.</p></div>
          </div>
        </div>

        <div class="sg-sec sg-final">
          <div class="sg-h2">Move on to the live orbit map</div>
          <p class="sg-p">3D Earth with real TLE data, close-approach screening and atmospheric re-entry simulation in a single command center.</p>
        </div>
        """
        foot_html = f"""
        <div class="sg-foot">NASA Space Apps Challenge · JPL Command Center · SP4C3 GR4V3 &nbsp;|&nbsp; Data: CelesTrak · Last update {CAT['fetched']}</div>
        """

    st.markdown(INTRO_CSS, unsafe_allow_html=True)
    with (st.container(key="sg_fg") if keyed else st.container()):
        st.markdown(_flat(nav_html), unsafe_allow_html=True)
        left, _ = st.columns([1.25, 1])
        with left:
            st.markdown(_flat(hero_html), unsafe_allow_html=True)
            render_live_threat_panel()
            b1, b2 = st.columns([1.15, 1])
            with b1:
                st.button(T("Canlı İzlemeyi Başlat  →", "Start Live Tracking  →"), key="sg_launch", type="primary", on_click=_enter_app, **stretch(st.button))
            with b2:
                st.markdown(f'<a class="sg-ghost" href="#features">{T("Özellikleri keşfet", "Explore features")}</a>', unsafe_allow_html=True)
            st.markdown(_flat(metrics_html), unsafe_allow_html=True)
        
        st.markdown(_flat(sections_html), unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.1, 1])
        with c2:
            st.button(T("Canlı İzlemeyi Başlat  →", "Start Live Tracking  →"), key="sg_launch_2", type="primary", on_click=_enter_app, **stretch(st.button))
        st.markdown(_flat(foot_html), unsafe_allow_html=True)
    render_lang_switch("24px", "12px")


if not st.session_state.intro_passed and _launch_from_url():
    st.session_state.intro_passed = True
    try:
        st.query_params.clear()
    except Exception:
        pass

if not st.session_state.intro_passed:
    render_intro()
    st.stop()

# --- ANA KOMUTA MERKEZİ ---
MAIN_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp {
  font-family:'Inter','Segoe UI',-apple-system,sans-serif;
  background:
    radial-gradient(1.2px 1.2px at 8% 14%, rgba(255,255,255,.55), transparent 100%),
    radial-gradient(1px 1px at 23% 61%, rgba(186,230,253,.5), transparent 100%),
    radial-gradient(1.4px 1.4px at 37% 27%, rgba(255,255,255,.45), transparent 100%),
    radial-gradient(1px 1px at 52% 83%, rgba(255,255,255,.5), transparent 100%),
    radial-gradient(1.2px 1.2px at 66% 9%, rgba(186,230,253,.55), transparent 100%),
    radial-gradient(1px 1px at 79% 47%, rgba(255,255,255,.45), transparent 100%),
    radial-gradient(1.3px 1.3px at 91% 72%, rgba(255,255,255,.5), transparent 100%),
    radial-gradient(1px 1px at 14% 92%, rgba(186,230,253,.45), transparent 100%),
    radial-gradient(ellipse at 78% -8%, #0f2347 0%, #060d22 42%, #02040d 100%) !important;
  background-attachment: fixed !important;
}
.block-container, [data-testid="stMainBlockContainer"] { padding-top:5.6rem !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family:'Space Grotesk','Inter','Segoe UI',sans-serif !important; letter-spacing:-.015em; }
.stApp h1 { color:#fff; font-weight:700 !important; }
.stApp h1, .stApp h3 { text-shadow:0 0 28px rgba(56,189,248,.18); }

/* Üst gezinme çubuğu (giriş ekranıyla aynı dil) */
.sg-mnav { position:fixed; top:0; left:0; right:0; height:64px; z-index:50; display:flex; align-items:center; justify-content:space-between;
  padding:0 clamp(16px,3vw,40px) 0 calc(clamp(16px,3vw,40px) + 168px); background:linear-gradient(180deg,rgba(4,8,23,.92),rgba(4,8,23,.62));
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border-bottom:1px solid rgba(56,189,248,.16); }
.sg-mlogo { display:flex; align-items:center; gap:11px; color:#fff; font-family:'Space Grotesk','Inter',sans-serif; font-weight:700; letter-spacing:.16em; font-size:15px; }
.sg-logo-mark { width:22px; height:22px; border-radius:50%; border:1.5px solid #38bdf8; position:relative; box-shadow:0 0 12px rgba(56,189,248,.4); display:inline-block; }
.sg-logo-mark::after { content:""; position:absolute; width:5px; height:5px; border-radius:50%; background:#fc3d21; top:-3px; right:1px; }
.sg-mlive { display:inline-flex; align-items:center; gap:9px; padding:7px 14px; border-radius:999px; border:1px solid rgba(56,189,248,.16);
  background:rgba(8,16,38,.65); color:#bae6fd; font-size:11.5px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; }
.sg-dot { width:7px; height:7px; border-radius:50%; background:#10b981; animation:sgPulse 2.2s infinite; }
.sg-dot.warn { background:#f59e0b; animation:none; }
@keyframes sgPulse { 0% { box-shadow:0 0 0 0 rgba(16,185,129,.55); } 70% { box-shadow:0 0 0 9px rgba(16,185,129,0); } 100% { box-shadow:0 0 0 0 rgba(16,185,129,0); } }
@media (max-width:700px) { .sg-mlogo { display:none; } .sg-mnav { padding-left:calc(clamp(16px,3vw,40px) + 168px); } }

/* Sol üstteki geri butonu */
.st-key-sg_back { position:fixed; top:11px; left:clamp(16px,3vw,40px); z-index:60; width:auto !important; }
.st-key-sg_back .stButton > button { height:42px; padding:0 20px; border-radius:999px; white-space:nowrap;
  border:1px solid rgba(56,189,248,.55); background:rgba(56,189,248,.10); box-shadow:none; transition:all .2s ease; }
.st-key-sg_back .stButton > button p { color:#fff; font-size:13px; font-weight:600; margin:0; }
.st-key-sg_back .stButton > button:hover { background:rgba(56,189,248,.24); border-color:#38bdf8; transform:translateY(-1px); }

/* Modül butonları */
[class*="st-key-sg_nav_"] .stButton > button { border-radius:999px; border:1px solid rgba(56,189,248,.28); background:rgba(8,16,38,.70);
  min-height:46px; transition:all .2s ease; backdrop-filter:blur(10px); }
[class*="st-key-sg_nav_"] .stButton > button p { color:#cbd5e1; font-weight:600; font-size:14px; }
[class*="st-key-sg_nav_"] .stButton > button:hover { border-color:#38bdf8; background:rgba(56,189,248,.14); transform:translateY(-2px); }

/* Kartlar, tablolar, grafikler */
.nasa-glass-card:hover { transform:translateY(-3px); border-color:rgba(56,189,248,.5); }
[data-testid="stMetric"] { background:linear-gradient(160deg,rgba(14,24,48,.80),rgba(6,11,26,.90)); border:1px solid rgba(56,189,248,.16);
  border-radius:16px; padding:14px 18px; }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border:1px solid rgba(56,189,248,.14); border-radius:14px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:14px; }
.stApp hr { border-color:rgba(56,189,248,.18) !important; }
.stApp [data-baseweb="select"] > div { border-radius:12px; }
.stApp .stButton > button[kind="primary"] { border-radius:999px; transition:transform .2s ease, box-shadow .2s ease; }
.stApp .stButton > button[kind="primary"]:hover { transform:translateY(-2px); }
.stApp iframe { border-radius:16px; }
@media (prefers-reduced-motion:reduce) { .sg-dot { animation:none; } }
</style>
"""


def _back_to_intro():
    st.session_state.intro_passed = False


st.markdown(MAIN_CSS, unsafe_allow_html=True)
_feed_word = FEED_LABEL.split(" ")[0]
st.markdown(_flat(f"""
    <div class="sg-mnav">
      <span class="sg-mlogo"><span class="sg-logo-mark"></span>SPACE GRAVE</span>
      <span class="sg-mlive"><span class="sg-dot{'' if _feed_word == 'LIVE' else ' warn'}"></span>{_feed_word}</span>
    </div>
"""), unsafe_allow_html=True)
with _keyed("sg_back"):
    st.button(T("← Giriş Sayfası", "← Home Page"), key="sg_back_btn", on_click=_back_to_intro)

st.title(T("🛰️ SPACE GRAVE [SP4C3 GR4V3] | Küresel Uzay Çöpü Merkezi", "🛰️ SPACE GRAVE [SP4C3 GR4V3] | Global Space Debris Center"))
st.markdown(T("### NASA Space Apps Challenge - JPL Komuta Merkezi", "### NASA Space Apps Challenge - JPL Command Center"))

PAGES = ["📊 General Status Overview", "🌍 3D Earth & Orbital Radar",
         "🚨 Collision Risk Analysis", "🔥 Re-Entry & Burn-Up Sim"]
_PAGES_TR = {
    PAGES[0]: "📊 Genel Durum Özeti", PAGES[1]: "🌍 3D Dünya ve Yörünge Radarı",
    PAGES[2]: "🚨 Çarpışma Riski Analizi", PAGES[3]: "🔥 Yeniden Giriş ve Yanma Sim.",
}


def page_label(p):
    return _PAGES_TR.get(p, p) if st.session_state.lang == "tr" else p


def _defense_txt():
    return T("AKTİF 🛡️", "ARMED 🛡️") if st.session_state.defense_active else T("BEKLEMEDE", "STANDBY")


def go_to(page):
    st.session_state.active_page = page


# on_click ile sayfa tek çalıştırmada değişir (st.rerun ile script iki kez çalışıyordu)
for _i, (col, page) in enumerate(zip(st.columns(len(PAGES)), PAGES)):
    with col:
        st.button(page_label(page), key=f"sg_nav_{_i}", on_click=go_to, args=(page,), **stretch(st.button))

_act_i = PAGES.index(st.session_state.active_page) if st.session_state.active_page in PAGES else 0
st.markdown(f"""<style>
.st-key-sg_nav_{_act_i} .stButton > button {{ background:rgba(56,189,248,.22) !important; border-color:#38bdf8 !important; }}
.st-key-sg_nav_{_act_i} .stButton > button p {{ color:#fff !important; }}
</style>""", unsafe_allow_html=True)

_lbl_mod, _lbl_feed, _lbl_upd = T("Aktif Modül", "Active Module"), T("Veri", "Feed"), T("Güncelleme", "Updated")
st.markdown(f"<p style='color: #38bdf8; font-size: 13px; margin-top: 5px;'>{_lbl_mod}: <b>{page_label(st.session_state.active_page)}</b> &nbsp;|&nbsp; {_lbl_feed}: <b>{FEED_LABEL}</b> &nbsp;|&nbsp; {_lbl_upd}: <b>{CAT['fetched']}</b></p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid rgba(56,189,248,0.25); margin: 15px 0;'>", unsafe_allow_html=True)

# =====================================================================
# 1. MODÜL: OVERVIEW
# =====================================================================
if st.session_state.active_page == PAGES[0]:
    n_active = len(CAT["groups"]["active"])
    n_debris = len(CAT["debris"])
    conj = st.session_state.conj_meta
    conj_txt = f"{conj['count']}" if conj else "—"

    c = st.columns(6)
    cards = [
        ("nasa-glass-card", T("Katalog Nesneleri", "Catalog Objects"), f"{n_active + n_debris:,}", "metric-value", ""),
        ("nasa-glass-card", T("Enkaz Parçaları", "Debris Fragments"), f"{n_debris:,}", "metric-value", "color:#fc3d21;"),
        ("nasa-glass-card", T("Aktif Uydular", "Active Sats"), f"{n_active:,}", "metric-value", ""),
        ("nasa-glass-card fin-card", T("Yakın Geçişler", "Close Approaches"), conj_txt, "metric-value-fin", "color:#f59e0b;"),
        ("nasa-glass-card fin-card", T("Veri Durumu", "Feed Status"), FEED_LABEL, "metric-value-fin", "font-size:16px;"),
        ("nasa-glass-card", T("Savunma Matrisi", "Defense Matrix"), _defense_txt(), "metric-value", "color:#10b981;font-size:20px;"),
    ]
    for col, (cls, title, val, vcls, style) in zip(c, cards):
        with col:
            st.markdown(f'<div class="{cls}"><div class="metric-title">{title}</div><div class="{vcls}" style="{style}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown(T("#### 🛰️ Yörünge Kabukları: İrtifa × Eğim", "#### 🛰️ Orbital Shells: Altitude × Inclination"))
    rng = np.random.default_rng(3)

    def sample(alt, inc, label, n):
        if len(alt) == 0:
            return pd.DataFrame()
        idx = rng.choice(len(alt), min(n, len(alt)), replace=False)
        return pd.DataFrame({"Altitude (km)": alt[idx], "Inclination (°)": inc[idx], "Class": label})

    shells = pd.concat([sample(CAT["a_alt"], CAT["a_inc"], "Active", 3500),
                        sample(CAT["d_alt"], CAT["d_inc"], "Debris", 3500)])
    if not shells.empty:
        shells = shells[shells["Altitude (km)"] < 2200]
        _xl, _yl, _cl = T("Eğim (°)", "Inclination (°)"), T("İrtifa (km)", "Altitude (km)"), T("Sınıf", "Class")
        _lab = {"Active": T("Aktif", "Active"), "Debris": T("Enkaz", "Debris")}
        shells = shells.rename(columns={"Inclination (°)": _xl, "Altitude (km)": _yl, "Class": _cl})
        shells[_cl] = shells[_cl].map(_lab)
        fig = px.scatter(shells, x=_xl, y=_yl, color=_cl, opacity=0.55,
                         color_discrete_map={_lab["Active"]: "#38bdf8", _lab["Debris"]: "#fc3d21"},
                         template="plotly_dark", height=430)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, **stretch(st.plotly_chart))
    else:
        st.info(T("Şu an gösterilecek yörünge verisi yok (yedek TLE'ler çok eski olabilir).",
                  "There is no orbit data to show right now (the fallback TLEs may be too old)."))

    st.markdown(T("#### 📡 Canlı Takip Tablosu", "#### 📡 Live Tracking Table"))
    st.dataframe(_disp(df_api), hide_index=True, **stretch(st.dataframe))

# =====================================================================
# 2. MODÜL: 3D RADAR
# =====================================================================
elif st.session_state.active_page == PAGES[1]:
    st.markdown(T("### 🌍 SPACE GRAVE [SP4C3 GR4V3] Profesyonel 3D Radar Merkezi", "### 🌍 SPACE GRAVE [SP4C3 GR4V3] Professional 3D Radar Hub"))
    st.caption(T("Her nesne gerçek TLE verisinden SGP4 ile propagate edilir.", "Every object is propagated from real TLE data with SGP4."))
    r1, r2 = st.columns(2)
    n_act = r1.slider(T("Aktif uydular", "Active satellites"), 0, 400, 120, key="sg_n_act")
    n_deb = r2.slider(T("Enkaz parçaları", "Debris fragments"), 0, 600, 200, key="sg_n_deb")

    @st.cache_data(ttl=300, show_spinner=False)
    def radar_payload(n_active, n_debris, steps=100, step_s=60):
        cat = load_catalog()
        rng = np.random.default_rng(7)

        def pick(lst, n):
            if not lst or n <= 0:
                return []
            return [lst[i] for i in rng.choice(len(lst), min(n, len(lst)), replace=False)]

        sel = ([("sat", t) for t in cat["groups"]["visual"]] +
               [("sat", t) for t in pick(cat["groups"]["active"], n_active)] +
               [("debris", t) for t in pick(cat["debris"], n_debris)])
        seen, objs, sats = set(), [], []
        for kind, (name, l1, l2) in sel:
            nid = norad_id(l1)
            if nid in seen:
                continue
            seen.add(nid)
            if kind == "sat" and name.startswith("ISS"):
                kind = "iss"
            objs.append({"n": name, "id": nid, "k": kind})
            sats.append(Satrec.twoline2rv(l1, l2))

        now = utc_now()
        jd0, fr0 = to_jd(now)
        gmst0 = math.radians((280.46061837 + 360.98564736629 * (jd0 + fr0 - 2451545.0)) % 360)
        if not sats:
            return {"steps": steps, "step_s": step_s, "gmst0": gmst0, "objs": [], "pos": []}

        jds, frs = to_jd_arrays(now + timedelta(seconds=step_s * k) for k in range(steps))
        e, r, _ = SatrecArray(sats).sgp4(jds, frs)
        good = (e == 0).all(axis=1)
        # TEME (x, y, z) -> three.js (x, z, -y), Dünya yarıçapı birimi
        r3 = np.stack([r[..., 0], r[..., 2], -r[..., 1]], axis=-1) / RE
        pos = [np.round(r3[i], 4).reshape(-1).tolist() for i in range(len(objs)) if good[i]]
        objs = [o for o, g in zip(objs, good) if g]
        return {"steps": steps, "step_s": step_s, "gmst0": gmst0, "objs": objs, "pos": pos}

    with st.spinner(T("Yörünge verisi propagate ediliyor...", "Propagating orbit data...")):
        payload = radar_payload(n_act, n_deb)
    flags = sorted(st.session_state.conj_meta["names"]) if st.session_state.conj_meta else []
    if not payload["objs"]:
        st.warning(T("Propagate edilebilen nesne yok; küre boş görünecek.", "No objects could be propagated; the globe will look empty."))

    html_earth_code = """
    <!DOCTYPE html><html><head>
    <style>
      body { margin:0; background:#020617; overflow:hidden; font-family:'Segoe UI',sans-serif; }
      #globe-container { width:100%; height:750px; cursor:crosshair; position:relative; }
      #hud-panel { position:absolute; top:20px; right:20px; width:360px; background:rgba(3,7,18,0.96); border:1px solid #003366; border-radius:14px; padding:18px; color:#f3f4f6; box-shadow:0 0 35px rgba(0,51,102,0.5); z-index:100; }
      #hud-panel h4 { margin:0 0 10px 0; color:#38bdf8; font-size:15px; border-bottom:1px solid #1e293b; padding-bottom:6px; }
      #hud-panel p { margin:6px 0; font-size:12px; color:#94a3b8; display:flex; justify-content:space-between; }
      #hud-panel span { color:#f8fafc; font-weight:bold; font-family:monospace; text-align:right; max-width:210px; }
      #ctl { position:absolute; left:20px; top:20px; z-index:100; color:#94a3b8; font-size:12px; background:rgba(3,7,18,0.9); padding:10px 14px; border:1px solid #003366; border-radius:10px; font-family:monospace; }
      select { background:#0b1329; color:#38bdf8; border:1px solid #003366; border-radius:6px; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head><body>
    <div id="globe-container"></div>
    <div id="ctl">Time warp: <select id="warp"><option value="60">60×</option><option value="120" selected>120×</option><option value="300">300×</option><option value="900">900×</option></select>
      <div id="clock" style="margin-top:6px;color:#38bdf8;">T+00:00</div>
    </div>
    <div id="hud-panel">
      <h4>🎯 LIVE ORBITAL TELEMETRY HUD</h4>
      <p>Target: <span id="h-name" style="color:#38bdf8;">HOVER OVER OBJECT</span></p>
      <p>NORAD Cat ID: <span id="h-id">---</span></p>
      <p>Class: <span id="h-class">---</span></p>
      <p>Altitude: <span id="h-alt" style="color:#10b981;">--- km</span></p>
      <p>Velocity: <span id="h-vel">--- km/s</span></p>
      <p>Conjunction watch: <span id="h-threat" style="color:#10b981;">---</span></p>
    </div>
    <script>
      const D = __DATA__;
      const FLAGGED = new Set(__FLAGS__);
      const RE_KM = 6378.137, OMEGA = 7.2921159e-5;
      const LOOP_S = (D.steps - 1) * D.step_s;
      const container = document.getElementById('globe-container');
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 1000);
      camera.position.set(0, 1.2, 3.2);
      const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      container.appendChild(renderer.domElement);
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true; controls.dampingFactor = 0.05;
      window.addEventListener('resize', () => {
        const w = container.clientWidth, h = container.clientHeight;
        camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
      });

      scene.add(new THREE.AmbientLight(0xffffff, 1.6));
      const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(5, 3, 5); scene.add(sun);
      const tl = new THREE.TextureLoader();
      const earthMat = new THREE.MeshStandardMaterial({ roughness:0.6, metalness:0.1, color:0x0b3d91 });
      const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), earthMat); scene.add(earth);
      tl.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg', (t) => { earthMat.map = t; earthMat.color.setHex(0xffffff); earthMat.needsUpdate = true; });
      const cloudMat = new THREE.MeshStandardMaterial({ color:0xffffff, transparent:true, opacity:0.3, blending:THREE.AdditiveBlending, depthWrite:false });
      const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.012, 64, 64), cloudMat); scene.add(clouds);
      tl.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_clouds_1024.png', (t) => { cloudMat.map = t; cloudMat.needsUpdate = true; });

      const bodyMat = new THREE.MeshStandardMaterial({ color:0xd1d5db, metalness:0.9, roughness:0.2 });
      const panelMat = new THREE.MeshStandardMaterial({ color:0x1e3a8a, metalness:0.8, roughness:0.3, emissive:0x0b2a6f });
      const debrisMat = new THREE.MeshStandardMaterial({ color:0xef4444, metalness:0.6, roughness:0.4, emissive:0x7f1d1d });
      const issMat = new THREE.MeshStandardMaterial({ color:0x10b981, emissive:0x065f46 });
      const hitMat = new THREE.MeshBasicMaterial({ visible:false });
      const objs = [];

      D.objs.forEach((o, idx) => {
        const g = new THREE.Group();
        if (o.k === 'debris') { g.add(new THREE.Mesh(new THREE.DodecahedronGeometry(0.010, 0), debrisMat)); }
        else {
          const s = o.k === 'iss' ? 2.2 : 1.0;
          const b = new THREE.Mesh(new THREE.BoxGeometry(0.012*s, 0.012*s, 0.02*s), o.k === 'iss' ? issMat : bodyMat);
          const p1 = new THREE.Mesh(new THREE.BoxGeometry(0.03*s, 0.002*s, 0.012*s), panelMat); p1.position.x = 0.02*s;
          const p2 = p1.clone(); p2.position.x = -0.02*s;
          g.add(b, p1, p2);
        }
        g.add(new THREE.Mesh(new THREE.SphereGeometry(0.03, 6, 6), hitMat));
        g.userData = { idx:idx, name:o.n, id:o.id, kind:o.k, flagged:FLAGGED.has(o.n) };
        if (g.userData.flagged) { g.add(new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 12), new THREE.MeshBasicMaterial({ color:0xfc3d21, transparent:true, opacity:0.45 }))); }
        scene.add(g); objs.push(g);
      });

      function sample(idx, t) {
        const arr = D.pos[idx], n = D.steps - 1, tt = ((t % n) + n) % n, i = Math.floor(tt), f = tt - i, a = i*3, b = (i+1)*3;
        return [arr[a] + (arr[b]-arr[a])*f, arr[a+1] + (arr[b+1]-arr[a+1])*f, arr[a+2] + (arr[b+2]-arr[a+2])*f];
      }
      function altKm(p) { const n = Math.hypot(p[0], p[1], p[2]); return (n - (1 - Math.pow(p[1]/n, 2)/298.257)) * RE_KM; }

      let orbitLine = null, hovered = -1;
      function showOrbit(idx) {
        if (orbitLine) { scene.remove(orbitLine); orbitLine.geometry.dispose(); orbitLine = null; }
        if (idx < 0) return;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(D.pos[idx], 3));
        orbitLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color:0x38bdf8, transparent:true, opacity:0.8 }));
        scene.add(orbitLine);
      }

      const raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2();
      renderer.domElement.addEventListener('mousemove', (ev) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((ev.clientX - rect.left)/rect.width)*2 - 1; mouse.y = -((ev.clientY - rect.top)/rect.height)*2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const hit = raycaster.intersectObjects(objs, true);
        if (hit.length) {
          const u = hit[0].object.parent.userData;
          if (u.idx !== hovered) { hovered = u.idx; showOrbit(hovered); }
        }
      });

      let simT = 0, last = performance.now();
      function animate(now) {
        requestAnimationFrame(animate);
        const dt = (now - last)/1000; last = now;
        const warp = parseFloat(document.getElementById('warp').value);
        simT += dt * warp / D.step_s;
        const loopSecs = (simT * D.step_s) % LOOP_S;
        earth.rotation.y = D.gmst0 + OMEGA * loopSecs;
        clouds.rotation.y = earth.rotation.y * 1.02;
        objs.forEach((g, i) => { const p = sample(i, simT); g.position.set(p[0], p[1], p[2]); g.rotation.y += 0.02; });
        const m = Math.floor(loopSecs / 60);
        document.getElementById('clock').innerText = 'T+' + String(Math.floor(m/60)).padStart(2,'0') + ':' + String(m%60).padStart(2,'0');
        if (hovered >= 0) {
          const u = objs[hovered].userData, h = 0.05, n = D.steps - 1;
          // Döngü sonunda ileri fark yerine geri fark kullan (sarma sıçramasını önler)
          const t0 = (simT % n) + h < n ? simT : simT - h;
          const p = sample(hovered, simT), pa = sample(hovered, t0), pb = sample(hovered, t0 + h);
          const v = Math.hypot(pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]) * RE_KM / (h * D.step_s);
          document.getElementById('h-name').innerText = u.name;
          document.getElementById('h-id').innerText = u.id;
          document.getElementById('h-class').innerText = u.kind === 'debris' ? 'DEBRIS' : (u.kind === 'iss' ? 'CREWED STATION' : 'SATELLITE');
          document.getElementById('h-alt').innerText = altKm(p).toFixed(0) + ' km';
          document.getElementById('h-vel').innerText = v.toFixed(2) + ' km/s';
          const t = document.getElementById('h-threat');
          t.innerText = u.flagged ? 'FLAGGED' : 'none flagged';
          t.style.color = u.flagged ? '#fc3d21' : '#10b981';
        }
        controls.update(); renderer.render(scene, camera);
      }
      requestAnimationFrame(animate);
    </script></body></html>
    """
    html_earth_code = (html_earth_code.replace("__DATA__", json.dumps(payload))
                       .replace("__FLAGS__", json.dumps(flags)))
    html_earth_code = _i18n_html(html_earth_code, [
        ("Time warp:", "Zaman hızı:"),
        ("🎯 LIVE ORBITAL TELEMETRY HUD", "🎯 CANLI YÖRÜNGE TELEMETRİ PANELİ"),
        ("Target: <span", "Hedef: <span"),
        ("HOVER OVER OBJECT", "BİR NESNENİN ÜSTÜNE GELİN"),
        ("NORAD Cat ID: <span", "NORAD Katalog No: <span"),
        ("Class: <span", "Sınıf: <span"),
        ("Altitude: <span", "İrtifa: <span"),
        ("Velocity: <span", "Hız: <span"),
        ("Conjunction watch: <span", "Yakınlaşma takibi: <span"),
        ("'DEBRIS' : (u.kind === 'iss' ? 'CREWED STATION' : 'SATELLITE')",
         "'ENKAZ' : (u.kind === 'iss' ? 'MÜRETTEBATLI İSTASYON' : 'UYDU')"),
        ("u.flagged ? 'FLAGGED' : 'none flagged'", "u.flagged ? 'İŞARETLİ' : 'işaretli değil'"),
    ])
    embed_html(html_earth_code, height=780)

# =====================================================================
# 3. MODÜL: COLLISION RISK
# =====================================================================
elif st.session_state.active_page == PAGES[2]:
    st.markdown(T("### 🚨 KRİTİK ÇARPIŞMA TEHDİT MATRİSİ", "### 🚨 CRITICAL COLLISION THREAT MATRIX"))

    def tles_for(choice):
        rng = np.random.default_rng(11)
        act, deb = CAT["groups"]["active"], CAT["debris"]

        def pick(lst, n):
            if not lst:
                return []
            return [lst[i] for i in rng.choice(len(lst), min(n, len(lst)), replace=False)]

        if choice.startswith("Active ("):
            return tuple(pick(act, 1200))
        if choice.startswith("Debris"):
            return tuple(pick(deb, 1200))
        return tuple(pick(act, 600) + pick(deb, 600))

    @st.cache_data(ttl=1800, show_spinner=False)
    def screen_conjunctions(tles, hours, thr, step_s=30):
        if len(tles) < 2:
            return pd.DataFrame()
        sats = [Satrec.twoline2rv(a, b) for _, a, b in tles]
        names = [t[0] for t in tles]
        ids = [norad_id(t[1]) for t in tles]
        arr = SatrecArray(sats)
        t0 = utc_now()
        n = int(hours * 3600 / step_s)
        best = {}
        # 1) Kaba tarama: her adımda KD-Tree ile eşik altındaki çiftler
        for c0 in range(0, n, 240):
            times = [t0 + timedelta(seconds=step_s * k) for k in range(c0, min(n, c0 + 240))]
            e, r, v = arr.sgp4(*to_jd_arrays(times))
            for j, t in enumerate(times):
                ok = np.where(e[:, j] == 0)[0]
                if len(ok) < 2:
                    continue
                p = r[ok, j, :]
                for a, b in cKDTree(p).query_pairs(thr):
                    i1, i2 = int(ok[a]), int(ok[b])
                    # Aynı yörüngede birlikte uçan nesneleri (ör. kenetli modüller) atla
                    if np.linalg.norm(v[i1, j] - v[i2, j]) < 0.5:
                        continue
                    d = float(np.linalg.norm(p[a] - p[b]))
                    if (i1, i2) not in best or d < best[(i1, i2)][0]:
                        best[(i1, i2)] = (d, t)
        # 2) İnce tarama: en yakın 40 çift için ±35 s pencerede 0.25 s adımla TCA
        top = sorted(best.items(), key=lambda kv: kv[1][0])[:40]
        offs = np.arange(-35, 35.01, 0.25)
        rows = []
        for (i1, i2), (_, t) in top:
            bd, brv, bt = math.inf, 0.0, t
            for o in offs:
                tt = t + timedelta(seconds=float(o))
                jd, fr = to_jd(tt)
                ea, ra, va = sats[i1].sgp4(jd, fr)
                eb, rb, vb = sats[i2].sgp4(jd, fr)
                if ea or eb:
                    continue
                d = math.dist(ra, rb)
                if d < bd:
                    bd, brv, bt = d, math.dist(va, vb), tt
            if math.isfinite(bd):
                rows.append({
                    "Object A": f"{names[i1]} ({ids[i1]})", "Object B": f"{names[i2]} ({ids[i2]})",
                    "Miss Distance (km)": round(bd, 3), "Rel. Velocity (km/s)": round(brv, 2),
                    "TCA (UTC)": bt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Risk": ("🔴 CRITICAL" if bd < 1 else "🟠 HIGH" if bd < 5 else "🟡 WATCH" if bd < 25 else "⚪ SCREENED"),
                    "_a": names[i1], "_b": names[i2],
                })
        df = pd.DataFrame(rows)
        return df.sort_values("Miss Distance (km)").reset_index(drop=True) if not df.empty else df

    c1, c2, c3 = st.columns(3)
    _CH = ["Active (random 1200)", "Debris fields (random 1200)", "Active × Debris (600 + 600)"]
    _CH_TR = {_CH[0]: "Aktif (rastgele 1200)", _CH[1]: "Enkaz alanları (rastgele 1200)", _CH[2]: "Aktif × Enkaz (600 + 600)"}
    choice = c1.selectbox(T("Taranacak katalog", "Catalog to screen"), _CH, index=2, key="sg_choice",
                          format_func=lambda x: _CH_TR[x] if st.session_state.lang == "tr" else x)
    hours = c2.slider(T("İleri bakış (saat)", "Look-ahead (hours)"), 1, 24, 12, key="sg_hours")
    thr = c3.slider(T("Kaba tarama eşiği (km)", "Coarse screening threshold (km)"), 20, 120, 60, key="sg_thr")

    if st.button(T("🛰️ YAKINLAŞMA TARAMASINI BAŞLAT", "🛰️ RUN CONJUNCTION SCREENING"), type="primary", **stretch(st.button)):
        with st.spinner(T("SGP4 propagasyonu + KD-Tree taraması çalışıyor...", "Running SGP4 propagation + KD-Tree screening...")):
            df = screen_conjunctions(tles_for(choice), hours, float(thr))
        st.session_state.conj_df = df
        if df.empty:
            st.session_state.conj_meta = {"count": 0, "names": set()}
            st.session_state.defense_active = False
        else:
            flagged = df[df["Miss Distance (km)"] < 25]
            st.session_state.conj_meta = {"count": len(df), "names": set(flagged["_a"]) | set(flagged["_b"])}
            st.session_state.defense_active = bool((df["Miss Distance (km)"] < 5).any())

    df = st.session_state.conj_df
    if df is None:
        st.info(T("Taramayı başlatmak için butona bas.", "Press the button to start the screening."))
    elif df.empty:
        st.success(T("Seçilen pencerede yaklaşma bulunamadı.", "No close approaches found in the selected window."))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T("Yakın geçişler", "Close approaches"), len(df))
        m2.metric(T("En yakın geçiş mesafesi", "Closest miss distance"), f"{df['Miss Distance (km)'].min():.2f} km")
        m3.metric("< 5 km", int((df["Miss Distance (km)"] < 5).sum()))
        m4.metric(T("En yüksek göreli hız", "Highest rel. velocity"), f"{df['Rel. Velocity (km/s)'].max():.1f} km/s")
        view = df.drop(columns=["_a", "_b"])
        st.dataframe(_disp(view), hide_index=True, **stretch(st.dataframe))

    # --- OrbitRisk (orbitrisk.py) ---
    st.markdown("<hr style='border: 1px solid rgba(56,189,248,0.25); margin: 25px 0;'>", unsafe_allow_html=True)
    # Koyu Tema Özel CSS Styling
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] { color: #06b6d4 !important; font-family: monospace; }
        .card {
            background-color: #111827;
            border: 1px solid #1f2937;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HEADER / ÜST BİLGİ ---
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.title(T("🛰️ ORBITRISK B2B SaaS Terminali", "🛰️ ORBITRISK B2B SaaS Terminal"))
        st.caption(T("Uzay Ekonomisi Finansal Risk ve Karar Destek Sistemi (NASA OTPS Uyumlu)",
                     "Space Economy Financial Risk & Decision Support System (NASA OTPS Aligned)"))

    with col_head2:
        st.metric(label=T("Toplam Takip Edilen Değer (LEO/GEO)", "Total Tracked Value (LEO/GEO)"),
                  value=T("$954,0 Milyar", "$954.0 Billion"))

    st.divider()

    # --- RED ALERT SİMÜLASYONU DURUMU ---
    if "red_alert" not in st.session_state:
        st.session_state.red_alert = False

    def trigger_alert():
        st.session_state.red_alert = True

    # --- ANA DÜZEN (LEFT & RIGHT COLUMNS) ---
    left_col, right_col = st.columns([2, 1])

    # ===============================
    # SOL SÜTUN: METRİKLER & GRAFİK
    # ===============================
    with left_col:
        # 3'lü Özet Kartları
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">{T("10 YILLIK FİNANSAL RİSK HAVUZU", "10-YR FINANCIAL RISK POOL")}</span>
                    <h3 style="color:white; margin:5px 0;">{T("25,8 – 42,3 Milyar &#36;", "&#36;25.8B – &#36;42.3B")}</h3>
                    <span style="color:#ef4444; font-size:11px;">{T("📈 Gizli operatör maliyet maruziyeti", "📈 Hidden operator cost exposure")}</span>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">{T("OPTİMİZE SENARYO KAYBI", "OPTIMIZED SCENARIO LOSS")}</span>
                    <h3 style="color:#06b6d4; margin:5px 0;">{T("14,2 Milyar &#36;", "&#36;14.2 Billion")}</h3>
                    <span style="color:#10b981; font-size:11px;">{T("🛡️ Erken uyarı ile 16,5 Milyar &#36; tasarruf", "🛡️ &#36;16.5B saved via early warning")}</span>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">{T("GÜVENLİK KÜRESİ EŞİĞİ", "SAFETY SPHERE THRESHOLD")}</span>
                    <h3 style="color:#f59e0b; margin:5px 0;">&lt; 50.0 km</h3>
                    <span style="color:#f59e0b; font-size:11px;">{T("🔔 Aktif 3D Yakınlaşma Matrisi", "🔔 Active 3D Conjunction Matrix")}</span>
                </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Financial Breakdown Grafiği (Plotly)
        st.subheader(T("Yörünge Riski Finansal Dökümü (10 Yıllık Ufuk)", "Orbital Risk Financial Breakdown (10-Year Horizon)"))
        st.caption(T("Temel Maruziyet ile Optimize SaaS Müdahalesinin Karşılaştırması (NASA OTPS Modeli)",
                     "Comparing Baseline Exposure vs. Optimized SaaS Intervention (NASA OTPS Model)"))

        categories = [T('Toplam Ekonomik Etki', 'Total Economic Impact'), T('Anomali Riski', 'Anomaly Risk'),
                      T('Toplam Kayıp Riski', 'Total Loss Risk'), T('Manevra Yakıt Maliyeti', 'Maneuver Fuel Cost')]
        baseline_data = [42.3, 30.7, 11.1, 0.56]
        optimized_data = [14.2, 9.1, 3.4, 0.18]

        fig = go.Figure(data=[
            go.Bar(name=T('Temel Maruziyet (Milyar $)', 'Baseline Exposure ($B)'), x=categories, y=baseline_data, marker_color='#ef4444'),
            go.Bar(name=T('Optimize SaaS Senaryosu (Milyar $)', 'Optimized SaaS Scenario ($B)'), x=categories, y=optimized_data, marker_color='#06b6d4')
        ])
        fig.update_layout(
            barmode='group',
            paper_bgcolor='#111827',
            plot_bgcolor='#111827',
            font=dict(color='#9ca3af'),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, **stretch(st.plotly_chart))

        # Canlı Yakınlaşma / Conjunction Tablosu
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.subheader(T("🌐 Aktif Yakınlaşmalar ve Güvenlik Küresi Durumu", "🌐 Active Conjunctions & Safety Sphere Status"))
        with col_t2:
            if st.session_state.red_alert:
                st.error(T("KIRMIZI ALARM: Güvenlik Küresi İhlal Edildi (<50 km)!", "RED ALARM: Safety Sphere Breached (<50km)!"))
            else:
                st.success(T("Tüm Sistemler Normal", "All Systems Nominal"))

        # Tablo Verisi
        k_sat, k_deb, k_app, k_kin, k_score, k_stat = (
            T("Hedef Uydu", "Target Satellite"), T("Enkaz / Nesne Kimliği", "Debris / Object ID"),
            T("En Yakın Geçiş", "Closest Approach"), T("Kinetik Potansiyel", "Kinetic Potential"),
            T("Dinamik Risk Skoru", "Dynamic Risk Score"), T("Durum", "Status"))
        table_data = [
            {k_sat: "Sat-Alpha (LEO 650km)", k_deb: "COSMOS 2251 Frg", k_app: T("18,4 km", "18.4 km"),
             k_kin: T("34,2 J/g (Yıkıcı Değil)", "34.2 J/g (Non-Catastrophic)"), k_score: "64 / 100", k_stat: T("Uyarı", "Warning")},
            {k_sat: "IRIDIUM-97 Fleet", k_deb: T("Roket Gövdesi (SL-8 R/B)", "Rocket Body (SL-8 R/B)"), k_app: T("72,1 km", "72.1 km"),
             k_kin: T("12,1 J/g", "12.1 J/g"), k_score: "22 / 100", k_stat: T("Temiz", "Clear")}
        ]

        # Red Alert tetiklendiyse yeni satır ekle
        if st.session_state.red_alert:
            table_data.insert(0, {
                k_sat: "Sat-Omega (LEO 800km)",
                k_deb: "IRIDIUM 33 Frg",
                k_app: T("11,2 km (KRİTİK)", "11.2 km (CRITICAL)"),
                k_kin: T("48,6 J/g (Yıkıcı)", "48.6 J/g (Catastrophic)"),
                k_score: "92 / 100",
                k_stat: T("KIRMIZI ALARM", "RED ALARM")
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, hide_index=True, **stretch(st.dataframe))


    # ===============================
    # SAĞ SÜTUN: HESAPLAMA VE FİYATLANDIRMA
    # ===============================
    with right_col:
        # Alarm Butonu
        st.button(T("🚨 Kırmızı Alarmı Simüle Et", "🚨 Simulate Red Alarm"), on_click=trigger_alert, type="primary", **stretch(st.button))

        st.write("")

        # InsurTech Policy Simulator
        st.subheader(T("🧮 InsurTech Poliçe Simülatörü", "🧮 InsurTech Policy Simulator"))
        st.caption(T("Uydu kütlesi ve yörünge yoğunluğuna göre dinamik prim ayarlarını hesaplayın.",
                     "Calculate dynamic premium adjustments based on satellite mass and orbit density."))

        sats = st.slider(T("Aktif Takımyıldız Büyüklüğü (Uydu):", "Active Constellation Size (Satellites):"),
                         min_value=10, max_value=1000, value=120, key="sg_sats")

        rocket_factor = st.selectbox(
            T("Yüksek Kütleli Roket Gövdesi Faktörü:", "High-Mass Rocket Body Factor:"),
            options=[1.0, 1.5], key="sg_rocket",
            format_func=lambda x: (T("Standart Konfigürasyon (1.0x)", "Standard Configuration (1.0x)") if x == 1.0
                                   else T("Ağır Roket Gövdesi Yakınlığı (1.5x)", "Heavy Rocket Body Proximity (1.5x)")),
            index=1
        )

        density_factor = st.selectbox(
            T("Yörünge Malzeme Yoğunluğu Çarpanı:", "Orbital Material Density Multiplier:"),
            options=[1.0, 1.25], key="sg_density",
            format_func=lambda x: (T("Düşük Yoğunluklu Enkaz Alanı (1.0x)", "Low Density Debris Field (1.0x)") if x == 1.0
                                   else T("Yüksek Yoğunluklu Parça Kümesi (1.25x)", "High Density Fragment Cluster (1.25x)")),
            index=1
        )

        # Hesaplama Algoritması
        base_score = min(98, round((sats / 10) * 0.4 + (rocket_factor * 20) + (density_factor * 15)))
        estimated_cost = (sats * 18500 * rocket_factor * density_factor) / 1000000

        st.markdown(f"""
            <div style="background-color: #1f2937; padding: 12px; border-radius: 8px; margin-top: 10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#9ca3af; font-size:12px;">{T("Dinamik Risk Skoru:", "Dynamic Risk Score:")}</span>
                    <span style="color:#f59e0b; font-weight:bold; font-family:monospace;">{base_score} / 100</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#9ca3af; font-size:12px;">{T("Tah. Yıllık Poliçe Riski:", "Est. Annual Policy Risk:")}</span>
                    <span style="color:#06b6d4; font-weight:bold; font-family:monospace; font-size:18px;">${estimated_cost:.2f}M</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.write("")

        # B2B SaaS Tiers
        st.subheader(T("📦 B2B SaaS Paketleri", "📦 B2B SaaS Tiers"))

        st.markdown(f"""
            <div class="card" style="border-left: 3px solid #3b82f6;">
                <strong style="color:white; font-size:12px;">BASIC / STANDART</strong> — <span style="color:#06b6d4; font-size:11px;">{T("Sabit Ücret / Abonelik", "Flat-Rate / Sub")}</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">{T("Hedef: Küçük Takımyıldız Operatörleri ve Üniversite Araştırmacıları. Manevra ve yakıt optimize edici.", "Target: Small Constellation Operators & University Researchers. Maneuver & fuel optimizer.")}</p>
            </div>
            <div class="card" style="border-left: 3px solid #06b6d4; background-color: #0e2433;">
                <span style="background-color:#06b6d4; color:#0a0f1d; font-size:9px; font-weight:bold; padding:2px 6px; border-radius:10px; float:right;">{T("POPÜLER", "POPULAR")}</span>
                <strong style="color:white; font-size:12px;">PRO / ENTERPRISE</strong> — <span style="color:#06b6d4; font-size:11px;">{T("Kullanıma Dayalı API", "Usage-Based API")}</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">{T("Hedef: Ticari Uydu Filo Yöneticileri. Gerçek zamanlı anomali tespiti ve kesinti maliyeti analizi.", "Target: Commercial Satellite Fleet Managers. Real-time anomaly detection & outage cost analysis.")}</p>
            </div>
            <div class="card" style="border-left: 3px solid #a855f7;">
                <strong style="color:white; font-size:12px;">INSTITUTIONAL / INSURTECH</strong> — <span style="color:#a855f7; font-size:11px;">{T("Yıllık Sözleşme", "Annual Contract")}</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">{T("Hedef: Uzay Savunma Ajansları ve Sigortacılar. Kümülatif risk modelleme ve özel poliçe fiyatlandırma matrisi.", "Target: Space Defense Agencies & Insurers. Cumulative risk modeling & custom policy pricing matrix.")}</p>
            </div>
        """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.divider()
    st.caption(T("OrbitRisk Finansal Terminali • NASA OTPS ve SpaceApps Veri Çerçeveleriyle Entegre • Akademik ve B2B Karar Destek Platformu",
                 "OrbitRisk Financial Terminal • Integrated with NASA OTPS & SpaceApps Data Frameworks • Academic & B2B Decision Support Platform"))

# =====================================================================
# 4. MODÜL: RE-ENTRY
# =====================================================================
elif st.session_state.active_page == PAGES[3]:
    # (taban irtifa km, yoğunluk kg/m³, ölçek yüksekliği km)
    ATM = [(0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682), (40, 3.972e-3, 7.554),
           (50, 1.057e-3, 8.382), (60, 3.206e-4, 7.714), (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799),
           (90, 3.396e-6, 5.382), (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473),
           (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523)]
    # Parçalanma için gereken özgül enerji (MJ/kg)
    MATERIALS = {"Aluminum alloy": 0.97, "Stainless steel": 1.0, "Titanium": 1.15, "Carbon composite (CFRP/C-C)": 3.0}
    _MAT_TR = {"Aluminum alloy": "Alüminyum alaşımı", "Stainless steel": "Paslanmaz çelik", "Titanium": "Titanyum",
               "Carbon composite (CFRP/C-C)": "Karbon kompozit (CFRP/C-C)"}

    def _mat_name(m):
        return _MAT_TR.get(m, m) if st.session_state.lang == "tr" else m

    def rho(h_km):
        h = max(h_km, 0.0)
        for hb, r0, H in reversed(ATM):
            if h >= hb:
                return r0 * math.exp(-(h - hb) / H)
        return 1.225

    def simulate_reentry(mass, area, gamma_deg, e_demise_mj, cd=2.2, rn=0.5, h0=120.0, v0=7.8):
        R, mu = 6371e3, 3.986004418e14
        sigma, eps, k = 5.670374e-8, 0.8, 1.7415e-4  # Stefan-Boltzmann, yayınım, Sutton-Graves sabiti
        h, v, g, psi, t, E = h0 * 1e3, v0 * 1e3, math.radians(gamma_deg), 0.0, 0.0, 0.0
        dt, rows, demise = 0.05, [], None
        while h > 0 and t < 2500 and v > 100:
            r = R + h
            rh = rho(h / 1e3)
            drag = 0.5 * rh * cd * area * v * v / mass
            grav = mu / r ** 2
            q = k * math.sqrt(rh / rn) * v ** 3
            T = max((q / (eps * sigma)) ** 0.25, 250.0)
            E += q * area * dt
            rows.append((t, h / 1e3, v / 1e3, math.degrees(g), drag / 9.80665, q / 1e6, T - 273.15,
                         E / mass / 1e6, psi * R / 1e3, psi))
            if E / mass >= e_demise_mj * 1e6:
                demise = len(rows) - 1
                break
            v += (-drag + grav * math.sin(g)) * dt
            g += (grav / v - v / r) * math.cos(g) * dt
            h -= v * math.sin(g) * dt
            psi += v * math.cos(g) / r * dt
            t += dt
        df = pd.DataFrame(rows, columns=["t_s", "alt_km", "v_kms", "gamma_deg", "decel_g", "flux_MWm2",
                                         "temp_C", "heat_MJkg", "downrange_km", "psi"])
        return df, demise

    st.markdown(T("### 🔥 SPACE GRAVE [SP4C3 GR4V3] Atmosfere Yeniden Giriş ve Parçalanma Simülatörü",
                  "### 🔥 SPACE GRAVE [SP4C3 GR4V3] Atmospheric Re-Entry & Disintegration Simulator"))
    st.markdown(f"<p style='color:#94a3b8;font-size:13px;'>{T('Üstel atmosfer + balistik sürükleme + Sutton-Graves ısı akısı modeli.', 'Exponential atmosphere + ballistic drag + Sutton-Graves heat flux model.')}</p>", unsafe_allow_html=True)
    st.markdown("---")

    sat_names = df_api["Satellite Name"].tolist() if not df_api.empty else []
    target_sat = st.selectbox(T("Hedef Enkaz / Uydu:", "Target Debris / Satellite:"), ["Custom object"] + sat_names, key="sg_target",
                              format_func=lambda x: T("Özel nesne", "Custom object") if x == "Custom object" else x)
    s1, s2, s3 = st.columns(3)
    mass = s1.slider(T("Nesne Kütlesi (kg)", "Object Mass (kg)"), 50, 5000, 1200, key="sg_mass")
    area = s2.slider(T("Kesit alanı (m²)", "Cross-section area (m²)"), 0.1, 30.0, 4.0, key="sg_area")
    angle = s3.slider(T("120 km'de uçuş yolu açısı (°)", "Flight-path angle at 120 km (°)"), 0.5, 45.0, 3.0, key="sg_angle")
    material = st.selectbox(T("Baskın yapısal malzeme", "Dominant structural material"), list(MATERIALS.keys()), key="sg_material",
                            format_func=_mat_name)
    start_sim = st.button(T("☄️ SPACE GRAVE YÖRÜNGEDEN ÇIKARMA SEKANSINI BAŞLAT", "☄️ INITIATE SPACE GRAVE DE-ORBIT SEQUENCE"), type="primary", **stretch(st.button))

    if start_sim:
        df, demise = simulate_reentry(mass, area, angle, MATERIALS[material])
        st.session_state.reentry = {"df": df, "demise": demise, "autoplay": True,
                                    "target": target_sat, "mass": mass, "area": area, "angle": angle, "material": material}

    res = st.session_state.reentry
    if res is None or res["df"].empty:
        frames = {"alt": [120.0], "v": [7.8], "temp": [0], "g": [0], "flux": [0], "psi": [0], "demise": -1, "autoplay": False}
    else:
        df, demise = res["df"], res["demise"]
        _tgt = T("Özel nesne", "Custom object") if res["target"] == "Custom object" else res["target"]
        st.caption(f"{T('Son simülasyon', 'Last simulation')}: {_tgt} · {res['mass']} kg · {res['area']} m² · {res['angle']}° · {_mat_name(res['material'])}")
        step = max(1, len(df) // 300)
        idx = list(range(0, len(df), step))
        if idx[-1] != len(df) - 1:
            idx.append(len(df) - 1)
        d = df.iloc[idx]
        pk = max(df["flux_MWm2"].max(), 1e-9)
        frames = {"alt": d["alt_km"].round(2).tolist(), "v": d["v_kms"].round(3).tolist(),
                  "temp": d["temp_C"].round(0).tolist(), "g": d["decel_g"].round(2).tolist(),
                  "flux": (d["flux_MWm2"] / pk).round(3).tolist(), "psi": d["psi"].round(4).tolist(),
                  "demise": (min(range(len(idx)), key=lambda j: abs(idx[j] - demise)) if demise is not None else -1),
                  "autoplay": res["autoplay"]}
        res["autoplay"] = False

    reentry_html = """
    <!DOCTYPE html><html><head>
    <style>
      body { margin:0; background:#020617; overflow:hidden; font-family:monospace; }
      #sim { width:100%; height:650px; border-radius:20px; border:1px solid rgba(0,51,102,0.8); position:relative; box-shadow:0 0 50px rgba(0,51,102,0.4); }
      .hud { position:absolute; top:20px; left:20px; color:#38bdf8; font-size:13px; background:rgba(3,7,18,0.9); padding:12px 18px; border-radius:12px; border:1px solid #003366; z-index:10; line-height:1.7; }
      #replay { position:absolute; bottom:20px; left:20px; z-index:10; background:#fc3d21; color:#fff; border:0; padding:10px 18px; border-radius:10px; font-weight:bold; cursor:pointer; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head><body>
    <div id="sim">
      <div class="hud">STATUS: <span id="st" style="color:#10b981;">STANDBY</span><br>
        ALT <span id="a">--</span> km · V <span id="v">--</span> km/s<br>TEMP <span id="t">--</span> °C · DECEL <span id="g">--</span> g</div>
      <button id="replay">▶ REPLAY</button>
    </div>
    <script>
      const R = __RES__;
      const box = document.getElementById('sim');
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, box.clientWidth/box.clientHeight, 0.01, 1000);
      camera.position.set(0, 1.9, 3.4); camera.lookAt(0, 1.1, 0);
      const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
      renderer.setSize(box.clientWidth, box.clientHeight); box.appendChild(renderer.domElement);
      window.addEventListener('resize', () => {
        const w = box.clientWidth, h = box.clientHeight;
        camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
      });
      scene.add(new THREE.AmbientLight(0xffffff, 1.8));
      const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(5, 3, 5); scene.add(sun);
      const tl = new THREE.TextureLoader();
      const earthMat = new THREE.MeshStandardMaterial({ roughness:0.5, metalness:0.1, color:0x0b3d91 });
      const earth = new THREE.Mesh(new THREE.SphereGeometry(1.2, 64, 64), earthMat); scene.add(earth);
      tl.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg', (t) => { earthMat.map = t; earthMat.color.setHex(0xffffff); earthMat.needsUpdate = true; });
      const cloudMat = new THREE.MeshStandardMaterial({ color:0xffffff, transparent:true, opacity:0.35, blending:THREE.AdditiveBlending, depthWrite:false });
      const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.215, 64, 64), cloudMat); scene.add(clouds);
      tl.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_clouds_1024.png', (t) => { cloudMat.map = t; cloudMat.needsUpdate = true; });
      scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.30, 64, 64), new THREE.MeshBasicMaterial({ color:0x38bdf8, transparent:true, opacity:0.16, side:THREE.BackSide })));

      const sat = new THREE.Group();
      sat.add(new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.075), new THREE.MeshStandardMaterial({ color:0xc0c0c0, metalness:0.9 })));
      const pm = new THREE.MeshStandardMaterial({ color:0x1e3a8a, emissive:0x0b2a6f });
      const p1 = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.004, 0.045), pm); p1.position.x = 0.085;
      const p2 = p1.clone(); p2.position.x = -0.085; sat.add(p1, p2);
      scene.add(sat);
      const glow = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 16), new THREE.MeshBasicMaterial({ color:0xff7a1a, transparent:true, opacity:0 }));
      scene.add(glow);

      const N = R.alt.length, trailPos = new Float32Array(N*3);
      const trailGeo = new THREE.BufferGeometry(); trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trailGeo.setDrawRange(0, 0);
      scene.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color:0xfc3d21 })));

      const frags = [];
      for (let i = 0; i < 28; i++) {
        const m = new THREE.Mesh(new THREE.TetrahedronGeometry(0.012), new THREE.MeshBasicMaterial({ color:0xffa040, transparent:true }));
        m.visible = false; scene.add(m); frags.push({ m:m, v:new THREE.Vector3() });
      }

      function place(i) {
        const r = 1.2 + 0.5 * Math.min(R.alt[i], 120) / 120, a = 0.22 - R.psi[i];
        return new THREE.Vector3(r*Math.sin(a), r*Math.cos(a), 0);
      }
      const DUR = 14; let f = 0, playing = R.autoplay, fragT = -1;
      document.getElementById('replay').onclick = () => { f = 0; playing = true; fragT = -1; sat.visible = true; frags.forEach(o => o.m.visible = false); };
      let last = performance.now();
      function setText(id, v) { document.getElementById(id).innerText = v; }

      function frame(now) {
        requestAnimationFrame(frame);
        const dt = (now - last)/1000; last = now;
        earth.rotation.y += 0.0008; clouds.rotation.y += 0.0011;
        if (playing) { f += dt * N / DUR; if (f >= N - 1) { f = N - 1; playing = false; } }
        const i = Math.floor(f), p = place(i);
        const dem = R.demise >= 0 && i >= R.demise;
        sat.position.copy(p); sat.rotation.y += 0.03; sat.rotation.x += 0.012;
        glow.position.copy(p); glow.material.opacity = Math.min(0.9, R.flux[i] * 1.2); glow.scale.setScalar(0.6 + R.flux[i] * 2.4);
        for (let k = 0; k <= i; k++) { const q = place(k); trailPos[k*3] = q.x; trailPos[k*3+1] = q.y; trailPos[k*3+2] = q.z; }
        trailGeo.setDrawRange(0, i + 1); trailGeo.attributes.position.needsUpdate = true;
        setText('a', R.alt[i].toFixed(1)); setText('v', R.v[i].toFixed(2)); setText('t', R.temp[i].toFixed(0)); setText('g', R.g[i].toFixed(1));
        const st = document.getElementById('st'); let s = 'STANDBY', c = '#10b981';
        if (playing || f > 0) {
          if (dem) { s = '💥 DEMISE'; c = '#fc3d21'; }
          else if (R.flux[i] > 0.1) { s = '🔥 PLASMA HEATING'; c = '#fc3d21'; }
          else { s = 'DESCENT'; c = '#38bdf8'; }
        }
        st.innerText = s; st.style.color = c;
        if (dem) {
          sat.visible = false; glow.material.opacity = 0;
          if (fragT < 0) { fragT = 0; frags.forEach(o => { o.m.position.copy(p); o.m.visible = true; o.v.set((Math.random()-0.5)*0.01, (Math.random()-0.5)*0.01, (Math.random()-0.5)*0.01); }); }
          fragT += dt;
          frags.forEach(o => { o.m.position.add(o.v); o.m.material.opacity = Math.max(0, 1 - fragT/2.5); });
        }
        renderer.render(scene, camera);
      }
      requestAnimationFrame(frame);
    </script></body></html>
    """.replace("__RES__", json.dumps(frames))
    reentry_html = _i18n_html(reentry_html, [
        ("STATUS: <span", "DURUM: <span"),
        (">STANDBY</span>", ">BEKLEMEDE</span>"),
        ("ALT <span", "İRTİFA <span"),
        ("· V <span", "· HIZ <span"),
        ("TEMP <span", "SICAKLIK <span"),
        ("· DECEL <span", "· YAVAŞLAMA <span"),
        ("▶ REPLAY", "▶ TEKRAR OYNAT"),
        ("let s = 'STANDBY'", "let s = 'BEKLEMEDE'"),
        ("'💥 DEMISE'", "'💥 PARÇALANMA'"),
        ("'🔥 PLASMA HEATING'", "'🔥 PLAZMA ISINMASI'"),
        ("'DESCENT'", "'ALÇALMA'"),
    ])
    embed_html(reentry_html, height=670)

# =====================================================================
# SAĞ ALT: MİNİ SORU-CEVAP BOTU (kural tabanlı; yapay zeka kullanmaz)
# =====================================================================
BOT_PLACEHOLDER = None  # dile göre _bot_ph() üretir


def _bot_conj():
    m = st.session_state.conj_meta
    if m:
        return T(f"{m['count']} yaklaşma", f"{m['count']} close approaches")
    return T("henüz tarama yapılmadı", "no screening run yet")


def _bot_defense():
    return _defense_txt()


# id: (butonda görünen soru, cevap üreten fonksiyon) — cevaplar sitedeki gerçek davranışa göre yazıldı
BOT_QA_TR = {
    # --- Genel ---
    "site": ("Bu site ne yapıyor?", lambda c: (
        "<b>SPACE GRAVE</b>, NASA Space Apps için hazırlanmış bir uzay çöpü komuta merkezi. "
        "CelesTrak'teki gerçek yörünge verilerini (TLE) SGP4 ile hesaplıyor. "
        f"Şu an <b>{c['n_active']:,}</b> aktif uydu ve <b>{c['n_debris']:,}</b> enkaz parçası izleniyor.")),
    "pages": ("Hangi sayfalar var?", lambda c: (
        "Dört modül var: <b>Genel Durum Özeti</b> (özet kartlar ve grafik), "
        "<b>3D Dünya ve Yörünge Radarı</b> (canlı 3D küre), <b>Çarpışma Riski Analizi</b> "
        "(yakınlaşma taraması + OrbitRisk mali panel) ve <b>Yeniden Giriş ve Yanma Sim.</b> "
        "(atmosfere giriş simülasyonu). Üstteki butonlarla geçiş yapabilirsin.")),
    "veri": ("Veriler nereden geliyor?", lambda c: (
        "Veriler <b>CelesTrak</b>'ten çekiliyor: aktif uydular, görünür uydular ve Cosmos 2251, "
        "Iridium 33, Fengyun-1C, Cosmos 1408 enkaz grupları. Veri 2 saat önbellekte tutulur. "
        f"Son güncelleme: <b>{c['fetched']}</b>. Bağlantı yoksa ISS, Hubble ve Tiangong yedek verisi kullanılır.")),
    "feed": ("Veri durumu ne demek?", lambda c: (
        "<b>LIVE 🟢</b> canlı veri, <b>CACHED 🟡</b> önbellek, <b>PARTIAL 🟡</b> bazı gruplar alınamadı, "
        f"<b>OFFLINE 🟠</b> yedek veri demek. Şu an: <b>{c['feed']}</b>.")),
    "sgp4": ("SGP4 nedir?", lambda c: (
        "<b>SGP4</b>, TLE satırlarından bir nesnenin belirli bir andaki konum ve hızını hesaplayan "
        "standart yörünge modelidir. Radar, çarpışma taraması ve tablo değerleri bununla üretiliyor.")),
    "kessler": ("Kessler Sendromu nedir?", lambda c: (
        "<b>Kessler Sendromu</b>: yörüngedeki enkaz yoğunluğu arttıkça çarpışmaların yeni parçalar "
        "üretmesi, bunların da yeni çarpışmalara yol açması zinciridir. Site, çarpışma taraması ve "
        "enkaz grafikleriyle bu riski görünür kılıyor.")),
    # --- Overview ---
    "cards": ("Kartlar ne anlama geliyor?", lambda c: (
        "<b>Katalog Nesneleri</b> aktif + enkaz toplamı, <b>Enkaz Parçaları</b> enkaz sayısı, "
        "<b>Aktif Uydular</b> aktif uydu sayısı. <b>Yakın Geçişler</b> Çarpışma sayfasında tarama "
        f"yapınca dolar (şu an: {_bot_conj()}). <b>Savunma Matrisi</b> şu an: <b>{_bot_defense()}</b>.")),
    "shells": ("Yörünge grafiği ne gösteriyor?", lambda c: (
        "Nesnelerin <b>irtifa × eğim</b> dağılımını: mavi noktalar aktif uydular, kırmızılar enkaz. "
        "Aynı yerde kümelenen noktalar aynı yörünge kabuğunu paylaşır. Okunabilirlik için rastgele "
        "örneklem ve 2200 km altı gösterilir.")),
    "table": ("Canlı tabloyu açıkla", lambda c: (
        "Tablo 'visual' grubundaki uyduları listeler; irtifa, eğim, periyot ve hız SGP4 ile anlık "
        "hesaplanır. ISS, CSS ve Tiangong <b>MÜRETTEBATLI</b> olarak işaretlenir.")),
    "decaying": ("ALÇALIYOR ne demek?", lambda c: (
        "<b>ALÇALIYOR ⚠️</b>, irtifası 300 km'nin altına inen nesneler için kullanılan etikettir. "
        "Bu yükseklikte atmosfer sürtünmesi yörüngeyi hızla düşürür.")),
    # --- 3D Radar ---
    "radar_use": ("Radarı nasıl kullanırım?", lambda c: (
        "Küreyi sürükleyerek döndür, tekerlekle yakınlaştır. Bir nesnenin üstüne gelince yörünge "
        "çizgisi belirir ve sağdaki panelde isim, NORAD ID, irtifa ve hız görünür. Sol üstteki "
        "seçici zaman hızını (60×–900×) değiştirir.")),
    "radar_colors": ("Renkler ne demek?", lambda c: (
        "Açık gri gövdeli küçük uydular aktif uydulardır, <b>yeşil</b> olan ISS, <b>kırmızı</b> parçalar "
        "enkazdır. Etrafında kırmızı halka olan nesne, çarpışma taramasında 25 km altında yaklaşma "
        "yapmıştır (panelde İŞARETLİ yazar).")),
    "radar_calc": ("Konumlar nasıl hesaplanıyor?", lambda c: (
        "Her nesne için gerçek TLE'den SGP4 ile 100 adımlık (60 sn aralıklı) bir yörünge hesaplanır; "
        "tarayıcıda akıcı hareket için ara değerler üretilir. Hareket yaklaşık 99 dakikada bir başa sarar.")),
    "radar_limit": ("Neden hepsi gösterilmiyor?", lambda c: (
        "Performans için slider'larla en fazla 400 aktif uydu ve 600 enkaz parçası (rastgele seçilmiş) "
        "gösterilir. 'visual' grubundaki uydular her zaman eklenir.")),
    # --- Collision ---
    "col_scan": ("Tarama nasıl çalışıyor?", lambda c: (
        "İki aşamalı: önce 30 sn adımlarla KD-Tree ile eşik mesafenin altındaki çiftler bulunur; sonra "
        "en yakın 40 çift için ±35 sn pencerede 0,25 sn adımla en yakın geçiş anı (TCA) hesaplanır. "
        "Birlikte uçan nesneler elenir.")),
    "col_risk": ("Risk seviyeleri neler?", lambda c: (
        "Geçiş mesafesine göre: <b>&lt; 1 km</b> 🔴 KRİTİK, <b>&lt; 5 km</b> 🟠 YÜKSEK, "
        "<b>&lt; 25 km</b> 🟡 İZLEME, daha uzağı ⚪ ELENDİ.")),
    "col_def": ("Savunma Matrisi nedir?", lambda c: (
        "5 km altında en az bir yaklaşma bulunursa <b>Savunma Matrisi</b> AKTİF olur (Genel Durum "
        f"kartında görünür). Şu an: <b>{_bot_defense()}</b>.")),
    "col_tca": ("Geçiş mesafesi ve TCA nedir?", lambda c: (
        "<b>Geçiş Mesafesi</b> iki nesnenin en yakın geçişteki uzaklığı, <b>TCA</b> bu geçişin UTC zamanı, "
        "<b>Göreli Hız</b> nesnelerin birbirine göre hızıdır. Göreli hız yükseldikçe çarpışmanın enerjisi de büyür.")),
    "col_red": ("Kırmızı Alarm ne yapar?", lambda c: (
        "Sayfanın altındaki OrbitRisk panelinde <b>🚨 Kırmızı Alarmı Simüle Et</b> demo senaryosunu başlatır: "
        "Sat-Omega için 11,2 km'lik kritik bir yaklaşma tabloya eklenir ve 50 km'lik güvenlik küresi "
        "durumu KIRMIZI ALARM'a geçer.")),
    "col_policy": ("Poliçe hesabı nasıl?", lambda c: (
        "<b>Risk Skoru</b> = min(98, uydu sayısı/10 × 0,4 + roket faktörü × 20 + yoğunluk faktörü × 15). "
        "<b>Yıllık poliçe riski</b> = uydu sayısı × 18.500 × roket faktörü × yoğunluk faktörü "
        "(milyon dolar olarak gösterilir).")),
    "col_fin": ("OrbitRisk paneli nedir?", lambda c: (
        "Çarpışma sayfasının alt bölümü <b>OrbitRisk</b> mali panelidir: 10 yıllık risk havuzu, "
        "optimize senaryo kazancı, temel durumla kıyaslayan grafik ve InsurTech simülatörü. "
        "Üstteki sabit finansal değerler örnek veridir, canlı taramadan hesaplanmaz.")),
    "col_tiers": ("SaaS paketleri neler?", lambda c: (
        "<b>Basic/Standart</b> (sabit abonelik, küçük operatörler), <b>Pro/Enterprise</b> (kullanıma "
        "dayalı API, ticari filolar) ve <b>Institutional/InsurTech</b> (yıllık sözleşme, savunma "
        "kurumları ve sigortacılar).")),
    # --- Re-entry ---
    "re_how": ("Simülasyon nasıl çalışıyor?", lambda c: (
        "Nesne 120 km'de 7,8 km/s ile başlar. Üstel atmosfer, balistik sürükleme ve Sutton-Graves ısı "
        "akısı modeliyle irtifa, hız, sıcaklık ve yavaşlama 0,05 sn adımlarla hesaplanır. "
        "▶ TEKRAR OYNAT animasyonu yeniden oynatır.")),
    "re_demise": ("Parçalanma neye bağlı?", lambda c: (
        "Biriken ısı (MJ/kg) malzemenin eşiğine ulaşınca nesne parçalanır (<b>PARÇALANMA</b>): alüminyum 0,97, "
        "çelik 1,0, titanyum 1,15, karbon kompozit 3,0 MJ/kg.")),
    "re_params": ("Hangi ayarlar sonucu değiştirir?", lambda c: (
        "Dört şey: <b>kütle</b>, <b>kesit alanı</b>, 120 km'deki <b>uçuş yolu açısı</b> ve <b>malzeme</b>. "
        "Dik açı daha kısa ve sert, sığ açı daha uzun bir giriş demektir.")),
    "re_phase": ("Animasyon aşamaları?", lambda c: (
        "<b>ALÇALMA</b> alçalma, <b>🔥 PLAZMA ISINMASI</b> ısı akısı zirvenin %10'unu aşınca, "
        "<b>💥 PARÇALANMA</b> parçalanma anıdır.")),
    "re_carbon": ("Karbon kompozit neden farklı?", lambda c: (
        "Parçalanması için gereken ısı 3,0 MJ/kg ile alüminyumun yaklaşık 3 katıdır; yani daha zor "
        "yanar ve yere ulaşan parça olasılığı artabilir.")),
}

BOT_QA_EN = {
    # --- General ---
    "site": ("What does this site do?", lambda c: (
        "<b>SPACE GRAVE</b> is a space debris command center built for NASA Space Apps. "
        "It computes real orbit data (TLE) from CelesTrak with SGP4. "
        f"Right now <b>{c['n_active']:,}</b> active satellites and <b>{c['n_debris']:,}</b> debris fragments are tracked.")),
    "pages": ("Which pages are there?", lambda c: (
        "There are four modules: <b>General Status Overview</b> (summary cards and chart), "
        "<b>3D Earth &amp; Orbital Radar</b> (live 3D globe), <b>Collision Risk Analysis</b> "
        "(conjunction screening + OrbitRisk financial panel) and <b>Re-Entry &amp; Burn-Up Sim</b> "
        "(atmospheric re-entry simulation). Use the buttons at the top to switch.")),
    "veri": ("Where does the data come from?", lambda c: (
        "Data is pulled from <b>CelesTrak</b>: active satellites, visible satellites and the Cosmos 2251, "
        "Iridium 33, Fengyun-1C and Cosmos 1408 debris groups. Data is cached for 2 hours. "
        f"Last update: <b>{c['fetched']}</b>. Without a connection, fallback data for ISS, Hubble and Tiangong is used.")),
    "feed": ("What does the feed status mean?", lambda c: (
        "<b>LIVE 🟢</b> means live data, <b>CACHED 🟡</b> cache, <b>PARTIAL 🟡</b> some groups could not be fetched, "
        f"<b>OFFLINE 🟠</b> fallback data. Right now: <b>{c['feed']}</b>.")),
    "sgp4": ("What is SGP4?", lambda c: (
        "<b>SGP4</b> is the standard orbit model that computes an object's position and velocity at a given moment "
        "from TLE lines. The radar, collision screening and table values are all produced with it.")),
    "kessler": ("What is Kessler Syndrome?", lambda c: (
        "<b>Kessler Syndrome</b> is the chain reaction in which, as debris density in orbit grows, collisions "
        "create new fragments that cause even more collisions. The site makes this risk visible with collision "
        "screening and debris charts.")),
    # --- Overview ---
    "cards": ("What do the cards mean?", lambda c: (
        "<b>Catalog Objects</b> is active + debris total, <b>Debris Fragments</b> the debris count, "
        "<b>Active Sats</b> the active satellite count. <b>Close Approaches</b> fills in once you run a screening "
        f"on the Collision page (right now: {_bot_conj()}). <b>Defense Matrix</b> is currently <b>{_bot_defense()}</b>.")),
    "shells": ("What does the orbit chart show?", lambda c: (
        "The <b>altitude × inclination</b> distribution of objects: blue dots are active satellites, red ones debris. "
        "Dots clustering in the same place share the same orbital shell. For readability, a random "
        "sample below 2200 km is shown.")),
    "table": ("Explain the live table", lambda c: (
        "The table lists satellites from the 'visual' group; altitude, inclination, period and speed are "
        "computed live with SGP4. ISS, CSS and Tiangong are marked <b>CREWED</b>.")),
    "decaying": ("What does DECAYING mean?", lambda c: (
        "<b>DECAYING ⚠️</b> is the label for objects whose altitude drops below 300 km. "
        "At that height atmospheric drag lowers the orbit quickly.")),
    # --- 3D Radar ---
    "radar_use": ("How do I use the radar?", lambda c: (
        "Drag the globe to rotate it and use the wheel to zoom. Hover over an object to see its orbit "
        "line and its name, NORAD ID, altitude and velocity in the HUD on the right. The selector at the "
        "top left changes the time warp (60×–900×).")),
    "radar_colors": ("What do the colors mean?", lambda c: (
        "Small satellites with light grey bodies are active satellites, the <b>green</b> one is the ISS and the "
        "<b>red</b> pieces are debris. An object with a red ring around it came within 25 km of another "
        "in the collision screening (the HUD shows FLAGGED).")),
    "radar_calc": ("How are positions computed?", lambda c: (
        "For each object a 100-step orbit (60 s apart) is computed from the real TLE with SGP4; "
        "the browser interpolates between steps for smooth motion. The motion loops back about every 99 minutes.")),
    "radar_limit": ("Why aren't all objects shown?", lambda c: (
        "For performance, the sliders show at most 400 active satellites and 600 debris fragments (randomly "
        "picked). Satellites in the 'visual' group are always added.")),
    # --- Collision ---
    "col_scan": ("How does the screening work?", lambda c: (
        "Two stages: first, pairs below the threshold distance are found with a KD-Tree at 30 s steps; then, "
        "for the 40 closest pairs, the time of closest approach (TCA) is computed in a ±35 s window with 0.25 s steps. "
        "Objects flying together are filtered out.")),
    "col_risk": ("What are the risk levels?", lambda c: (
        "By miss distance: <b>&lt; 1 km</b> 🔴 CRITICAL, <b>&lt; 5 km</b> 🟠 HIGH, "
        "<b>&lt; 25 km</b> 🟡 WATCH, anything farther ⚪ SCREENED.")),
    "col_def": ("What is the Defense Matrix?", lambda c: (
        "If at least one approach below 5 km is found, the <b>Defense Matrix</b> becomes ARMED (shown on the "
        f"Overview card). Right now: <b>{_bot_defense()}</b>.")),
    "col_tca": ("What are miss distance & TCA?", lambda c: (
        "<b>Miss Distance</b> is how far apart two objects are at closest approach, <b>TCA</b> is the UTC time of that pass, "
        "and <b>Rel. Velocity</b> is their relative speed. The higher the relative velocity, the bigger the collision energy.")),
    "col_red": ("What does Red Alarm do?", lambda c: (
        "In the OrbitRisk panel at the bottom of the page, <b>🚨 Simulate Red Alarm</b> starts a demo scenario: "
        "a critical 11.2 km approach for Sat-Omega is added to the table and the 50 km safety sphere "
        "status switches to RED ALARM.")),
    "col_policy": ("How is the policy calculated?", lambda c: (
        "<b>Risk Score</b> = min(98, satellite count/10 × 0.4 + rocket factor × 20 + density factor × 15). "
        "<b>Annual policy risk</b> = satellite count × 18,500 × rocket factor × density factor "
        "(shown in millions of dollars).")),
    "col_fin": ("What is the OrbitRisk panel?", lambda c: (
        "The lower part of the Collision page is the <b>OrbitRisk</b> financial panel: 10-year risk pool, "
        "optimized-scenario savings, a chart comparing against the baseline and the InsurTech simulator. "
        "The fixed financial values at the top are sample data, not computed from the live screening.")),
    "col_tiers": ("What are the SaaS tiers?", lambda c: (
        "<b>Basic/Standard</b> (flat subscription, small operators), <b>Pro/Enterprise</b> (usage-based "
        "API, commercial fleets) and <b>Institutional/InsurTech</b> (annual contract, defense "
        "agencies and insurers).")),
    # --- Re-entry ---
    "re_how": ("How does the simulation work?", lambda c: (
        "The object starts at 120 km at 7.8 km/s. Altitude, velocity, temperature and deceleration are computed "
        "in 0.05 s steps with an exponential atmosphere, ballistic drag and a Sutton-Graves heat flux model. "
        "The ▶ REPLAY button plays the animation again.")),
    "re_demise": ("What does disintegration depend on?", lambda c: (
        "When accumulated heat (MJ/kg) reaches the material's threshold, the object breaks up (<b>DEMISE</b>): aluminum 0.97, "
        "steel 1.0, titanium 1.15, carbon composite 3.0 MJ/kg.")),
    "re_params": ("Which settings change the result?", lambda c: (
        "Four things: <b>mass</b>, <b>cross-section area</b>, the <b>flight-path angle</b> at 120 km and the <b>material</b>. "
        "A steep angle means a shorter, harsher entry, a shallow angle a longer one.")),
    "re_phase": ("What are the animation phases?", lambda c: (
        "<b>DESCENT</b> is the fall, <b>🔥 PLASMA HEATING</b> starts when heat flux exceeds 10% of its peak, "
        "and <b>💥 DEMISE</b> is the moment of break-up.")),
    "re_carbon": ("Why is carbon composite different?", lambda c: (
        "The heat needed to break it up is 3.0 MJ/kg, about 3 times that of aluminum; so it burns up "
        "less easily and the chance of fragments reaching the ground can increase.")),
}


def _bot_qa():
    return BOT_QA_TR if st.session_state.lang == "tr" else BOT_QA_EN


# Sayfa indeksine göre öne çıkan sorular (+ her sayfada 2 genel soru)
BOT_FEATURED = {
    0: ["cards", "shells", "table", "decaying"],
    1: ["radar_use", "radar_colors", "radar_calc", "radar_limit"],
    2: ["col_scan", "col_risk", "col_red", "col_policy"],
    3: ["re_how", "re_demise", "re_params", "re_phase"],
}
BOT_GENERAL = ["site", "veri"]


def bot_toggle():
    st.session_state.bot_open = not st.session_state.bot_open


def bot_ask(qid):
    log = st.session_state.bot_log
    log.append({"r": "user", "q": qid})
    log.append({"r": "bot", "q": qid})
    del log[:-30]


def bot_pick():
    choice = st.session_state.sg_bot_pick
    for qid, (label, _) in _bot_qa().items():
        if label == choice:
            bot_ask(qid)
            break
    st.session_state.sg_bot_pick = _bot_ph()


def _bot_bubbles(log):
    # column-reverse kullanıldığı için en yeni mesaj önce yazılır (kutu hep en altta açılır)
    # Mesajlar seçili dile göre gösterim anında üretilir -> dil değişince geçmiş de çevrilir
    qa, ctx, out = _bot_qa(), st.session_state.bot_ctx, []
    for m in reversed(log):
        if m.get("g"):
            txt = T("Selam! Ben <b>SG-Bot</b> 🤖 Bu sayfayla ilgili merak ettiklerini aşağıdaki sorulara dokunarak sorabilirsin.",
                    "Hi! I'm <b>SG-Bot</b> 🤖 Tap the questions below to ask anything about this page.")
        elif m["q"] not in qa:
            continue
        elif m["r"] == "user":
            txt = html_escape(qa[m["q"]][0])
        else:
            txt = qa[m["q"]][1](ctx)
        txt = txt.replace("$", "&#36;")  # $...$ LaTeX gibi yorumlanmasın
        out.append(f'<div class="sg-msg sg-{"user" if m["r"] == "user" else "bot"}">{txt}</div>')
    return "".join(out)


def _bot_container(key):
    # st.container(key=...) eski Streamlit sürümlerinde yok -> sayfa altında normal kutu olarak çalışır
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


st.session_state.bot_ctx = {
    "n_active": len(CAT["groups"]["active"]),
    "n_debris": len(CAT["debris"]),
    "feed": FEED_LABEL,
    "fetched": CAT["fetched"],
    "page_i": PAGES.index(st.session_state.active_page) if st.session_state.active_page in PAGES else 0,
}

if st.session_state.bot_open:
    render_lang_switch("calc(min(340px, 92vw) + 36px)", "22px")
else:
    render_lang_switch("96px", "31px")

st.markdown("""
    <style>
    .st-key-sg_bot_fab { position: fixed; bottom: 22px; right: 22px; width: 60px; z-index: 999999; }
    .st-key-sg_bot_fab button {
        width: 60px; height: 60px; border-radius: 50%; padding: 0; font-size: 26px;
        background: linear-gradient(135deg, #0b4b94 0%, #0a2a5c 100%);
        border: 1px solid #38bdf8; box-shadow: 0 0 25px rgba(56,189,248,0.6);
    }
    .st-key-sg_bot_win {
        position: fixed; bottom: 22px; right: 22px; width: min(340px, 92vw); max-height: 88vh;
        overflow-y: auto; z-index: 999999; gap: 0.45rem !important; padding: 12px; border-radius: 16px;
        background: linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(7,11,25,0.99) 100%);
        border: 1px solid rgba(0,102,204,0.8);
        box-shadow: 0 12px 40px rgba(0,0,0,0.85), 0 0 25px rgba(0,102,204,0.35);
    }
    .st-key-sg_bot_win [data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }
    .st-key-sg_bot_win button {
        background: #0f1b33; border: 1px solid #1e3a5f; border-radius: 10px;
        padding: 4px 8px; min-height: 34px; line-height: 1.2;
    }
    .st-key-sg_bot_win button p { font-size: 11.5px; color: #cbd5e1; margin: 0; }
    .st-key-sg_bot_win button:hover { border-color: #38bdf8; }
    .st-key-sg_bot_win [data-baseweb="select"] > div { background: #0b1329; border-color: #1e3a5f; }
    .st-key-sg_bot_win [data-baseweb="select"] * { color: #e2e8f0 !important; font-size: 12px; }
    .sg-title { color: #38bdf8; font-weight: 800; font-size: 13px; letter-spacing: 1px; padding-top: 6px; }
    .sg-log {
        display: flex; flex-direction: column-reverse; gap: 6px; height: 210px; overflow-y: auto;
        padding: 6px; background: rgba(2,6,23,0.6); border: 1px solid #12315a; border-radius: 10px;
    }
    .sg-msg { font-size: 12px; line-height: 1.45; padding: 7px 10px; border-radius: 10px; max-width: 90%; color: #e2e8f0; }
    .sg-bot { align-self: flex-start; background: #0e2433; border-left: 3px solid #38bdf8; }
    .sg-user { align-self: flex-end; background: #1e3a8a; }
    </style>
""", unsafe_allow_html=True)

if st.session_state.bot_open:
    with _bot_container("sg_bot_win"):
        hc1, hc2 = st.columns([5, 1])
        hc1.markdown(f'<div class="sg-title">🛰️ SG-BOT · {T("HIZLI YARDIM", "QUICK HELP")}</div>', unsafe_allow_html=True)
        hc2.button("✕", key="sg_bot_close", on_click=bot_toggle)
        st.markdown(f'<div class="sg-log">{_bot_bubbles(st.session_state.bot_log)}</div>', unsafe_allow_html=True)
        qcols = st.columns(2)
        for i, qid in enumerate(BOT_FEATURED.get(st.session_state.bot_ctx["page_i"], []) + BOT_GENERAL):
            qcols[i % 2].button(_bot_qa()[qid][0], key=f"sg_q_{qid}", on_click=bot_ask, args=(qid,), **stretch(st.button))
        st.selectbox(T("Tüm sorular", "All questions"), [_bot_ph()] + [v[0] for v in _bot_qa().values()],
                     key="sg_bot_pick", on_change=bot_pick, label_visibility="collapsed")
else:
    with _bot_container("sg_bot_fab"):
        st.button("💬", key="sg_bot_open", on_click=bot_toggle)
