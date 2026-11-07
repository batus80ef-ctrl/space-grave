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

# --- SESSION STATE ---
for key, default in {
    "intro_passed": False,
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
    st.session_state.bot_log = [
        {"r": "bot", "t": "Selam! Ben <b>SG-Bot</b> 🤖 Bu sayfayla ilgili merak ettiklerini aşağıdaki sorulara dokunarak sorabilirsin."}
    ]

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
    """Dönüş: (tles, kaynak) — kaynak: 'live' | 'cache' | 'stale' | 'fail'"""
    path = os.path.join(CACHE_DIR, f"{g}.tle")
    has_cache = os.path.exists(path)
    if has_cache and time.time() - os.path.getmtime(path) < CACHE_TTL_S:
        with open(path, encoding="utf-8") as f:
            return parse_tles(f.read()), "cache"
    try:
        r = requests.get(
            f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle",
            timeout=(5, 30), headers=HEADERS
        )
        r.raise_for_status()
        tles = parse_tles(r.text)
        if not tles:
            raise ValueError("boş yanıt")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
        return tles, "live"
    except Exception:
        # İndirme olmadıysa eski de olsa gerçek veriyi kullan
        if has_cache:
            with open(path, encoding="utf-8") as f:
                return parse_tles(f.read()), "stale"
        return [], "fail"


@st.cache_data(ttl=CACHE_TTL_S, show_spinner="🚀 Hızlı veri yükleniyor...")
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


CAT = load_catalog()
df_api = build_table(CAT["groups"]["visual"])
FEED_LABEL = CAT["status_label"]

# --- SPLASH EKRANI ---
if not st.session_state.intro_passed:
    st.markdown("<br>", unsafe_allow_html=True)
    col_i1, col_i2, col_i3 = st.columns([0.6, 3.2, 0.6])
    with col_i2:
        intro_html = f"""
        <div style="background: linear-gradient(135deg, rgba(11, 25, 53, 0.98) 0%, rgba(2, 6, 23, 0.99) 100%); border: 2px solid rgba(0, 102, 204, 0.8); padding: 50px; border-radius: 35px; box-shadow: 0 0 110px rgba(0, 102, 204, 0.65); text-align: center;">
            <h4 style="color: #fc3d21; font-size: 12px; text-transform: uppercase; letter-spacing: 7px; margin-bottom: 8px; font-weight: 800;">NASA SPACE APPS CHALLENGE // JPL COMMAND CENTER</h4>
            <h1 style="color: #ffffff; font-size: 52px; font-weight: 900; letter-spacing: 6px; margin-bottom: 4px; text-shadow: 0 0 40px rgba(0,102,204,0.9);">SPACE GRAVE</h1>
            <h2 style="color: #38bdf8; font-size: 22px; font-weight: 700; letter-spacing: 5px; margin-bottom: 20px; text-shadow: 0 0 25px rgba(56,189,248,0.7);">SP4C3 GR4V3</h2>
            <p style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 24px;">Orbital Telemetry, Conjunction Screening & Atmospheric Re-Entry Matrix</p>
            <hr style="border: 1px solid rgba(0, 102, 204, 0.5); margin-bottom: 24px;">
            <p style="color: #cbd5e1; font-size: 13.5px; line-height: 1.6; margin-bottom: 35px;">
                Welcome to <b>SPACE GRAVE (SP4C3 GR4V3)</b>. CelesTrak feed status: <b>{FEED_LABEL}</b>.
            </p>
        </div>
        """
        st.markdown(intro_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 LAUNCH SP4C3 GR4V3 // SPACE GRAVE COMMAND", type="primary", **stretch(st.button)):
            st.session_state.intro_passed = True
            st.rerun()
    st.stop()

# --- ANA KOMUTA MERKEZİ ---
st.title("🛰️ SPACE GRAVE [SP4C3 GR4V3] | Global Space Debris Center")
st.markdown("### NASA Space Apps Challenge - JPL Command Center")

PAGES = ["📊 General Status Overview", "🌍 3D Earth & Orbital Radar",
         "🚨 Collision Risk Analysis", "🔥 Re-Entry & Burn-Up Sim"]

def go_to(page):
    st.session_state.active_page = page


# on_click ile sayfa tek çalıştırmada değişir (st.rerun ile script iki kez çalışıyordu)
for col, page in zip(st.columns(len(PAGES)), PAGES):
    with col:
        st.button(page, on_click=go_to, args=(page,), **stretch(st.button))

st.markdown(f"<p style='color: #38bdf8; font-size: 13px; margin-top: 5px;'>Active Module: <b>{st.session_state.active_page}</b> &nbsp;|&nbsp; Feed: <b>{FEED_LABEL}</b> &nbsp;|&nbsp; Updated: <b>{CAT['fetched']}</b></p>", unsafe_allow_html=True)
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
        ("nasa-glass-card", "Catalog Objects", f"{n_active + n_debris:,}", "metric-value", ""),
        ("nasa-glass-card", "Debris Fragments", f"{n_debris:,}", "metric-value", "color:#fc3d21;"),
        ("nasa-glass-card", "Active Sats", f"{n_active:,}", "metric-value", ""),
        ("nasa-glass-card fin-card", "Close Approaches", conj_txt, "metric-value-fin", "color:#f59e0b;"),
        ("nasa-glass-card fin-card", "Feed Status", FEED_LABEL, "metric-value-fin", "font-size:16px;"),
        ("nasa-glass-card", "Defense Matrix", "ARMED 🛡️" if st.session_state.defense_active else "STANDBY", "metric-value", "color:#10b981;font-size:20px;"),
    ]
    for col, (cls, title, val, vcls, style) in zip(c, cards):
        with col:
            st.markdown(f'<div class="{cls}"><div class="metric-title">{title}</div><div class="{vcls}" style="{style}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("#### 🛰️ Orbital Shells: Altitude × Inclination")
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
        fig = px.scatter(shells, x="Inclination (°)", y="Altitude (km)", color="Class", opacity=0.55,
                         color_discrete_map={"Active": "#38bdf8", "Debris": "#fc3d21"},
                         template="plotly_dark", height=430)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, **stretch(st.plotly_chart))
    else:
        st.info("Şu an gösterilecek yörünge verisi yok (yedek TLE'ler çok eski olabilir).")

    st.markdown("#### 📡 Live Tracking Table")
    st.dataframe(df_api, hide_index=True, **stretch(st.dataframe))

# =====================================================================
# 2. MODÜL: 3D RADAR
# =====================================================================
elif st.session_state.active_page == PAGES[1]:
    st.markdown("### 🌍 SPACE GRAVE [SP4C3 GR4V3] Professional 3D Radar Hub")
    st.caption("Her nesne gerçek TLE verisinden SGP4 ile propagate edilir.")
    r1, r2 = st.columns(2)
    n_act = r1.slider("Active satellites", 0, 400, 120)
    n_deb = r2.slider("Debris fragments", 0, 600, 200)

    @st.cache_data(ttl=300, show_spinner="Yörünge verisi propagate ediliyor...")
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

    payload = radar_payload(n_act, n_deb)
    flags = sorted(st.session_state.conj_meta["names"]) if st.session_state.conj_meta else []
    if not payload["objs"]:
        st.warning("Propagate edilebilen nesne yok; küre boş görünecek.")

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
    embed_html(html_earth_code, height=780)

# =====================================================================
# 3. MODÜL: COLLISION RISK
# =====================================================================
elif st.session_state.active_page == PAGES[2]:
    st.markdown("### 🚨 CRITICAL COLLISION THREAT MATRIX")

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

    @st.cache_data(ttl=1800, show_spinner="SGP4 propagasyonu + KD-Tree taraması çalışıyor...")
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
    choice = c1.selectbox("Catalog to screen", ["Active (random 1200)", "Debris fields (random 1200)", "Active × Debris (600 + 600)"], index=2)
    hours = c2.slider("Look-ahead (hours)", 1, 24, 12)
    thr = c3.slider("Coarse screening threshold (km)", 20, 120, 60)

    if st.button("🛰️ RUN CONJUNCTION SCREENING", type="primary", **stretch(st.button)):
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
        st.info("Taramayı başlatmak için butona bas.")
    elif df.empty:
        st.success("Seçilen pencerede yaklaşma bulunamadı.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close approaches", len(df))
        m2.metric("Closest miss distance", f"{df['Miss Distance (km)'].min():.2f} km")
        m3.metric("< 5 km", int((df["Miss Distance (km)"] < 5).sum()))
        m4.metric("Highest rel. velocity", f"{df['Rel. Velocity (km/s)'].max():.1f} km/s")
        view = df.drop(columns=["_a", "_b"])
        st.dataframe(view, hide_index=True, **stretch(st.dataframe))

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
        st.title("🛰️ ORBITRISK B2B SaaS Terminal")
        st.caption("Space Economy Financial Risk & Decision Support System (NASA OTPS Aligned)")

    with col_head2:
        st.metric(label="Total Tracked Value (LEO/GEO)", value="$954.0 Billion")

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
            st.markdown("""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">10-YR FINANCIAL RISK POOL</span>
                    <h3 style="color:white; margin:5px 0;">$25.8B – $42.3B</h3>
                    <span style="color:#ef4444; font-size:11px;">📈 Hidden operator cost exposure</span>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">OPTIMIZED SCENARIO LOSS</span>
                    <h3 style="color:#06b6d4; margin:5px 0;">$14.2 Billion</h3>
                    <span style="color:#10b981; font-size:11px;">🛡️ $16.5B saved via early warning</span>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown("""
                <div class="card">
                    <span style="color:#9ca3af; font-size:11px;">SAFETY SPHERE THRESHOLD</span>
                    <h3 style="color:#f59e0b; margin:5px 0;">&lt; 50.0 km</h3>
                    <span style="color:#f59e0b; font-size:11px;">🔔 Active 3D Conjunction Matrix</span>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
    
        # Financial Breakdown Grafiği (Plotly)
        st.subheader("Orbital Risk Financial Breakdown (10-Year Horizon)")
        st.caption("Comparing Baseline Exposure vs. Optimized SaaS Intervention (NASA OTPS Model)")
    
        categories = ['Total Economic Impact', 'Anomaly Risk', 'Total Loss Risk', 'Maneuver Fuel Cost']
        baseline_data = [42.3, 30.7, 11.1, 0.56]
        optimized_data = [14.2, 9.1, 3.4, 0.18]

        fig = go.Figure(data=[
            go.Bar(name='Baseline Exposure ($B)', x=categories, y=baseline_data, marker_color='#ef4444'),
            go.Bar(name='Optimized SaaS Scenario ($B)', x=categories, y=optimized_data, marker_color='#06b6d4')
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
            st.subheader("🌐 Active Conjunctions & Safety Sphere Status")
        with col_t2:
            if st.session_state.red_alert:
                st.error("RED ALARM: Safety Sphere Breached (<50km)!")
            else:
                st.success("All Systems Nominal")

        # Tablo Verisi
        table_data = [
            {"Target Satellite": "Sat-Alpha (LEO 650km)", "Debris / Object ID": "COSMOS 2251 Frg", "Closest Approach": "18.4 km", "Kinetic Potential": "34.2 J/g (Non-Catastrophic)", "Dynamic Risk Score": "64 / 100", "Status": "Warning"},
            {"Target Satellite": "IRIDIUM-97 Fleet", "Debris / Object ID": "Rocket Body (SL-8 R/B)", "Closest Approach": "72.1 km", "Kinetic Potential": "12.1 J/g", "Dynamic Risk Score": "22 / 100", "Status": "Clear"}
        ]

        # Red Alert tetiklendiyse yeni satır ekle
        if st.session_state.red_alert:
            table_data.insert(0, {
                "Target Satellite": "Sat-Omega (LEO 800km)", 
                "Debris / Object ID": "IRIDIUM 33 Frg", 
                "Closest Approach": "11.2 km (CRITICAL)", 
                "Kinetic Potential": "48.6 J/g (Catastrophic)", 
                "Dynamic Risk Score": "92 / 100", 
                "Status": "RED ALARM"
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, hide_index=True, **stretch(st.dataframe))


    # ===============================
    # SAĞ SÜTUN: HESAPLAMA VE FİYATLANDIRMA
    # ===============================
    with right_col:
        # Alarm Butonu
        st.button("🚨 Simulate Red Alarm", on_click=trigger_alert, type="primary", **stretch(st.button))

        st.write("")

        # InsurTech Policy Simulator
        st.subheader("🧮 InsurTech Policy Simulator")
        st.caption("Calculate dynamic premium adjustments based on satellite mass and orbit density.")

        sats = st.slider("Active Constellation Size (Satellites):", min_value=10, max_value=1000, value=120)
    
        rocket_factor = st.selectbox(
            "High-Mass Rocket Body Factor:",
            options=[1.0, 1.5],
            format_func=lambda x: "Standard Configuration (1.0x)" if x == 1.0 else "Heavy Rocket Body Proximity (1.5x)",
            index=1
        )

        density_factor = st.selectbox(
            "Orbital Material Density Multiplier:",
            options=[1.0, 1.25],
            format_func=lambda x: "Low Density Debris Field (1.0x)" if x == 1.0 else "High Density Fragment Cluster (1.25x)",
            index=1
        )

        # Hesaplama Algoritması
        base_score = min(98, round((sats / 10) * 0.4 + (rocket_factor * 20) + (density_factor * 15)))
        estimated_cost = (sats * 18500 * rocket_factor * density_factor) / 1000000

        st.markdown(f"""
            <div style="background-color: #1f2937; padding: 12px; border-radius: 8px; margin-top: 10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#9ca3af; font-size:12px;">Dynamic Risk Score:</span>
                    <span style="color:#f59e0b; font-weight:bold; font-family:monospace;">{base_score} / 100</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#9ca3af; font-size:12px;">Est. Annual Policy Risk:</span>
                    <span style="color:#06b6d4; font-weight:bold; font-family:monospace; font-size:18px;">${estimated_cost:.2f}M</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.write("")

        # B2B SaaS Tiers
        st.subheader("📦 B2B SaaS Tiers")
    
        st.markdown("""
            <div class="card" style="border-left: 3px solid #3b82f6;">
                <strong style="color:white; font-size:12px;">BASIC / STANDART</strong> — <span style="color:#06b6d4; font-size:11px;">Flat-Rate / Sub</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">Target: Small Constellation Operators & University Researchers. Maneuver & fuel optimizer.</p>
            </div>
            <div class="card" style="border-left: 3px solid #06b6d4; background-color: #0e2433;">
                <span style="background-color:#06b6d4; color:#0a0f1d; font-size:9px; font-weight:bold; padding:2px 6px; border-radius:10px; float:right;">POPULAR</span>
                <strong style="color:white; font-size:12px;">PRO / ENTERPRISE</strong> — <span style="color:#06b6d4; font-size:11px;">Usage-Based API</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">Target: Commercial Satellite Fleet Managers. Real-time anomaly detection & outage cost analysis.</p>
            </div>
            <div class="card" style="border-left: 3px solid #a855f7;">
                <strong style="color:white; font-size:12px;">INSTITUTIONAL / INSURTECH</strong> — <span style="color:#a855f7; font-size:11px;">Annual Contract</span>
                <p style="color:#9ca3af; font-size:11px; margin-top:4px;">Target: Space Defense Agencies & Insurers. Cumulative risk modeling & custom policy pricing matrix.</p>
            </div>
        """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.divider()
    st.caption("OrbitRisk Financial Terminal • Integrated with NASA OTPS & SpaceApps Data Frameworks • Academic & B2B Decision Support Platform")

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

    st.markdown("### 🔥 SPACE GRAVE [SP4C3 GR4V3] Atmospheric Re-Entry & Disintegration Simulator")
    st.markdown("<p style='color:#94a3b8;font-size:13px;'>Üstel atmosfer + balistik sürükleme + Sutton-Graves ısı akısı modeli.</p>", unsafe_allow_html=True)
    st.markdown("---")

    sat_names = df_api["Satellite Name"].tolist() if not df_api.empty else []
    target_sat = st.selectbox("Target Debris / Satellite:", ["Custom object"] + sat_names)
    s1, s2, s3 = st.columns(3)
    mass = s1.slider("Object Mass (kg)", 50, 5000, 1200)
    area = s2.slider("Cross-section area (m²)", 0.1, 30.0, 4.0)
    angle = s3.slider("Flight-path angle at 120 km (°)", 0.5, 45.0, 3.0)
    material = st.selectbox("Dominant structural material", list(MATERIALS.keys()))
    start_sim = st.button("☄️ INITIATE SPACE GRAVE DE-ORBIT SEQUENCE", type="primary", **stretch(st.button))

    if start_sim:
        df, demise = simulate_reentry(mass, area, angle, MATERIALS[material])
        st.session_state.reentry = {"df": df, "demise": demise, "autoplay": True,
                                    "label": f"{target_sat} · {mass} kg · {area} m² · {angle}° · {material}"}

    res = st.session_state.reentry
    if res is None or res["df"].empty:
        frames = {"alt": [120.0], "v": [7.8], "temp": [0], "g": [0], "flux": [0], "psi": [0], "demise": -1, "autoplay": False}
    else:
        df, demise = res["df"], res["demise"]
        st.caption(f"Son simülasyon: {res['label']}")
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
    embed_html(reentry_html, height=670)

# =====================================================================
# SAĞ ALT: MİNİ SORU-CEVAP BOTU (kural tabanlı; yapay zeka kullanmaz)
# =====================================================================
BOT_PLACEHOLDER = "💬 Başka bir soru seç…"


def _bot_conj():
    m = st.session_state.conj_meta
    return f"{m['count']} yaklaşma" if m else "henüz tarama yapılmadı"


def _bot_defense():
    return "ARMED 🛡️" if st.session_state.defense_active else "STANDBY"


# id: (butonda görünen soru, cevap üreten fonksiyon) — cevaplar sitedeki gerçek davranışa göre yazıldı
BOT_QA = {
    # --- Genel ---
    "site": ("Bu site ne yapıyor?", lambda c: (
        "<b>SPACE GRAVE</b>, NASA Space Apps için hazırlanmış bir uzay çöpü komuta merkezi. "
        "CelesTrak'teki gerçek yörünge verilerini (TLE) SGP4 ile hesaplıyor. "
        f"Şu an <b>{c['n_active']:,}</b> aktif uydu ve <b>{c['n_debris']:,}</b> enkaz parçası izleniyor.")),
    "pages": ("Hangi sayfalar var?", lambda c: (
        "Dört modül var: <b>General Status Overview</b> (özet kartlar ve grafik), "
        "<b>3D Earth &amp; Orbital Radar</b> (canlı 3D küre), <b>Collision Risk Analysis</b> "
        "(yakınlaşma taraması + OrbitRisk mali panel) ve <b>Re-Entry &amp; Burn-Up Sim</b> "
        "(atmosfere giriş simülasyonu). Üstteki butonlarla geçiş yapabilirsin.")),
    "veri": ("Veriler nereden geliyor?", lambda c: (
        "Veriler <b>CelesTrak</b>'ten çekiliyor: aktif uydular, görünür uydular ve Cosmos 2251, "
        "Iridium 33, Fengyun-1C, Cosmos 1408 enkaz grupları. Veri 2 saat önbellekte tutulur. "
        f"Son güncelleme: <b>{c['fetched']}</b>. Bağlantı yoksa ISS, Hubble ve Tiangong yedek verisi kullanılır.")),
    "feed": ("Feed durumu ne demek?", lambda c: (
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
        "<b>Catalog Objects</b> aktif + enkaz toplamı, <b>Debris Fragments</b> enkaz sayısı, "
        "<b>Active Sats</b> aktif uydu sayısı. <b>Close Approaches</b> Collision sayfasında tarama "
        f"yapınca dolar (şu an: {_bot_conj()}). <b>Defense Matrix</b> şu an: <b>{_bot_defense()}</b>.")),
    "shells": ("Yörünge grafiği ne gösteriyor?", lambda c: (
        "Nesnelerin <b>irtifa × eğim</b> dağılımını: mavi noktalar aktif uydular, kırmızılar enkaz. "
        "Aynı yerde kümelenen noktalar aynı yörünge kabuğunu paylaşır. Okunabilirlik için rastgele "
        "örneklem ve 2200 km altı gösterilir.")),
    "table": ("Canlı tabloyu açıkla", lambda c: (
        "Tablo 'visual' grubundaki uyduları listeler; irtifa, eğim, periyot ve hız SGP4 ile anlık "
        "hesaplanır. ISS, CSS ve Tiangong <b>CREWED</b> olarak işaretlenir.")),
    "decaying": ("DECAYING ne demek?", lambda c: (
        "<b>DECAYING ⚠️</b>, irtifası 300 km'nin altına inen nesneler için kullanılan etikettir. "
        "Bu yükseklikte atmosfer sürtünmesi yörüngeyi hızla düşürür.")),
    # --- 3D Radar ---
    "radar_use": ("Radarı nasıl kullanırım?", lambda c: (
        "Küreyi sürükleyerek döndür, tekerlekle yakınlaştır. Bir nesnenin üstüne gelince yörünge "
        "çizgisi belirir ve sağdaki HUD'da isim, NORAD ID, irtifa ve hız görünür. Sol üstteki "
        "seçici zaman hızını (60×–900×) değiştirir.")),
    "radar_colors": ("Renkler ne demek?", lambda c: (
        "Açık gri gövdeli küçük uydular aktif uydulardır, <b>yeşil</b> olan ISS, <b>kırmızı</b> parçalar "
        "enkazdır. Etrafında kırmızı halka olan nesne, çarpışma taramasında 25 km altında yaklaşma "
        "yapmıştır (HUD'da FLAGGED yazar).")),
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
        "Miss distance'a göre: <b>&lt; 1 km</b> 🔴 CRITICAL, <b>&lt; 5 km</b> 🟠 HIGH, "
        "<b>&lt; 25 km</b> 🟡 WATCH, daha uzağı ⚪ SCREENED.")),
    "col_def": ("Defense Matrix nedir?", lambda c: (
        "5 km altında en az bir yaklaşma bulunursa <b>Defense Matrix</b> ARMED olur (Overview kartında "
        f"görünür). Şu an: <b>{_bot_defense()}</b>.")),
    "col_tca": ("Miss distance & TCA nedir?", lambda c: (
        "<b>Miss Distance</b> iki nesnenin en yakın geçişteki uzaklığı, <b>TCA</b> bu geçişin UTC zamanı, "
        "<b>Rel. Velocity</b> göreli hızdır. Göreli hız yükseldikçe çarpışmanın enerjisi de büyür.")),
    "col_red": ("Red Alarm ne yapar?", lambda c: (
        "Sayfanın altındaki OrbitRisk panelinde <b>🚨 Simulate Red Alarm</b> demo senaryosunu başlatır: "
        "Sat-Omega için 11.2 km'lik kritik bir yaklaşma tabloya eklenir ve 50 km'lik güvenlik küresi "
        "durumu RED ALARM'a geçer.")),
    "col_policy": ("Poliçe hesabı nasıl?", lambda c: (
        "<b>Risk Skoru</b> = min(98, uydu sayısı/10 × 0,4 + roket faktörü × 20 + yoğunluk faktörü × 15). "
        "<b>Yıllık poliçe riski</b> = uydu sayısı × 18.500 × roket faktörü × yoğunluk faktörü "
        "(milyon dolar olarak gösterilir).")),
    "col_fin": ("OrbitRisk paneli nedir?", lambda c: (
        "Collision sayfasının alt bölümü <b>OrbitRisk</b> mali panelidir: 10 yıllık risk havuzu, "
        "optimize senaryo kazancı, baseline ile kıyaslayan grafik ve InsurTech simülatörü. "
        "Üstteki sabit finansal değerler örnek veridir, canlı taramadan hesaplanmaz.")),
    "col_tiers": ("SaaS paketleri neler?", lambda c: (
        "<b>Basic/Standart</b> (sabit abonelik, küçük operatörler), <b>Pro/Enterprise</b> (kullanıma "
        "dayalı API, ticari filolar) ve <b>Institutional/InsurTech</b> (yıllık sözleşme, savunma "
        "kurumları ve sigortacılar).")),
    # --- Re-entry ---
    "re_how": ("Simülasyon nasıl çalışıyor?", lambda c: (
        "Nesne 120 km'de 7,8 km/s ile başlar. Üstel atmosfer, balistik sürükleme ve Sutton-Graves ısı "
        "akısı modeliyle irtifa, hız, sıcaklık ve yavaşlama 0,05 sn adımlarla hesaplanır. "
        "▶ REPLAY animasyonu tekrar oynatır.")),
    "re_demise": ("Parçalanma neye bağlı?", lambda c: (
        "Biriken ısı (MJ/kg) malzemenin eşiğine ulaşınca nesne parçalanır (<b>DEMISE</b>): alüminyum 0,97, "
        "çelik 1,0, titanyum 1,15, karbon kompozit 3,0 MJ/kg.")),
    "re_params": ("Hangi ayarlar sonucu değiştirir?", lambda c: (
        "Dört şey: <b>kütle</b>, <b>kesit alanı</b>, 120 km'deki <b>uçuş yolu açısı</b> ve <b>malzeme</b>. "
        "Dik açı daha kısa ve sert, sığ açı daha uzun bir giriş demektir.")),
    "re_phase": ("Animasyon aşamaları?", lambda c: (
        "<b>DESCENT</b> alçalma, <b>🔥 PLASMA HEATING</b> ısı akısı zirvenin %10'unu aşınca, "
        "<b>💥 DEMISE</b> parçalanma anıdır.")),
    "re_carbon": ("Karbon kompozit neden farklı?", lambda c: (
        "Parçalanması için gereken ısı 3,0 MJ/kg ile alüminyumun yaklaşık 3 katıdır; yani daha zor "
        "yanar ve yere ulaşan parça olasılığı artabilir.")),
}

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
    label, answer = BOT_QA[qid]
    log = st.session_state.bot_log
    log.append({"r": "user", "t": html_escape(label)})
    log.append({"r": "bot", "t": answer(st.session_state.bot_ctx)})
    del log[:-30]


def bot_pick():
    choice = st.session_state.sg_bot_pick
    for qid, (label, _) in BOT_QA.items():
        if label == choice:
            bot_ask(qid)
            break
    st.session_state.sg_bot_pick = BOT_PLACEHOLDER


def _bot_bubbles(log):
    # column-reverse kullanıldığı için en yeni mesaj önce yazılır (kutu hep en altta açılır)
    out = []
    for m in reversed(log):
        txt = m["t"].replace("$", "&#36;")  # $...$ LaTeX gibi yorumlanmasın
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
        hc1.markdown('<div class="sg-title">🛰️ SG-BOT · HIZLI YARDIM</div>', unsafe_allow_html=True)
        hc2.button("✕", key="sg_bot_close", on_click=bot_toggle)
        st.markdown(f'<div class="sg-log">{_bot_bubbles(st.session_state.bot_log)}</div>', unsafe_allow_html=True)
        qcols = st.columns(2)
        for i, qid in enumerate(BOT_FEATURED.get(st.session_state.bot_ctx["page_i"], []) + BOT_GENERAL):
            qcols[i % 2].button(BOT_QA[qid][0], key=f"sg_q_{qid}", on_click=bot_ask, args=(qid,), **stretch(st.button))
        st.selectbox("Tüm sorular", [BOT_PLACEHOLDER] + [v[0] for v in BOT_QA.values()],
                     key="sg_bot_pick", on_change=bot_pick, label_visibility="collapsed")
else:
    with _bot_container("sg_bot_fab"):
        st.button("💬", key="sg_bot_open", on_click=bot_toggle)
