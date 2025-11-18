import streamlit as st
import base64


# ===============================================================
# 1. Iconițe SVG premium (folosite în meniu + KPI)
# ===============================================================

ICONS = {
    "sentiment": """
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"
        fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10"/>
            <path d="M8 15s1.5 2 4 2 4-2 4-2"/>
            <line x1="9" y1="9" x2="9.01" y2="9"/>
            <line x1="15" y1="9" x2="15.01" y2="9"/>
        </svg>
    """,
    "emotion": """
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"
        fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round" viewBox="0 0 24 24">
            <path d="M4 4v5h.582a2 2 0 0 1 1.84 1.18l1.297 2.815a2 2 0 0 0 1.84 1.18H14"/>
            <circle cx="19" cy="12" r="2"/>
            <path d="M17 4h1a2 2 0 0 1 2 2v4"/>
        </svg>
    """,
    "crisis": """
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"
        fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round" viewBox="0 0 24 24">
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.29 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
    """,
    "upload": """
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"
        fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round" viewBox="0 0 24 24">
            <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>
            <polyline points="7 9 12 4 17 9"/>
            <line x1="12" y1="4" x2="12" y2="16"/>
        </svg>
    """,
    "download": """
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"
        fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round" viewBox="0 0 24 24">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
    """
}


def icon(name, size=28):
    """Returnează un SVG icon ca HTML embed."""
    svg = ICONS.get(name, "")
    svg = svg.replace('width="28"', f'width="{size}"').replace('height="28"', f'height="{size}"')
    return svg


# ===============================================================
# 2. Animații Lottie (locale, embed Base64)
# ===============================================================

def load_lottie(path: str):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:application/json;base64,{b64}"


# PUTEȚI include fișiere .json de animație lottie în folderul assets/
# exemplu:
# spinner = load_lottie("assets/loader.json")


# ===============================================================
# 3. COMPONENTE UI CUSTOMIZATE
# ===============================================================

def section_title(icon_name, text):
    """Titlu de secțiune cu icon + text"""
    svg = icon(icon_name, size=30)
    return f"""
    <div style="
        display:flex;
        align-items:center;
        gap:12px;
        margin-top:25px;
        margin-bottom:10px;">
        {svg}
        <h3 style="margin:0;">{text}</h3>
    </div>
    """


def kpi_card(label, value, icon_name=None, color="#fbbf24"):
    """Returnează HTML pentru un card KPI modern."""
    svg = icon(icon_name, size=26) if icon_name else ""

    return f"""
    <div style="
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-radius: 18px;
        padding: 18px 22px;
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        color:#e5e7eb;">
        
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            {svg}
            <div style="font-size:0.75rem;letter-spacing:0.09em;color:#94a3b8;text-transform:uppercase;">
                {label}
            </div>
        </div>

        <div style="font-size:2rem;font-weight:700;color:{color}; margin-left:4px;">
            {value}
        </div>
    </div>
    """


# ===============================================================
# 4. MESAJE PRESETATE (texte premium)
# ===============================================================

INTRO_TEXT = """
<p style="font-size:1.1rem;color:#9ca3af;line-height:1.6;margin-top:-10px;">
Această aplicație folosește modele Transformer avansate pentru
analiza sentimentului, identificarea emoțiilor,
detecția automată a crizelor și vizualizarea evoluției tonului instituțional
în timp. Design modern, interactiv, optimizat pentru prezentări oficiale.
</p>
"""


UPLOAD_HELP = """
<p style="color:#94a3b8;font-size:0.9rem;">
Încarcă un fișier Excel cu postările unei instituții (CrowdTangle sau format similar).
Aplicația detectează automat coloanele <b>Date</b> și <b>Message</b>.
</p>
"""


CRISIS_EXPLAIN = """
<p style="color:#9ca3af;font-size:0.95rem;line-height:1.65;">
Scorul de criză este compus din:<br>
• <b>Negativitate</b> (procent mesaje negative)<br>
• <b>Volatilitate</b> (variația tonului în acea zi)<br>
• <b>Volum</b> (numărul total de postări publicate)<br><br>
Toate sunt convertite în <i>z-scores</i> și combinate prin media lor pozitivă.
Zilele care depășesc pragul setat sunt marcate drept potențiale crize.
</p>
"""


# ===============================================================
# READY TO USE
# ===============================================================

__all__ = [
    "ICONS",
    "icon",
    "load_lottie",
    "section_title",
    "kpi_card",
    "INTRO_TEXT",
    "UPLOAD_HELP",
    "CRISIS_EXPLAIN"
]
