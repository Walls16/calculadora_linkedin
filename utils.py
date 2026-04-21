"""
utils.py
--------
Utilidades compartidas para toda la app.
Centraliza: temas de color, CSS, instancia del engine y helpers de UI.
"""

import streamlit as st
from financial_engine import FinancialMathEngine

# =============================================================================
# INSTANCIA GLOBAL DEL MOTOR
# =============================================================================
@st.cache_resource
def get_engine() -> FinancialMathEngine:
    """Devuelve una instancia cacheada del motor financiero."""
    return FinancialMathEngine()


# =============================================================================
# DEFINICION DE TEMAS
# Cada tema tiene las claves base + title_color, subtitle_color, input_bg,
# text_color (color principal de texto) y bg_main (fondo global de la app).
# =============================================================================
THEMES = {
    # -------------------------------------------------------------------------
    # TEMAS CLAROS (9 Temas)
    # -------------------------------------------------------------------------
    "Azul & Rosa": {
        "primary":       "#E84797", "secondary":     "#94C2DA", "accent":        "#E84797",
        "success":       "#4E7CB2", "danger":        "#C62828", "warning":       "#D97706",
        "bg_light":      "#EFE8E0", "border":        "#EAB4CC", "text_muted":    "#64748B",
        "sidebar_bg":    "#F8F4F0", "title_color":   "#E84797", "subtitle_color":"#94C2DA",
        "input_bg":      "#F2C8D8", "text_color":    "#1A1A2A", "bg_main":       "#FFFFFF",
        "info_bg":       "#FFEBEE", "info_border":   "#E84797", "info_text":     "#1A3A5C",
        "success_bg":    "#E0EEF8", "success_border":"#4E7CB2", "success_text":  "#1A3A5C",
        "warning_bg":    "#F2C8D8", "warning_border":"#E84797", "warning_text":  "#5A3E00",
        "error_bg":      "#FFEBEE", "error_border":  "#C62828", "error_text":    "#7B0000",
    },
    "Azul Clasico": {
        "primary":       "#1E3A8A", "secondary":     "#2563EB", "accent":        "#3B82F6",
        "success":       "#2E7D32", "danger":        "#C62828", "warning":       "#D97706",
        "bg_light":      "#F3F4F6", "border":        "#E2E8F0", "text_muted":    "#64748B",
        "sidebar_bg":    "#F8FAFC", "title_color":   "#1E3A8A", "subtitle_color":"#2563EB",
        "input_bg":      "#EFF6FF", "text_color":    "#0F172A", "bg_main":       "#FFFFFF",
        "info_bg":       "#EFF6FF", "info_border":   "#2563EB", "info_text":     "#1E3A8A",
        "success_bg":    "#F0FDF4", "success_border":"#2E7D32", "success_text":  "#14532D",
        "warning_bg":    "#FFFBEB", "warning_border":"#D97706", "warning_text":  "#78350F",
        "error_bg":      "#FEF2F2", "error_border":  "#C62828", "error_text":    "#7F1D1D",
    },
    "Rojo Ejecutivo": {
        "primary":       "#7B0000", "secondary":     "#C62828", "accent":        "#EF5350",
        "success":       "#2E7D32", "danger":        "#B71C1C", "warning":       "#F57F17",
        "bg_light":      "#FFF5F5", "border":        "#FFCDD2", "text_muted":    "#78909C",
        "sidebar_bg":    "#FFF8F8", "title_color":   "#7B0000", "subtitle_color":"#C62828",
        "input_bg":      "#FFF0F0", "text_color":    "#1A0000", "bg_main":       "#FFFFFF",
        "info_bg":       "#FFF0F0", "info_border":   "#C62828", "info_text":     "#7B0000",
        "success_bg":    "#F0FDF4", "success_border":"#2E7D32", "success_text":  "#14532D",
        "warning_bg":    "#FFFBEB", "warning_border":"#F57F17", "warning_text":  "#78350F",
        "error_bg":      "#FEF2F2", "error_border":  "#B71C1C", "error_text":    "#7F1D1D",
    },
    "Verde Esmeralda": {
        "primary":       "#064E3B", "secondary":     "#059669", "accent":        "#10B981",
        "success":       "#047857", "danger":        "#DC2626", "warning":       "#D97706",
        "bg_light":      "#F0FDF4", "border":        "#A7F3D0", "text_muted":    "#6B7280",
        "sidebar_bg":    "#ECFDF5", "title_color":   "#064E3B", "subtitle_color":"#059669",
        "input_bg":      "#DCFCE7", "text_color":    "#022C22", "bg_main":       "#FFFFFF",
        "info_bg":       "#ECFDF5", "info_border":   "#10B981", "info_text":     "#064E3B",
        "success_bg":    "#F0FDF4", "success_border":"#047857", "success_text":  "#14532D",
        "warning_bg":    "#FFFBEB", "warning_border":"#D97706", "warning_text":  "#78350F",
        "error_bg":      "#FEF2F2", "error_border":  "#DC2626", "error_text":    "#7F1D1D",
    },
    "Violeta Profundo": {
        "primary":       "#3B0764", "secondary":     "#7C3AED", "accent":        "#A855F7",
        "success":       "#166534", "danger":        "#B91C1C", "warning":       "#B45309",
        "bg_light":      "#FAF5FF", "border":        "#E9D5FF", "text_muted":    "#94A3B8",
        "sidebar_bg":    "#F5F0FF", "title_color":   "#3B0764", "subtitle_color":"#7C3AED",
        "input_bg":      "#EDE9FE", "text_color":    "#1E0037", "bg_main":       "#FFFFFF",
        "info_bg":       "#F5F3FF", "info_border":   "#7C3AED", "info_text":     "#3B0764",
        "success_bg":    "#F0FDF4", "success_border":"#166534", "success_text":  "#14532D",
        "warning_bg":    "#FFFBEB", "warning_border":"#B45309", "warning_text":  "#78350F",
        "error_bg":      "#FEF2F2", "error_border":  "#B91C1C", "error_text":    "#7F1D1D",
    },
    "Teal Minimalista": {
        "primary":       "#134E4A", "secondary":     "#0D9488", "accent":        "#14B8A6",
        "success":       "#166534", "danger":        "#DC2626", "warning":       "#CA8A04",
        "bg_light":      "#F0FDFA", "border":        "#99F6E4", "text_muted":    "#64748B",
        "sidebar_bg":    "#EAFAF8", "title_color":   "#134E4A", "subtitle_color":"#0D9488",
        "input_bg":      "#CCFBF1", "text_color":    "#042F2E", "bg_main":       "#FFFFFF",
        "info_bg":       "#F0FDFA", "info_border":   "#14B8A6", "info_text":     "#134E4A",
        "success_bg":    "#F0FDF4", "success_border":"#166534", "success_text":  "#14532D",
        "warning_bg":    "#FFFBEB", "warning_border":"#CA8A04", "warning_text":  "#78350F",
        "error_bg":      "#FEF2F2", "error_border":  "#DC2626", "error_text":    "#7F1D1D",
    },
    "Blanco Limpio": {
        "primary":       "#111827", "secondary":     "#374151", "accent":        "#6B7280",
        "success":       "#15803D", "danger":        "#B91C1C", "warning":       "#D97706",
        "bg_light":      "#F9FAFB", "border":        "#E5E7EB", "text_muted":    "#9CA3AF",
        "sidebar_bg":    "#FFFFFF", "title_color":   "#111827", "subtitle_color":"#374151",
        "input_bg":      "#F3F4F6", "text_color":    "#111827", "bg_main":       "#FFFFFF",
        "info_bg":       "#F0F9FF", "info_border":   "#6B7280", "info_text":     "#1F2937",
        "success_bg":    "#F0FDF4", "success_border":"#15803D", "success_text":  "#14532D",
        "warning_bg":    "#FFFBEB", "warning_border":"#D97706", "warning_text":  "#78350F",
        "error_bg":      "#FEF2F2", "error_border":  "#B91C1C", "error_text":    "#7F1D1D",
    },
    "Café Latte Soft": {
        "primary":       "#92400E", "secondary":     "#B45309", "accent":        "#D97706",
        "success":       "#15803D", "danger":        "#B91C1C", "warning":       "#B45309",
        "bg_light":      "#FAFAF9", "border":        "#E7E5E4", "text_muted":    "#78716C",
        "sidebar_bg":    "#F5F5F4", "title_color":   "#78350F", "subtitle_color":"#92400E",
        "input_bg":      "#F5F5F4", "text_color":    "#292524", "bg_main":       "#FFFFFF",
        "info_bg":       "#FDF2F2", "info_border":   "#D97706", "info_text":     "#78350F",
        "success_bg":    "#F0FDF4", "success_border":"#15803D", "success_text":  "#14532D",
        "warning_bg":    "#FFFBEB", "warning_border":"#D97706", "warning_text":  "#78350F",
        "error_bg":      "#FEF2F2", "error_border":  "#B91C1C", "error_text":    "#7F1D1D",
    },
    "Océano & Arena": {
        "primary":       "#0369A1", "secondary":     "#0284C7", "accent":        "#D97706",
        "success":       "#047857", "danger":        "#BE123C", "warning":       "#EA580C",
        "bg_light":      "#FFFFFF", "border":        "#E0F2FE", "text_muted":    "#64748B",
        "sidebar_bg":    "#F8FAFC", "title_color":   "#0369A1", "subtitle_color":"#0284C7",
        "input_bg":      "#F0F9FF", "text_color":    "#0F172A", "bg_main":       "#FDFDF8",
        "info_bg":       "#F0F9FF", "info_border":   "#0284C7", "info_text":     "#0369A1",
        "success_bg":    "#ECFDF5", "success_border":"#047857", "success_text":  "#064E3B",
        "warning_bg":    "#FFF7ED", "warning_border":"#EA580C", "warning_text":  "#9A3412",
        "error_bg":      "#FFF1F2", "error_border":  "#BE123C", "error_text":    "#881337",
    },

    # -------------------------------------------------------------------------
    # TEMAS OSCUROS (9 Temas)
    # -------------------------------------------------------------------------
    "TRON Legacy": {
        "primary":       "#00E5FF", "secondary":     "#008291", "accent":        "#FF8C00",
        "success":       "#00E5FF", "danger":        "#FF2E2E", "warning":       "#FF8C00",
        "bg_light":      "#0A1A2F", "border":        "#00E5FF33", "text_muted":    "#5080A0",
        "sidebar_bg":    "#020A12", "title_color":   "#00E5FF", "subtitle_color":"#00A3B4",
        "input_bg":      "#0D253F", "text_color":    "#E0F7FA", "bg_main":       "#01050A",
        "info_bg":       "#002B36", "info_border":   "#00E5FF", "info_text":     "#80F1FF",
        "success_bg":    "#002B36", "success_border":"#00E5FF", "success_text":  "#80F1FF",
        "warning_bg":    "#1A1100", "warning_border":"#FF8C00", "warning_text":  "#FFD080",
        "error_bg":      "#2A0000", "error_border":  "#FF2E2E", "error_text":    "#FFB2B2",
    },
    "Negro & Dorado": {
        "primary":       "#D4AF37", "secondary":     "#AA8439", "accent":        "#FFD700",
        "success":       "#D4AF37", "danger":        "#DC3545", "warning":       "#AA8439",
        "bg_light":      "#1A1A1A", "border":        "#333333", "text_muted":    "#888888",
        "sidebar_bg":    "#0A0A0A", "title_color":   "#D4AF37", "subtitle_color":"#AA8439",
        "input_bg":      "#262626", "text_color":    "#FFFFFF", "bg_main":       "#000000",
        "info_bg":       "#1A1608", "info_border":   "#D4AF37", "info_text":     "#F3E5AB",
        "success_bg":    "#0D2B1F", "success_border":"#198754", "success_text":  "#6EE7B7",
        "warning_bg":    "#261A00", "warning_border":"#AA8439", "warning_text":  "#FFD700",
        "error_bg":      "#2A0000", "error_border":  "#DC3545", "error_text":    "#FFB2B2",
    },
    "Cyberpunk Night": {
        "primary":       "#FDE047", "secondary":     "#FF00FF", "accent":        "#00FFFF",
        "success":       "#00FFFF", "danger":        "#FF003C", "warning":       "#FDE047",
        "bg_light":      "#120422", "border":        "#FF00FF44", "text_muted":    "#8B5CF6",
        "sidebar_bg":    "#0B0118", "title_color":   "#FDE047", "subtitle_color":"#FF00FF",
        "input_bg":      "#1E0836", "text_color":    "#F3E8FF", "bg_main":       "#05010D",
        "info_bg":       "#1A0033", "info_border":   "#FF00FF", "info_text":     "#FFB3FF",
        "success_bg":    "#002222", "success_border":"#00FFFF", "success_text":  "#B3FFFF",
        "warning_bg":    "#221E00", "warning_border":"#FDE047", "warning_text":  "#FFF7B3",
        "error_bg":      "#33000C", "error_border":  "#FF003C", "error_text":    "#FFB3C6",
    },
    "Gris Espacial": {
        "primary":       "#A1A1AA", "secondary":     "#D4D4D8", "accent":        "#3B82F6",
        "success":       "#22C55E", "danger":        "#EF4444", "warning":       "#F59E0B",
        "bg_light":      "#27272A", "border":        "#3F3F46", "text_muted":    "#A1A1AA",
        "sidebar_bg":    "#1F1F22", "title_color":   "#FFFFFF", "subtitle_color":"#D4D4D8",
        "input_bg":      "#3F3F46", "text_color":    "#F4F4F5", "bg_main":       "#18181B",
        "info_bg":       "#1E293B", "info_border":   "#3B82F6", "info_text":     "#93C5FD",
        "success_bg":    "#064E3B", "success_border":"#10B981", "success_text":  "#6EE7B7",
        "warning_bg":    "#422006", "warning_border":"#F59E0B", "warning_text":  "#FCD34D",
        "error_bg":      "#4C0519", "error_border":  "#EF4444", "error_text":    "#FCA5A5",
    },
    "Azul Medianoche": {
        "primary":       "#38BDF8", "secondary":     "#7DD3FC", "accent":        "#0EA5E9",
        "success":       "#10B981", "danger":        "#F43F5E", "warning":       "#F59E0B",
        "bg_light":      "#1E293B", "border":        "#334155", "text_muted":    "#94A3B8",
        "sidebar_bg":    "#152033", "title_color":   "#38BDF8", "subtitle_color":"#7DD3FC",
        "input_bg":      "#334155", "text_color":    "#F8FAFC", "bg_main":       "#0F172A",
        "info_bg":       "#0C4A6E", "info_border":   "#0EA5E9", "info_text":     "#BAE6FD",
        "success_bg":    "#064E3B", "success_border":"#10B981", "success_text":  "#6EE7B7",
        "warning_bg":    "#451A03", "warning_border":"#F59E0B", "warning_text":  "#FDE68A",
        "error_bg":      "#4C0519", "error_border":  "#F43F5E", "error_text":    "#FECDD3",
    },
    "Tema Drácula": {
        "primary":       "#FF79C6", "secondary":     "#BD93F9", "accent":        "#50FA7B",
        "success":       "#50FA7B", "danger":        "#FF5555", "warning":       "#F1FA8C",
        "bg_light":      "#44475A", "border":        "#6272A4", "text_muted":    "#6272A4",
        "sidebar_bg":    "#21222C", "title_color":   "#FF79C6", "subtitle_color":"#BD93F9",
        "input_bg":      "#44475A", "text_color":    "#F8F8F2", "bg_main":       "#282A36",
        "info_bg":       "#21222C", "info_border":   "#8BE9FD", "info_text":     "#8BE9FD",
        "success_bg":    "#21222C", "success_border":"#50FA7B", "success_text":  "#50FA7B",
        "warning_bg":    "#21222C", "warning_border":"#F1FA8C", "warning_text":  "#F1FA8C",
        "error_bg":      "#21222C", "error_border":  "#FF5555", "error_text":    "#FF5555",
    },
    "Bosque Luminiscente": {
        "primary":       "#A3E635", "secondary":     "#84CC16", "accent":        "#2DD4BF",
        "success":       "#4ADE80", "danger":        "#F87171", "warning":       "#FBBF24",
        "bg_light":      "#0B291C", "border":        "#064E3B", "text_muted":    "#6EE7B7",
        "sidebar_bg":    "#04140D", "title_color":   "#A3E635", "subtitle_color":"#D9F99D",
        "input_bg":      "#133E2B", "text_color":    "#ECFCCB", "bg_main":       "#061E14",
        "info_bg":       "#134E4A", "info_border":   "#2DD4BF", "info_text":     "#99F6E4",
        "success_bg":    "#064E3B", "success_border":"#4ADE80", "success_text":  "#A7F3D0",
        "warning_bg":    "#331A00", "warning_border":"#FBBF24", "warning_text":  "#FDE68A",
        "error_bg":      "#450A0A", "error_border":  "#F87171", "error_text":    "#FECACA",
    },
    "Obsidiana Roja": {
        "primary":       "#FF1E1E", "secondary":     "#950101", "accent":        "#FF4D4D",
        "success":       "#FF1E1E", "danger":        "#950101", "warning":       "#FF4D4D",
        "bg_light":      "#1A0000", "border":        "#4D0000", "text_muted":    "#804040",
        "sidebar_bg":    "#0D0000", "title_color":   "#FF1E1E", "subtitle_color":"#950101",
        "input_bg":      "#260000", "text_color":    "#FFDADA", "bg_main":       "#050000",
        "info_bg":       "#260000", "info_border":   "#FF1E1E", "info_text":     "#FFB3B3",
        "success_bg":    "#260000", "success_border":"#FF1E1E", "success_text":  "#FFB3B3",
        "warning_bg":    "#331100", "warning_border":"#FF4D4D", "warning_text":  "#FFD1B3",
        "error_bg":      "#4D0000", "error_border":  "#950101", "error_text":    "#FF8080",
    },
    "Espacio Profundo": {
        "primary":       "#A855F7", "secondary":     "#7C3AED", "accent":        "#C084FC",
        "success":       "#22D3EE", "danger":        "#F43F5E", "warning":       "#FB923C",
        "bg_light":      "#1E1B4B", "border":        "#312E81", "text_muted":    "#94A3B8",
        "sidebar_bg":    "#0F0E23", "title_color":   "#C084FC", "subtitle_color":"#A855F7",
        "input_bg":      "#1E1B4B", "text_color":    "#F5F3FF", "bg_main":       "#020617",
        "info_bg":       "#1E1B4B", "info_border":   "#A855F7", "info_text":     "#E9D5FF",
        "success_bg":    "#083344", "success_border":"#22D3EE", "success_text":  "#A5F3FC",
        "warning_bg":    "#431407", "warning_border":"#FB923C", "warning_text":  "#FFEDD5",
        "error_bg":      "#450A0A", "error_border":  "#F43F5E", "error_text":    "#FECACA",
    }
}

_DEFAULT_THEME = "Azul Clasico"

# =============================================================================
# SELECTOR DE TEMA
# =============================================================================
def get_current_theme() -> dict:
    selected = st.session_state.get("theme_name", _DEFAULT_THEME)
    return THEMES.get(selected, THEMES[_DEFAULT_THEME])

def render_theme_selector():
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### Tema de Color")

        temas_oscuros = [
            "TRON Legacy", 
            "Negro & Dorado", 
            "Cyberpunk Night", 
            "Gris Espacial", 
            "Azul Medianoche", 
            "Tema Drácula", 
            "Bosque Luminiscente", 
            "Obsidiana Roja", 
            "Espacio Profundo"
        ]

        temas_claros = [
            "Azul & Rosa (Predeterminado)", 
            "Azul Clasico", 
            "Rojo Ejecutivo", 
            "Verde Esmeralda", 
            "Violeta Profundo", 
            "Teal Minimalista", 
            "Blanco Limpio", 
            "Café Latte Soft", 
            "Océano & Arena"
        ]
        # 2. Selector de categoría (filtro)
        categoria = st.radio(
            "Filtrar por:",
            ["Todos", "Claros", "Oscuros"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # 3. Construimos la lista de opciones basada en el filtro
        todas_opciones = list(THEMES.keys())
        if categoria == "Claros":
            opciones_filtradas = [t for t in todas_opciones if t in temas_claros]
        elif categoria == "Oscuros":
            opciones_filtradas = [t for t in todas_opciones if t in temas_oscuros]
        else:
            opciones_filtradas = todas_opciones

        # 4. Obtenemos el tema actual desde session_state
        tema_actual = st.session_state.get("theme_name", todas_opciones[0])

        # 5. Prevención de errores: si el usuario filtra "Claros" pero tenía un tema oscuro, 
        # seleccionamos el primer elemento de la lista filtrada temporalmente.
        if tema_actual not in opciones_filtradas:
            idx_actual = 0
        else:
            idx_actual = opciones_filtradas.index(tema_actual)

        # 6. Renderizamos el Selectbox final
        tema_elegido = st.selectbox(
            "Elige tu estilo:",
            opciones_filtradas,
            index=idx_actual,
            key="_theme_selector_box", # Cambiamos ligeramente la key
            label_visibility="collapsed",
        )
        
        # 7. Guardamos el resultado en la variable global del estado
        st.session_state["theme_name"] = tema_elegido
        st.caption("El tema se aplica a todos los módulos.")

# =============================================================================
# CSS GLOBAL DINAMICO
# Fuerza todos los colores desde el tema, ignorando completamente el modo
# oscuro/claro del sistema operativo del usuario.
# =============================================================================
def _build_css(c: dict) -> str:
    return f"""
<style>
    /* ══════════════════════════════════════════════════════════
       1. VARIABLES CSS — anulan las de Streamlit (dark/light mode)
       ══════════════════════════════════════════════════════════ */
    :root, [data-theme], [data-theme="dark"], [data-theme="light"] {{
        --background-color:           {c['bg_main']}    !important;
        --secondary-background-color: {c['bg_light']}   !important;
        --text-color:                 {c['text_color']}  !important;
        --primary-color:              {c['accent']}      !important;
    }}

    /* ══════════════════════════════════════════════════════════
       2. HEADER SUPERIOR (barra negra de Streamlit)
       ══════════════════════════════════════════════════════════ */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    header[data-testid="stHeader"],
    .stApp > header {{
        background-color: {c['sidebar_bg']} !important;
        border-bottom: 1px solid {c['border']} !important;
        box-shadow: none !important;
    }}
    /* Iconos del header */
    [data-testid="stHeader"] svg,
    [data-testid="stToolbar"] svg {{
        fill: {c['text_muted']} !important;
        color: {c['text_muted']} !important;
    }}
    /* Decoracion superior (barra de color) */
    [data-testid="stDecoration"] {{
        background: linear-gradient(90deg, {c['accent']}, {c['primary']}) !important;
        height: 3px !important;
    }}

    /* ══════════════════════════════════════════════════════════
       3. FONDO GLOBAL + TEXTO (cubre modo oscuro del SO)
       ══════════════════════════════════════════════════════════ */
    html, body,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMain"] > div,
    [data-testid="stAppViewBlockContainer"],
    .block-container,
    .main,
    section.main {{
        background-color: {c['bg_main']} !important;
        color:            {c['text_color']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       4. SIDEBAR
       ══════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {{
        background-color: {c['sidebar_bg']} !important;
        border-right: 1px solid {c['border']} !important;
    }}
    [data-testid="stSidebar"] *,
    [data-testid="stSidebarContent"] * {{
        color: {c['text_color']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       5. TEXTO GENERAL — fuerza colores del tema
       ══════════════════════════════════════════════════════════ */
    p, li, td, th,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {{
        color: {c['text_color']} !important;
    }}
    /* Headings h3/h4 (ej. "Resultados de la Valuacion") */
    h1, h2, h3, h4, h5, h6 {{
        color: {c['subtitle_color']} !important;
    }}
    /* Captions */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {{
        color: {c['text_muted']} !important;
    }}
    /* Labels de inputs */
    label, [data-testid="stWidgetLabel"] * {{
        color: {c['text_color']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       6. TITULOS PERSONALIZADOS (main-title, section-header)
       ══════════════════════════════════════════════════════════ */
    .main-title {{
        font-size: 36px;
        font-weight: 900;
        color: {c['title_color']} !important;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }}
    .section-header {{
        font-size: 23px;
        font-weight: 700;
        color: {c['subtitle_color']} !important;
        margin-top: 16px;
        padding-bottom: 6px;
        border-bottom: 3px solid {c['accent']};
    }}
    .formula-box {{
        background-color: {c['input_bg']} !important;
        color: {c['text_color']} !important;
        padding: 14px 18px;
        border-radius: 8px;
        border-left: 5px solid {c['accent']};
        margin: 12px 0;
    }}

    /* ══════════════════════════════════════════════════════════
       7. EXPANDERS
       ══════════════════════════════════════════════════════════ */
    [data-testid="stExpander"],
    [data-testid="stExpander"] > details,
    [data-baseweb="accordion"],
    [data-baseweb="accordion"] > div {{
        background-color: {c['bg_light']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details summary,
    [data-baseweb="accordion"] button {{
        background-color: {c['bg_light']} !important;
        color: {c['primary']} !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] details > div,
    [data-testid="stExpander"] > details > div {{
        background-color: {c['bg_light']} !important;
        color: {c['text_color']} !important;
    }}
    [data-testid="stExpander"] svg {{
        fill: {c['primary']} !important;
        stroke: {c['primary']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       8. INPUTS, SELECTS, DATE INPUTS
       ══════════════════════════════════════════════════════════ */
    input, textarea {{
        background-color: {c['input_bg']} !important;
        color: {c['text_color']} !important;
        border-color: {c['border']} !important;
    }}
    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] > div,
    [data-testid="stTextInput"] > div,
    [data-testid="stDateInput"] > div {{
        background-color: {c['input_bg']} !important;
        border-color: {c['border']} !important;
    }}
    /* Botones +/- de number_input */
    [data-testid="stNumberInput"] button {{
        background-color: {c['primary']} !important;
        color: #ffffff !important;
        border-color: {c['primary']} !important;
    }}
    [data-testid="stNumberInput"] button:hover {{
        background-color: {c['accent']} !important;
        border-color: {c['accent']} !important;
    }}
    /* Select / dropdown */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div {{
        background-color: {c['input_bg']} !important;
        color: {c['text_color']} !important;
        border-color: {c['border']} !important;
    }}
    /* Opciones del dropdown */
    [data-baseweb="menu"],
    [data-baseweb="menu"] li,
    [role="listbox"],
    [role="option"] {{
        background-color: {c['bg_light']} !important;
        color: {c['text_color']} !important;
    }}
    [role="option"]:hover {{
        background-color: {c['border']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       9. BOTONES
       ══════════════════════════════════════════════════════════ */
    [data-testid="stButton"] > button,
    [data-testid="stFormSubmitButton"] > button {{
        background-color: {c['primary']} !important;
        color: #ffffff !important;
        border-color: {c['primary']} !important;
        border-radius: 6px !important;
    }}
    [data-testid="stButton"] > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {{
        background-color: {c['accent']} !important;
        border-color: {c['accent']} !important;
    }}
    [data-testid="stLinkButton"] > a {{
        background-color: {c['primary']} !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }}

    /* ══════════════════════════════════════════════════════════
       10. ALERTAS — st.info / st.success / st.warning / st.error
           Cada una usa el par _bg + _border + _text del tema.
       ══════════════════════════════════════════════════════════ */
    /* Wrapper base */
    [data-testid="stAlert"] {{
        border-radius: 8px !important;
        border: none !important;
        border-left-width: 4px !important;
        border-left-style: solid !important;
    }}
    /* INFO */
    [data-testid="stAlert"][data-type="info"],
    div[data-baseweb="notification"][data-type="info"] {{
        background-color: {c['info_bg']} !important;
        border-left-color: {c['info_border']} !important;
    }}
    [data-testid="stAlert"][data-type="info"] *,
    [data-testid="stAlert"][data-type="info"] p,
    [data-testid="stAlert"][data-type="info"] span {{
        color: {c['info_text']} !important;
    }}
    [data-testid="stAlert"][data-type="info"] svg {{
        fill: {c['info_border']} !important;
        color: {c['info_border']} !important;
    }}
    /* SUCCESS */
    [data-testid="stAlert"][data-type="success"],
    div[data-baseweb="notification"][data-type="success"] {{
        background-color: {c['success_bg']} !important;
        border-left-color: {c['success_border']} !important;
    }}
    [data-testid="stAlert"][data-type="success"] *,
    [data-testid="stAlert"][data-type="success"] p,
    [data-testid="stAlert"][data-type="success"] span {{
        color: {c['success_text']} !important;
    }}
    [data-testid="stAlert"][data-type="success"] svg {{
        fill: {c['success_border']} !important;
        color: {c['success_border']} !important;
    }}
    /* WARNING */
    [data-testid="stAlert"][data-type="warning"],
    div[data-baseweb="notification"][data-type="warning"] {{
        background-color: {c['warning_bg']} !important;
        border-left-color: {c['warning_border']} !important;
    }}
    [data-testid="stAlert"][data-type="warning"] *,
    [data-testid="stAlert"][data-type="warning"] p,
    [data-testid="stAlert"][data-type="warning"] span {{
        color: {c['warning_text']} !important;
    }}
    [data-testid="stAlert"][data-type="warning"] svg {{
        fill: {c['warning_border']} !important;
        color: {c['warning_border']} !important;
    }}
    /* ERROR */
    [data-testid="stAlert"][data-type="error"],
    div[data-baseweb="notification"][data-type="error"] {{
        background-color: {c['error_bg']} !important;
        border-left-color: {c['error_border']} !important;
    }}
    [data-testid="stAlert"][data-type="error"] *,
    [data-testid="stAlert"][data-type="error"] p,
    [data-testid="stAlert"][data-type="error"] span {{
        color: {c['error_text']} !important;
    }}
    [data-testid="stAlert"][data-type="error"] svg {{
        fill: {c['error_border']} !important;
        color: {c['error_border']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       11. METRICAS
       ══════════════════════════════════════════════════════════ */
    [data-testid="stMetricValue"] {{
        color: {c['title_color']} !important;
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{
        color: {c['text_muted']} !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: {c['text_muted']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       12. TABS
       ══════════════════════════════════════════════════════════ */
    [data-testid="stTabs"],
    [data-testid="stTabsContent"] {{
        background-color: {c['bg_main']} !important;
    }}
    [data-testid="stTabs"] button {{
        color: {c['text_muted']} !important;
        background-color: transparent !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {c['accent']} !important;
        border-bottom: 2px solid {c['accent']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       13. RADIO, CHECKBOX, SLIDER
       ══════════════════════════════════════════════════════════ */
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {{
        color: {c['primary']} !important;
        font-weight: 600;
    }}

    /* ══════════════════════════════════════════════════════════
       14. TABLAS / DATAFRAME
       NOTA: st.dataframe usa un canvas WebGL (glide-data-grid).
       NO se aplica CSS sobre canvas/iframe/dvn porque taparía el contenido.
       Solo se estiliza el wrapper exterior y st.table (HTML estático).
       ══════════════════════════════════════════════════════════ */
    /* Wrapper exterior del dataframe — solo el borde/fondo del contenedor */
    [data-testid="stDataFrameResizable"],
    [data-testid="stDataFrame"] {{
        border-radius: 8px !important;
        border: 1px solid {c['border']} !important;
        overflow: hidden;
    }}
    /* st.table — tabla HTML estática (sí se puede estilizar con CSS) */
    [data-testid="stTable"] table {{
        background-color: {c['bg_main']} !important;
        color: {c['text_color']} !important;
        border-collapse: collapse;
        width: 100%;
    }}
    [data-testid="stTable"] th {{
        background-color: {c['bg_light']} !important;
        color: {c['subtitle_color']} !important;
        border-bottom: 2px solid {c['accent']} !important;
        padding: 8px 12px;
    }}
    [data-testid="stTable"] td {{
        background-color: {c['bg_main']} !important;
        color: {c['text_color']} !important;
        border-bottom: 1px solid {c['border']} !important;
        padding: 7px 12px;
    }}
    [data-testid="stTable"] tr:nth-child(even) td {{
        background-color: {c['bg_light']} !important;
    }}

    /* ══════════════════════════════════════════════════════════
       15. TARJETAS CALL / PUT
       ══════════════════════════════════════════════════════════ */
    .result-call {{
        background-color: {c['success_bg']} !important;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid {c['success']};
    }}
    .result-call h3 {{
        color: {c['success']} !important;
        margin: 0;
    }}
    .result-put {{
        background-color: {c['error_bg']} !important;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid {c['danger']};
    }}
    .result-put h3 {{
        color: {c['danger']} !important;
        margin: 0;
    }}
    .result-card {{
        background: {c['bg_light']} !important;
        color: {c['text_color']} !important;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid {c['border']};
        margin: 10px 0;
    }}

    /* ══════════════════════════════════════════════════════════
       16. SEPARADORES Y MISCELANEOS
       ══════════════════════════════════════════════════════════ */
    hr {{
        border-color: {c['border']} !important;
    }}
    ::-webkit-scrollbar-track {{
        background: {c['bg_light']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {c['border']};
        border-radius: 4px;
    }}

    /* Ocultar menu deploy y footer */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
"""


def inject_global_css():
    """Inyecta el CSS del tema activo."""
    c = get_current_theme()
    st.markdown(_build_css(c), unsafe_allow_html=True)


# =============================================================================
# CONSTANTES DE DISENO (compatibilidad con codigo existente)
# =============================================================================
COLORS = THEMES[_DEFAULT_THEME]


# =============================================================================
# HELPERS DE UI
# =============================================================================

def page_header(titulo: str, subtitulo: str = ""):
    inject_global_css()
    render_theme_selector()
    st.markdown(
        '<div class="main-title">Calculadora Financiera y Actuarial</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-header">{titulo}</div>',
        unsafe_allow_html=True,
    )
    if subtitulo:
        st.caption(subtitulo)
    st.write("---")


def result_call(label: str, valor: float, decimales: int = 4):
    st.markdown(
        f"<div class='result-call'><h3>{label}: ${valor:,.{decimales}f}</h3></div>",
        unsafe_allow_html=True,
    )


def result_put(label: str, valor: float, decimales: int = 4):
    st.markdown(
        f"<div class='result-put'><h3>{label}: ${valor:,.{decimales}f}</h3></div>",
        unsafe_allow_html=True,
    )


def resultado_metrica(label: str, valor: str, ayuda: str = ""):
    st.metric(label=label, value=valor, help=ayuda if ayuda else None)


def paso_a_paso(titulo: str = "Ver desarrollo paso a paso"):
    return st.expander(titulo)


def alerta_metodo_numerico():
    st.warning(
        "**Aviso: Calculo mediante Metodos Numericos (Iteracion)**  \n"
        "La variable buscada no puede despejarse algebraicamente. \n"
        "Se ha implementado un metodo de iteracion para aproximar el resultado. \n"
        "Medainte newton-raphson o biseccion, dependiendo del caso.  \n"
    )


def separador():
    st.write("---")


# =============================================================================
# HELPERS TEMATICOS
# Reemplazan st.info / st.success / st.warning con HTML puro que siempre
# respeta el tema activo sin depender de los selectores de Streamlit.
# =============================================================================

def _bold_to_html(texto: str) -> str:
    """Convierte **negrita** a <strong>negrita</strong>."""
    import re
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', texto)


def themed_info(texto: str):
    """Equivalente tematico de st.info() — usa colores del tema."""
    c = get_current_theme()
    html = _bold_to_html(texto).replace("\n", "<br>")
    st.markdown(
        f"<div style='"
        f"background:{c['info_bg']};color:{c['info_text']};"
        f"border-left:4px solid {c['info_border']};border-radius:8px;"
        f"padding:14px 18px;margin:6px 0;line-height:1.6;"
        f"'>{html}</div>",
        unsafe_allow_html=True,
    )


def themed_success(texto: str):
    """Equivalente tematico de st.success() — usa colores del tema."""
    c = get_current_theme()
    html = _bold_to_html(texto).replace("\n", "<br>")
    st.markdown(
        f"<div style='"
        f"background:{c['success_bg']};color:{c['success_text']};"
        f"border-left:4px solid {c['success_border']};border-radius:8px;"
        f"padding:14px 18px;margin:6px 0;line-height:1.6;"
        f"'>{html}</div>",
        unsafe_allow_html=True,
    )


def themed_warning(texto: str):
    """Equivalente tematico de st.warning() — usa colores del tema."""
    c = get_current_theme()
    html = _bold_to_html(texto).replace("\n", "<br>")
    st.markdown(
        f"<div style='"
        f"background:{c['warning_bg']};color:{c['warning_text']};"
        f"border-left:4px solid {c['warning_border']};border-radius:8px;"
        f"padding:14px 18px;margin:6px 0;line-height:1.6;"
        f"'>{html}</div>",
        unsafe_allow_html=True,
    )


def themed_error(texto: str):
    """Equivalente tematico de st.error() — usa colores del tema."""
    c = get_current_theme()
    html = _bold_to_html(texto).replace("\n", "<br>")
    st.markdown(
        f"<div style='"
        f"background:{c['error_bg']};color:{c['error_text']};"
        f"border-left:4px solid {c['error_border']};border-radius:8px;"
        f"padding:14px 18px;margin:6px 0;line-height:1.6;"
        f"'>{html}</div>",
        unsafe_allow_html=True,
    )


def index_card(numero: str, titulo: str, descripcion: str, variante: str = "a"):
    """
    Tarjeta del indice principal de la app. Usa colores del tema activo.
    variante:
      'a' -> primary  (col 1: fundamentos)
      'b' -> success  (col 2: valuacion)
      'c' -> accent   (col 3: derivados)
    """
    c = get_current_theme()
    colores = {
        "a": (c["info_bg"],    c["primary"],  c["info_text"]),
        "b": (c["success_bg"], c["success"],  c["success_text"]),
        "c": (c["warning_bg"], c["accent"],   c["warning_text"]),
    }
    bg, borde, texto = colores.get(variante, colores["a"])
    st.markdown(
        f"<div style='"
        f"background:{bg};color:{texto};"
        f"border-left:5px solid {borde};border-radius:10px;"
        f"padding:16px 20px;margin-bottom:12px;min-height:85px;"
        f"'>"
        f"<div style='font-weight:700;font-size:15px;color:{borde};"
        f"margin-bottom:5px;'>{numero}. {titulo}</div>"
        f"<div style='font-size:14px;line-height:1.5;'>{descripcion}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# HELPER DE PLOTLY — layout tematico
# Usar asi:  fig.update_layout(**plotly_layout())
# O pasar directamente a px/go:  layout=plotly_layout()
# =============================================================================

def plotly_layout(**extra) -> dict:
    """
    Devuelve un dict de layout para Plotly que usa los colores del tema activo.
    Combina con kwargs extra: fig.update_layout(**plotly_layout(height=500))
    """
    c = get_current_theme()
    base = dict(
        paper_bgcolor = c["bg_main"],
        plot_bgcolor  = c["bg_light"],
        font          = dict(color=c["text_color"], family="Source Sans Pro, sans-serif"),
        title_font    = dict(color=c["subtitle_color"]),
        legend        = dict(
            bgcolor     = c["bg_light"],
            bordercolor = c["border"],
            borderwidth = 1,
            font        = dict(color=c["text_color"]),
        ),
        xaxis = dict(
            gridcolor    = c["border"],
            linecolor    = c["border"],
            tickfont     = dict(color=c["text_muted"]),
            title_font   = dict(color=c["text_color"]),
            zerolinecolor= c["border"],
        ),
        yaxis = dict(
            gridcolor    = c["border"],
            linecolor    = c["border"],
            tickfont     = dict(color=c["text_muted"]),
            title_font   = dict(color=c["text_color"]),
            zerolinecolor= c["border"],
        ),
        margin = dict(l=40, r=20, t=50, b=40),
    )
    base.update(extra)
    return base


def apply_plotly_theme(fig, **extra):
    """
    Aplica el tema activo a una figura Plotly in-place y la devuelve.
    Uso: fig = apply_plotly_theme(fig, height=500)
    """
    fig.update_layout(**plotly_layout(**extra))
    # Quitar el template que sobreescribiria nuestros colores
    fig.update_layout(template="none")
    return fig


# =============================================================================
# HELPER PLOTLY TEMATICO
# Devuelve un dict de layout y colores para usar en todas las graficas plotly,
# garantizando que siempre respondan al tema activo.
# =============================================================================

def plotly_theme() -> dict:
    """
    Devuelve un dict con layout base para fig.update_layout(**plotly_theme()).
    Sustituye template='plotly_white' con colores del tema activo.

    Uso:
        fig.update_layout(**plotly_theme())
    """
    c = get_current_theme()
    return dict(
        paper_bgcolor=c["bg_main"],
        plot_bgcolor=c["bg_light"],
        font=dict(color=c["text_color"], family="Source Sans Pro, sans-serif"),
        title_font=dict(color=c["subtitle_color"]),
        legend=dict(
            bgcolor=c["bg_light"],
            bordercolor=c["border"],
            borderwidth=1,
            font=dict(color=c["text_color"]),
        ),
        xaxis=dict(
            gridcolor=c["border"],
            linecolor=c["border"],
            tickcolor=c["border"],
            tickfont=dict(color=c["text_muted"]),
            title_font=dict(color=c["text_color"]),
            zerolinecolor=c["border"],
        ),
        yaxis=dict(
            gridcolor=c["border"],
            linecolor=c["border"],
            tickcolor=c["border"],
            tickfont=dict(color=c["text_muted"]),
            title_font=dict(color=c["text_color"]),
            zerolinecolor=c["border"],
        ),
        colorway=[
            c["primary"], c["accent"], c["success"],
            c["secondary"], c["warning"], c["danger"],
            "#8B5CF6", "#06B6D4", "#F59E0B",
        ],
    )


def plotly_colors() -> list:
    """Lista de colores del tema para usar en color_discrete_sequence."""
    c = get_current_theme()
    return [
        c["primary"], c["accent"], c["success"],
        c["secondary"], c["warning"], c["danger"],
        "#8B5CF6", "#06B6D4", "#F59E0B", "#EC4899",
    ]


def df_style(df):
    """
    Aplica estilos tematicos a un DataFrame de pandas para st.dataframe().
    Uso: st.dataframe(df_style(df), ...)
    """
    c = get_current_theme()
    return (
        df.style
        .set_properties(**{
            "background-color": c["bg_light"],
            "color":            c["text_color"],
            "border-color":     c["border"],
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", c["primary"]),
                ("color",            "#ffffff"),
                ("font-weight",      "700"),
                ("border-color",     c["border"]),
            ]},
            {"selector": "tr:hover td", "props": [
                ("background-color", c["border"]),
            ]},
        ])
    )
