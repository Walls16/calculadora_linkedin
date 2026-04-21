"""
app.py
------
Punto de entrada de la aplicación.
Solo contiene la Portada y el Índice visual.
La navegación real la maneja Streamlit mediante la carpeta /pages.
"""

import streamlit as st
from utils import (
    inject_global_css, render_theme_selector,
    get_current_theme, index_card, COLORS,
)

# =============================================================================
# CONFIGURACIÓN GLOBAL DE LA APP
# =============================================================================
st.set_page_config(
    page_title="Calculadora Financiera y Actuarial",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
render_theme_selector()

# Colores del tema activo para inline styles
c = get_current_theme()

# =============================================================================
# PORTADA
# =============================================================================
st.markdown(
    f"<h1 style='text-align:center;color:{c['title_color']};"
    f"font-size:42px;font-weight:800;margin-bottom:4px;'>"
    f"Calculadora Financiera y Actuarial</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:{c['text_muted']};font-size:16px;'>"
    f"Herramienta interactiva de matemáticas financieras · Python + Streamlit"
    f"</p>",
    unsafe_allow_html=True,
)

st.write("---")

# =============================================================================
# CRÉDITOS
# =============================================================================
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"<h4 style='color:{c['subtitle_color']};'>Desarrollado por:</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "- **[Owen Paredes Conde](https://www.linkedin.com/in/owen-conde-a731b9249/)**"
    )
with c2:
    st.markdown(
        f"<h4 style='color:{c['subtitle_color']};'>Dirigido por:</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "- **[Dr. Francisco García Castillo]"
        "(https://www.linkedin.com/in/dr-francisco-garcía-castillo/)**"
    )

st.write("---")

# =============================================================================
# DESCRIPCIÓN
# =============================================================================
st.markdown(
    f"<p style='color:{c['text_color']};font-size:15px;line-height:1.7;'>"
    f"Esta herramienta fue desarrollada en Python "
    f"para automatizar y visualizar los cálculos más rigurosos de las matemáticas "
    f"financieras: desde el valor del dinero en el tiempo hasta la valuación de "
    f"derivados con Black-Scholes-Merton y Árboles Binomiales CRR."
    f"</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='background:{c['warning_bg']};color:{c['warning_text']};"
    f"border-left:4px solid {c['warning_border']};border-radius:8px;"
    f"padding:10px 16px;font-size:14px;margin-top:8px;'>"
    f"Esta herramienta tiene fines netamente académicos y de demostración."
    f"</div>",
    unsafe_allow_html=True,
)

st.write("---")

# =============================================================================
# ÍNDICE VISUAL (3 columnas con tarjetas temáticas)
# =============================================================================
st.markdown(
    f"<h3 style='color:{c['subtitle_color']};font-weight:700;'>Mapa de la Calculadora</h3>",
    unsafe_allow_html=True,
)
st.caption("Usa el menú lateral para navegar entre módulos.")

col1, col2, col3 = st.columns(3)

with col1:
    index_card("1", "Tasas de Interés",
               "Conversión entre tasas nominales, efectivas e instantáneas.", "a")
    index_card("2", "Valor del Dinero",
               "VP, VF, tasa de rendimiento y número de periodos.", "a")
    index_card("3", "Rentas y Anualidades",
               "Constantes, geométricas y aritméticas (vencidas, anticipadas, continuas).", "a")
    index_card("4", "Tabla de Amortización",
               "Tablas dinámicas para créditos con enganche y fondos de amortización.", "a")

with col2:
    index_card("5", "Valuación de Bonos",
               "Precio limpio, precio sucio y Yield to Maturity (YTM).", "b")
    index_card("6", "Valuación de Acciones",
               "Gordon-Shapiro y valuación relativa por múltiplos de mercado.", "b")
    index_card("7", "Portafolios Eficientes",
               "Optimización de Markowitz con datos reales de Yahoo Finance.", "b")
    index_card("8", "Riesgo de Portafolios",
               "VaR y CVaR paramétrico y Monte Carlo.", "b")

with col3:
    index_card("9",  "Forwards",
               "Precio teórico y valuación de contratos forward.", "c")
    index_card("10", "Derivados Vanilla",
               "Primas y Griegas con BSM y Árbol Binomial CRR.", "c")
    index_card("11", "Derivados Exóticos",
               "Barrera, Asiáticas, Lookback, Compuestas e Intercambio.", "c")
    index_card("12", "Formulario",
               "Cheat-sheet descargable en HTML con todas las ecuaciones.", "c")

st.write("---")

# =============================================================================
# PIE DE PÁGINA
# =============================================================================
c_foot1, c_foot2 = st.columns(2)

with c_foot1:
    st.markdown(
        f"<h4 style='color:{c['subtitle_color']};'>Código Abierto</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{c['text_color']};'>¿Experimentas lentitud? "
        f"Descarga el código y ejecútalo localmente.</p>",
        unsafe_allow_html=True,
    )
    st.link_button(
        "Descargar desde GitHub",
        "https://github.com/Walls16/calculadora-actuarial",
        use_container_width=True,
    )

with c_foot2:
    st.markdown(
        f"<h4 style='color:{c['subtitle_color']};'>Contacto y Feedback</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{c['text_color']};'>¿Tienes comentarios, dudas o sugerencias?</p>",
        unsafe_allow_html=True,
    )
    st.link_button(
        "Conectar en LinkedIn",
        "https://www.linkedin.com/in/owen-conde-a731b9249/",
        use_container_width=True,
    )
