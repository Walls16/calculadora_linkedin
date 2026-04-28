"""
app.py
------
Punto de entrada de la aplicación.
Define la navegación agrupada por secciones y la Portada.
La navegación se maneja con st.navigation() para mostrar
secciones colapsables en el sidebar.
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

# Inyectar CSS y selector de tema UNA VEZ aquí para todas las páginas
inject_global_css()
render_theme_selector()

# Marcamos que el sidebar ya fue configurado para evitar que
# page_header() en cada página lo renderice de nuevo (duplicado)
st.session_state["_nav_setup_done"] = True


# =============================================================================
# PORTADA — función que se usa como página por defecto
# =============================================================================
def portada():
    inject_global_css()  # Re-inyectar CSS por si el tema cambia
    c = get_current_theme()

    # ── Título ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<h1 style='text-align:center;color:{c['title_color']};"
        f"font-size:42px;font-weight:800;margin-bottom:4px;'>"
        f"Calculadora Financiera y Actuarial</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:{c['text_muted']};font-size:16px;'>"
        f"Herramienta interactiva de matemáticas financieras"
        f"</p>",
        unsafe_allow_html=True,
    )
    st.write("---")

    # ── Créditos ──────────────────────────────────────────────────────────────
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

    # ── Descripción ───────────────────────────────────────────────────────────
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

    # ── Mapa de la calculadora ────────────────────────────────────────────────
    st.markdown(
        f"<h3 style='color:{c['subtitle_color']};font-weight:700;'>Mapa de la Calculadora</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Usa el menú lateral para navegar entre módulos.")

    def seccion_label(icono: str, texto: str, color: str):
        st.markdown(
            f"<p style='font-weight:700;color:{color};font-size:12px;"
            f"letter-spacing:0.8px;text-transform:uppercase;"
            f"margin:14px 0 6px 0;border-bottom:2px solid {color};"
            f"padding-bottom:4px;'>{icono} {texto}</p>",
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        seccion_label("","Matemáticas Financieras", c["primary"])
        index_card("1", "Tasas de Interés",
                   "Conversión entre tasas nominales, efectivas e instantáneas.", "a")
        index_card("2", "Valor del Dinero",
                   "VP, VF, tasa de rendimiento y número de periodos.", "a")
        index_card("3", "Rentas y Anualidades",
                   "Constantes, geométricas y aritméticas (vencidas, anticipadas, continuas).", "a")
        index_card("4", "Tabla de Amortización",
                   "Tablas dinámicas para créditos con enganche y fondos de amortización.", "a")

    with col2:
        seccion_label("","Valuación de Activos y Riesgo", c["success"])
        index_card("5",  "Valuación de Bonos",
                   "Precio limpio, precio sucio y Yield to Maturity (YTM).", "b")
        index_card("6",  "Valuación de Acciones",
                   "Gordon-Shapiro y valuación relativa por múltiplos de mercado.", "b")
        index_card("7",  "Portafolios Eficientes",
                   "Optimización de Markowitz con datos reales de Yahoo Finance.", "b")
        index_card("8",  "Valuación Corporativa",
                   "DCF, WACC y valuación de empresas por flujos descontados.", "b")
        index_card("9",  "Riesgo de Portafolios",
                   "VaR y CVaR paramétrico y Monte Carlo.", "b")

    with col3:
        seccion_label("","Derivados Financieros", c["accent"])
        index_card("10", "Forwards",
                   "Precio teórico y valuación de contratos forward.", "c")
        index_card("11", "Derivados Vanilla",
                   "Primas y Griegas con BSM y Árbol Binomial CRR.", "c")
        index_card("12", "Derivados Exóticos",
                   "Barrera, Asiáticas, Lookback, Compuestas e Intercambio.", "c")

        seccion_label("","Referencia", c["text_muted"])
        index_card("13", "Formulario",
                   "Cheat-sheet descargable en HTML con todas las ecuaciones.", "a")

    st.write("---")

    # ── Pie de página ─────────────────────────────────────────────────────────
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


# =============================================================================
# NAVEGACIÓN AGRUPADA
# Streamlit 1.36+ permite pasar un dict {sección: [st.Page(...)]}
# para generar secciones colapsables en el sidebar automáticamente.
# =============================================================================
pg = st.navigation(
    {
        "Inicio": [
            st.Page(portada, title="Portada", default=True),
        ],
        "Matemáticas Financieras": [
            st.Page("pages/1_Tasas.py",        title="Tasas de Interés"         ),
            st.Page("pages/2_Valor_Dinero.py",  title="Valor del Dinero"        ),
            st.Page("pages/3_Rentas.py",        title="Rentas y Anualidades"    ),
            st.Page("pages/4_Amortizacion.py",  title="Tabla de Amortización"   ),
        ],
        "Valuación de Activos y Riesgo": [
            st.Page("pages/5_Bonos.py",                 title="Bonos"                   ),
            st.Page("pages/6_Acciones.py",              title="Acciones"                ),
            st.Page("pages/7_Portafolios.py",           title="Portafolios"             ),
            st.Page("pages/8_Valuacion_Corporativa.py", title="Valuación Corporativa"   ),
            st.Page("pages/9_Riesgo.py",                title="Riesgo"                  ),
        ],
        "Derivados Financieros": [
            st.Page("pages/10_Forwards.py",           title="Forwards"              ),
            st.Page("pages/11_Derivados_Vanilla.py",  title="Derivados Vanilla"     ),
            st.Page("pages/12_Derivados_Exoticos.py", title="Derivados Exóticos"    ),
            #st.Page("pages/13_Derivados_Credito.py",   title="Derivados de Crédito"  ),
        ],
        "Referencia": [
            st.Page("pages/13_Formulario.py", title="Formulario"    ),
        ],
    },
    position="sidebar",
)

pg.run()

# =============================================================================
# El proyecto de esta calculadora se trabajo de la mano con el Dr. Francisco García Castillo
# la intención siempre fue ayudar a todos los estudiantes a entender y aplicar los conceptos de matemáticas financieras, valuación de activos, riesgo y derivados de una manera práctica e interactiva.
# La calculadora se diseñó para ser una herramienta de apoyo académico, con explicaciones claras, visualizaciones y ejemplos que faciliten el aprendizaje de estos temas complejos.
# A lo largo del desarrollo, se priorizó la precisión de los cálculos y la claridad de la interfaz, con el objetivo de que los estudiantes puedan experimentar y comprender mejor los conceptos financieros
# Agradezco al Dr. García Castillo por su guía y apoyo durante todo el proceso, y espero que esta calculadora sea de gran utilidad para los estudiantes en su aprendizaje de las matemáticas financieras y la valuación de activos.
# Agradezco a mis padres por su apoyo incondicional durante el desarrollo de este proyecto, y a mis amigos por su paciencia y comprensión mientras me sumergía en el código y los cálculos financieros.
# Agradezco a mi novia por su apoyo durante el desarrollo de esta calculadora, por su comprensión y ánimo mientras me dedicaba a este proyecto académico. Su apoyo ha sido fundamental para mantenerme motivado y enfocado en la creación de esta herramienta que espero sea de gran ayuda para los estudiantes.
# Si alguien llega a ver esto, gracias por tu interés en el proyecto. Espero que la calculadora sea útil para tu aprendizaje y te ayude a entender mejor las matemáticas financieras y la valuación de activos. Si tienes alguna pregunta o sugerencia, no dudes en contactarme. ¡Mucho éxito en tus estudios!
