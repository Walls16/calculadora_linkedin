"""
pages/6_Acciones.py
-------------------
Módulo 6: Valuación de Acciones.
Cubre:
  - Modelo de Gordon-Shapiro (precio y rendimiento requerido)
  - Valuación relativa por múltiplos: P/E, P/S, EV/EBITDA, P/B
"""

import streamlit as st

from utils import get_engine, page_header, paso_a_paso, separador, themed_info, themed_success, themed_warning, themed_error

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Valuación de Acciones · Calculadora Financiera",
    page_icon="📈",
    layout="wide",
)

engine = get_engine()

# --- Estilos globales para métricas destacadas ---
math_style = "font-family: 'Times New Roman', Times, serif; font-style: italic; font-weight: normal; padding: 0 2px;"
css_titulo = "font-size: 20px; opacity: 0.85; font-weight: 500;"
css_valor = "font-size: 28px; font-weight: bold;"
css_contenedor = "display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 12px 0;"
css_paso = "text-align: center; font-size: 22px; font-weight: bold; padding: 4px 0; margin: 0;"

page_header(
    titulo="6. Valuación de Acciones",
    subtitulo="Modelo de Gordon-Shapiro · Rendimiento requerido · Valuación relativa por múltiplos"
)

# =============================================================================
# PESTAÑAS
# =============================================================================
tab_gordon, tab_rendimiento, tab_multiplos = st.tabs([
    "Gordon-Shapiro (D₁, k, g)",
    "Rendimiento Requerido (k)",
    "Valuación por Múltiplos",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — GORDON-SHAPIRO: PRECIO
# ─────────────────────────────────────────────────────────────────────────────
with tab_gordon:
    st.markdown("### Modelo de Crecimiento Constante de Dividendos")
    themed_success(
        "El **Modelo de Gordon-Shapiro** calcula el precio justo de una acción asumiendo que sus dividendos crecerán a un ritmo constante "
        "(<span style='font-family: serif; font-style: italic;'>g</span>) para siempre. Es muy útil para valuar empresas maduras y estables que pagan dividendos regularmente. <br><br>"
    )

    c1, c2 = st.columns(2)

    with c1:
        d1_gs = st.number_input(
            "Dividendo esperado el próximo año ($D_1$)",
            min_value=0.01, value=5.0, step=0.5, key="gs_d1"
        )
        k_gs  = st.number_input(
            "Tasa de rendimiento requerida ($k$) %",
            value=12.0, step=0.1, key="gs_k"
        ) / 100
        g_gs  = st.number_input(
            "Tasa de crecimiento constante ($g$) %",
            value=4.0, step=0.1, key="gs_g"
        ) / 100

    with c2:
        if k_gs <= g_gs:
            themed_error(
                "El modelo **no converge** cuando <span style='font-family: serif; font-style: italic;'>k ≤ g</span>. "
                "La tasa de rendimiento debe superar obligatoriamente a la de crecimiento."
            )
        else:
            precio_gs = engine.valuacion_gordon_shapiro(d1_gs, k_gs, g_gs)
            
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Precio Teórico (<span style='{math_style}'>P<sub>0</sub></span>)</span>"
                f"<span style='{css_valor}'>${precio_gs:,.4f}</span>"
                f"</div>"
            )
            st.latex(r"P_0 = \frac{D_1}{k - g}")

            separador()

            with paso_a_paso():
                st.latex(r"P_0 = \frac{D_1}{k - g}")
                st.latex(rf"P_0 = \frac{{{d1_gs:,.2f}}}{{{k_gs:.4f} - {g_gs:.4f}}}")
                st.latex(rf"P_0 = \frac{{{d1_gs:,.2f}}}{{{k_gs - g_gs:.4f}}}")
                themed_success(f"<div style='{css_paso}'><span style='{math_style}'>P<sub>0</sub></span> = ${precio_gs:,.4f}</div>")

            # ── Interpretación automática ──────────────────────────────────
            separador()
            st.markdown("#### Desglose de Retornos")
            col_i1, col_i2, col_i3 = st.columns(3)
            col_i1.metric("Dividendo Yield ($D_1/P_0$)", f"{(d1_gs / precio_gs)*100:.2f}%")
            col_i2.metric("Tasa de crecimiento ($g$)", f"{g_gs*100:.2f}%")
            col_i3.metric("Rendimiento total implícito ($k$)", f"{k_gs*100:.2f}%")

            st.caption(
                "El rendimiento total que exige el inversionista ($k$) se compone "
                "del dividendo yield más la apreciación esperada del precio ($g$): "
                f"$k = D_1/P_0 + g = {(d1_gs/precio_gs)*100:.2f}\\% + {g_gs*100:.2f}\\%$"
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RENDIMIENTO REQUERIDO
# ─────────────────────────────────────────────────────────────────────────────
with tab_rendimiento:
    st.markdown("### Cálculo del Rendimiento Requerido ($k$)")
    themed_info(
        "El **Rendimiento Requerido (<span style='font-family: serif; font-style: italic;'>k</span>)** te dice qué porcentaje de ganancia anual "
        "está esperando el mercado de esta acción, basándose en su precio actual y sus dividendos. <br><br>"
        "Esta métrica es clave para saber si la inversión cumple con tus expectativas personales de ganancia frente al riesgo que estás tomando."
    )

    c1, c2 = st.columns(2)

    with c1:
        p0_rr = st.number_input(
            "Precio actual de la acción ($P_0$)",
            min_value=0.01, value=150.0, step=5.0, key="rr_p0"
        )
        d1_rr = st.number_input(
            "Dividendo esperado ($D_1$)",
            min_value=0.01, value=7.5, step=0.5, key="rr_d1"
        )
        g_rr  = st.number_input(
            "Tasa de crecimiento ($g$) %",
            value=5.0, step=0.1, key="rr_g"
        ) / 100

    with c2:
        k_calc = engine.rendimiento_requerido_accion(d1_rr, p0_rr, g_rr)
        
        themed_info(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>Rendimiento Requerido (<span style='{math_style}'>k</span>)</span>"
            f"<span style='{css_valor}'>{k_calc*100:.4f}%</span>"
            f"</div>"
        )
        st.latex(r"k = \frac{D_1}{P_0} + g")

        separador()

        with paso_a_paso():
            div_yield = d1_rr / p0_rr
            st.latex(r"k = \frac{D_1}{P_0} + g")
            st.latex(rf"k = \frac{{{d1_rr:,.2f}}}{{{p0_rr:,.2f}}} + {g_rr:.4f}")
            st.latex(rf"k = {div_yield:.6f} + {g_rr:.4f}")
            themed_info(f"<div style='{css_paso}'><span style='{math_style}'>k</span> = {k_calc*100:.4f}%</div>")

        separador()

        # Desglose del rendimiento
        st.markdown("#### Desglose del rendimiento")
        col_d1, col_d2 = st.columns(2)
        col_d1.metric("Componente Dividendo ($D_1/P_0$)",
                      f"{(d1_rr/p0_rr)*100:.2f}%",
                      help="Rendimiento por cobro de dividendos directamente en efectivo.")
        col_d2.metric("Componente Crecimiento ($g$)",
                      f"{g_rr*100:.2f}%",
                      help="Rendimiento por la apreciación esperada del precio de la acción.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MÚLTIPLOS
# ─────────────────────────────────────────────────────────────────────────────
with tab_multiplos:
    st.markdown("### Valuación Relativa por Múltiplos de Mercado")
    themed_success(
        "La **Valuación por Múltiplos** es una forma rápida de estimar cuánto debería valer una empresa comparándola con otras similares. <br><br>"
        "Por ejemplo, si sabes que en promedio las empresas de tu sector se venden a 15 veces sus ganancias, "
        "puedes usar esa misma regla matemática para calcular el precio justo de la acción que estás analizando."
    )

    # Configuración del múltiplo
    metodo = st.selectbox(
        "Selecciona el múltiplo a utilizar para la comparación:",
        [
            "Precio / Utilidad (P/E Ratio)",
            "Precio / Ventas (P/S Ratio)",
            "EV / EBITDA",
            "Precio / Valor en Libros (P/B Ratio)",
        ],
        key="sel_multiplo",
    )
    separador()

    # Configuración dinámica según múltiplo
    MULTIPLOS_CONFIG = {
        "Precio / Utilidad (P/E Ratio)": {
            "label_metrica":  "Utilidad por Acción (UPA / EPS) $",
            "label_multiplo": "Múltiplo P/E objetivo",
            "formula":        r"P_0 = \text{UPA} \times \left(\frac{P}{E}\right)",
            "var_nombre":     "P_0",
            "es_ev":          False,
            "default_met":    10.0,
            "default_mul":    15.0,
        },
        "Precio / Ventas (P/S Ratio)": {
            "label_metrica":  "Ventas por Acción (VPA) $",
            "label_multiplo": "Múltiplo P/S objetivo",
            "formula":        r"P_0 = \text{VPA} \times \left(\frac{P}{S}\right)",
            "var_nombre":     "P_0",
            "es_ev":          False,
            "default_met":    8.0,
            "default_mul":    3.0,
        },
        "EV / EBITDA": {
            "label_metrica":  "EBITDA por Acción $",
            "label_multiplo": "Múltiplo EV/EBITDA objetivo",
            "formula":        r"EV = \text{EBITDA} \times \left(\frac{EV}{\text{EBITDA}}\right)",
            "var_nombre":     "EV",
            "es_ev":          True,
            "default_met":    5.0,
            "default_mul":    10.0,
        },
        "Precio / Valor en Libros (P/B Ratio)": {
            "label_metrica":  "Valor en Libros por Acción (VLA) $",
            "label_multiplo": "Múltiplo P/B objetivo",
            "formula":        r"P_0 = \text{VLA} \times \left(\frac{P}{B}\right)",
            "var_nombre":     "P_0",
            "es_ev":          False,
            "default_met":    12.0,
            "default_mul":    2.5,
        },
    }

    cfg = MULTIPLOS_CONFIG[metodo]

    c1, c2 = st.columns(2)

    with c1:
        val_metrica  = st.number_input(
            cfg["label_metrica"],
            min_value=0.01,
            value=cfg["default_met"],
            step=1.0,
            key="val_met",
        )
        val_multiplo = st.number_input(
            cfg["label_multiplo"],
            min_value=0.1,
            value=cfg["default_mul"],
            step=0.5,
            key="val_mul",
        )

    with c2:
        resultado_mul = engine.valuacion_multiplos(val_metrica, val_multiplo)

        titulo_res = (
            f"Valor de la Empresa (<span style='{math_style}'>EV</span>)"
            if cfg["es_ev"]
            else f"Precio Estimado (<span style='{math_style}'>P<sub>0</sub></span>)"
        )
        
        var_display = (
            f"<span style='{math_style}'>EV</span>"
            if cfg["es_ev"]
            else f"<span style='{math_style}'>P<sub>0</sub></span>"
        )
        
        themed_success(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>{titulo_res}</span>"
            f"<span style='{css_valor}'>${resultado_mul:,.4f}</span>"
            f"</div>"
        )
        st.latex(cfg["formula"])

        separador()

        with paso_a_paso():
            st.latex(cfg["formula"])
            st.latex(rf"{cfg['var_nombre']} = {val_metrica:,.2f} \times {val_multiplo:,.2f}")
            themed_success(f"<div style='{css_paso}'>{var_display} = ${resultado_mul:,.4f}</div>")

