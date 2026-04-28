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
            themed_success(f"<h3 style='margin:0; color:inherit;'>Precio Teórico (P₀): ${precio_gs:,.4f}</h3>")
            st.latex(r"P_0 = \frac{D_1}{k - g}")

            separador()

            with paso_a_paso():
                st.latex(r"P_0 = \frac{D_1}{k - g}")
                st.latex(rf"P_0 = \frac{{{d1_gs:,.2f}}}{{{k_gs:.4f} - {g_gs:.4f}}}")
                st.latex(rf"P_0 = \frac{{{d1_gs:,.2f}}}{{{k_gs - g_gs:.4f}}}")
                themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>P_0 = ${precio_gs:,.4f}</h4>")

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
        themed_info(f"<h3 style='margin:0; color:inherit;'>Rendimiento Requerido (k): {k_calc*100:.4f}%</h3>")
        st.latex(r"k = \frac{D_1}{P_0} + g")

        separador()

        with paso_a_paso():
            div_yield = d1_rr / p0_rr
            st.latex(r"k = \frac{D_1}{P_0} + g")
            st.latex(rf"k = \frac{{{d1_rr:,.2f}}}{{{p0_rr:,.2f}}} + {g_rr:.4f}")
            st.latex(rf"k = {div_yield:.6f} + {g_rr:.4f}")
            themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>k = {k_calc*100:.4f}\%</h4>")

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
            "Valor de la Empresa (EV)"
            if cfg["es_ev"]
            else "Precio Estimado (P₀)"
        )
        
        themed_success(f"<h3 style='margin:0; color:inherit;'>{titulo_res}: ${resultado_mul:,.4f}</h3>")
        st.latex(cfg["formula"])

        separador()

        with paso_a_paso():
            st.latex(cfg["formula"])
            st.latex(rf"{cfg['var_nombre']} = {val_metrica:,.2f} \times {val_multiplo:,.2f}")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>{cfg['var_nombre']} = ${resultado_mul:,.4f}</h4>")

    # ── Tabla comparativa de múltiplos ────────────────────────────────────────
    separador()
    with st.expander("Referencia: Rangos típicos de múltiplos por sector"):
        st.markdown("""
| Sector            | P/E típico | P/S típico | EV/EBITDA típico | P/B típico |
| :---------------- | :--------: | :--------: | :--------------: | :--------: |
| Tecnología        | 25 – 50x   | 5 – 15x    | 20 – 40x         | 5 – 15x    |
| Banca / Finanzas  | 10 – 15x   | 2 – 4x     | 8 – 12x          | 1 – 2x     |
| Consumo básico    | 15 – 25x   | 1 – 3x     | 10 – 16x         | 3 – 6x     |
| Energía           | 8 – 15x    | 0.5 – 2x   | 5 – 10x          | 1 – 2x     |
| Salud             | 20 – 35x   | 3 – 8x     | 12 – 20x         | 3 – 7x     |
| Industrial        | 15 – 25x   | 1 – 3x     | 8 – 14x          | 2 – 4x     |

>  *Estos rangos son orientativos y cambian constantemente. Siempre es necesario comparar contra el promedio específico del sector al momento de evaluar.*
        """)