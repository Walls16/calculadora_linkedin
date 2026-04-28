"""
pages/5_Bonos.py
----------------
Módulo 5: Valuación de Bonos y Obligaciones.
Cubre:
  - Precio del bono dado su tasa de rendimiento (YTM → P)
  - Tasa de rendimiento al vencimiento dado su precio (P → YTM)
  - Análisis de riesgo: Duración Macaulay, Duración Modificada, Convexidad
  - Simulador de estrés de tasas de interés
"""

import numpy as np
import streamlit as st

from utils import get_engine, page_header, paso_a_paso, separador, alerta_metodo_numerico, themed_info, themed_success, themed_warning, themed_error

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Valuación de Bonos · Calculadora Financiera",
    page_icon="📄",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="5. Valuación de Bonos y Obligaciones",
    subtitulo="Precio limpio · Yield to Maturity (YTM) · Duración · Convexidad · Simulador de estrés"
)

# =============================================================================
# SELECTOR DE MODO
# =============================================================================
st.markdown("### ¿Qué deseas calcular?")
themed_info(
    "Un bono es un instrumento de deuda. Su valuación depende de la relación inversa entre su **Precio (<span style='font-family: serif; font-style: italic;'>P</span>)** "
    "y su **Tasa de Rendimiento o YTM (<span style='font-family: serif; font-style: italic;'>i</span>)**. <br><br>"
    "• Si conoces qué tasa de interés exige el mercado, calculamos cuál es el precio justo para comprar/vender el bono hoy.<br>"
    "• Si conoces a qué precio se está vendiendo el bono en el mercado hoy, calculamos qué tasa de rendimiento ganarás si lo conservas hasta el final."
)

modo_bono = st.radio(
    "Selecciona un modo de cálculo:",
    [
        "Precio del Bono (P)  →  conozco la Tasa de Rendimiento",
        "Tasa de Rendimiento (YTM)  →  conozco el Precio (P)",
    ],
    horizontal=True,
    key="modo_bono",
)
separador()

# =============================================================================
# INPUTS COMUNES
# =============================================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Características del Bono**")
    F_bono      = st.number_input("Valor Nominal ($F$)",
                                   min_value=0.01, value=1_000.0,
                                   step=100.0, key="bono_f")
    r_nom_bono  = st.number_input("Tasa Cupón Nominal Anual ($r^{(m)}$) %",
                                   value=8.0, step=0.1, key="bono_r") / 100
    igual_C     = st.checkbox("Valor de Redención ($C$) = Valor Nominal",
                               value=True, key="bono_check_c")
    C_bono      = F_bono if igual_C else st.number_input(
                    "Valor de Redención ($C$)",
                    min_value=0.01, value=1_000.0, step=100.0, key="bono_c"
                  )

with c2:
    st.markdown("**Condiciones de Mercado**")

    if modo_bono.startswith("Precio"):
        tipo_tasa_bono = st.radio(
            "Ingresar tasa de rendimiento como:",
            ["Tasa efectiva periódica", "Tasa nominal anual"],
            key="tipo_tasa_b",
        )
        if tipo_tasa_bono == "Tasa efectiva periódica":
            i_mercado     = st.number_input("Tasa efectiva periódica ($i_m$) %",
                                             value=5.0, step=0.1, key="bono_ieff") / 100
            str_i_bono    = r"i_m"
        else:
            i_nom_mercado = st.number_input("Tasa nominal anual ($i^{(m)}$) %",
                                             value=10.0, step=0.1, key="bono_inom") / 100
            str_i_bono    = r"\frac{i^{(m)}}{m}"
    else:
        precio_mercado = st.number_input("Precio actual en el mercado ($P$)",
                                          min_value=0.01, value=950.0,
                                          step=10.0, key="bono_p_mercado")

with c3:
    st.markdown("**Plazos**")
    n_anios_bono = st.number_input("Años al vencimiento ($n$)",
                                    min_value=0.1, value=5.0,
                                    step=1.0, key="bono_n")
    m_bono       = st.number_input("Cupones por año ($m$)",
                                    min_value=1.0, value=2.0, step=1.0,
                                    help="Ej. 2 = semestral, 4 = trimestral",
                                    key="bono_m")

# =============================================================================
# CÁLCULOS BASE
# =============================================================================
n_periodos_bono = int(n_anios_bono * m_bono)
r_periodo       = r_nom_bono / m_bono
cupon_Fr        = F_bono * r_periodo

separador()
st.markdown("### Resultados de la Valuación")

# Variables que se reutilizan en el bloque de riesgo
i_final = None
p_final = None

# ─────────────────────────────────────────────────────────────────────────────
# CASO A: CALCULAR PRECIO  (P)
# ─────────────────────────────────────────────────────────────────────────────
if modo_bono.startswith("Precio"):

    if tipo_tasa_bono == "Tasa efectiva periódica":
        i_periodo_bono    = i_mercado
        str_val_i_mercado = f"{i_mercado:.6f}"
    else:
        i_periodo_bono    = i_nom_mercado / m_bono
        str_val_i_mercado = f"{i_periodo_bono:.6f}"

    precio_P, _, vp_cup, vp_red = engine.precio_bono(
        F_bono, r_periodo, C_bono, i_periodo_bono, n_periodos_bono
    )
    i_final = i_periodo_bono
    p_final = precio_P

    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        themed_success(f"<h3 style='margin:0; color:inherit;'>Precio del Bono (P): ${precio_P:,.4f}</h3>")
        st.metric("VP de Cupones", f"${vp_cup:,.4f}")
        st.metric("VP de Redención", f"${vp_red:,.4f}")
        
        # Estado del bono con los themes
        if precio_P > C_bono:
            themed_success("Estado: Se vende con **PRIMA** (sobre la par)")
        elif precio_P < C_bono:
            themed_error("Estado: Se vende con **DESCUENTO** (bajo la par)")
        else:
            themed_info("Estado: Se vende **A LA PAR**")

    with col_res2:
        if tipo_tasa_bono == "Tasa efectiva periódica":
            st.latex(r"P = Fr \left[ \frac{1-(1+i_m)^{-nm}}{i_m} \right] + C(1+i_m)^{-nm}")
        else:
            st.latex(r"P = Fr \left[ \frac{1-\left(1+\frac{i^{(m)}}{m}\right)^{-nm}}{\frac{i^{(m)}}{m}} \right] + C\left(1+\frac{i^{(m)}}{m}\right)^{-nm}")

    with paso_a_paso():
        st.latex(r"Fr = F \times \frac{r^{(m)}}{m}")
        st.latex(rf"Fr = {F_bono:,.2f} \times \frac{{{r_nom_bono:.4f}}}{{{m_bono:g}}} = {cupon_Fr:,.4f}")
        st.write("---")
        
        st.latex(r"P = Fr \left[ \frac{1-(1+i_m)^{-nm}}{i_m} \right] + C(1+i_m)^{-nm}")
        st.latex(rf"P = {cupon_Fr:,.4f} \left[ \frac{{1-(1+{str_val_i_mercado})^{{-{n_periodos_bono:g}}}}}{{{str_val_i_mercado}}} \right] + {C_bono:,.2f}(1+{str_val_i_mercado})^{{-{n_periodos_bono:g}}}")
        
        factor_d = (1 + i_periodo_bono) ** (-n_periodos_bono)
        factor_a = (1 - factor_d) / i_periodo_bono
        
        st.latex(rf"P = {cupon_Fr:,.4f} \left[ \frac{{1 - {factor_d:.6f}}}{{{str_val_i_mercado}}} \right] + {C_bono:,.2f}({factor_d:.6f})")
        st.latex(rf"P = {cupon_Fr:,.4f} [{factor_a:.6f}] + {C_bono:,.2f}({factor_d:.6f})")
        st.latex(rf"P = {vp_cup:,.4f} + {vp_red:,.4f}")
        
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>P = ${precio_P:,.4f}</h4>")

# ─────────────────────────────────────────────────────────────────────────────
# CASO B: CALCULAR YTM
# ─────────────────────────────────────────────────────────────────────────────
else:

    i_periodo_res = engine.tasa_rendimiento_bono(
        precio_mercado, F_bono, r_periodo, C_bono, n_periodos_bono
    )

    if np.isnan(i_periodo_res):
        themed_error("No se encontró una tasa válida para este precio. Verifica los parámetros.")
        st.stop()

    i_nom_res = i_periodo_res * m_bono
    i_final   = i_periodo_res
    p_final   = precio_mercado

    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        themed_info(f"<h3 style='margin:0; color:inherit;'>YTM Nominal Anual: {i_nom_res*100:.4f}%</h3>")
        st.metric("Tasa Efectiva Periódica ($i_m$)", f"{i_periodo_res*100:.4f}%")
        
        # Estado del bono
        if precio_mercado > C_bono:
            themed_success("Estado: Se vende con **PRIMA** (Rendimiento < Tasa Cupón)")
        elif precio_mercado < C_bono:
            themed_error("Estado: Se vende con **DESCUENTO** (Rendimiento > Tasa Cupón)")
        else:
            themed_info("Estado: Se vende **A LA PAR** (Rendimiento = Tasa Cupón)")

    with col_res2:
        st.latex(r"P_{mdo} = Fr \left[ \frac{1-(1+i_m)^{-nm}}{i_m} \right] + C(1+i_m)^{-nm}")
        st.caption(f"Se resuelve numéricamente (Método de Brent) para $i_m$")

    with paso_a_paso():
        st.latex(r"P_{mdo} = Fr \left[ \frac{1-(1+i_m)^{-nm}}{i_m} \right] + C(1+i_m)^{-nm}")
        st.latex(rf"{precio_mercado:,.2f} = {cupon_Fr:,.4f} \left[ \frac{{1-(1+i_m)^{{-{n_periodos_bono:g}}}}}{{i_m}} \right] + {C_bono:,.2f}(1+i_m)^{{-{n_periodos_bono:g}}}")
        alerta_metodo_numerico()
        
        st.latex(rf"i_m \approx {i_periodo_res:.6f} \implies {i_periodo_res*100:.4f}\%")
        st.write("---")
        st.latex(rf"\text{{YTM}} = i_m \times m")
        st.latex(rf"\text{{YTM}} = {i_periodo_res:.6f} \times {m_bono:g} = {i_nom_res:.6f}")
        
        themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>\text{{YTM}} = {i_nom_res*100:.4f}\%</h4>")


# =============================================================================
# ANÁLISIS DE RIESGO DE TASA DE INTERÉS
# =============================================================================
if i_final is not None and p_final is not None:

    separador()
    st.markdown("### Análisis de Riesgo de Tasas de Interés")
    themed_warning(
        "Si las tasas del mercado suben, el precio del bono cae (y viceversa). La **Duración Modificada** mide de forma lineal qué tanto "
        "cae el precio por cada 1% que suba la tasa. Como esta relación no es una línea recta sino una curva, la **Convexidad** "
        "corrige ese error para darnos una estimación casi perfecta."
    )

    mac_d, mod_d, conv = engine.riesgo_bono(
        F_bono, r_periodo, C_bono, i_final, n_periodos_bono, m_bono
    )

    cr1, cr2, cr3 = st.columns(3)
    cr1.metric(
        "Duración de Macaulay ($D_{Mac}$)",
        f"{mac_d:.4f} años",
        help="Tiempo promedio ponderado para recuperar tu inversión.",
    )
    cr2.metric(
        "Duración Modificada ($D_{Mod}$)",
        f"{mod_d:.4f}",
        help="Sensibilidad lineal: caída % del precio ante subida de 1% en tasas.",
    )
    cr3.metric(
        "Convexidad ($C$)",
        f"{conv:.4f}",
        help="Curvatura del precio. Mayor convexidad = menor riesgo ante cambios bruscos.",
    )

    # ── Fórmulas de riesgo ────────────────────────────────────────────────────
    with st.expander("Ver fórmulas de Duración y Convexidad"):
        st.latex(r"D_{Mac} = \frac{\sum_{t=1}^{nm} t \cdot VP(CF_t)}{P} \div m")
        st.latex(r"D_{Mod} = \frac{D_{Mac}}{1 + i_m}")
        st.latex(r"C = \frac{\sum_{t=1}^{nm} t(t+1) \cdot VP(CF_t)}{P \cdot m^2 \cdot (1+i_m)^2}")
        st.latex(r"\frac{\Delta P}{P} \approx -D_{Mod}(\Delta y) + \frac{1}{2}C(\Delta y)^2")

    separador()

    # ── Simulador de estrés ───────────────────────────────────────────────────
    with st.expander("Simulador de Estrés de Tasas", expanded=True):
        st.markdown("#### Simulador")
        st.markdown(
            "Utiliza el control deslizante para simular una variación brusca (<span style='font-family: serif; font-style: italic;'>Δy</span>) en las tasas de rendimiento exigidas por el mercado hoy mismo "
            "y observa cómo impacta matemáticamente al precio actual de tu bono.",
            unsafe_allow_html=True
        )

        delta_y_pct = st.slider(
            "Variación de tasas de interés (Δy):",
            min_value=-5.0, max_value=5.0,
            value=1.0, step=0.1, format="%f%%",
            key="stress_slider",
        )
        delta_y = delta_y_pct / 100.0

        impacto_dur  = -mod_d * delta_y
        impacto_conv = 0.5 * conv * (delta_y ** 2)
        impacto_tot  = impacto_dur + impacto_conv
        nuevo_precio = p_final * (1 + impacto_tot)
        variacion    = nuevo_precio - p_final

        col_s1, col_s2 = st.columns([1, 1])

        with col_s1:
            themed_warning(f"<h3 style='margin:0; color:inherit;'>Nuevo Precio: ${nuevo_precio:,.4f}</h3>")
            st.metric(
                "Impacto Total (Variación)",
                f"${variacion:+,.4f}",
                delta=f"{impacto_tot*100:+.2f}%",
                delta_color="inverse"
            )
            st.caption(
                f"Efecto lineal (Duración): **{impacto_dur*100:+.4f}%**<br>"
                f"Corrección curva (Convexidad): **{impacto_conv*100:+.4f}%**",
                unsafe_allow_html=True
            )

        with col_s2:
            st.latex(r"\frac{\Delta P}{P} \approx -D_{Mod}(\Delta y) + \frac{1}{2}C(\Delta y)^2")
            st.latex(rf"\frac{{\Delta P}}{{P}} \approx -({mod_d:.4f})({delta_y:.4f}) + \frac{{1}}{{2}}({conv:.4f})({delta_y:.4f})^2")
            st.latex(rf"\frac{{\Delta P}}{{P}} \approx {impacto_dur:.6f} + {impacto_conv:.6f} = {impacto_tot:.6f}")
            st.latex(rf"\text{{Nuevo P}} = {p_final:,.4f} \times (1 {impacto_tot:+.6f}) = {nuevo_precio:,.4f}")