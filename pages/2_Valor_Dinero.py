"""
pages/2_Valor_Dinero.py
-----------------------
Módulo 2: Valor del Dinero en el Tiempo.
Cubre: Valor Futuro, Valor Presente, Número de Periodos y Tasa de Rendimiento.
Soporta tasas efectivas, nominales e instantáneas (continuas).
"""

import numpy as np
import streamlit as st

from utils import get_engine, page_header, paso_a_paso, separador, themed_info, themed_success, themed_warning, themed_error

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Valor del Dinero · Calculadora Financiera",
    page_icon="⏳",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="2. Valor del Dinero en el Tiempo",
    subtitulo="Interés compuesto · VP · VF · Tasa de rendimiento · Número de periodos"
)

# =============================================================================
# PESTAÑAS PRINCIPALES
# =============================================================================
t1, t2, t3, t4 = st.tabs([
    "Valor Futuro",
    "Valor Presente",
    "Número de Periodos",
    "Tasa de Rendimiento",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: VALOR FUTURO
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown("### Valor futuro de una inversión inicial")
    themed_success(
        "El **Valor Futuro (VF)** proyecta cuánto valdrá un capital inicial (<span style='font-family: serif; font-style: italic;'>C<sub>0</sub></span>) "
        "en el futuro, sumando los rendimientos generados por el interés compuesto a una tasa dada."
    )

    escenario_vf = st.radio(
        "Tipo de tasa:",
        ["Tasa efectiva", "Tasa nominal", "Tasa instantánea"],
        horizontal=True,
        key="radio_vf",
    )
    separador()

    c1, c2 = st.columns(2)

    # ── Efectiva ──────────────────────────────────────────
    if escenario_vf == "Tasa efectiva":
        with c1:
            C0_vf = st.number_input("Capital Inicial ($C_0$)", min_value=0.0, value=20_000.0, step=1_000.0, key="vf_c0_1")
            i_vf  = st.number_input("Tasa efectiva anual ($i$) %", value=6.8, step=0.1, key="vf_i") / 100
            n_vf  = st.number_input("Años ($n$)", min_value=0.0, value=6.0, step=1.0, key="vf_n")
        
        vf_res     = engine.valor_futuro(C0_vf, i_vf, n_vf)
        formula_vf = r"VF = C_0 (1+i)^n"

        with c2:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Valor Futuro: ${vf_res:,.2f}</h3>")
            st.latex(formula_vf)

        with paso_a_paso():
            st.latex(formula_vf)
            st.latex(rf"VF = {C0_vf:,.2f} (1 + {i_vf:.4f})^{{{n_vf:g}}}")
            st.latex(rf"VF = {C0_vf:,.2f} ({(1+i_vf)**n_vf:.6f})")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>VF = ${vf_res:,.2f}</h4>")

    # ── Nominal ───────────────────────────────────────────
    elif escenario_vf == "Tasa nominal":
        with c1:
            C0_vf    = st.number_input("Capital Inicial ($C_0$)", min_value=0.0, value=54_000.0, step=1_000.0, key="vf_c0_2")
            i_nom_vf = st.number_input("Tasa nominal anual ($i^{(m)}$) %", value=11.25, step=0.1, key="vf_inom") / 100
            n_vf2    = st.number_input("Años ($n$)", min_value=0.0, value=8.0, step=1.0, key="vf_n2")
            m_vf     = st.number_input("Periodos por año ($m$)", min_value=1.0, value=4.0, step=1.0, key="vf_m")

        im_vf      = i_nom_vf / m_vf
        nm_vf      = n_vf2 * m_vf
        vf_res     = engine.valor_futuro(C0_vf, im_vf, nm_vf)
        formula_vf = r"VF = C_0 \left(1+\frac{i^{(m)}}{m}\right)^{nm}"

        with c2:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Valor Futuro: ${vf_res:,.2f}</h3>")
            st.latex(formula_vf)

        with paso_a_paso():
            st.latex(formula_vf)
            st.latex(rf"VF = {C0_vf:,.2f} \left(1 + \frac{{{i_nom_vf:.4f}}}{{{m_vf:g}}}\right)^{{{n_vf2:g} \times {m_vf:g}}}")
            st.latex(rf"VF = {C0_vf:,.2f} (1 + {im_vf:.6f})^{{{nm_vf:g}}}")
            st.latex(rf"VF = {C0_vf:,.2f} ({(1+im_vf)**nm_vf:.6f})")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>VF = ${vf_res:,.2f}</h4>")

    # ── Instantánea ───────────────────────────────────────
    else:
        with c1:
            C0_vf = st.number_input("Capital Inicial ($C_0$)", min_value=0.0, value=20_000.0, step=1_000.0, key="vf_c0_3")
            d_vf  = st.number_input("Tasa instantánea ($\\delta$) %", value=5.0, step=0.1, key="vf_d") / 100
            n_vf3 = st.number_input("Años ($n$)", min_value=0.0, value=10.0, step=1.0, key="vf_n3")

        vf_res     = engine.valor_futuro_continuo(C0_vf, d_vf, n_vf3)
        formula_vf = r"VF = C_0 e^{\delta n}"

        with c2:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Valor Futuro: ${vf_res:,.2f}</h3>")
            st.latex(formula_vf)

        with paso_a_paso():
            st.latex(formula_vf)
            st.latex(rf"VF = {C0_vf:,.2f} e^{{({d_vf:.4f})({n_vf3:g})}}")
            st.latex(rf"VF = {C0_vf:,.2f} e^{{{d_vf*n_vf3:.6f}}}")
            st.latex(rf"VF = {C0_vf:,.2f} ({np.exp(d_vf*n_vf3):.6f})")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>VF = ${vf_res:,.2f}</h4>")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: VALOR PRESENTE
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown("### Valor presente de una cantidad de dinero futura")
    themed_info(
        "El **Valor Presente (VP)** es el proceso inverso (descuento). Determina cuánto dinero "
        "necesitas invertir hoy para alcanzar un monto objetivo (<span style='font-family: serif; font-style: italic;'>C<sub>n</sub></span>) "
        "en el futuro, descontando el efecto de la tasa de interés."
    )

    escenario_vp = st.radio(
        "Tipo de tasa:",
        ["Tasa efectiva", "Tasa nominal", "Tasa instantánea"],
        horizontal=True,
        key="radio_vp",
    )
    separador()

    c1, c2 = st.columns(2)

    # ── Efectiva ──────────────────────────────────────────
    if escenario_vp == "Tasa efectiva":
        with c1:
            Cn_vp = st.number_input("Valor Futuro ($C_n$)", min_value=0.0, value=245_000.0, step=1_000.0, key="vp_cn_1")
            i_vp  = st.number_input("Tasa efectiva anual ($i$) %", value=11.2, step=0.1, key="vp_i") / 100
            n_vp  = st.number_input("Años ($n$)", min_value=0.0, value=9.0, step=1.0, key="vp_n")

        vp_res     = engine.valor_presente(Cn_vp, i_vp, n_vp)
        formula_vp = r"VP = C_n (1+i)^{-n}"

        with c2:
            themed_info(f"<h3 style='margin:0; color:inherit;'>Valor Presente: ${vp_res:,.2f}</h3>")
            st.latex(formula_vp)

        with paso_a_paso():
            st.latex(formula_vp)
            st.latex(rf"VP = {Cn_vp:,.2f} (1 + {i_vp:.4f})^{{-{n_vp:g}}}")
            st.latex(rf"VP = {Cn_vp:,.2f} ({(1+i_vp)**(-n_vp):.6f})")
            themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>VP = ${vp_res:,.2f}</h4>")

    # ── Nominal ───────────────────────────────────────────
    elif escenario_vp == "Tasa nominal":
        with c1:
            Cn_vp    = st.number_input("Valor Futuro ($C_n$)", min_value=0.0, value=1_000.0, step=100.0, key="vp_cn_2")
            i_nom_vp = st.number_input("Tasa nominal anual ($i^{(m)}$) %", value=10.0, step=0.1, key="vp_inom") / 100
            n_vp2    = st.number_input("Años ($n$)", min_value=0.0, value=10.0, step=1.0, key="vp_n2")
            m_vp     = st.number_input("Periodos por año ($m$)", min_value=1.0, value=2.0, step=1.0, key="vp_m")

        im_vp      = i_nom_vp / m_vp
        nm_vp      = n_vp2 * m_vp
        vp_res     = engine.valor_presente(Cn_vp, im_vp, nm_vp)
        formula_vp = r"VP = C_n \left(1+\frac{i^{(m)}}{m}\right)^{-nm}"

        with c2:
            themed_info(f"<h3 style='margin:0; color:inherit;'>Valor Presente: ${vp_res:,.2f}</h3>")
            st.latex(formula_vp)

        with paso_a_paso():
            st.latex(formula_vp)
            st.latex(rf"VP = {Cn_vp:,.2f} \left(1 + \frac{{{i_nom_vp:.4f}}}{{{m_vp:g}}}\right)^{{-({n_vp2:g} \times {m_vp:g})}}")
            st.latex(rf"VP = {Cn_vp:,.2f} (1 + {im_vp:.6f})^{{-{nm_vp:g}}}")
            st.latex(rf"VP = {Cn_vp:,.2f} ({(1+im_vp)**(-nm_vp):.6f})")
            themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>VP = ${vp_res:,.2f}</h4>")

    # ── Instantánea ───────────────────────────────────────
    else:
        with c1:
            Cn_vp = st.number_input("Valor Futuro ($C_n$)", min_value=0.0, value=1_000.0, step=100.0, key="vp_cn_3")
            d_vp  = st.number_input("Tasa instantánea ($\\delta$) %", value=5.0, step=0.1, key="vp_d") / 100
            n_vp3 = st.number_input("Años ($n$)", min_value=0.0, value=10.0, step=1.0, key="vp_n3")

        vp_res     = engine.valor_presente_continuo(Cn_vp, d_vp, n_vp3)
        formula_vp = r"VP = C_n e^{-\delta n}"

        with c2:
            themed_info(f"<h3 style='margin:0; color:inherit;'>Valor Presente: ${vp_res:,.2f}</h3>")
            st.latex(formula_vp)

        with paso_a_paso():
            st.latex(formula_vp)
            st.latex(rf"VP = {Cn_vp:,.2f} e^{{-({d_vp:.4f})({n_vp3:g})}}")
            st.latex(rf"VP = {Cn_vp:,.2f} e^{{-{d_vp*n_vp3:.6f}}}")
            st.latex(rf"VP = {Cn_vp:,.2f} ({np.exp(-d_vp*n_vp3):.6f})")
            themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>VP = ${vp_res:,.2f}</h4>")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: NÚMERO DE PERIODOS
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown("### Determinación del número de periodos de una inversión")
    themed_info(
        "Despeja la variable de tiempo (<span style='font-family: serif; font-style: italic;'>n</span>) "
        "de la ecuación de interés compuesto utilizando las propiedades de los logaritmos."
    )

    c1, c2 = st.columns(2)

    with c1:
        va_nper = st.number_input("Valor Inicial ($C_0$)", min_value=0.01, value=50_000.0, step=1_000.0, key="nper_va")
        vf_nper = st.number_input("Valor Final ($C_n$)",   min_value=0.01, value=245_000.0, step=1_000.0, key="nper_vf")
        i_nper  = st.number_input("Tasa Efectiva ($i$) %", min_value=0.0001, value=4.3, step=0.1, key="nper_i") / 100

    with c2:
        n_res = engine.numero_periodos(va_nper, vf_nper, i_nper)
        themed_info(f"<h3 style='margin:0; color:inherit;'>Número de Periodos (n): {n_res:.5f}</h3>")
        st.latex(r"n = \frac{\ln(C_n/C_0)}{\ln(1+i)}")

    separador()

    with paso_a_paso():
        st.latex(r"C_n = C_0(1+i)^n \implies \frac{C_n}{C_0} = (1+i)^n")
        st.latex(r"n = \frac{\ln(C_n/C_0)}{\ln(1+i)}")
        st.write("---")

        ratio = vf_nper / va_nper
        num   = np.log(ratio)
        den   = np.log(1 + i_nper)

        st.latex(rf"n = \frac{{\ln({vf_nper:,.2f} / {va_nper:,.2f})}}{{\ln(1 + {i_nper:.4f})}}")
        st.latex(rf"n = \frac{{\ln({ratio:.6f})}}{{\ln({1+i_nper:.6f})}}")
        st.latex(rf"n = \frac{{{num:.6f}}}{{{den:.6f}}}")
        themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>n = {n_res:.5f} \text{{ periodos}}</h4>")

    # Desglose del tiempo exacto
    separador()
    st.markdown("#### Desglose temporal exacto")
    df_desglose = engine.desglosar_periodos(n_res)
    st.dataframe(
        df_desglose.style.set_properties(**{
            "background-color": "#F3F4F6",
            "color":            "#1E3A8A",
            "font-weight":      "bold",
            "text-align":       "center",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: TASA DE RENDIMIENTO
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown("### Tasa de rendimiento (Tasa Anual de Crecimiento Compuesto)")
    themed_success(
        "Calcula la tasa media anual (<span style='font-family: serif; font-style: italic;'>i</span>) a la que "
        "creció una inversión desde un valor inicial hasta un valor final a lo largo de un plazo determinado."
    )

    c1, c2 = st.columns(2)

    with c1:
        va_rate = st.number_input("Valor Inicial ($C_0$)", min_value=0.01, value=4_582_500.0, step=1_000.0, key="rate_va")
        vf_rate = st.number_input("Valor Final ($C_n$)",   min_value=0.01, value=9_360_000.0, step=1_000.0, key="rate_vf")
        n_rate  = st.number_input("Periodos ($n$)",        min_value=0.1, value=10.0, step=1.0, key="rate_n")

    with c2:
        i_res = engine.tasa_rendimiento(va_rate, vf_rate, n_rate)
        themed_success(f"<h3 style='margin:0; color:inherit;'>Tasa de Rendimiento (i): {i_res*100:.4f}%</h3>")
        st.latex(r"i = \left(\frac{C_n}{C_0}\right)^{\frac{1}{n}} - 1")

    separador()

    with paso_a_paso():
        st.latex(r"C_n = C_0(1+i)^n \implies 1+i = \sqrt[n]{\frac{C_n}{C_0}}")
        st.latex(r"i = \left(\frac{C_n}{C_0}\right)^{\frac{1}{n}} - 1")
        st.write("---")

        ratio   = vf_rate / va_rate
        exp_val = 1 / n_rate

        st.latex(rf"i = \left(\frac{{{vf_rate:,.2f}}}{{{va_rate:,.2f}}}\right)^{{\frac{{1}}{{{n_rate:g}}}}} - 1")
        st.latex(rf"i = ({ratio:.6f})^{{{exp_val:.6f}}} - 1")
        st.latex(rf"i = {ratio**exp_val:.6f} - 1")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i = {i_res*100:.4f}\% \text{{ por periodo}}</h4>")