"""
pages/1_Tasas.py
----------------
Módulo 1: Conversión de Tasas de Interés.
Patrón estándar de todas las páginas:
    1. Importar utils (no repetir CSS ni instanciar engine aquí)
    2. Llamar page_header()
    3. Lógica de la página
"""

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Imports locales ---
from utils import get_engine, page_header, paso_a_paso, separador, themed_info, themed_success, themed_warning, themed_error, apply_plotly_theme

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Tasas de Interés · Calculadora Financiera",
    page_icon="📐",
    layout="wide",
)

# Instancia cacheada del motor
engine = get_engine()

# Encabezado estándar
page_header(
    titulo="1. Conversión de Tasas de Interés",
    subtitulo="Triple igualdad: tasa efectiva · nominal · instantánea (fuerza de interés)"
)

# =============================================================================
# PESTAÑAS
# =============================================================================
tabs = st.tabs([
    "Triple Igualdad",
    "De i⁽ᵐ⁾ → i, δ",
    "De δ → i",
    "De δ → i⁽ᵐ⁾",
    "De i⁽ᵐ⁾ → i⁽ᵖ⁾",
    "Reinversión"
])

# ─────────────────────────────────────────────
# TAB 0: Triple igualdad
# ─────────────────────────────────────────────
with tabs[0]:
    st.markdown("### La triple igualdad de tasas de interés")
    st.markdown(
        "Esta ecuación es la regla de oro de las matemáticas financieras. Establece que el dinero puede "
        "crecer exactamente al mismo ritmo anual utilizando tres estructuras de capitalización distintas:"
    )
    themed_success(
        "• <b>Tasa Efectiva (<span style='font-family: serif; font-style: italic;'>i</span>):</b> Rendimiento real cobrado una sola vez al final del año.<br>"
        "• <b>Tasa Nominal (<span style='font-family: serif; font-style: italic;'>i<sup>(m)</sup></span>):</b> Tasa anual dividida y cobrada en <span style='font-family: serif; font-style: italic;'>m</span> periodos iguales."
    )
    themed_info(
        "• <b>Tasa Instantánea (<span style='font-family: serif; font-style: italic;'>δ</span>):</b> Fuerza de interés donde la capitalización es continua (a cada instante)."
    )
    st.latex(r"1 + i = \left(1 + \frac{i^{(m)}}{m}\right)^m = e^\delta")

    separador()

    with st.expander("¿Cómo funciona esta equivalencia?"):
        st.markdown(r"""
        El objetivo es que, sin importar la frecuencia de pago, el rendimiento anual real sea idéntico para que no exista arbitraje.

        * Si inviertes **$1$** a una tasa efectiva, terminas con **$(1 + i)$**.
        * Si te pagan en $m$ periodos y reinviertes el dinero, terminas con **$(1 + \frac{i^{(m)}}{m})^m$**.
        * Si la capitalización es continua (infinita), terminas con **$e^\delta$**.

        Como el dinero final debe ser exactamente el mismo en los tres escenarios, podemos despejar cualquier tasa a partir de las otras.
        """)

# ─────────────────────────────────────────────
# TAB 1: Nominal → Efectiva e Instantánea
# ─────────────────────────────────────────────
with tabs[1]:
    st.markdown("### De tasa nominal $i^{(m)}$ → efectiva $i$ e instantánea $\\delta$")
    
    col_t1a, col_t1b = st.columns(2)
    with col_t1a:
        themed_success(
            "La **tasa efectiva** (<span style='font-family: serif; font-style: italic;'>i</span>) representa el rendimiento real "
            "generado por el efecto del interés compuesto anualizado."
        )
    with col_t1b:
        themed_info(
            "La **tasa instantánea** (<span style='font-family: serif; font-style: italic;'>&delta;</span>) es el análogo teórico continuo "
            "de esa misma tasa, donde la capitalización ocurre a cada instante."
        )
    
    separador()

    c1, c2 = st.columns(2)

    with c1:
        j    = st.number_input("Tasa Nominal $i^{(m)}$ %", value=20.0, step=0.1, key="t1_j") / 100
        m    = st.number_input("Frecuencia de pagos por año (m)", min_value=0.0001, value=12.0, step=0.5, format="%.2f", key="t1_m")

    with c2:
        i_eff = engine.tasa_nominal_a_efectiva(j, m)
        delta = engine.tasa_nominal_a_instantanea(j, m)
        
        themed_success(f"<h3 style='margin:0; color:inherit;'>Tasa Efectiva Anual (i): {i_eff*100:.4f}%</h3>")
        themed_info(f"<h3 style='margin:0; color:inherit;'>Tasa Instantánea (δ): {delta*100:.4f}%</h3>")

    separador()

    with paso_a_paso():
        st.latex(r"i = \left(1 + \frac{i^{(m)}}{m}\right)^m - 1")
        st.latex(rf"i = \left(1 + \frac{{{j:.4f}}}{{{m:g}}}\right)^{{{m:g}}} - 1")
        st.latex(rf"i = (1 + {j/m:.6f})^{{{m:g}}} - 1 = {i_eff:.6f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i = {i_eff*100:.4f}\%</h4>")

        st.write("---")
        
        st.latex(r"\delta = m \ln\left(1 + \frac{i^{(m)}}{m}\right)")
        st.latex(rf"\delta = {m:g} \ln\left(1 + \frac{{{j:.4f}}}{{{m:g}}}\right)")
        st.latex(rf"\delta = {m:g} \times \ln(1 + {j/m:.6f})")
        st.latex(rf"\delta = {m:g} \times {np.log(1+j/m):.6f} = {delta:.6f}")
        themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>\delta = {delta*100:.4f}\%</h4>")

# ─────────────────────────────────────────────
# TAB 2: Instantánea → Efectiva
# ─────────────────────────────────────────────
with tabs[2]:
    st.markdown("### De tasa instantánea $\\delta$ → efectiva $i$")
    themed_success(
        "Convierte el concepto matemático de la fuerza de interés continua (<span style='font-family: serif; font-style: italic;'>&delta;</span>) "
        "en una **tasa efectiva anual** (<span style='font-family: serif; font-style: italic;'>i</span>) comprobable en periodos discretos del mundo real."
    )
    separador()

    c1, c2 = st.columns(2)

    with c1:
        d2 = st.number_input("Tasa Instantánea δ %", value=18.0, step=0.1, key="t2_d") / 100

    with c2:
        i2 = engine.tasa_instantanea_a_efectiva(d2)
        themed_success(f"<h3 style='margin:0; color:inherit;'>Tasa Efectiva Anual (i): {i2*100:.4f}%</h3>")

    separador()

    with paso_a_paso():
        st.latex(r"1 + i = e^\delta \quad \Rightarrow \quad i = e^\delta - 1")
        st.latex(rf"i = e^{{{d2:.4f}}} - 1")
        st.latex(rf"i = {np.exp(d2):.6f} - 1 = {i2:.6f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i = {i2*100:.4f}\%</h4>")

# ─────────────────────────────────────────────
# TAB 3: Instantánea → Nominal
# ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("### De tasa instantánea $\\delta$ → nominal $i^{(m)}$")
    themed_success(
        "Toma la tasa continua de un modelo teórico (<span style='font-family: serif; font-style: italic;'>&delta;</span>) y "
        "encuentra la **tasa comercial** con capitalización discreta (<span style='font-family: serif; font-style: italic;'>i<sup>(m)</sup></span>) "
        "que produciría exactamente la misma riqueza al vencimiento."
    )
    separador()

    c1, c2 = st.columns(2)

    with c1:
        d3 = st.number_input("Tasa Instantánea δ %", value=18.0, step=0.1, key="t3_d") / 100
        m3 = st.number_input("Frecuencia de pagos deseada (m)", min_value=0.0001, value=12.0, step=0.5, format="%.4f", key="t3_m")

    with c2:
        i3 = engine.tasa_instantanea_a_nominal(d3, m3)
        themed_success(f"<h3 style='margin:0; color:inherit;'>Tasa Nominal i^({m3:g}): {i3*100:.4f}%</h3>")

    separador()

    with paso_a_paso():
        st.latex(r"\left(1 + \frac{i^{(m)}}{m}\right)^m = e^\delta")
        st.latex(r"i^{(m)} = m \left(e^{\delta/m} - 1\right)")
        st.latex(rf"i^{{({m3:g})}} = {m3:g} \left(e^{{\frac{{{d3:.4f}}}{{{m3:g}}}}} - 1\right)")
        st.latex(rf"i^{{({m3:g})}} = {m3:g} \left(e^{{{d3/m3:.6f}}} - 1\right)")
        st.latex(rf"i^{{({m3:g})}} = {m3:g} ({np.exp(d3/m3):.6f} - 1) = {i3:.6f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i^{{({m3:g})}} = {i3*100:.4f}\%</h4>")

# ─────────────────────────────────────────────
# TAB 4: Nominal i(m) → Nominal i(p)
# ─────────────────────────────────────────────
with tabs[4]:
    st.markdown("### De tasa nominal $i^{(m)}$ → nominal $i^{(p)}$")
    themed_success(
        "Calcula la **tasa nominal equivalente** al cambiar la frecuencia de pagos "
        "(de <span style='font-family: serif; font-style: italic;'>m</span> a <span style='font-family: serif; font-style: italic;'>p</span> periodos por año), "
        "garantizando que el monto final acumulado sea idéntico en ambos escenarios para prevenir el arbitraje."
    )
    separador()

    c1, c2 = st.columns(2)

    with c1:
        i_orig = st.number_input("Tasa Nominal Origen $i^{(m)}$ %",  value=10.0, step=0.1, key="t4_i") / 100
        m_orig = st.number_input("Frecuencia Origen (m)",  min_value=0.0001, value=2.0,  step=0.5, key="t4_m")
        p_dest = st.number_input("Frecuencia Destino (p)", min_value=0.0001, value=3.0,  step=0.5, key="t4_p")

    with c2:
        i_p = engine.tasa_nominal_m_a_nominal_p(i_orig, m_orig, p_dest)
        themed_success(f"<h3 style='margin:0; color:inherit;'>Tasa Nominal i^({p_dest:g}): {i_p*100:.4f}%</h3>")

    separador()

    with paso_a_paso():
        frac_mp       = m_orig / p_dest
        tasa_per_orig = i_orig / m_orig
        
        st.latex(r"\left(1 + \frac{i^{(p)}}{p}\right)^p = \left(1 + \frac{i^{(m)}}{m}\right)^m")
        st.latex(r"i^{(p)} = p \left[ \left(1 + \frac{i^{(m)}}{m}\right)^{\frac{m}{p}} - 1 \right]")
        
        st.latex(rf"i^{{({p_dest:g})}} = {p_dest:g} \left[ \left(1 + \frac{{{i_orig:.4f}}}{{{m_orig:g}}}\right)^{{\frac{{{m_orig:g}}}{{{p_dest:g}}}}} - 1 \right]")
        st.latex(rf"i^{{({p_dest:g})}} = {p_dest:g} \left[ (1 + {tasa_per_orig:.6f})^{{{frac_mp:.4f}}} - 1 \right]")
        st.latex(rf"i^{{({p_dest:g})}} = {p_dest:g} ({((1+tasa_per_orig)**frac_mp):.6f} - 1) = {i_p:.6f}")
        
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i^{{({p_dest:g})}} = {i_p*100:.4f}\%</h4>")

# ─────────────────────────────────────────────
# TAB 5: Reinversión
# ─────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Ilustración de la reinversión: convergencia al límite $e$")
    themed_info(
        "Muestra gráficamente cómo al incrementar la frecuencia de pago, el efecto del interés compuesto "
        "hace que el dinero crezca cada vez más lento, hasta topar con un límite matemático absoluto: "
        "la **capitalización continua** gobernada por el número de Euler (<span style='font-family: serif; font-style: italic;'>e</span>)."
    )
    separador()

    c1, c2, c3 = st.columns(3)
    C0       = c1.number_input("Capital Inicial ($C_0$)", min_value=0.0, value=100_000.0, step=1_000.0, key="t5_c0")
    tasa_ref = c2.number_input("Tasa Nominal ($i$) %",   value=10.0, step=0.1, key="t5_ref") / 100
    n_anios  = c3.number_input("Periodos ($n$)",          min_value=0.1, value=1.0, step=1.0, key="t5_n")

    separador()

    col_t, col_g = st.columns([1, 2])
    
    df_reinv = engine.generar_tabla_reinversion(C0, tasa_ref, n_anios)

    with col_t:
        st.markdown("##### Tabla de Acumulación")
        st.dataframe(
            df_reinv.style.format({
                "Monto acumulado":      "${:,.2f}",
                "Rendimiento Acumulado": "{:.6f}",
            }).set_properties(**{
                "background-color": "#F8FAFC",
                "color": "#0F172A"
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col_g:
        st.markdown("##### Convergencia del monto acumulado")
        fig = px.scatter(
            df_reinv,
            x="Periodo de reinversión",
            y="Monto acumulado",
            color="Periodo de reinversión",
            labels={
                "Periodo de reinversión": "Frecuencia",
                "Monto acumulado": "Monto ($)",
            },
        )
        fig.update_traces(marker=dict(size=14))
        fig.add_trace(go.Scatter(
            x=df_reinv["Periodo de reinversión"],
            y=df_reinv["Monto acumulado"],
            mode="lines",
            line=dict(color="#cbd5e1", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.data = fig.data[::-1]
        fig.update_layout(yaxis=dict(tickformat="$.2f"), template="none", height=400, margin=dict(t=10))
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)