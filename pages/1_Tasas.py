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
    themed_info(
        "Esta ecuación relaciona la tasa efectiva anual (<span style='font-family: serif; font-style: italic;'>i</span>), "
        "la tasa nominal capitalizable <span style='font-family: serif; font-style: italic;'>m</span> veces al año (<span style='font-family: serif; font-style: italic;'>i<sup>(m)</sup></span>) "
        "y la tasa instantánea o fuerza de interés (<span style='font-family: serif; font-style: italic;'>δ</span>)."
    )
    st.latex(r"1 + i = \left(1 + \frac{i^{(m)}}{m}\right)^m = e^\delta")

    separador()

    with st.expander("Ver Explicación de la Triple Igualdad"):
        st.markdown(r"""
        **Concepto Clave:**
        El objetivo es que, sin importar con qué frecuencia se reinviertan los intereses,
        un inversionista gane exactamente el mismo rendimiento a final de año.

        * Si inviertes **$1$** y te pagan el interés solo una vez al final del año, terminas con **$(1 + i)$**.
        * Si te lo pagan en $m$ pedacitos al año, terminas con **$(1 + \frac{i^{(m)}}{m})^m$**.
        * Si la capitalización es continua (infinita), terminas con **$e^\delta$**.

        Como el rendimiento real anual debe ser el mismo para que no haya arbitraje,
        estas tres expresiones son matemáticamente idénticas.
        """)

# ─────────────────────────────────────────────
# TAB 1: Nominal → Efectiva e Instantánea
# ─────────────────────────────────────────────
with tabs[1]:
    st.markdown("### De tasa nominal $i^{(m)}$ → efectiva $i$ e instantánea $\\delta$")
    c1, c2 = st.columns(2)

    with c1:
        j    = st.number_input("Tasa Nominal $i^{(m)}$ %", value=20.0, step=0.1, key="t1_j") / 100
        m    = st.number_input("Frecuencia (m)", min_value=0.0001, value=12.0, step=0.5, format="%.2f", key="t1_m")

    with c2:
        i_eff = engine.tasa_nominal_a_efectiva(j, m)
        delta = engine.tasa_nominal_a_instantanea(j, m)
        
        # Diseño Premium para los resultados
        themed_success(f"<h3 style='margin:0; color:inherit;'>Efectiva (i): {i_eff*100:.4f}%</h3>")
        themed_info(f"<h3 style='margin:0; color:inherit;'>Instantánea (δ): {delta*100:.4f}%</h3>")

    separador()

    with paso_a_paso("Ver desarrollo paso a paso"):
        themed_info("**1. Cálculo de la Tasa Efectiva (<span style='font-family: serif; font-style: italic;'>i</span>):**")
        st.latex(r"i = \left(1 + \frac{i^{(m)}}{m}\right)^m - 1")
        st.latex(rf"i = \left(1 + \frac{{{j:.4f}}}{{{m:g}}}\right)^{{{m:g}}} - 1")
        st.latex(rf"i = (1 + {j/m:.6f})^{{{m:g}}} - 1 = {i_eff:.6f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i = {i_eff*100:.4f}%</h4>")

        st.write("---")
        
        themed_info("**2. Cálculo de la Fuerza de Interés (<span style='font-family: serif; font-style: italic;'>δ</span>):**")
        st.latex(r"\delta = m \ln\left(1 + \frac{i^{(m)}}{m}\right)")
        st.latex(rf"\delta = {m:g} \ln\left(1 + \frac{{{j:.4f}}}{{{m:g}}}\right)")
        st.latex(rf"\delta = {m:g} \times \ln(1 + {j/m:.6f}) = {m:g} \times {np.log(1+j/m):.6f} = {delta:.6f}")
        themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>δ = {delta*100:.4f}%</h4>")

# ─────────────────────────────────────────────
# TAB 2: Instantánea → Efectiva
# ─────────────────────────────────────────────
with tabs[2]:
    st.markdown("### De tasa instantánea $\\delta$ → efectiva $i$")
    c1, c2 = st.columns(2)

    with c1:
        d2 = st.number_input("Tasa Instantánea δ %", value=18.0, step=0.1, key="t2_d") / 100

    with c2:
        i2 = engine.tasa_instantanea_a_efectiva(d2)
        themed_success(f"<h3 style='margin:0; color:inherit;'>Efectiva (i): {i2*100:.4f}%</h3>")

    separador()

    with paso_a_paso("Ver desarrollo paso a paso"):
        st.latex(r"1 + i = e^\delta \quad \Rightarrow \quad i = e^\delta - 1")
        st.latex(rf"i = e^{{{d2:.4f}}} - 1")
        st.latex(rf"i = {np.exp(d2):.6f} - 1 = {i2:.6f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i = {i2*100:.4f}%</h4>")

# ─────────────────────────────────────────────
# TAB 3: Instantánea → Nominal
# ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("### De tasa instantánea $\\delta$ → nominal $i^{(m)}$")
    c1, c2 = st.columns(2)

    with c1:
        d3 = st.number_input("Tasa Instantánea δ %", value=18.0, step=0.1, key="t3_d") / 100
        m3 = st.number_input("Frecuencia deseada (m)", min_value=0.0001, value=12.0, step=0.5, format="%.4f", key="t3_m")

    with c2:
        i3 = engine.tasa_instantanea_a_nominal(d3, m3)
        themed_success(f"<h3 style='margin:0; color:inherit;'>Nominal i^({m3:g}): {i3*100:.4f}%</h3>")

    separador()

    with paso_a_paso("Ver desarrollo paso a paso"):
        st.latex(r"\left(1 + \frac{i^{(m)}}{m}\right)^m = e^\delta")
        st.latex(r"i^{(m)} = m \left(e^{\delta/m} - 1\right)")
        st.latex(rf"i^{{({m3:g})}} = {m3:g} \left(e^{{\frac{{{d3:.4f}}}{{{m3:g}}}}} - 1\right)")
        st.latex(rf"i^{{({m3:g})}} = {m3:g} \left(e^{{{d3/m3:.6f}}} - 1\right)")
        st.latex(rf"i^{{({m3:g})}} = {m3:g} \times ({np.exp(d3/m3):.6f} - 1) = {i3:.6f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i^({m3:g}) = {i3*100:.4f}%</h4>")

# ─────────────────────────────────────────────
# TAB 4: Nominal i(m) → Nominal i(p)
# ─────────────────────────────────────────────
with tabs[4]:
    st.markdown("### De tasa nominal $i^{(m)}$ → nominal $i^{(p)}$")
    c1, c2 = st.columns(2)

    with c1:
        i_orig = st.number_input("Tasa Nominal Origen $i^{(m)}$ %",  value=10.0, step=0.1, key="t4_i") / 100
        m_orig = st.number_input("Frecuencia Origen (m)",  min_value=0.0001, value=2.0,  step=0.5, key="t4_m")
        p_dest = st.number_input("Frecuencia Destino (p)", min_value=0.0001, value=3.0,  step=0.5, key="t4_p")

    with c2:
        i_p = engine.tasa_nominal_m_a_nominal_p(i_orig, m_orig, p_dest)
        themed_success(f"<h3 style='margin:0; color:inherit;'>Nominal i^({p_dest:g}): {i_p*100:.4f}%</h3>")

    separador()

    with paso_a_paso("Ver desarrollo paso a paso"):
        frac_mp       = m_orig / p_dest
        tasa_per_orig = i_orig / m_orig
        
        st.latex(r"\left(1 + \frac{i^{(p)}}{p}\right)^p = \left(1 + \frac{i^{(m)}}{m}\right)^m")
        st.latex(r"i^{(p)} = p \left[ \left(1 + \frac{i^{(m)}}{m}\right)^{\frac{m}{p}} - 1 \right]")
        
        st.latex(rf"i^{{({p_dest:g})}} = {p_dest:g} \left[ \left(1 + \frac{{{i_orig:.4f}}}{{{m_orig:g}}}\right)^{{\frac{{{m_orig:g}}}{{{p_dest:g}}}}} - 1 \right]")
        st.latex(rf"i^{{({p_dest:g})}} = {p_dest:g} \times \left[(1 + {tasa_per_orig:.6f})^{{{frac_mp:.4f}}} - 1\right]")
        st.latex(rf"i^{{({p_dest:g})}} = {p_dest:g} \times ({((1+tasa_per_orig)**frac_mp):.6f} - 1) = {i_p:.6f}")
        
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>i^({p_dest:g}) = {i_p*100:.4f}%</h4>")

# ─────────────────────────────────────────────
# TAB 5: Reinversión
# ─────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Ilustración de la reinversión: convergencia al límite $e$")

    c1, c2, c3 = st.columns(3)
    C0       = c1.number_input("Capital Inicial ($C_0$)", min_value=0.0, value=100_000.0, step=1_000.0, key="t5_c0")
    tasa_ref = c2.number_input("Tasa Nominal ($i$) %",   value=10.0, step=0.1, key="t5_ref") / 100
    n_anios  = c3.number_input("Periodos ($n$)",          min_value=0.1, value=1.0, step=1.0, key="t5_n")

    # Tabla
    st.subheader("Tabla de Acumulación")
    df_reinv = engine.generar_tabla_reinversion(C0, tasa_ref, n_anios)
    
    # Formateo mejorado del DataFrame
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

    # Gráfica
    st.subheader("Convergencia del monto acumulado")
    fig = px.scatter(
        df_reinv,
        x="Periodo de reinversión",
        y="Monto acumulado",
        color="Periodo de reinversión",
        title="Monto Acumulado vs. Frecuencia de Capitalización",
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
    fig.update_layout(yaxis=dict(tickformat="$.2f"), template="none")
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("¿Cómo interpretar esta gráfica?"):
        themed_info(
            "**La magia del límite matemático (<span style='font-family: serif; font-style: italic;'>e</span>)**\n\n"
            "Aumentar la frecuencia de capitalización (por ejemplo, pasar de meses a días, y luego a horas) "
            "genera saltos en las ganancias cada vez menores. "
            "Existe un techo invisible marcado por el número de Euler <span style='font-family: serif; font-style: italic;'>e</span>: "
            "el interés compuesto tiene un límite máximo absoluto de crecimiento, "
            "al cual llamamos **capitalización continua**."
        )