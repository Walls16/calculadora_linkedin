"""
pages/4_Amortizacion.py
-----------------------
Módulo 4: Tabla de Amortización de Pagos Fijos.
Cubre:
  - Cálculo del pago fijo (R) conociendo el préstamo (VP)
  - Cálculo del préstamo (VP) conociendo el pago fijo (R)
  - Soporte para enganche (monto fijo o porcentaje)
  - Tabla de amortización detallada
  - Gráfica de composición interés / amortización por periodo
"""
import streamlit as st
import plotly.express as px
import numpy as np

# Asegúrate de que todas estas funciones existan en tu utils.py
from utils import get_engine, page_header, paso_a_paso, separador, themed_info, themed_success, themed_warning, themed_error, apply_plotly_theme

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Amortización · Calculadora Financiera",
    page_icon="🏦",
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
    titulo="4. Tabla de Amortización (Pagos Fijos)",
    subtitulo="Calcula la cuota periódica o el monto del préstamo y genera la tabla completa"
)

# =============================================================================
# CONTROLES PRINCIPALES
# =============================================================================
st.markdown("### Configuración del Crédito")
themed_info(
    "Una **Tabla de Amortización** desglosa cada uno de los pagos de un crédito para mostrarte "
    "exactamente qué cantidad se destina a pagar los intereses y qué cantidad reduce tu deuda real (capital).<br><br>"
    "Aquí puedes calcular de cuánto será tu mensualidad (<span style='font-family: serif; font-style: italic;'>R</span>) si pides "
    "un préstamo (<span style='font-family: serif; font-style: italic;'>VP</span>), o viceversa."
)
separador()

col_modo, col_tasa = st.columns(2)

with col_modo:
    modo = st.radio(
        "¿Qué deseas calcular?",
        [
            "Pago Fijo (R)  →  conozco el Préstamo (VP)",
            "Préstamo (VP)  →  conozco el Pago Fijo (R)",
        ],
        key="modo_am",
    )

with col_tasa:
    tipo_tasa = st.radio(
        "Tipo de tasa:",
        ["Tasa efectiva periódica", "Tasa nominal anual"],
        horizontal=True,
        key="tipo_tasa_am",
    )

separador()

# =============================================================================
# INPUTS SEGÚN MODO
# =============================================================================
c1, c2, c3 = st.columns(3)

# ── Columna 1: monto principal ────────────────────────────────────────────────
with c1:
    if "Pago Fijo (R)  →  conozco el Préstamo (VP)" in modo:
        st.markdown("**Valor del bien y enganche**")
        vp_bruto = st.number_input(
            "Valor total del bien ($VP_{bruto}$)",
            min_value=0.01, value=500_000.0, step=10_000.0, key="am_vp_bruto"
        )

        tipo_eng = st.radio(
            "Tipo de enganche:",
            ["Monto fijo ($)", "Porcentaje (%)"],
            horizontal=True,
            key="tipo_eng",
        )

        if tipo_eng == "Monto fijo ($)":
            enganche = st.number_input(
                "Enganche ($)", min_value=0.0,
                value=50_000.0, step=5_000.0, key="am_eng_monto"
            )
        else:
            pct_eng  = st.number_input(
                "Enganche (%)", min_value=0.0,
                max_value=99.9, value=10.0, step=1.0, key="am_eng_pct"
            )
            enganche = vp_bruto * (pct_eng / 100)

        vp_neto = vp_bruto - enganche

        if vp_neto <= 0:
            themed_error("El enganche debe ser menor al valor total del bien.")
            st.stop()

        themed_info(f"**Monto a financiar (VP):** ${vp_neto:,.2f}")

    else:
        st.markdown("**Pago periódico conocido**")
        R_conocido = st.number_input(
            "Pago periódico fijo ($R$)",
            min_value=0.01, value=15_000.0, step=500.0, key="am_r"
        )

# ── Columna 3: plazos (Lo movemos antes para poder usar 'm_am' en la tasa) ───
with c3:
    st.markdown("**Plazos**")
    n_am = st.number_input(
        "Años del préstamo ($n$)",
        min_value=0.1, value=5.0, step=1.0, key="am_n"
    )
    m_am = st.number_input(
        "Pagos por año ($m$)",
        min_value=1.0, value=12.0, step=1.0, key="am_m"
    )

# ── Columna 2: tasa ──────────────────────────────────────────────────────────
with c2:
    st.markdown("**Tasa de interés**")

    if tipo_tasa == "Tasa efectiva periódica":
        tasa_input = st.number_input(
            "Tasa efectiva periódica ($i_m$) %",
            value=1.5, step=0.1, key="am_ieff"
        ) / 100
        str_tasa  = r"i_m"
        str_val_t = f"{tasa_input:.4f}"
    else:
        tasa_input = st.number_input(
            "Tasa nominal anual ($i^{(m)}$) %",
            value=18.0, step=0.1, key="am_inom"
        ) / 100
        str_tasa  = r"\frac{i^{(m)}}{m}"
        str_val_t = f"\\frac{{{tasa_input:.4f}}}{{{int(m_am)}}}"

# =============================================================================
# CÁLCULOS BASE
# =============================================================================
nm_am = int(n_am * m_am)

if tipo_tasa == "Tasa efectiva periódica":
    tasa_periodo = tasa_input
else:
    tasa_periodo = tasa_input / m_am

separador()

# =============================================================================
# RESULTADO PRINCIPAL: R o VP
# =============================================================================
st.markdown("### Cálculo Matemático")

col_form, col_res = st.columns([2, 1])

# ── Modo A: calcular R ───────────────────────────────────────────────────────
if "Pago Fijo (R)  →  conozco el Préstamo (VP)" in modo:

    if tasa_periodo > 0:
        pago_R = vp_neto * (tasa_periodo / (1 - (1 + tasa_periodo) ** (-nm_am)))
    else:
        pago_R = vp_neto / nm_am

    vp_final = vp_neto
    formula_amort = (
        r"R = VP \left[ \frac{" + str_tasa + r"}{1 - \left(1+"
        + str_tasa + r"\right)^{-nm}} \right]"
    )

    with col_form:
        themed_success(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>Pago Periódico Fijo (<span style='{math_style}'>R</span>)</span>"
            f"<span style='{css_valor}'>${pago_R:,.2f}</span>"
            f"</div>"
        )
        
        with paso_a_paso():
            st.latex(formula_amort)
            st.latex(
                rf"R = {vp_neto:,.2f} \left[ \frac{{{str_val_t}}}"
                rf"{{1 - \left(1+{str_val_t}\right)^{{-{nm_am:g}}}}} \right]"
            )
            if tasa_periodo > 0:
                v_n = (1 + tasa_periodo) ** (-nm_am)
                factor = tasa_periodo / (1 - v_n)
                st.latex(
                    rf"R = {vp_neto:,.2f} \left[ \frac{{{tasa_periodo:.6f}}}"
                    rf"{{1 - (1 + {tasa_periodo:.6f})^{{-{nm_am:g}}}}} \right]"
                )
                st.latex(rf"R = {vp_neto:,.2f} \left[ \frac{{{tasa_periodo:.6f}}}{{1 - {v_n:.6f}}} \right]")
                st.latex(rf"R = {vp_neto:,.2f} ({factor:.6f}) = {pago_R:.2f}")
                
            themed_success(f"<div style='{css_paso}'><span style='{math_style}'>R</span> = ${pago_R:,.2f}</div>")

    with col_res:
        st.metric("Total pagado al final", f"${pago_R * nm_am:,.2f}")
        st.metric("Total intereses generados", f"${(pago_R * nm_am) - vp_neto:,.2f}")

# ── Modo B: calcular VP ──────────────────────────────────────────────────────
else:

    if tasa_periodo > 0:
        vp_calc = R_conocido * ((1 - (1 + tasa_periodo) ** (-nm_am)) / tasa_periodo)
    else:
        vp_calc = R_conocido * nm_am

    pago_R   = R_conocido
    vp_final = vp_calc
    formula_amort = (
        r"VP = R \left[ \frac{1 - \left(1+"
        + str_tasa + r"\right)^{-nm}}{" + str_tasa + r"} \right]"
    )

    with col_form:
        themed_info(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>Préstamo a Financiar (<span style='{math_style}'>VP</span>)</span>"
            f"<span style='{css_valor}'>${vp_calc:,.2f}</span>"
            f"</div>"
        )
        
        with paso_a_paso():
            st.latex(formula_amort)
            st.latex(
                rf"VP = {R_conocido:,.2f} \left[ \frac{{1 - \left(1+{str_val_t}\right)"
                rf"^{{-{nm_am:g}}}}}{{{str_val_t}}} \right]"
            )
            if tasa_periodo > 0:
                v_n = (1 + tasa_periodo) ** (-nm_am)
                factor_vp = (1 - v_n) / tasa_periodo
                st.latex(
                    rf"VP = {R_conocido:,.2f} \left[ \frac{{1 - "
                    rf"(1 + {tasa_periodo:.6f})^{{-{nm_am:g}}}}}{{{tasa_periodo:.6f}}} \right]"
                )
                st.latex(rf"VP = {R_conocido:,.2f} \left[ \frac{{1 - {v_n:.6f}}}{{{tasa_periodo:.6f}}} \right]")
                st.latex(rf"VP = {R_conocido:,.2f} ({factor_vp:.6f}) = {vp_calc:.2f}")
                
            themed_info(f"<div style='{css_paso}'><span style='{math_style}'>VP</span> = ${vp_calc:,.2f}</div>")

    with col_res:
        st.metric("Total pagado al final", f"${pago_R * nm_am:,.2f}")
        st.metric("Total intereses generados", f"${(pago_R * nm_am) - vp_calc:,.2f}")

# =============================================================================
# TABLA Y GRÁFICA DE AMORTIZACIÓN
# =============================================================================
separador()
st.markdown("### Resultados Detallados")

# Asumimos que tu engine devuelve un DataFrame de pandas
df_amort = engine.tabla_amortizacion(vp_final, tasa_periodo, nm_am)

tab_tabla, tab_grafica = st.tabs(["Tabla de Amortización", "Composición del Pago"])

with tab_tabla:
    # Resumen rápido antes de la tabla
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Periodos totales",  f"{nm_am}")
    r2.metric("Pago fijo ($R$)",   f"${pago_R:,.2f}")
    r3.metric("Capital financiado", f"${vp_final:,.2f}")
    
    # Manejo seguro por si el DataFrame cambia de nombre de columna
    total_interes = df_amort['Interés'].sum() if 'Interés' in df_amort.columns else (pago_R * nm_am) - vp_final
    r4.metric("Intereses totales",  f"${total_interes:,.2f}")

    separador()

    # Formateo dinámico: Aplica formato de moneda a todas las columnas excepto 'Periodo'
    format_dict = {col: "${:,.2f}" for col in df_amort.columns if col != "Periodo"}
    
    st.dataframe(
        df_amort.style.format(format_dict),
        use_container_width=True,
        hide_index=True,
    )

with tab_grafica:
    if "Periodo" in df_amort.columns and "Amortización" in df_amort.columns and "Interés" in df_amort.columns:
        fig = px.bar(
            df_amort,
            x="Periodo",
            y=["Amortización", "Interés"],
            title="Composición de tu cuota a lo largo del tiempo",
            labels={"value": "Monto ($)", "variable": "Componente"},
            color_discrete_map={
                "Amortización": "#4ECDC4",
                "Interés":      "#FF6B6B",
            },
            template="none",
        )
        fig.update_layout(
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        
        try:
            fig = apply_plotly_theme(fig)
        except Exception:
            pass 
            
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("¿Cómo interpretar esta gráfica?"):
            st.markdown(
                "**El comportamiento clásico de un crédito:**\n\n"
                "Aunque el pago que le haces al banco siempre es de la misma cantidad, "
                "por dentro, su composición cambia con cada periodo que pasa:\n\n"
                "- **Al inicio:** El saldo de tu deuda es muy grande, por lo que casi todo tu pago se destina a cubrir intereses (zona roja). Se abona muy poco a la deuda real.\n"
                "- **Al final:** Como el saldo de la deuda ya es pequeño, los intereses bajan considerablemente y casi todo tu pago se va directamente a liquidar el capital (zona verde)."
            )
    else:
        themed_warning("No se pudo generar la gráfica. Asegúrate de que tu motor devuelve las columnas: 'Periodo', 'Amortización' e 'Interés'.")