"""
pages/13_Derivados_de_Credito_CDS.py
--------------------------------------
Módulo 13: Valuación de Credit Default Swaps (CDS).
Orden pedagógico:
  1. Conceptos y mecánica del CDS
  2. Cálculo de probabilidades de incumplimiento y sobrevivencia
  3. VPC_CDS — Valor Presente de la Pata Fija (comprador)
  4. VPV_CDS — Valor Presente de la Pata Contingente (vendedor)
  5. VPPP_CDS — Valor Presente Prima Prorrateada
  6. Prima (spread) CDS = s
  7. Valuación a mercado (mark-to-market)
  8. Análisis de sensibilidad: spread vs Recovery Rate
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import (
    get_engine, page_header, paso_a_paso, separador,
    themed_info, themed_success, themed_warning, themed_error,
    apply_plotly_theme, plotly_theme, plotly_colors, plotly_color,
    get_current_theme,
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Derivados de Crédito CDS · Calculadora Financiera",
    page_icon="🛡️",
    layout="wide",
)

engine = get_engine()

# --- Estilos globales ---
math_style     = "font-family: 'Times New Roman', Times, serif; font-style: italic; font-weight: normal; padding: 0 2px;"
css_titulo     = "font-size: 20px; opacity: 0.85; font-weight: 500;"
css_valor      = "font-size: 28px; font-weight: bold;"
css_contenedor = "display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 12px 0;"
css_paso       = "text-align: center; font-size: 22px; font-weight: bold; padding: 4px 0; margin: 0;"

# Variante para columnas estrechas (Agregado)
css_contenedor_col = "display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; padding: 8px 0;"
css_titulo_col = "font-size: 16px; opacity: 0.85; font-weight: 500; margin-bottom: 4px; text-align: center;"
css_valor_col = "font-size: 24px; font-weight: bold; text-align: center;"

page_header(
    titulo="13. Derivados de Crédito — CDS",
    subtitulo="Credit Default Swap · Prima CDS · Pata Fija · Pata Contingente · Mark-to-Market"
)

# =============================================================================
# PESTAÑAS
# =============================================================================
tab_prima, tab_mtm, tab_sens = st.tabs([
    "Prima del CDS (Pricing)",
    "Valuación a Mercado (MTM)",
    "Análisis de Sensibilidad",
])


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES DE CÁLCULO
# ─────────────────────────────────────────────────────────────────────────────

def tabla_probabilidades(hazard_rate: float, T: int) -> pd.DataFrame:
    """
    Genera tabla de probabilidades anuales.
    Columnas: t, S(t) Sobrevivencia, F(t) Incump. Acumulada, q(t) Incump. Marginal
    """
    rows = []
    for t in range(1, T + 1):
        ps_t    = np.exp(-hazard_rate * t)
        ps_prev = np.exp(-hazard_rate * (t - 1))
        pd_acu  = 1.0 - ps_t          # F(t) acumulada
        pd_marg = ps_prev - ps_t      # q(t) marginal individual
        rows.append({
            "t (años)":               t,
            "S(t) Sobrevivencia":     ps_t,
            "F(t) Incump. Acumulada": pd_acu,
            "q(t) Incump. Marginal":  pd_marg,
        })
    return pd.DataFrame(rows)


def tabla_vpc_cds(hazard_rate: float, r: float, T: int) -> pd.DataFrame:
    """
    VPC_CDS — Valor Presente de Pagos del Comprador SIN incumplimiento.
    Pago esperado en t = S(t) * s  →  VP = S(t) * s * e^{-r*t}
    (el spread s se factoriza fuera, se trabaja con 1 unidad de s)
    """
    rows = []
    for t in range(1, T + 1):
        ps_t  = np.exp(-hazard_rate * t)
        fvp_t = np.exp(-r * t)
        vp_t  = ps_t * fvp_t
        rows.append({
            "Tiempo (años)":           t,
            "Prob. Sobrevivencia S(t)": ps_t,
            "Pago Esperado (×s)":      ps_t,
            "Factor VP  e^{-rt}":      fvp_t,
            "VP Pago Esperado (×s)":   vp_t,
        })
    df = pd.DataFrame(rows)
    df.loc["Total"] = df[["VP Pago Esperado (×s)"]].sum()
    return df


def tabla_vpv_cds(hazard_rate: float, r: float, T: int, rr: float) -> pd.DataFrame:
    """
    VPV_CDS — Valor Presente de Pagos del Vendedor (pata contingente).
    Pagos en los puntos medios t - 0.5; LGD = 1 - RR.
    """
    lgd = 1.0 - rr
    rows = []
    for t in range(1, T + 1):
        t_mid   = t - 0.5
        ps_prev = np.exp(-hazard_rate * (t - 1))
        ps_t    = np.exp(-hazard_rate * t)
        pd_cond = ps_prev - ps_t
        pago_p  = lgd * pd_cond
        fvp     = np.exp(-r * t_mid)
        vp_pago = pago_p * fvp
        rows.append({
            "Tiempo (años)":                      t_mid,
            "Prob. Incumplimiento Marginal q(t)": pd_cond,
            "Tasa Recuperación (RR)":             rr,
            "LGD (1-RR)":                         lgd,
            "Pago Parcial Esperado":              pago_p,
            "Factor VP  e^{-rt_mid}":             fvp,
            "VP Pago Parcial Esperado":           vp_pago,
        })
    df = pd.DataFrame(rows)
    df.loc["Total"] = df[["VP Pago Parcial Esperado"]].sum()
    return df


def tabla_vppp_cds(hazard_rate: float, r: float, T: int) -> pd.DataFrame:
    """
    VPPP_CDS — Valor Presente de la Prima Prorrateada.
    En caso de incumplimiento en el año t, el comprador paga
    la prima acumulada hasta el punto medio (0.5 × s).
    """
    rows = []
    for t in range(1, T + 1):
        t_mid   = t - 0.5
        ps_prev = np.exp(-hazard_rate * (t - 1))
        ps_t    = np.exp(-hazard_rate * t)
        pd_cond = ps_prev - ps_t
        pago_p  = 0.5 * pd_cond
        fvp     = np.exp(-r * t_mid)
        vp_pago = pago_p * fvp
        rows.append({
            "Tiempo (años)":                          t_mid,
            "Prob. Incumplimiento Marginal q(t)":     pd_cond,
            "Pago Prorrateado (×s)":                  0.5,
            "Pago Prorrateado Esperado (×s)":         pago_p,
            "Factor VP  e^{-rt_mid}":                 fvp,
            "VP Pago Prorrateado Esperado (×s)":      vp_pago,
        })
    df = pd.DataFrame(rows)
    df.loc["Total"] = df[["VP Pago Prorrateado Esperado (×s)"]].sum()
    return df


def prima_cds(vpc_total: float, vppp_total: float, vpv_total: float) -> float:
    """
    s = VPV_CDS / (VPC_CDS + VPPP_CDS)
    Condición de no-arbitraje: Pata fija = Pata contingente.
    """
    denominador = vpc_total + vppp_total
    if denominador == 0:
        return 0.0
    return vpv_total / denominador


def format_pct(v, decimales=4):
    return f"{v*100:.{decimales}f}%"


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRIMA DEL CDS
# ═════════════════════════════════════════════════════════════════════════════
with tab_prima:
    st.markdown("### Valuación del Credit Default Swap (CDS)")

    themed_info(
        "Un **Credit Default Swap (CDS)** es un contrato de protección crediticia. "
        f"El **comprador** paga una prima periódica (<span style='{math_style}'>s</span>) "
        "y el **vendedor** paga el Nocional × LGD en caso de que el emisor de referencia incumpla. "
        "La prima de equilibrio se obtiene igualando la **Pata Fija** (comprador) "
        "con la **Pata Contingente** (vendedor): "
        "<br><br>"
        "<div style='text-align:center;font-family:serif;font-size:1.1em;'>"
        "VPC<sub>CDS</sub> + VPPP<sub>CDS</sub> = VPV<sub>CDS</sub>"
        "</div>"
    )

    separador()

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.markdown("#### Parámetros del CDS")

    col_riesgo, col_rr, col_r, col_T = st.columns(4)

    with col_riesgo:
        lam_input = st.number_input(
            "λ — Hazard Rate (%)",
            min_value=0.001, max_value=50.0, value=0.14, step=0.001,
            format="%.3f", key="cds_lam_direct",
            help=(
                "Tasa de riesgo instantánea continua bajo el supuesto de distribución "
                "exponencial del tiempo de incumplimiento.\n\n"
                "S(t) = e^{−λt} | F(t) = 1 − e^{−λt}"
            ),
        ) / 100.0

    with col_rr:
        rr_input = st.number_input(
            "RR — Recovery Rate (%)",
            min_value=0.0, max_value=99.99, value=80.0, step=1.0,
            format="%.2f", key="cds_rr",
            help="Porcentaje del nocional que se recupera en caso de default."
        ) / 100.0

    with col_r:
        r_input = st.number_input(
            "r — Tasa Libre de Riesgo (%)",
            min_value=0.001, max_value=30.0, value=5.0, step=0.1,
            format="%.3f", key="cds_r",
            help="Tasa de descuento libre de riesgo (continua). Se usa e^{−rt}."
        ) / 100.0

    with col_T:
        T_input = st.number_input(
            "T — Plazo (años)",
            min_value=1, max_value=30, value=5, step=1,
            key="cds_T",
            help="Vencimiento del contrato CDS en años enteros."
        )

    # ── Variables derivadas ───────────────────────────────────────────────────
    lam = lam_input
    lgd = 1.0 - rr_input

    separador()

    # ── Paso 1 · Hazard Rate ──────────────────────────────────────────────────
    st.markdown("#### Paso 1 · Hazard Rate (λ)")

    themed_info(
        "La **hazard rate** (tasa de riesgo instantánea) define la distribución exponencial "
        "del tiempo de incumplimiento t*:"
        "<br><br>"
        "<div style='text-align:center;font-family:serif;font-size:1.05em;'>"
        "P[t* &lt; t] = F(t) = 1 − e<sup>−λt</sup> &nbsp;&nbsp;|&nbsp;&nbsp; "
        "P[t* ≥ t] = S(t) = e<sup>−λt</sup>"
        "</div>"
    )

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'><span style='{math_style}'>&lambda;</span> — Hazard Rate</span>"
            f"<span style='{css_valor_col}'>{lam*100:.4f}%</span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'><span style='{math_style}'>S(1)</span> — Prob. sobrevivir 1 año</span>"
            f"<span style='{css_valor_col}'>{np.exp(-lam)*100:.4f}%</span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col_h3:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>LGD = 1 &minus; RR</span>"
            f"<span style='{css_valor_col}'>{format_pct(lgd, 2)}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )

    with paso_a_paso("Ver desarrollo — Hazard Rate"):
        st.markdown("**Distribución del tiempo de incumplimiento:**")
        st.latex(r"P[t^* < t] = F(t) = 1 - e^{-\lambda t}")
        st.latex(r"P[t^* \geq t] = S(t) = e^{-\lambda t}")
        st.write("---")
        st.markdown(f"**Con λ = {lam*100:.4f}%**, probabilidades de sobrevivencia:")
        for t in range(1, min(int(T_input), 5) + 1):
            ps_t = np.exp(-lam * t)
            st.latex(
                rf"S({t}) = e^{{-{lam:.6f} \times {t}}} = {ps_t:.6f}"
            )
        themed_success(
            f"<div style='{css_paso}'>λ = {lam*100:.4f}% &nbsp;|&nbsp; LGD = {lgd*100:.2f}%</div>"
        )

    separador()

    # ── Paso 2 · Tabla de Probabilidades ──────────────────────────────────────
    st.markdown("#### Paso 2 · Tabla de Probabilidades")

    themed_info(
        "**S(t) = e<sup>−λt</sup>** — probabilidad de sobrevivir hasta el año t. &nbsp;|&nbsp; "
        "**F(t) = 1 − S(t)** — probabilidad <i>acumulada</i> de incumplimiento hasta t. &nbsp;|&nbsp; "
        "**q(t) = S(t−1) − S(t)** — probabilidad <i>marginal</i>: incumplir exactamente en el año t."
    )

    df_probs = tabla_probabilidades(lam, int(T_input))
    df_probs_show = df_probs.set_index("t (años)")

    st.dataframe(
        df_probs_show.style.format({
            "S(t) Sobrevivencia":     "{:.6f}",
            "F(t) Incump. Acumulada": "{:.6f}",
            "q(t) Incump. Marginal":  "{:.6f}",
        }).bar(
            subset=["S(t) Sobrevivencia"],
            color="#2ecc71", vmin=0, vmax=1,
        ).bar(
            subset=["F(t) Incump. Acumulada", "q(t) Incump. Marginal"],
            color="#e74c3c", vmin=0, vmax=df_probs["F(t) Incump. Acumulada"].max(),
        ),
        use_container_width=True,
    )

    with paso_a_paso("Ver desarrollo — Tabla de Probabilidades"):
        st.latex(r"S(t) = e^{-\lambda \cdot t} \quad \text{(sobrevivencia acumulada)}")
        st.latex(r"F(t) = 1 - S(t) \quad \text{(incumplimiento acumulado hasta t)}")
        st.latex(r"q(t) = S(t-1) - S(t) \quad \text{(incumplimiento marginal en el año t)}")
        st.markdown("---")
        st.markdown(
            "**Diferencia clave:** F(t) es la probabilidad de haber incumplido *en cualquier momento* "
            "hasta t; q(t) es la probabilidad de incumplir *exactamente en el año t*, "
            "dado que sobrevivió hasta t−1. Nótese que Σ q(t) = F(T)."
        )
        st.markdown("**Ejemplo numérico:**")
        for t in range(1, min(int(T_input), 3) + 1):
            ps_t    = np.exp(-lam * t)
            ps_prev = np.exp(-lam * (t - 1))
            pd_marg = ps_prev - ps_t
            st.latex(
                rf"t={t}:\quad "
                rf"S({t}) = e^{{-{lam:.6f} \times {t}}} = {ps_t:.6f},\quad "
                rf"F({t}) = {1 - ps_t:.6f},\quad "
                rf"q({t}) = {pd_marg:.6f}"
            )

    separador()

    # ── Paso 3 · VPC_CDS ──────────────────────────────────────────────────────
    st.markdown("#### Paso 3 · VPC_CDS — Pata Fija (Comprador)")

    themed_info(
        f"El comprador paga la prima <span style='{math_style}'>s</span> "
        "al final de cada año, <b>siempre que no haya habido incumplimiento</b>. "
        "El pago esperado en el año t es S(t)·s, y su VP es:"
        "<br><br>"
        "<div style='text-align:center;font-family:serif;font-size:1.05em;'>"
        "VPC<sub>CDS</sub> = s · Σ S(t) · e<sup>−r·t</sup>"
        "</div>"
    )

    df_vpc = tabla_vpc_cds(lam, r_input, int(T_input))
    vpc_total = df_vpc.loc["Total", "VP Pago Esperado (×s)"]

    st.dataframe(
        df_vpc.style.format({
            "Prob. Sobrevivencia S(t)": "{:.6f}",
            "Pago Esperado (×s)":       "{:.6f}",
            "Factor VP  e^{-rt}":       "{:.6f}",
            "VP Pago Esperado (×s)":    "{:.6f}",
        }, na_rep=""),
        use_container_width=True,
    )

    themed_success(
        f"<div style='{css_paso}'>VPC<sub>CDS</sub> = <b>{vpc_total:.6f} &times; <span style='{math_style}'>s</span></b></div>"
    )

    with paso_a_paso("Ver desarrollo — VPC_CDS"):
        st.latex(r"\text{VPC}_{\text{CDS}} = s \cdot \sum_{t=1}^{T} S(t) \cdot e^{-r \cdot t}")
        st.write("Desglose para cada año:")
        for t in range(1, int(T_input) + 1):
            ps_t  = np.exp(-lam * t)
            fvp_t = np.exp(-r_input * t)
            vp_t  = ps_t * fvp_t
            st.latex(
                rf"t={t}: \quad S({t}) \cdot e^{{-r \cdot {t}}} = "
                rf"{ps_t:.6f} \times {fvp_t:.6f} = {vp_t:.6f} \cdot s"
            )
        st.write("---")
        themed_success(
            f"<div style='{css_paso}'>VPC<sub>CDS</sub> = {vpc_total:.6f} &times; <span style='{math_style}'>s</span></div>"
        )

    separador()

    # ── Paso 4 · VPV_CDS ──────────────────────────────────────────────────────
    st.markdown("#### Paso 4 · VPV_CDS — Pata Contingente (Vendedor)")

    themed_info(
        "El vendedor paga LGD = (1 − RR) si ocurre el incumplimiento. Se asume que "
        "el default ocurre en el <b>punto medio</b> del período (t − 0.5). El VP del pago esperado es:"
        "<br><br>"
        "<div style='text-align:center;font-family:serif;font-size:1.05em;'>"
        "VPV<sub>CDS</sub> = Σ q(t) · LGD · e<sup>−r·(t−0.5)</sup>"
        "</div>"
    )

    df_vpv = tabla_vpv_cds(lam, r_input, int(T_input), rr_input)
    vpv_total = df_vpv.loc["Total", "VP Pago Parcial Esperado"]

    st.dataframe(
        df_vpv.style.format({
            "Prob. Incumplimiento Marginal q(t)": "{:.6f}",
            "Tasa Recuperación (RR)":             "{:.0%}",
            "LGD (1-RR)":                         "{:.0%}",
            "Pago Parcial Esperado":              "{:.6f}",
            "Factor VP  e^{-rt_mid}":             "{:.6f}",
            "VP Pago Parcial Esperado":           "{:.6f}",
        }, na_rep=""),
        use_container_width=True,
    )

    themed_error(
        f"<div style='{css_paso}'>VPV<sub>CDS</sub> = <b>{vpv_total:.6f}</b></div>"
    )

    with paso_a_paso("Ver desarrollo — VPV_CDS"):
        st.latex(r"\text{VPV}_{\text{CDS}} = \sum_{t=1}^{T} q(t) \cdot (1-RR) \cdot e^{-r \cdot (t-0.5)}")
        st.latex(rf"LGD = 1 - RR = 1 - {rr_input:.2f} = {lgd:.2f}")
        st.write("Desglose para cada período:")
        for t in range(1, int(T_input) + 1):
            t_mid   = t - 0.5
            ps_prev = np.exp(-lam * (t - 1))
            ps_t    = np.exp(-lam * t)
            pd_cond = ps_prev - ps_t
            pago_p  = lgd * pd_cond
            fvp     = np.exp(-r_input * t_mid)
            vp_pago = pago_p * fvp
            st.latex(
                rf"t={t}: \quad q({t}) \cdot LGD \cdot e^{{-r \cdot {t_mid}}} = "
                rf"{pd_cond:.6f} \times {lgd:.4f} \times {fvp:.6f} = {vp_pago:.6f}"
            )
        st.write("---")
        themed_error(
            f"<div style='{css_paso}'>VPV<sub>CDS</sub> = {vpv_total:.6f}</div>"
        )

    separador()

    # ── Paso 5 · VPPP_CDS ─────────────────────────────────────────────────────
    st.markdown("#### Paso 5 · VPPP_CDS — Prima Prorrateada (Accrued)")

    themed_info(
        "Si el default ocurre en el año t (en el punto medio t−0.5), el comprador debe "
        f"pagar la prima acumulada hasta ese punto: <b>0.5 &times; <span style='{math_style}'>s</span></b> por cada período. "
        "El VP de estos pagos es:"
        "<br><br>"
        "<div style='text-align:center;font-family:serif;font-size:1.05em;'>"
        "VPPP<sub>CDS</sub> = s · Σ 0.5 · q(t) · e<sup>−r·(t−0.5)</sup>"
        "</div>"
    )

    df_vppp = tabla_vppp_cds(lam, r_input, int(T_input))
    vppp_total = df_vppp.loc["Total", "VP Pago Prorrateado Esperado (×s)"]

    st.dataframe(
        df_vppp.style.format({
            "Prob. Incumplimiento Marginal q(t)":  "{:.6f}",
            "Pago Prorrateado (×s)":               "{:.1f}",
            "Pago Prorrateado Esperado (×s)":      "{:.6f}",
            "Factor VP  e^{-rt_mid}":              "{:.6f}",
            "VP Pago Prorrateado Esperado (×s)":   "{:.6f}",
        }, na_rep=""),
        use_container_width=True,
    )

    themed_warning(
        f"<div style='{css_paso}'>VPPP<sub>CDS</sub> = <b>{vppp_total:.6f} &times; <span style='{math_style}'>s</span></b></div>"
    )

    with paso_a_paso("Ver desarrollo — VPPP_CDS"):
        st.latex(r"\text{VPPP}_{\text{CDS}} = s \cdot \sum_{t=1}^{T} 0.5 \cdot q(t) \cdot e^{-r \cdot (t-0.5)}")
        st.write("Desglose para cada período:")
        for t in range(1, int(T_input) + 1):
            t_mid   = t - 0.5
            ps_prev = np.exp(-lam * (t - 1))
            ps_t    = np.exp(-lam * t)
            pd_cond = ps_prev - ps_t
            pago_p  = 0.5 * pd_cond
            fvp     = np.exp(-r_input * t_mid)
            vp_pago = pago_p * fvp
            st.latex(
                rf"t={t}: \quad 0.5 \cdot q({t}) \cdot e^{{-r \cdot {t_mid}}} = "
                rf"0.5 \times {pd_cond:.6f} \times {fvp:.6f} = {vp_pago:.6f} \cdot s"
            )
        st.write("---")
        themed_warning(
            f"<div style='{css_paso}'>VPPP<sub>CDS</sub> = {vppp_total:.6f} &times; <span style='{math_style}'>s</span></div>"
        )

    separador()

    # ── Paso 6 · Prima s ──────────────────────────────────────────────────────
    st.markdown("#### Paso 6 · Prima del CDS — Spread *s*")

    themed_info(
        "La condición de **no-arbitraje** establece que el valor del CDS en el inicio es cero, "
        "es decir, el VP de la Pata Fija (comprador) debe igualar el VP de la Pata Contingente (vendedor):"
        "<br><br>"
        "<div style='text-align:center;font-family:serif;font-size:1.1em;'>"
        "(VPC<sub>CDS</sub> + VPPP<sub>CDS</sub>) · s = VPV<sub>CDS</sub>"
        "<br><br>"
        "s = VPV<sub>CDS</sub> / (VPC<sub>CDS</sub> + VPPP<sub>CDS</sub>)"
        "</div>"
    )

    s_prima = prima_cds(vpc_total, vppp_total, vpv_total)
    s_pb    = s_prima * 10_000  # en puntos base

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>VPC<sub>CDS</sub></span>"
            f"<span style='{css_valor_col}'>{vpc_total:.6f} &times; <span style='{math_style}'>s</span></span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col_p2:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>VPPP<sub>CDS</sub></span>"
            f"<span style='{css_valor_col}'>{vppp_total:.6f} &times; <span style='{math_style}'>s</span></span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col_p3:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>VPV<sub>CDS</sub></span>"
            f"<span style='{css_valor_col}'>{vpv_total:.6f}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col_p4:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>Pata Fija Total</span>"
            f"<span style='{css_valor_col}'>{vpc_total + vppp_total:.6f} &times; <span style='{math_style}'>s</span></span>"
            f"</div>", 
            unsafe_allow_html=True
        )

    separador()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        themed_success(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>Prima CDS (<span style='{math_style}'>s</span>)</span>"
            f"<span style='{css_valor}'>{s_prima*100:.4f}% &nbsp;=&nbsp; {s_pb:.1f} pb</span>"
            f"</div>"
        )
    with col_s2:
        themed_info(
            f"Es decir, la prima anual del CDS sería de <b>{round(s_pb)} puntos base</b>. "
            f"El comprador paga <b>{s_prima*100:.4f}%</b> del nocional cada año a cambio de "
            f"protección ante el incumplimiento del emisor de referencia."
        )

    with paso_a_paso("Ver desarrollo — Prima s"):
        st.latex(r"s = \frac{\text{VPV}_{\text{CDS}}}{\text{VPC}_{\text{CDS}} + \text{VPPP}_{\text{CDS}}}")
        st.latex(
            rf"s = \frac{{{vpv_total:.6f}}}{{{vpc_total:.6f} + {vppp_total:.6f}}}"
            rf" = \frac{{{vpv_total:.6f}}}{{{vpc_total + vppp_total:.6f}}}"
            rf" = {s_prima:.6f}"
        )
        st.latex(
            rf"s = {s_prima*100:.4f}\% = {s_pb:.2f} \text{{ puntos base}}"
        )
        themed_success(
            f"<div style='{css_paso}'>"
            f"<span style='{math_style}'>s</span> = {s_prima*100:.4f}% = <b>{round(s_pb, 1)} pb</b>"
            f"</div>"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALUACIÓN A MERCADO (MTM)
# ═════════════════════════════════════════════════════════════════════════════
with tab_mtm:
    st.markdown("### Valuación a Mercado (Mark-to-Market)")

    themed_info(
        f"Si el CDS fue emitido con una prima <span style='{math_style}'>s<sub>0</sub></span> y ahora el mercado cotiza una prima "
        f"<span style='{math_style}'>s<sub>1</sub></span> diferente, el contrato tiene un **valor de mercado** (positivo o negativo). "
        "La variación en la Pata Contingente refleja el cambio en el riesgo de crédito percibido:"
        "<br><br>"
        "<div style='text-align:center;font-family:serif;font-size:1.05em;'>"
        "MTM = (s<sub>1</sub> − s<sub>0</sub>) × (VPC<sub>CDS</sub> + VPPP<sub>CDS</sub>)"
        "</div>"
    )

    separador()

    st.markdown("#### Parámetros MTM")
    st.caption("Los parámetros de hazard rate, RR y T se toman de la pestaña **Prima del CDS**.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        s0_pb = st.number_input(
            "s₀ — Prima original del CDS (puntos base)",
            min_value=0.0, value=15.0, step=0.5, format="%.1f",
            key="cds_s0",
            help="La prima en puntos base a la que se emitió originalmente el CDS."
        )
    with col_m2:
        s1_pb = st.number_input(
            "s₁ — Prima actual de mercado (puntos base)",
            min_value=0.0, value=float(round(s_pb, 1)), step=0.5, format="%.1f",
            key="cds_s1",
            help="La prima CDS que cotiza actualmente el mercado para el mismo riesgo."
        )

    separador()

    s0 = s0_pb / 10_000
    s1 = s1_pb / 10_000

    pata_fija = vpc_total + vppp_total
    vpc_s0    = pata_fija * s0
    vpc_s1    = pata_fija * s1
    mtm       = (s1 - s0) * pata_fija

    col_mt1, col_mt2, col_mt3 = st.columns(3)
    
    with col_mt1:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>Pata Fija (&times;1)</span>"
            f"<span style='{css_valor_col}'>{pata_fija:.6f}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col_mt2:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>Pata Fija con <span style='{math_style}'>s<sub>0</sub></span></span>"
            f"<span style='{css_valor_col}'>{vpc_s0:.6f}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )
    with col_mt3:
        st.markdown(
            f"<div style='{css_contenedor_col}'>"
            f"<span style='{css_titulo_col}'>Pata Fija con <span style='{math_style}'>s<sub>1</sub></span></span>"
            f"<span style='{css_valor_col}'>{vpc_s1:.6f}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )

    separador()

    if mtm >= 0:
        themed_success(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>MTM del CDS (posición compradora)</span>"
            f"<span style='{css_valor}'>+{mtm:.6f}</span>"
            f"</div>"
        )
        themed_info(
            f"El spread de mercado subió de <b>{s0_pb:.1f} pb → {s1_pb:.1f} pb</b>. "
            f"El comprador de protección <b>gana {mtm:.6f} por unidad de nocional</b> "
            f"porque ahora la protección es más cara en el mercado."
        )
    else:
        themed_error(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>MTM del CDS (posición compradora)</span>"
            f"<span style='{css_valor}'>{mtm:.6f}</span>"
            f"</div>"
        )
        themed_info(
            f"El spread de mercado bajó de <b>{s0_pb:.1f} pb → {s1_pb:.1f} pb</b>. "
            f"El comprador de protección <b>pierde {abs(mtm):.6f} por unidad de nocional</b> "
            f"porque ahora la protección es más barata en el mercado."
        )

    with paso_a_paso("Ver desarrollo — MTM"):
        st.latex(r"\text{MTM} = (s_1 - s_0) \times (\text{VPC}_{\text{CDS}} + \text{VPPP}_{\text{CDS}})")
        st.latex(
            rf"\text{{MTM}} = ({s1:.6f} - {s0:.6f}) \times {pata_fija:.6f}"
        )
        st.latex(
            rf"\text{{MTM}} = {s1 - s0:.6f} \times {pata_fija:.6f} = {mtm:.6f}"
        )
        if mtm >= 0:
            themed_success(f"<div style='{css_paso}'>MTM = +{mtm:.6f}</div>")
        else:
            themed_error(f"<div style='{css_paso}'>MTM = {mtm:.6f}</div>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANÁLISIS DE SENSIBILIDAD
# ═════════════════════════════════════════════════════════════════════════════
with tab_sens:
    st.markdown("### Análisis de Sensibilidad")

    themed_info(
        "Exploramos cómo varía el **spread del CDS** ante cambios en el "
        "**Recovery Rate (RR)** y en la **Hazard Rate (λ)**. "
        "Mientras mayor sea la pérdida esperada (menor RR o mayor λ), "
        "mayor será la prima que el comprador debe pagar."
    )

    separador()

    tab_s1, tab_s2 = st.tabs([
        "Spread vs Recovery Rate",
        "Spread vs Hazard Rate",
    ])

    # ── Spread vs RR ──────────────────────────────────────────────────────────
    with tab_s1:
        st.markdown("#### Spread CDS en función del Recovery Rate")
        themed_info(
            "Con RR = 100% (recuperación total), el vendedor nunca pierde → spread ≈ 0. "
            "Con RR = 0% (pérdida total), el spread es máximo. "
            "La relación es **lineal** en el LGD."
        )

        rr_vals = np.linspace(0, 0.999, 200)
        spreads_rr = []
        for rr_v in rr_vals:
            df_vpv_s  = tabla_vpv_cds(lam, r_input, int(T_input), rr_v)
            df_vppp_s = tabla_vppp_cds(lam, r_input, int(T_input))
            df_vpc_s  = tabla_vpc_cds(lam, r_input, int(T_input))
            vpv_s     = df_vpv_s.loc["Total", "VP Pago Parcial Esperado"]
            vppp_s    = df_vppp_s.loc["Total", "VP Pago Prorrateado Esperado (×s)"]
            vpc_s_v   = df_vpc_s.loc["Total", "VP Pago Esperado (×s)"]
            s_v       = prima_cds(vpc_s_v, vppp_s, vpv_s) * 10_000
            spreads_rr.append(s_v)

        c_tema2 = get_current_theme()
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Scatter(
            x=rr_vals * 100,
            y=spreads_rr,
            mode="lines",
            name="Spread CDS",
            line=dict(color=c_tema2["primary"], width=2.5),
        ))
        fig_rr.add_trace(go.Scatter(
            x=[rr_input * 100],
            y=[s_pb],
            mode="markers",
            name=f"RR actual = {rr_input*100:.0f}% → s = {s_pb:.1f} pb",
            marker=dict(color=c_tema2["accent"], size=10, symbol="circle"),
        ))
        fig_rr.update_layout(
            **plotly_theme(),
            title=dict(
                text=f"Spread CDS (pb) vs Recovery Rate | λ={lam*100:.4f}% | T={int(T_input)}a | r={r_input*100:.2f}%",
                font=dict(size=13),
            ),
            xaxis_title="Recovery Rate (%)",
            yaxis_title="Spread CDS (puntos base)",
            height=420,
            margin=dict(l=50, r=20, t=60, b=42),
        )
        st.plotly_chart(fig_rr, use_container_width=True)

        # Tabla resumen cada 5%
        rr_tbl = np.arange(0, 1.01, 0.05)
        rows_tbl = []
        for rr_v in rr_tbl:
            if rr_v >= 1.0:
                rr_v = 0.999
            df_vpv_s  = tabla_vpv_cds(lam, r_input, int(T_input), rr_v)
            df_vppp_s = tabla_vppp_cds(lam, r_input, int(T_input))
            df_vpc_s  = tabla_vpc_cds(lam, r_input, int(T_input))
            vpv_s     = df_vpv_s.loc["Total", "VP Pago Parcial Esperado"]
            vppp_s    = df_vppp_s.loc["Total", "VP Pago Prorrateado Esperado (×s)"]
            vpc_s_v   = df_vpc_s.loc["Total", "VP Pago Esperado (×s)"]
            s_v_pb    = prima_cds(vpc_s_v, vppp_s, vpv_s) * 10_000
            rows_tbl.append({"RR": f"{rr_v*100:.0f}%", "Spread CDS (pb)": round(s_v_pb)})

        df_tbl_rr = pd.DataFrame(rows_tbl)
        rr_ref = f"{round(rr_input * 100):.0f}%"

        def highlight_rr(row):
            if row["RR"] == rr_ref:
                color = c_tema2["warning_bg"]
                return [f"background-color: {color}"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_tbl_rr.style.apply(highlight_rr, axis=1),
            use_container_width=False,
            height=380,
        )

    # ── Spread vs λ ───────────────────────────────────────────────────────────
    with tab_s2:
        st.markdown("#### Spread CDS en función de la Hazard Rate (λ)")
        themed_info(
            "Cuanto mayor sea la **hazard rate**, mayor es la probabilidad de incumplimiento "
            "y por tanto mayor es la prima que exige el comprador de protección. "
            "La relación no es perfectamente lineal debido al efecto de los descuentos."
        )

        lam_vals = np.linspace(0.0001, 0.20, 200)
        spreads_lam = []
        for lam_v in lam_vals:
            df_vpv_s  = tabla_vpv_cds(lam_v, r_input, int(T_input), rr_input)
            df_vppp_s = tabla_vppp_cds(lam_v, r_input, int(T_input))
            df_vpc_s  = tabla_vpc_cds(lam_v, r_input, int(T_input))
            vpv_s     = df_vpv_s.loc["Total", "VP Pago Parcial Esperado"]
            vppp_s    = df_vppp_s.loc["Total", "VP Pago Prorrateado Esperado (×s)"]
            vpc_s_v   = df_vpc_s.loc["Total", "VP Pago Esperado (×s)"]
            s_v_pb    = prima_cds(vpc_s_v, vppp_s, vpv_s) * 10_000
            spreads_lam.append(s_v_pb)

        fig_lam = go.Figure()
        fig_lam.add_trace(go.Scatter(
            x=lam_vals * 100,
            y=spreads_lam,
            mode="lines",
            name="Spread CDS",
            line=dict(color=c_tema2["success"], width=2.5),
        ))
        fig_lam.add_trace(go.Scatter(
            x=[lam * 100],
            y=[s_pb],
            mode="markers",
            name=f"λ actual = {lam*100:.4f}% → s = {s_pb:.1f} pb",
            marker=dict(color=c_tema2["accent"], size=10, symbol="circle"),
        ))
        fig_lam.update_layout(
            **plotly_theme(),
            title=dict(
                text=f"Spread CDS (pb) vs Hazard Rate | RR={rr_input*100:.0f}% | T={int(T_input)}a | r={r_input*100:.2f}%",
                font=dict(size=13),
            ),
            xaxis_title="Hazard Rate λ (%)",
            yaxis_title="Spread CDS (puntos base)",
            height=420,
            margin=dict(l=50, r=20, t=60, b=42),
        )
        st.plotly_chart(fig_lam, use_container_width=True)

        # Mapa de calor: RR x λ → spread
        st.markdown("#### Mapa de Calor: Spread CDS (pb) por RR × λ")
        themed_info(
            "Cada celda muestra el spread del CDS para una combinación de Recovery Rate y Hazard Rate. "
            "Las zonas oscuras (alto spread) corresponden a emisores de alto riesgo con baja recuperación."
        )

        rr_hm  = np.arange(0.0, 1.0, 0.10)
        lam_hm = np.arange(0.0005, 0.051, 0.005)
        z_hm   = []
        for lam_v in lam_hm:
            row_hm = []
            for rr_v in rr_hm:
                df_vpv_s  = tabla_vpv_cds(lam_v, r_input, int(T_input), rr_v)
                df_vppp_s = tabla_vppp_cds(lam_v, r_input, int(T_input))
                df_vpc_s  = tabla_vpc_cds(lam_v, r_input, int(T_input))
                vpv_s     = df_vpv_s.loc["Total", "VP Pago Parcial Esperado"]
                vppp_s    = df_vppp_s.loc["Total", "VP Pago Prorrateado Esperado (×s)"]
                vpc_s_v   = df_vpc_s.loc["Total", "VP Pago Esperado (×s)"]
                s_pb_hm   = prima_cds(vpc_s_v, vppp_s, vpv_s) * 10_000
                row_hm.append(round(s_pb_hm, 1))
            z_hm.append(row_hm)

        fig_hm = go.Figure(data=go.Heatmap(
            z=z_hm,
            x=[f"{v*100:.0f}%" for v in rr_hm],
            y=[f"{v*100:.2f}%" for v in lam_hm],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Spread (pb)"),
            text=[[f"{val:.0f}" for val in row] for row in z_hm],
            texttemplate="%{text}",
        ))
        fig_hm.update_layout(
            **plotly_theme(),
            title=dict(
                text=f"Spread CDS (pb) | T={int(T_input)}a | r={r_input*100:.2f}%",
                font=dict(size=13),
            ),
            xaxis_title="Recovery Rate (RR)",
            yaxis_title="Hazard Rate (λ)",
            height=480,
            margin=dict(l=60, r=20, t=60, b=42),
        )
        st.plotly_chart(fig_hm, use_container_width=True)