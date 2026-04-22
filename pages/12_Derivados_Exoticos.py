"""
pages/11_Derivados_Exoticos.py
------------------------------
Módulo 11: Valuación de Derivados Exóticos.
Cubre (en orden de complejidad creciente):
  - Opciones Gap
  - Opciones Binarias (Cash-or-Nothing, Asset-or-Nothing)
  - Opciones de Barrera (Down-and-Out, con paridad KI/KO)
  - Opciones Asiáticas (Media Aritmética y Geométrica)
  - Opciones Lookback (precio mínimo / máximo flotante)
  - Opciones Compuestas (opción sobre opción)
  - Opciones de Intercambio (Exchange / Margrabe)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm as _norm

from utils import (
    get_engine, page_header, paso_a_paso,
    separador, alerta_metodo_numerico,
    result_call, result_put,
    themed_info, themed_success, themed_warning, themed_error,
    apply_plotly_theme, plotly_theme, plotly_colors, get_current_theme)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Derivados Exóticos · Calculadora Financiera",
    page_icon="🧪",
    layout="wide")

engine = get_engine()

page_header(
    titulo="11. Derivados Exóticos",
    subtitulo="Gap · Binarias · Barrera · Asiáticas · Lookback · Compuestas · Intercambio"
)

# =============================================================================
# PESTAÑAS
# =============================================================================
tabs = st.tabs([
    "Gap",
    "Binarias",
    "Barrera",
    "Asiaticas",
    "Lookback",
    "Compuestas",
    "Intercambio",
    "Activos Reales (Yahoo Finance)",
])

tab_gap, tab_bin, tab_bar, tab_asi, tab_look, tab_comp, tab_int, tab_real = tabs


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: bloque de inputs BSM reutilizable
# ─────────────────────────────────────────────────────────────────────────────
def _inputs_bsm_base(sufijo: str, default_S=100.0, default_K=100.0,
                     default_r=5.0, default_sig=20.0, default_T=1.0,
                     default_q=0.0, mostrar_tipo=True):
    """Devuelve (S, K, r, sigma, T, q, es_call)."""
    S   = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                           value=default_S, step=1.0, key=f"S_{sufijo}")
    K   = st.number_input("Precio de Ejercicio ($K$)", min_value=0.01,
                           value=default_K, step=1.0, key=f"K_{sufijo}")
    r   = st.number_input("Tasa libre de riesgo ($r$) %",
                           value=default_r, step=0.1, key=f"r_{sufijo}") / 100
    sig = st.number_input("Volatilidad ($\\sigma$) %",
                           min_value=0.01, value=default_sig,
                           step=0.5, key=f"sig_{sufijo}") / 100
    T   = st.number_input("Tiempo al vencimiento ($T$) años",
                           min_value=0.001, value=default_T,
                           step=0.25, key=f"T_{sufijo}")
    q   = st.number_input("Dividendo continuo ($q$) %",
                           value=default_q, step=0.1, key=f"q_{sufijo}") / 100
    es_call = True
    if mostrar_tipo:
        tipo = st.radio("Tipo:", ["Call", "Put"],
                        horizontal=True, key=f"tipo_{sufijo}")
        es_call = (tipo == "Call")
    return S, K, r, sig, T, q, es_call


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPCIONES GAP
# ═════════════════════════════════════════════════════════════════════════════
with tab_gap:
    st.markdown("### Opciones Gap")
    themed_info(
        "Una opción Gap tiene **dos strikes**: el strike de activación <span style='font-family: serif; font-style: italic;'>K<sub>1</sub></span> "
        "que determina si se ejerce, y el strike de pago <span style='font-family: serif; font-style: italic;'>K<sub>2</sub></span> "
        "que determina el monto. <br><br><b> Payoff (Call):</b> Se recibe <span style='font-family: serif; font-style: italic;'>S<sub>T</sub> − K<sub>2</sub></span> "
        "únicamente si <span style='font-family: serif; font-style: italic;'>S<sub>T</sub> > K<sub>1</sub></span>. "
        "Si <span style='font-family: serif; font-style: italic;'>K<sub>1</sub> ≠ K<sub>2</sub></span>, existe un salto brusco (gap) en la ganancia."
    )

    c1, c2 = st.columns(2)
    with c1:
        S_g, _, r_g, sig_g, T_g, q_g, es_call_g = _inputs_bsm_base("gap")
        K1_g = st.number_input("Strike de activacion ($K_1$)", min_value=0.01,
                                value=100.0, step=1.0, key="K1_gap")
        K2_g = st.number_input("Strike de pago ($K_2$)", min_value=0.01,
                                value=90.0, step=1.0, key="K2_gap")

    with c2:
        tipo_gap = "call" if es_call_g else "put"
        prima_gap = engine.opciones_gap(S_g, K2_g, K1_g, T_g, r_g, sig_g, q_g, tipo_gap)
        tipo_txt  = "Call" if es_call_g else "Put"

        if es_call_g:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Gap {tipo_txt}: ${prima_gap:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>Gap {tipo_txt}: ${prima_gap:,.4f}</h3>")

        st.latex(
            r"c_{gap} = S_0 e^{-qT} N(d_1) - K_2 e^{-rT} N(d_2)"
            if es_call_g else
            r"p_{gap} = K_2 e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)"
        )

    with paso_a_paso():
        st.latex(r"d_1 = \frac{\ln(S_0/K_1)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}}")
        st.latex(rf"d_1 = \frac{{\ln({S_g:.2f}/{K1_g:.2f}) + ({r_g:.4f} - {q_g:.4f} + \frac{{{sig_g:.4f}^2}}{{2}}){T_g:.4f}}}{{{sig_g:.4f}\sqrt{{{T_g:.4f}}}}}")
        d1_g = (np.log(S_g/K1_g) + (r_g - q_g + sig_g**2/2)*T_g) / (sig_g*np.sqrt(T_g))
        d2_g = d1_g - sig_g*np.sqrt(T_g)
        st.latex(rf"d_1 = {d1_g:.6f}")
        st.latex(rf"d_2 = d_1 - \sigma\sqrt{{T}} = {d1_g:.6f} - {sig_g*np.sqrt(T_g):.6f} = {d2_g:.6f}")
        
        st.write("---")
        
        if es_call_g:
            st.latex(r"c_{gap} = S_0 e^{-qT} N(d_1) - K_2 e^{-rT} N(d_2)")
            st.latex(rf"c_{{gap}} = {S_g:.2f} e^{{-{q_g:.4f}({T_g:.4f})}} N({d1_g:.6f}) - {K2_g:.2f} e^{{-{r_g:.4f}({T_g:.4f})}} N({d2_g:.6f})")
            t1 = S_g*np.exp(-q_g*T_g)*_norm.cdf(d1_g)
            t2 = K2_g*np.exp(-r_g*T_g)*_norm.cdf(d2_g)
            st.latex(rf"c_{{gap}} = {S_g*np.exp(-q_g*T_g):.4f}({_norm.cdf(d1_g):.6f}) - {K2_g*np.exp(-r_g*T_g):.4f}({_norm.cdf(d2_g):.6f})")
            st.latex(rf"c_{{gap}} = {t1:.4f} - {t2:.4f}")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>c_{{gap}} = {prima_gap:,.4f}</h4>")
        else:
            st.latex(r"p_{gap} = K_2 e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)")
            st.latex(rf"p_{{gap}} = {K2_g:.2f} e^{{-{r_g:.4f}({T_g:.4f})}} N({-d2_g:.6f}) - {S_g:.2f} e^{{-{q_g:.4f}({T_g:.4f})}} N({-d1_g:.6f})")
            t1 = K2_g*np.exp(-r_g*T_g)*_norm.cdf(-d2_g)
            t2 = S_g*np.exp(-q_g*T_g)*_norm.cdf(-d1_g)
            st.latex(rf"p_{{gap}} = {K2_g*np.exp(-r_g*T_g):.4f}({_norm.cdf(-d2_g):.6f}) - {S_g*np.exp(-q_g*T_g):.4f}({_norm.cdf(-d1_g):.6f})")
            st.latex(rf"p_{{gap}} = {t1:.4f} - {t2:.4f}")
            themed_error(f"<h4 style='margin:0; color:inherit; text-align:center;'>p_{{gap}} = {prima_gap:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPCIONES BINARIAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_bin:
    st.markdown("### Opciones Binarias (Digitales)")
    themed_info(
        "Las opciones binarias (o digitales) pagan una cantidad fija o el activo subyacente si terminan In-The-Money. "
        "Son los bloques atómicos con los que se construyen los demás derivados. <br><br>"
        "• <b>Payoff Cash-or-Nothing (Call):</b> Paga un monto fijo <span style='font-family: serif; font-style: italic;'>Q</span> si <span style='font-family: serif; font-style: italic;'>S<sub>T</sub> > K</span>. <br>"
        "• <b>Payoff Asset-or-Nothing (Call):</b> Paga una unidad del activo <span style='font-family: serif; font-style: italic;'>S<sub>T</sub></span> si <span style='font-family: serif; font-style: italic;'>S<sub>T</sub> > K</span>."
    )

    subtipo_bin = st.radio(
        "Subtipo:",
        ["Cash-or-Nothing", "Asset-or-Nothing"],
        horizontal=True,
        key="bin_subtipo")
    separador()

    c1, c2 = st.columns(2)
    with c1:
        S_b, K_b, r_b, sig_b, T_b, q_b, es_call_b = _inputs_bsm_base("bin")
        if subtipo_bin == "Cash-or-Nothing":
            Q_b = st.number_input("Monto fijo a pagar ($Q$)", min_value=0.01,
                                   value=100.0, step=10.0, key="bin_Q")

    with c2:
        d1_b = (np.log(S_b/K_b) + (r_b - q_b + sig_b**2/2)*T_b) / (sig_b*np.sqrt(T_b))
        d2_b = d1_b - sig_b*np.sqrt(T_b)
        tipo_b = "call" if es_call_b else "put"

        if subtipo_bin == "Cash-or-Nothing":
            prima_bin = engine.opciones_cash_or_nothing(S_b, K_b, Q_b, T_b, r_b, sig_b, q_b, tipo_b)
            lbl = "Cash-or-Nothing Call" if es_call_b else "Cash-or-Nothing Put"
            formula_b = r"c_{CoN} = Q e^{-rT} N(d_2)" if es_call_b else r"p_{CoN} = Q e^{-rT} N(-d_2)"
        else:
            prima_bin = engine.opciones_asset_or_nothing(S_b, K_b, T_b, r_b, sig_b, q_b, tipo_b)
            lbl = "Asset-or-Nothing Call" if es_call_b else "Asset-or-Nothing Put"
            formula_b = r"c_{AoN} = S_0 e^{-qT} N(d_1)" if es_call_b else r"p_{AoN} = S_0 e^{-qT} N(-d_1)"

        if es_call_b:
            themed_success(f"<h3 style='margin:0; color:inherit;'>{lbl}: ${prima_bin:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>{lbl}: ${prima_bin:,.4f}</h3>")

        st.latex(formula_b)

    with paso_a_paso():
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}")
        st.latex(rf"d_1 = {d1_b:.6f}, \quad d_2 = {d2_b:.6f}")
        st.write("---")
        
        if subtipo_bin == "Cash-or-Nothing":
            st.latex(formula_b)
            nd  = _norm.cdf(d2_b) if es_call_b else _norm.cdf(-d2_b)
            fac = Q_b * np.exp(-r_b * T_b)
            sign_d = "d_2" if es_call_b else "-d_2"
            
            st.latex(rf"\text{{Prima}} = {Q_b:.2f} e^{{-{r_b:.4f}({T_b:.4f})}} N({sign_d})")
            st.latex(rf"\text{{Prima}} = {fac:.4f} \times {_norm.cdf(d2_b if es_call_b else -d2_b):.6f}")
        else:
            st.latex(formula_b)
            nd  = _norm.cdf(d1_b) if es_call_b else _norm.cdf(-d1_b)
            fac = S_b * np.exp(-q_b * T_b)
            sign_d = "d_1" if es_call_b else "-d_1"
            
            st.latex(rf"\text{{Prima}} = {S_b:.2f} e^{{-{q_b:.4f}({T_b:.4f})}} N({sign_d})")
            st.latex(rf"\text{{Prima}} = {fac:.4f} \times {_norm.cdf(d1_b if es_call_b else -d1_b):.6f}")
            
        if es_call_b:
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_bin:,.4f}</h4>")
        else:
            themed_error(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_bin:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPCIONES DE BARRERA
# ═════════════════════════════════════════════════════════════════════════════
with tab_bar:
    st.markdown("### Opciones de Barrera (Down-and-Out / Down-and-In)")
    themed_info(
        "Las opciones de barrera se activan (Knock-In) o desactivan (Knock-Out) cuando el precio del "
        "subyacente cruza el nivel <span style='font-family: serif; font-style: italic;'>H</span>. <br><br>"
        "• <b>Payoff Down-and-Out (Call):</b> Igual a la vanilla si <span style='font-family: serif; font-style: italic;'>S<sub>t</sub> > H</span> en todo momento; si toca <span style='font-family: serif; font-style: italic;'>H</span> se vuelve <span style='font-family: serif; font-style: italic;'>0</span>. <br>"
        "• <b>Paridad Estricta:</b> <span style='font-family: serif; font-style: italic;'>c<sub>vanilla</sub> = c<sub>KI</sub> + c<sub>KO</sub></span>."
    )

    c1, c2 = st.columns(2)
    with c1:
        S_ba, K_ba, r_ba, sig_ba, T_ba, q_ba, es_call_ba = _inputs_bsm_base("bar")
        H_ba = st.number_input("Nivel de barrera ($H < S_0$)", min_value=0.01,
                                value=85.0, step=1.0, key="bar_H")
        tipo_bar_sel = st.radio(
            "Tipo de barrera:",
            ["Down-and-Out (se desactiva al tocar H)",
             "Down-and-In  (se activa al tocar H)"],
            key="bar_tipo")
        es_out = tipo_bar_sel.startswith("Down-and-Out")

        if H_ba >= S_ba:
            themed_warning("La barrera H debe ser menor que el precio spot S0 para Down barriers.")

    with c2:
        tipo_b_str = "call" if es_call_ba else "put"

        prima_ko = engine.barrera_down_and_out(S_ba, K_ba, H_ba, T_ba, r_ba, sig_ba, q_ba, tipo_b_str)
        prima_vanilla = engine.black_scholes(S_ba, K_ba, r_ba, sig_ba, T_ba, es_call_ba, q_ba)
        prima_ki = max(0.0, prima_vanilla - prima_ko)

        if es_out:
            prima_bar = prima_ko
            lbl_bar = f"Down-and-Out {'Call' if es_call_ba else 'Put'}"
        else:
            prima_bar = prima_ki
            lbl_bar = f"Down-and-In {'Call' if es_call_ba else 'Put'}"

        if es_call_ba:
            themed_success(f"<h3 style='margin:0; color:inherit;'>{lbl_bar}: ${prima_bar:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>{lbl_bar}: ${prima_bar:,.4f}</h3>")

        separador()
        col_ba1, col_ba2, col_ba3 = st.columns(3)
        col_ba1.metric("Vanilla BSM (referencia)", f"${prima_vanilla:,.4f}")
        col_ba2.metric("Down-and-Out ($c_{KO}$)",  f"${prima_ko:,.4f}")
        col_ba3.metric("Down-and-In ($c_{KI}$)", f"${prima_ki:,.4f}")

    with paso_a_paso():
        st.latex(r"c_{vanilla} = c_{KO} + c_{KI}")
        st.write("El modelo analítico de Reiner y Rubinstein (1991) aplica un factor de descuento por el riesgo de tocar la barrera, determinado por $\mu$:")
        mu_b = (r_ba - q_ba - (sig_ba**2)/2) / (sig_ba**2)
        st.latex(rf"\mu = \frac{{r - q - \sigma^2/2}}{{\sigma^2}} = \frac{{{r_ba:.4f} - {q_ba:.4f} - {(sig_ba**2)/2:.6f}}}{{{sig_ba**2:.6f}}} = {mu_b:.4f}")
        st.latex(rf"\text{{Factor de castigo}} \propto \left(\frac{{H}}{{S_0}}\right)^{{2\mu}} = \left(\frac{{{H_ba:.2f}}}{{{S_ba:.2f}}}\right)^{{2({mu_b:.4f})}}")
        st.write("---")
        st.latex(r"c_{KI} = c_{vanilla} - c_{KO}")
        st.latex(rf"c_{{KO}} = {prima_ko:.6f}")
        st.latex(rf"c_{{KI}} = {prima_vanilla:.6f} - {prima_ko:.6f} = {prima_ki:.6f}")
        if es_call_ba:
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_bar:,.4f}</h4>")
        else:
            themed_error(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_bar:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — OPCIONES ASIÁTICAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_asi:
    st.markdown("### Opciones Asiaticas")
    themed_info(
        "El pago depende del **precio promedio** del subyacente (<span style='font-family: serif; font-style: italic;'>S<sub>avg</sub></span>) durante la vida del contrato. <br><br>"
        "• <b>Payoff Call Asiática:</b> <span style='font-family: serif; font-style: italic;'>max(S<sub>avg</sub> − K, 0)</span>. <br>"
        "• <b>Payoff Put Asiática:</b> <span style='font-family: serif; font-style: italic;'>max(K − S<sub>avg</sub>, 0)</span>. <br>"
        "Son más baratas que las vanilla porque promediar los precios reduce la volatilidad efectiva del derivado."
    )

    subtipo_asi = st.radio(
        "Metodo de promediacion:",
        ["Media Geometrica (formula cerrada)", "Media Aritmetica (Turnbull-Wakeman)"],
        horizontal=True,
        key="asi_subtipo")
    separador()

    c1, c2 = st.columns(2)
    with c1:
        S_as, K_as, r_as, sig_as, T_as, q_as, es_call_as = _inputs_bsm_base("asi")

    with c2:
        tipo_as = "call" if es_call_as else "put"

        if subtipo_asi.startswith("Media Geometrica"):
            prima_asi = engine.opciones_asiaticas_geometricas(
                S_as, K_as, T_as, r_as, sig_as, q_as, tipo_as
            )
            lbl_as = "Asiatica Geometrica Call" if es_call_as else "Asiatica Geometrica Put"
            
            if es_call_as:
                themed_success(f"<h3 style='margin:0; color:inherit;'>{lbl_as}: ${prima_asi:,.4f}</h3>")
            else:
                themed_error(f"<h3 style='margin:0; color:inherit;'>{lbl_as}: ${prima_asi:,.4f}</h3>")

        else:
            prima_asi = engine.opciones_asiaticas_aritmeticas(
                S_as, K_as, T_as, r_as, sig_as, q_as, tipo_as
            )
            lbl_as = "Asiatica Aritmetica Call" if es_call_as else "Asiatica Aritmetica Put"
            if es_call_as:
                themed_success(f"<h3 style='margin:0; color:inherit;'>{lbl_as}: ${prima_asi:,.4f}</h3>")
            else:
                themed_error(f"<h3 style='margin:0; color:inherit;'>{lbl_as}: ${prima_asi:,.4f}</h3>")

        # Comparativa con vanilla
        prima_van_as = engine.black_scholes(S_as, K_as, r_as, sig_as, T_as, es_call_as, q_as)
        st.metric("Vanilla BSM (referencia)",  f"${prima_van_as:,.4f}")
        descuento_as = (1 - prima_asi / prima_van_as) * 100 if prima_van_as > 0 else 0
        st.metric("Descuento vs Vanilla", f"{descuento_as:.2f}%")

    with paso_a_paso():
        if subtipo_asi.startswith("Media Geometrica"):
            sig_star = sig_as / np.sqrt(3)
            b_star   = 0.5 * (r_as - q_as - sig_as**2/6)
            q_star   = r_as - b_star

            st.latex(rf"\sigma^* = \frac{{\sigma}}{{\sqrt{{3}}}} = \frac{{{sig_as:.4f}}}{{\sqrt{{3}}}} = {sig_star:.6f}")
            st.latex(rf"b^* = \frac{{1}}{{2}}\left(r - q - \frac{{\sigma^2}}{{6}}\right) = \frac{{1}}{{2}}\left({r_as:.4f} - {q_as:.4f} - \frac{{{sig_as**2:.6f}}}{{6}}\right) = {b_star:.6f}")
            st.write("---")
            st.latex(r"\text{Sustituimos en BSM con volatilidad } \sigma^* \text{ y dividendo equivalente } q^* = r - b^*:")
            st.latex(rf"q^* = {r_as:.4f} - {b_star:.6f} = {q_star:.6f}")
            d1_as = (np.log(S_as/K_as) + (r_as - q_star + sig_star**2/2)*T_as) / (sig_star*np.sqrt(T_as))
            d2_as = d1_as - sig_star*np.sqrt(T_as)
            st.latex(rf"d_1 = \frac{{\ln({S_as:.2f}/{K_as:.2f}) + ({r_as:.4f} - {q_star:.6f} + \frac{{{sig_star:.6f}^2}}{{2}}){T_as:.4f}}}{{{sig_star:.6f}\sqrt{{{T_as:.4f}}}}} = {d1_as:.6f}")
            st.latex(rf"d_2 = {d1_as:.6f} - {sig_star*np.sqrt(T_as):.6f} = {d2_as:.6f}")
            st.write("---")
            if es_call_as:
                st.latex(rf"c_{{asi}} = {S_as:.2f} e^{{-{q_star:.6f}({T_as:.4f})}} N({d1_as:.4f}) - {K_as:.2f} e^{{-{r_as:.4f}({T_as:.4f})}} N({d2_as:.4f}) = {prima_asi:.4f}")
            else:
                st.latex(rf"p_{{asi}} = {K_as:.2f} e^{{-{r_as:.4f}({T_as:.4f})}} N({-d2_as:.4f}) - {S_as:.2f} e^{{-{q_star:.6f}({T_as:.4f})}} N({-d1_as:.4f}) = {prima_asi:.4f}")
        else:
            st.latex(r"\text{Aproximación de Turnbull-Wakeman: cálculo de momentos } M_1 \text{ y } M_2")
            b_tw = r_as - q_as
            if abs(b_tw) < 1e-6:
                M1 = S_as
                M2 = (2 * S_as**2 / (sig_as**2 * T_as**2)) * (np.exp(sig_as**2 * T_as) - 1 - sig_as**2 * T_as)
            else:
                M1 = S_as * (np.exp(b_tw * T_as) - 1) / (b_tw * T_as)
                num1 = (np.exp((2*b_tw + sig_as**2)*T_as) - 1) / (2*b_tw + sig_as**2)
                num2 = (np.exp(b_tw*T_as) - 1) / b_tw
                M2 = (2 * S_as**2 / ((b_tw + sig_as**2) * T_as**2)) * (num1 - num2)
            sig_tw = np.sqrt(max(0, np.log(M2 / (M1**2)) / T_as))
            st.latex(rf"M_1 = \text{{E}}[A_T] = {M1:.6f}")
            st.latex(rf"M_2 = \text{{E}}[A_T^2] = {M2:.6f}")
            st.latex(rf"\sigma_{{TW}} = \sqrt{{\frac{{\ln(M_2 / M_1^2)}}{{T}}}} = {sig_tw:.6f}")
            st.write("---")
            st.latex(rf"\text{{Prima}} = \text{{BSM}}(S'={M1:.4f}, K={K_as:.2f}, \sigma={sig_tw:.6f}) = {prima_asi:.4f}")
            
        if es_call_as:
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_asi:,.4f}</h4>")
        else:
            themed_error(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_asi:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — OPCIONES LOOKBACK
# ═════════════════════════════════════════════════════════════════════════════
with tab_look:
    st.markdown("### Opciones Lookback (Precio Flotante)")
    themed_info(
        "El payoff utiliza el precio extremo (<span style='font-family: serif; font-style: italic;'>S<sub>min</sub></span> o <span style='font-family: serif; font-style: italic;'>S<sub>max</sub></span>) alcanzado durante el periodo. "
        "Al garantizar comprar en el mínimo o vender en el máximo, son los derivados más caros del mercado.<br><br>"
        "• <b>Payoff Call (Mínimo Flotante):</b> <span style='font-family: serif; font-style: italic;'>max(S<sub>T</sub> − S<sub>min</sub>, 0)</span>. <br>"
        "• <b>Payoff Put (Máximo Flotante):</b> <span style='font-family: serif; font-style: italic;'>max(S<sub>max</sub> − S<sub>T</sub>, 0)</span>."
    )

    c1, c2 = st.columns(2)
    with c1:
        S_lk, _, r_lk, sig_lk, T_lk, q_lk, es_call_lk = _inputs_bsm_base(
            "look", mostrar_tipo=True
        )
        S_min_max = st.number_input(
            "Minimo observado (S_min) Call  /  Maximo observado (S_max) Put",
            min_value=0.01,
            value=95.0,
            step=1.0,
            key="look_ext",
            help="Si la opcion acaba de emitirse, usa el valor de S0.")

    with c2:
        tipo_lk = "call" if es_call_lk else "put"
        prima_lk = engine.opciones_lookback_flotante(
            S_lk, S_min_max, T_lk, r_lk, sig_lk, q_lk, tipo_lk
        )
        lbl_lk = "Lookback Call (mínimo flotante)" if es_call_lk else "Lookback Put (máximo flotante)"

        if es_call_lk:
            themed_success(f"<h3 style='margin:0; color:inherit;'>{lbl_lk}: ${prima_lk:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>{lbl_lk}: ${prima_lk:,.4f}</h3>")

        prima_van_lk = engine.black_scholes(S_lk, S_min_max, r_lk, sig_lk, T_lk,
                                             es_call_lk, q_lk)
        st.metric("Vanilla BSM (referencia, K = extremo)", f"${prima_van_lk:,.4f}")

    with paso_a_paso():
        st.markdown("Fórmula analítica de Goldman, Sosin & Gatto (1979) para extremos flotantes:")
        if es_call_lk:
            a1 = (np.log(S_lk/S_min_max) + (r_lk - q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            a2 = a1 - sig_lk*np.sqrt(T_lk)
            a3 = (np.log(S_lk/S_min_max) + (-r_lk + q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            st.latex(rf"a_1 = \frac{{\ln(S_0/S_{{min}}) + (r-q+\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a1:.6f}")
            st.latex(rf"a_2 = a_1 - \sigma\sqrt{{T}} = {a2:.6f}")
            st.latex(rf"a_3 = \frac{{\ln(S_0/S_{{min}}) - (r-q-\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a3:.6f}")
            st.write("---")
            st.latex(r"c_{LB} = S_0 e^{-qT} N(a_1) - S_{min} e^{-rT} N(a_2) + \text{Término de Extremo}")
            st.latex(rf"c_{{LB}} = {prima_lk:.4f}")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>c_{{LB}} = ${prima_lk:,.4f}</h4>")
        else:
            a1 = (np.log(S_min_max/S_lk) + (-r_lk + q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            a2 = a1 - sig_lk*np.sqrt(T_lk)
            a3 = (np.log(S_min_max/S_lk) + (r_lk - q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            st.latex(rf"a_1 = \frac{{\ln(S_{{max}}/S_0) + (-r+q+\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a1:.6f}")
            st.latex(rf"a_2 = a_1 - \sigma\sqrt{{T}} = {a2:.6f}")
            st.latex(rf"a_3 = \frac{{\ln(S_{{max}}/S_0) + (r-q+\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a3:.6f}")
            st.write("---")
            st.latex(r"p_{LB} = S_{max} e^{-rT} N(a_1) - S_0 e^{-qT} N(a_2) + \text{Término de Extremo}")
            st.latex(rf"p_{{LB}} = {prima_lk:.4f}")
            themed_error(f"<h4 style='margin:0; color:inherit; text-align:center;'>p_{{LB}} = ${prima_lk:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — OPCIONES COMPUESTAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_comp:
    st.markdown("### Opciones Compuestas (Opcion sobre Opcion)")
    themed_info(
        "Una opción compuesta da el derecho a **comprar o vender otra opción** en una fecha futura "
        "<span style='font-family: serif; font-style: italic;'>T<sub>1</sub></span> pagando un strike <span style='font-family: serif; font-style: italic;'>K<sub>out</sub></span>. <br><br>"
        "• <b>Payoff Call sobre Call en T<sub>1</sub>:</b> <span style='font-family: serif; font-style: italic;'>max( C(T<sub>2</sub>, K<sub>in</sub>) − K<sub>out</sub>, 0)</span>.<br>"
        "Se utilizan habitualmente para cubrir el riesgo de financiamiento de coberturas complejas."
    )

    subtipo_comp = st.selectbox(
        "Tipo de opcion compuesta:",
        [
            "Call sobre Call",
            "Call sobre Put",
            "Put sobre Call",
            "Put sobre Put",
        ],
        key="comp_subtipo")
    es_call_outer = subtipo_comp.startswith("Call")
    es_call_inner = "Call" in subtipo_comp.split("sobre")[1]

    tipo_map = {
        (True,  True):  "call_on_call",
        (True,  False): "call_on_put",
        (False, True):  "put_on_call",
        (False, False): "put_on_put",
    }
    tipo_comp_str = tipo_map[(es_call_outer, es_call_inner)]

    separador()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Opcion exterior (la que compras hoy)**")
        S_cp2   = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                                   value=100.0, step=1.0, key="comp_S")
        K_out   = st.number_input("Strike exterior ($K_{out}$)",
                                   min_value=0.01, value=5.0, step=0.5, key="comp_Kout")
        T_out   = st.number_input("Vencimiento exterior ($T_1$) años",
                                   min_value=0.001, value=0.5, step=0.25, key="comp_T1")
        st.markdown("**Opcion interior (la opcion subyacente)**")
        K_in    = st.number_input("Strike interior ($K_{in}$)", min_value=0.01,
                                   value=100.0, step=1.0, key="comp_Kin")
        T_in    = st.number_input("Vencimiento interior ($T_2 > T_1$) años",
                                   min_value=T_out + 0.001, value=1.0,
                                   step=0.25, key="comp_T2")
        r_cp2   = st.number_input("Tasa libre de riesgo ($r$) %",
                                   value=5.0, step=0.1, key="comp_r") / 100
        sig_cp2 = st.number_input("Volatilidad ($\\sigma$) %",
                                   min_value=0.01, value=20.0,
                                   step=0.5, key="comp_sig") / 100
        q_cp2   = st.number_input("Dividendo continuo ($q$) %",
                                   value=0.0, step=0.1, key="comp_q") / 100

    with c2:
        try:
            prima_comp = engine.opciones_compuestas(
                S_cp2, K_out, K_in, T_out, T_in, r_cp2, sig_cp2, q_cp2, tipo_comp_str
            )
            if es_call_outer:
                themed_success(f"<h3 style='margin:0; color:inherit;'>{subtipo_comp}: ${prima_comp:,.4f}</h3>")
            else:
                themed_error(f"<h3 style='margin:0; color:inherit;'>{subtipo_comp}: ${prima_comp:,.4f}</h3>")

            prima_inner = engine.black_scholes(S_cp2, K_in, r_cp2, sig_cp2, T_in,
                                                es_call_inner, q_cp2)
            st.metric("Prima de la opcion interior sola (BSM)", f"${prima_inner:,.4f}")
        except Exception as e:
            themed_error(f"Error en el calculo: {e}")

    with paso_a_paso():
        rho = np.sqrt(T_out / T_in)
        st.latex(r"\rho = \sqrt{\frac{T_1}{T_2}}")
        st.latex(rf"\rho = \sqrt{{\frac{{{T_out:.4f}}}{{{T_in:.4f}}}}} = {rho:.6f}")
        st.write("---")
        st.latex(r"S^* \text{ tal que } \text{BSM}(S^*, T_2-T_1) = K_{out}")
        st.latex(r"a_{1,2} = \frac{\ln(S_0/S^*) + (r-q \pm \sigma^2/2)T_1}{\sigma\sqrt{T_1}}")
        st.latex(r"b_{1,2} = \frac{\ln(S_0/K_{in}) + (r-q \pm \sigma^2/2)T_2}{\sigma\sqrt{T_2}}")
        st.write("---")
        if es_call_outer and es_call_inner:
            st.latex(r"c_{cc} = S_0 e^{-qT_2} M(a_1, b_1; \rho) - K_{in} e^{-rT_2} M(a_2, b_2; \rho) - K_{out} e^{-rT_1} N(a_2)")
        else:
            st.latex(r"\text{Evaluación de la integral de Geske usando Distribución Normal Bivariada } M(x,y;\rho)")
            
        if es_call_outer:
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_comp:,.4f}</h4>")
        else:
            themed_error(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_comp:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — OPCIONES DE INTERCAMBIO (MARGRABE)
# ═════════════════════════════════════════════════════════════════════════════
with tab_int:
    st.markdown("### Opciones de Intercambio (Margrabe, 1978)")
    themed_info(
        "Otorga el derecho a entregar el activo <span style='font-family: serif; font-style: italic;'>S<sub>1</sub></span> para recibir el activo <span style='font-family: serif; font-style: italic;'>S<sub>2</sub></span>. "
        "No existe un strike fijo en moneda.<br><br>"
        "• <b>Payoff al vencimiento:</b> <span style='font-family: serif; font-style: italic;'>max(S<sub>2</sub> − S<sub>1</sub>, 0)</span>. <br>"
        "Su precio depende fundamentalmente de la correlación (<span style='font-family: serif; font-style: italic;'>ρ</span>) entre ambos activos."
    )

    # ── Cantidades ──────────────────────────────────────────────────────────
    col_qty1, col_qty2 = st.columns(2)
    with col_qty1:
        n1_ex = st.number_input("Unidades a ENTREGAR de S1 (n1):", min_value=0.001, value=1.0, step=0.5, key="int_n1")
    with col_qty2:
        n2_ex = st.number_input("Unidades a RECIBIR de S2 (n2):", min_value=0.001, value=1.0, step=0.5, key="int_n2")

    separador()
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        S1_int = st.number_input("Spot S1 (Entregar)", min_value=0.01, value=100.0, step=1.0, key="int_S1")
        sig1_in = st.number_input("Volatilidad sigma1 %", min_value=0.01, value=20.0, step=0.5, key="int_sig1") / 100
        q1_int = st.number_input("Dividendo q1 %", value=3.0, step=0.1, key="int_q1") / 100

        S2_int = st.number_input("Spot S2 (Recibir)", min_value=0.01, value=110.0, step=1.0, key="int_S2")
        sig2_in = st.number_input("Volatilidad sigma2 %", min_value=0.01, value=25.0, step=0.5, key="int_sig2") / 100
        q2_int = st.number_input("Dividendo q2 %", value=2.0, step=0.1, key="int_q2") / 100

        rho_int = st.slider("Correlacion rho:", min_value=-1.0, max_value=1.0, value=0.5, step=0.01, key="int_rho")
        T_int = st.number_input("Vencimiento (T) años", min_value=0.001, value=1.0, step=0.25, key="int_T")

    with col_m2:
        U_eff = n1_ex * S1_int
        V_eff = n2_ex * S2_int
        sig_comb = np.sqrt(sig1_in**2 + sig2_in**2 - 2*rho_int*sig1_in*sig2_in)

        if V_eff > 0 and U_eff > 0 and sig_comb > 0 and T_int > 0:
            d1_int = (np.log(V_eff / U_eff) + (q1_int - q2_int + sig_comb**2 / 2) * T_int) / (sig_comb * np.sqrt(T_int))
            d2_int = d1_int - sig_comb * np.sqrt(T_int)
            prima_int = max(V_eff * np.exp(-q2_int * T_int) * _norm.cdf(d1_int) - U_eff * np.exp(-q1_int * T_int) * _norm.cdf(d2_int), 0.0)
        else:
            prima_int = 0.0

        themed_success(f"<h3 style='margin:0; color:inherit;'>Intercambio (Recibir S2, Entregar S1): ${prima_int:,.4f}</h3>")
        c1r, c2r, c3r = st.columns(3)
        c1r.metric("U = n1 * S1", f"${U_eff:,.4f}")
        c2r.metric("V = n2 * S2", f"${V_eff:,.4f}")
        c3r.metric("sigma*", f"{sig_comb*100:.4f}%")

    with paso_a_paso():
        st.latex(r"\sigma^* = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}")
        st.latex(rf"\sigma^* = \sqrt{{{sig1_in:.4f}^2 + {sig2_in:.4f}^2 - 2({rho_int:.2f})({sig1_in:.4f})({sig2_in:.4f})}} = {sig_comb:.6f}")
        st.write("---")
        st.latex(r"d_1 = \frac{\ln(V/U) + (q_1 - q_2 + \sigma^{*2}/2)T}{\sigma^*\sqrt{T}}")
        st.latex(rf"d_1 = \frac{{\ln({V_eff:.2f}/{U_eff:.2f}) + ({q1_int:.4f} - {q2_int:.4f} + \frac{{{sig_comb:.6f}^2}}{{2}}){T_int:.4f}}}{{{sig_comb:.6f}\sqrt{{{T_int:.4f}}}}}")
        st.latex(rf"d_1 = {d1_int:.6f}, \quad d_2 = d_1 - \sigma^*\sqrt{{T}} = {d2_int:.6f}")
        st.write("---")
        st.latex(r"c = V e^{-q_2 T} N(d_1) - U e^{-q_1 T} N(d_2)")
        st.latex(rf"c = {V_eff:.2f} e^{{-{q2_int:.4f}({T_int:.4f})}} N({d1_int:.4f}) - {U_eff:.2f} e^{{-{q1_int:.4f}({T_int:.4f})}} N({d2_int:.4f})")
        st.latex(rf"c = {prima_int:.4f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${prima_int:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — ACTIVOS REALES (YAHOO FINANCE)
# Valúa los 7 tipos de exóticos con datos reales de mercado
# ═════════════════════════════════════════════════════════════════════════════
with tab_real:
    st.markdown("### Derivados Exóticos sobre Activos Reales")
    themed_info(
        "Carga el precio spot y la volatilidad histórica de cualquier activo real desde Yahoo Finance, "
        "luego valuá cualquiera de los 7 tipos de derivados exóticos con parámetros de mercado reales."
    )
    separador()

    # ── BUSCADOR ─────────────────────────────────────────────────────────────
    st.markdown("#### Paso 1 — Cargar datos de mercado")
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        ticker_ex = st.text_input(
            "Ticker de Yahoo Finance:",
            value="AAPL",
            key="ex_ticker",
            help="Ejemplos: AAPL, TSLA, MSFT, SPY, GLD, CEMEXCPO.MX",
            placeholder="Ej. AAPL, TSLA, SPY...",
        ).strip().upper()
    with col_t2:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_ex = st.button("Buscar en Yahoo Finance", use_container_width=True, key="btn_ex")

    if btn_ex:
        with st.spinner(f"Consultando {ticker_ex}..."):
            spot_ex, vol_ex = engine.obtener_datos_subyacente(ticker_ex)
            if spot_ex is not None:
                st.session_state["ex_spot"] = float(spot_ex)
                st.session_state["ex_vol"]  = float(vol_ex * 100)
                st.session_state["ex_ticker_ok"] = ticker_ex
                st.session_state["ex_S"]   = float(spot_ex)
                st.session_state["ex_K"]   = float(spot_ex)   
                st.session_state["ex_sig"] = float(vol_ex * 100)
                themed_success(
                    f"**{ticker_ex}** cargado correctamente.  \n"
                    f"S₀ = **${spot_ex:,.2f}** · K (ATM) = **${spot_ex:,.2f}** · "
                    f"σ = **{vol_ex*100:.2f}%** — campos actualizados automaticamente."
                )
                st.rerun()
            else:
                st.session_state.pop("ex_spot", None)
                themed_error(f"No se encontró **{ticker_ex}**. Verifica el símbolo.")

    if "ex_spot" not in st.session_state: st.session_state["ex_spot"] = 100.0
    if "ex_vol"  not in st.session_state: st.session_state["ex_vol"]  = 20.0
    if "ex_ticker_ok" not in st.session_state: st.session_state["ex_ticker_ok"] = "ACTIVO"
    if "ex_S"   not in st.session_state: st.session_state["ex_S"]   = st.session_state["ex_spot"]
    if "ex_K"   not in st.session_state: st.session_state["ex_K"]   = st.session_state["ex_spot"]
    if "ex_sig" not in st.session_state: st.session_state["ex_sig"] = st.session_state["ex_vol"]

    separador()

    # ── PARÁMETROS BASE ───────────────────────────────────────────────────────
    st.markdown("#### Paso 2 — Parámetros base del contrato")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        S_ex  = st.number_input("Precio Spot ($S_0$)", min_value=0.01, step=1.0, key="ex_S")
        K_ex  = st.number_input("Strike ($K$)", min_value=0.01, step=1.0, key="ex_K")
        tipo_ex = st.radio("Tipo:", ["Call", "Put"], horizontal=True, key="ex_tipo")
        es_call_ex = (tipo_ex == "Call")
    with col_b2:
        sig_ex = st.number_input("Volatilidad ($\\sigma$) %", min_value=0.01, step=0.5, key="ex_sig") / 100
        r_ex   = st.number_input("Tasa libre de riesgo ($r$) %", value=5.0, step=0.1, key="ex_r") / 100
        q_ex   = st.number_input("Dividendo continuo ($q$) %", value=0.0, step=0.1, key="ex_q") / 100
        T_ex   = st.number_input("Vencimiento ($T$) años", min_value=0.01, value=0.5, step=0.25, key="ex_T")
    with col_b3:
        prima_van_ex = engine.black_scholes(S_ex, K_ex, r_ex, sig_ex, T_ex, es_call_ex, q_ex)
        ticker_lbl   = st.session_state.get("ex_ticker_ok", "ACTIVO")
        st.markdown(f"**Referencia — {ticker_lbl} Vanilla BSM**")
        if es_call_ex:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Vanilla Call: ${prima_van_ex:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>Vanilla Put: ${prima_van_ex:,.4f}</h3>")
            
        moneyness_ex = ((S_ex - K_ex) / K_ex) * 100
        if moneyness_ex > 1:
            themed_success(f"**ITM** — Moneyness: {moneyness_ex:+.2f}%")
        elif moneyness_ex < -1:
            themed_warning(f"**OTM** — Moneyness: {moneyness_ex:+.2f}%")
        else:
            themed_info(f"**ATM** — Moneyness: {moneyness_ex:+.2f}%")

    separador()

    # ── SELECTOR DE TIPO EXÓTICO ──────────────────────────────────────────────
    st.markdown("#### Paso 3 — Elige el tipo de derivado exótico")
    tipo_exotico = st.selectbox(
        "Tipo de derivado exótico:",
        [
            "Gap — Strike de activación vs pago",
            "Binaria Cash-or-Nothing — paga monto fijo",
            "Binaria Asset-or-Nothing — paga el activo",
            "Barrera Down-and-Out / Down-and-In",
            "Asiática Geométrica — precio promedio",
            "Asiática Aritmética — Turnbull-Wakeman",
            "Lookback Flotante — precio extremo",
            "Compuesta — opción sobre opción",
            "Intercambio (Margrabe) — entregar S1 recibir S2",
        ],
        key="ex_tipo_exotico",
    )
    separador()

    c_th = get_current_theme()
    tipo_str_ex = "call" if es_call_ex else "put"

    # ── GAP REAL ───────────────────────────────────────────────────────────────────
    if tipo_exotico.startswith("Gap"):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            K1_ex = st.number_input("Strike activación ($K_1$)", min_value=0.01, value=K_ex, step=1.0, key="ex_K1_gap")
            K2_ex = st.number_input("Strike de pago ($K_2$)", min_value=0.01, value=K_ex * 0.9, step=1.0, key="ex_K2_gap")
        with col_g2:
            prima_gap_ex = engine.opciones_gap(S_ex, K2_ex, K1_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)
            if es_call_ex:
                themed_success(f"<h3 style='margin:0; color:inherit;'>Gap Call: ${prima_gap_ex:,.4f}</h3>")
            else:
                themed_error(f"<h3 style='margin:0; color:inherit;'>Gap Put: ${prima_gap_ex:,.4f}</h3>")
        with paso_a_paso():
            d1_g = (np.log(S_ex/K1_ex) + (r_ex - q_ex + sig_ex**2/2)*T_ex) / (sig_ex*np.sqrt(T_ex))
            d2_g = d1_g - sig_ex*np.sqrt(T_ex)
            st.latex(rf"d_1 = {d1_g:.6f}, \quad d_2 = {d2_g:.6f}")
            st.write("---")
            if es_call_ex:
                st.latex(rf"c_{{gap}} = {S_ex:.2f} e^{{-{q_ex:.4f}({T_ex:.4f})}} N({d1_g:.6f}) - {K2_ex:.2f} e^{{-{r_ex:.4f}({T_ex:.4f})}} N({d2_g:.6f}) = {prima_gap_ex:.4f}")
            else:
                st.latex(rf"p_{{gap}} = {K2_ex:.2f} e^{{-{r_ex:.4f}({T_ex:.4f})}} N({-d2_g:.6f}) - {S_ex:.2f} e^{{-{q_ex:.4f}({T_ex:.4f})}} N({-d1_g:.6f}) = {prima_gap_ex:.4f}")

    # ── BINARIA CASH REAL ───────────────────────────────────────────────
    elif tipo_exotico.startswith("Binaria Cash"):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            Q_ex = st.number_input("Monto fijo a pagar ($Q$)", min_value=0.01, value=100.0, step=10.0, key="ex_Q_bin")
        with col_c2:
            prima_bin_ex = engine.opciones_cash_or_nothing(S_ex, K_ex, Q_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)
            if es_call_ex:
                themed_success(f"<h3 style='margin:0; color:inherit;'>Cash-o-N Call: ${prima_bin_ex:,.4f}</h3>")
            else:
                themed_error(f"<h3 style='margin:0; color:inherit;'>Cash-o-N Put: ${prima_bin_ex:,.4f}</h3>")
        with paso_a_paso():
            d2_b = (np.log(S_ex/K_ex) + (r_ex - q_ex - sig_ex**2/2)*T_ex) / (sig_ex*np.sqrt(T_ex))
            st.latex(rf"d_2 = {d2_b:.6f}")
            if es_call_ex:
                st.latex(rf"c_{{CoN}} = {Q_ex:.2f} e^{{-{r_ex:.4f}({T_ex:.4f})}} N({d2_b:.6f}) = {prima_bin_ex:.4f}")
            else:
                st.latex(rf"p_{{CoN}} = {Q_ex:.2f} e^{{-{r_ex:.4f}({T_ex:.4f})}} N({-d2_b:.6f}) = {prima_bin_ex:.4f}")

    # ── BINARIA ASSET REAL ──────────────────────────────────────────────
    elif tipo_exotico.startswith("Binaria Asset"):
        prima_aon_ex = engine.opciones_asset_or_nothing(S_ex, K_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)
        if es_call_ex:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Asset-o-N Call: ${prima_aon_ex:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>Asset-o-N Put: ${prima_aon_ex:,.4f}</h3>")
        with paso_a_paso():
            d1_b = (np.log(S_ex/K_ex) + (r_ex - q_ex + sig_ex**2/2)*T_ex) / (sig_ex*np.sqrt(T_ex))
            st.latex(rf"d_1 = {d1_b:.6f}")
            if es_call_ex:
                st.latex(rf"c_{{AoN}} = {S_ex:.2f} e^{{-{q_ex:.4f}({T_ex:.4f})}} N({d1_b:.6f}) = {prima_aon_ex:.4f}")
            else:
                st.latex(rf"p_{{AoN}} = {S_ex:.2f} e^{{-{q_ex:.4f}({T_ex:.4f})}} N({-d1_b:.6f}) = {prima_aon_ex:.4f}")

    # ── BARRERA REAL ───────────────────────────────────────────────────────────────
    elif tipo_exotico.startswith("Barrera"):
        col_bar1, col_bar2 = st.columns(2)
        with col_bar1:
            H_ex = st.number_input("Barrera ($H$)", min_value=0.01, step=1.0, value=round(S_ex * 0.85, 2), key="ex_H")
            tipo_bar_ex = st.radio("Tipo:", ["Down-and-Out", "Down-and-In"], horizontal=True, key="ex_bar_tipo")
        with col_bar2:
            prima_ko_ex = engine.barrera_down_and_out(S_ex, K_ex, H_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)
            prima_ki_ex = max(0.0, prima_van_ex - prima_ko_ex)
            es_out = tipo_bar_ex.startswith("Down-and-Out")
            prima_bar_ex = prima_ko_ex if es_out else prima_ki_ex
            
            if es_call_ex:
                themed_success(f"<h3 style='margin:0; color:inherit;'>{tipo_bar_ex} Call: ${prima_bar_ex:,.4f}</h3>")
            else:
                themed_error(f"<h3 style='margin:0; color:inherit;'>{tipo_bar_ex} Put: ${prima_bar_ex:,.4f}</h3>")
                
            st.metric("Descuento vs Vanilla", f"{(1 - prima_bar_ex / prima_van_ex) * 100:.1f}%")
        with paso_a_paso():
            mu_b = (r_ex - q_ex - (sig_ex**2)/2) / (sig_ex**2)
            st.latex(rf"\mu = \frac{{{r_ex:.4f} - {q_ex:.4f} - {(sig_ex**2)/2:.6f}}}{{{sig_ex**2:.6f}}} = {mu_b:.4f}")
            st.latex(rf"c_{{KO}} = {prima_ko_ex:.6f}")
            st.latex(rf"c_{{KI}} = {prima_van_ex:.6f} - {prima_ko_ex:.6f} = {prima_ki_ex:.6f}")

    # ── ASIÁTICA GEOMÉTRICA REAL ───────────────────────────────────────────────────
    elif tipo_exotico.startswith("Asiática Geométrica"):
        prima_asi_g = engine.opciones_asiaticas_geometricas(S_ex, K_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)
        if es_call_ex:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Asiática Geo Call: ${prima_asi_g:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>Asiática Geo Put: ${prima_asi_g:,.4f}</h3>")
        with paso_a_paso():
            sig_star = sig_ex / np.sqrt(3)
            b_star   = 0.5 * (r_ex - q_ex - sig_ex**2/6)
            st.latex(rf"\sigma^* = \frac{{{sig_ex:.4f}}}{{\sqrt{{3}}}} = {sig_star:.6f}")
            st.latex(rf"b^* = \frac{{1}}{{2}}\left({r_ex:.4f} - {q_ex:.4f} - \frac{{{sig_ex:.4f}^2}}{{6}}\right) = {b_star:.6f}")
            st.latex(rf"\text{{Prima}} = \text{{BSM}}(S_0, K, r, \sigma^*, T, q=(r - b^*)) = {prima_asi_g:.4f}")

    # ── ASIÁTICA ARITMÉTICA REAL ───────────────────────────────────────────────────
    elif tipo_exotico.startswith("Asiática Aritmética"):
        prima_asi_a = engine.opciones_asiaticas_aritmeticas(S_ex, K_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)
        if es_call_ex:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Asiática Aritm Call: ${prima_asi_a:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>Asiática Aritm Put: ${prima_asi_a:,.4f}</h3>")
        with paso_a_paso():
            b_tw = r_ex - q_ex
            if abs(b_tw) < 1e-6:
                M1 = S_ex
                M2 = (2 * S_ex**2 / (sig_ex**2 * T_ex**2)) * (np.exp(sig_ex**2 * T_ex) - 1 - sig_ex**2 * T_ex)
            else:
                M1 = S_ex * (np.exp(b_tw * T_ex) - 1) / (b_tw * T_ex)
                num1 = (np.exp((2*b_tw + sig_ex**2)*T_ex) - 1) / (2*b_tw + sig_ex**2)
                num2 = (np.exp(b_tw*T_ex) - 1) / b_tw
                M2 = (2 * S_ex**2 / ((b_tw + sig_ex**2) * T_ex**2)) * (num1 - num2)
            sig_tw = np.sqrt(max(0, np.log(M2 / (M1**2)) / T_ex))
            st.latex(rf"M_1 = {M1:.6f}, \quad M_2 = {M2:.6f}")
            st.latex(rf"\sigma_{{TW}} = \sqrt{{\frac{{\ln(M_2 / M_1^2)}}{{T}}}} = {sig_tw:.6f}")
            st.latex(rf"\text{{Prima}} = \text{{BSM}}(S'={M1:.4f}, K={K_ex:.2f}, \sigma={sig_tw:.6f}) = {prima_asi_a:.4f}")

    # ── LOOKBACK REAL ──────────────────────────────────────────────────────────────
    elif tipo_exotico.startswith("Lookback"):
        S_ext_ex = st.number_input("Mínimo observado (Call) / Máximo (Put)", min_value=0.01, step=1.0,
                                   value=round(S_ex * 0.95, 2) if es_call_ex else round(S_ex * 1.05, 2), key="ex_Sext")
        prima_lk_ex = engine.opciones_lookback_flotante(S_ex, S_ext_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)
        if es_call_ex:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Lookback Call: ${prima_lk_ex:,.4f}</h3>")
        else:
            themed_error(f"<h3 style='margin:0; color:inherit;'>Lookback Put: ${prima_lk_ex:,.4f}</h3>")
        with paso_a_paso():
            if es_call_ex:
                a1 = (np.log(S_ex/S_ext_ex) + (r_ex - q_ex + sig_ex**2/2)*T_ex) / (sig_ex*np.sqrt(T_ex))
                a2 = a1 - sig_ex*np.sqrt(T_ex)
                a3 = (np.log(S_ex/S_ext_ex) + (-r_ex + q_ex + sig_ex**2/2)*T_ex) / (sig_ex*np.sqrt(T_ex))
            else:
                a1 = (np.log(S_ext_ex/S_ex) + (-r_ex + q_ex + sig_ex**2/2)*T_ex) / (sig_ex*np.sqrt(T_ex))
                a2 = a1 - sig_ex*np.sqrt(T_ex)
                a3 = (np.log(S_ext_ex/S_ex) + (r_ex - q_ex + sig_ex**2/2)*T_ex) / (sig_ex*np.sqrt(T_ex))
            st.latex(rf"a_1 = {a1:.6f}, \quad a_2 = {a2:.6f}, \quad a_3 = {a3:.6f}")
            st.latex(rf"\text{{Prima}} = {prima_lk_ex:.4f}")

    # ── COMPUESTA REAL ─────────────────────────────────────────────────────────────
    elif tipo_exotico.startswith("Compuesta"):
        subtipo_comp_ex = st.selectbox("Subtipo:", ["Call sobre Call", "Call sobre Put", "Put sobre Call", "Put sobre Put"], key="ex_comp_sub")
        es_call_outer_ex = subtipo_comp_ex.startswith("Call")
        tipo_comp_str_ex = {"Call sobre Call": "call_on_call", "Call sobre Put": "call_on_put", "Put sobre Call": "put_on_call", "Put sobre Put": "put_on_put"}[subtipo_comp_ex]
        
        col_cp1, col_cp2 = st.columns(2)
        with col_cp1:
            K_out_ex = st.number_input("Strike exterior ($K_{out}$)", min_value=0.01, value=round(prima_van_ex, 2), step=0.5, key="ex_Kout")
            T_out_ex = st.number_input("Vencimiento exterior ($T_1$)", min_value=0.01, value=T_ex / 2, step=0.25, key="ex_T1")
            T_in_ex  = st.number_input("Vencimiento interior ($T_2 > T_1$)", min_value=T_out_ex + 0.01, value=T_ex, step=0.25, key="ex_T2")
        with col_cp2:
            prima_comp_ex = engine.opciones_compuestas(S_ex, K_out_ex, K_ex, T_out_ex, T_in_ex, r_ex, sig_ex, q_ex, tipo_comp_str_ex)
            if es_call_outer_ex:
                themed_success(f"<h3 style='margin:0; color:inherit;'>Compuesta ({subtipo_comp_ex}): ${prima_comp_ex:,.4f}</h3>")
            else:
                themed_error(f"<h3 style='margin:0; color:inherit;'>Compuesta ({subtipo_comp_ex}): ${prima_comp_ex:,.4f}</h3>")
        with paso_a_paso():
            st.latex(rf"\rho = \sqrt{{\frac{{{T_out_ex:.4f}}}{{{T_in_ex:.4f}}}}} = {np.sqrt(T_out_ex/T_in_ex):.6f}")
            st.latex(rf"\text{{Prima}} = {prima_comp_ex:.4f}")

    # ── INTERCAMBIO REAL ─────────────────────────────────────────────────
    elif tipo_exotico.startswith("Intercambio"):
        
        col_qty1, col_qty2 = st.columns(2)
        with col_qty1:
            n1_ex = st.number_input("Unidades a ENTREGAR de S1 (n1):", min_value=0.001, value=1.0, step=0.5, key="ex_n1")
        with col_qty2:
            n2_ex = st.number_input("Unidades a RECIBIR de S2 (n2):", min_value=0.001, value=1.0, step=0.5, key="ex_n2")

        separador()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**S1 — Activo que se entrega ({ticker_lbl})**")
            q1_ex = st.number_input("Dividendo q1 %", value=q_ex*100, step=0.1, key="ex_q1") / 100

            st.markdown("**S2 — Activo que se recibe**")
            ticker_ex2 = st.text_input("Ticker Activo 2:", value="MSFT", key="ex_ticker2").strip().upper()
            btn_ex2 = st.button("Cargar S2 y calcular correlacion real", key="btn_ex2", use_container_width=True)

            if btn_ex2:
                with st.spinner(f"Descargando historial de {ticker_lbl} y {ticker_ex2}..."):
                    try:
                        import yfinance as _yf
                        import datetime as _dtt
                        _hoy = _dtt.date.today()
                        _ini = _hoy - _dtt.timedelta(days=365)
                        _h1 = _yf.download(ticker_lbl, start=_ini, end=_hoy, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
                        _h2 = _yf.download(ticker_ex2, start=_ini, end=_hoy, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
                        _s2v, _v2v = engine.obtener_datos_subyacente(ticker_ex2)

                        if _s2v is None or len(_h1) < 20 or len(_h2) < 20:
                            themed_error(f"No se encontró {ticker_ex2} o datos insuficientes.")
                        else:
                            import pandas as _pd
                            _df = _pd.concat([_h1.rename("S1"), _h2.rename("S2")], axis=1, join="inner").dropna()
                            _b1 = _df["S1"] / _df["S1"].iloc[0] * 100
                            _b2 = _df["S2"] / _df["S2"].iloc[0] * 100
                            _r1 = np.log(_b1 / _b1.shift(1)).dropna()
                            _r2 = np.log(_b2 / _b2.shift(1)).dropna()
                            _rho_calc = float(_r1.corr(_r2))
                            st.session_state["ex_S2"] = float(_s2v)
                            st.session_state["ex_sig2"] = float(_v2v * 100)
                            st.session_state["ex_ticker2_ok"] = ticker_ex2
                            st.session_state["ex_rho_real"] = _rho_calc
                            st.session_state["ex_b100_1"] = _b1
                            st.session_state["ex_b100_2"] = _b2
                            themed_success(f"**{ticker_ex2}** S2=${_s2v:,.4f} | sigma2={_v2v*100:.2f}% | **rho = {_rho_calc:.4f}**")
                            st.rerun()
                    except Exception as _e:
                        themed_error(f"Error: {_e}")

            if "ex_S2" not in st.session_state: st.session_state["ex_S2"] = S_ex * 1.1
            if "ex_sig2" not in st.session_state: st.session_state["ex_sig2"] = sig_ex * 100
            if "ex_ticker2_ok" not in st.session_state: st.session_state["ex_ticker2_ok"] = "ACTIVO2"
            if "ex_rho_real" not in st.session_state: st.session_state["ex_rho_real"] = 0.5

            S2_ex   = st.number_input("Spot S2", min_value=0.001, value=float(st.session_state["ex_S2"]), step=1.0, key="ex_S2_inp")
            sig2_ex = st.number_input("Volatilidad sigma2 %", min_value=0.01, value=float(st.session_state["ex_sig2"]), step=0.5, key="ex_sig2_inp") / 100
            q2_ex   = st.number_input("Dividendo q2 %", value=0.0, step=0.1, key="ex_q2") / 100
            rho_mode = st.radio("Correlacion rho:", ["Automatica (real)", "Manual"], horizontal=True, key="ex_rho_mode")
            if rho_mode.startswith("Auto"):
                rho_ex = st.session_state["ex_rho_real"]
                st.metric("rho calculado", f"{rho_ex:.4f}")
            else:
                rho_ex = st.slider("rho manual:", min_value=-1.0, max_value=1.0, value=float(st.session_state["ex_rho_real"]), step=0.01, key="ex_rho_slider")

        ticker2_lbl = st.session_state.get("ex_ticker2_ok", "ACTIVO2")

        with col_m2:
            U_eff = n1_ex * S_ex
            V_eff = n2_ex * S2_ex
            sig_comb = np.sqrt(sig_ex**2 + sig2_ex**2 - 2*rho_ex*sig_ex*sig2_ex)

            if V_eff > 0 and U_eff > 0 and sig_comb > 0 and T_ex > 0:
                d1_int = (np.log(V_eff / U_eff) + (q1_ex - q2_ex + sig_comb**2 / 2) * T_ex) / (sig_comb * np.sqrt(T_ex))
                d2_int = d1_int - sig_comb * np.sqrt(T_ex)
                prima_int_ex = max(V_eff * np.exp(-q2_ex * T_ex) * _norm.cdf(d1_int) - U_eff * np.exp(-q1_ex * T_ex) * _norm.cdf(d2_int), 0.0)
            else:
                prima_int_ex = 0.0

            themed_success(f"<h3 style='margin:0; color:inherit;'>Intercambio: ${prima_int_ex:,.4f}</h3>")
            c1r, c2r, c3r = st.columns(3)
            c1r.metric("U = n1 * S1", f"${U_eff:,.4f}")
            c2r.metric("V = n2 * S2", f"${V_eff:,.4f}")
            c3r.metric("sigma*", f"{sig_comb*100:.4f}%")
            
            if "ex_b100_1" in st.session_state and "ex_b100_2" in st.session_state:
                import plotly.graph_objects as _go_i
                _b1 = st.session_state["ex_b100_1"]
                _b2 = st.session_state["ex_b100_2"]
                _fig_b = _go_i.Figure()
                _fig_b.add_trace(_go_i.Scatter(x=_b1.index.astype(str), y=_b1.values, name=ticker_lbl, mode="lines", line=dict(color=c_th["primary"], width=1.5)))
                _fig_b.add_trace(_go_i.Scatter(x=_b2.index.astype(str), y=_b2.values, name=ticker2_lbl, mode="lines", line=dict(color=c_th["accent"], width=1.5)))
                _fig_b.update_layout(title=f"Precios normalizados base 100 | rho={rho_ex:.4f}", xaxis_title="Fecha", yaxis_title="Indice (base 100)", height=300, **plotly_theme())
                st.plotly_chart(_fig_b, use_container_width=True)

        with paso_a_paso():
            st.latex(rf"\sigma^* = \sqrt{{{sig_ex:.4f}^2 + {sig2_ex:.4f}^2 - 2({rho_ex:.2f})({sig_ex:.4f})({sig2_ex:.4f})}} = {sig_comb:.6f}")
            st.latex(rf"d_1 = \frac{{\ln({V_eff:.2f}/{U_eff:.2f}) + ({q1_ex:.4f} - {q2_ex:.4f} + \frac{{{sig_comb:.6f}^2}}{{2}}){T_ex:.4f}}}{{{sig_comb:.6f}\sqrt{{{T_ex:.4f}}}}}")
            st.latex(rf"d_1 = {d1_int:.6f}, \quad d_2 = {d2_int:.6f}")
            st.latex(rf"c = {V_eff:.2f} e^{{-{q2_ex:.4f}({T_ex:.4f})}} N({d1_int:.4f}) - {U_eff:.2f} e^{{-{q1_ex:.4f}({T_ex:.4f})}} N({d2_int:.4f}) = {prima_int_ex:.4f}")

    # ── GRÁFICA COMPARATIVA DE TODOS LOS EXÓTICOS ─────────────────────────────
    separador()
    st.markdown("#### Comparativa de Primas — Todos los Exóticos")

    H_comp = round(S_ex * 0.85, 2)
    S_ext_comp = round(S_ex * 0.95, 2) if es_call_ex else round(S_ex * 1.05, 2)

    comparativa = [
        ("Vanilla BSM",             prima_van_ex),
        ("Gap (K2=0.9·K)",          engine.opciones_gap(S_ex, K_ex*0.9, K_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)),
        ("Cash-or-Nothing (Q=100)", engine.opciones_cash_or_nothing(S_ex, K_ex, 100, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)),
        ("Asset-or-Nothing",        engine.opciones_asset_or_nothing(S_ex, K_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)),
        ("Barrera Down-and-Out",    engine.barrera_down_and_out(S_ex, K_ex, H_comp, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)),
        ("Asiática Geométrica",     engine.opciones_asiaticas_geometricas(S_ex, K_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)),
        ("Asiática Aritmética",     engine.opciones_asiaticas_aritmeticas(S_ex, K_ex, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)),
        ("Lookback Flotante",       engine.opciones_lookback_flotante(S_ex, S_ext_comp, T_ex, r_ex, sig_ex, q_ex, tipo_str_ex)),
    ]

    nombres_comp = [r[0] for r in comparativa]
    primas_comp  = [r[1] for r in comparativa]
    colors_comp  = [c_th["accent"] if n == "Vanilla BSM" else c_th["primary"]
                    for n in nombres_comp]

    fig_comp_ex = go.Figure(go.Bar(
        x=primas_comp, y=nombres_comp,
        orientation="h",
        marker_color=colors_comp,
        text=[f"${p:,.4f}" for p in primas_comp],
        textposition="outside",
    ))
    fig_comp_ex.update_layout(
        title=f"Primas — {tipo_ex} sobre {ticker_lbl} (S={S_ex:.2f}, K={K_ex:.2f}, σ={sig_ex*100:.1f}%, T={T_ex})",
        xaxis_title="Prima ($)",
        height=420,
        margin=dict(l=180),
        **plotly_theme(),
    )
    st.plotly_chart(fig_comp_ex, use_container_width=True)

    df_comp_ex = pd.DataFrame({
        "Tipo de Exótico":   nombres_comp,
        "Prima ($)":         [f"${p:,.4f}" for p in primas_comp],
        "vs Vanilla":        [f"{(p/prima_van_ex - 1)*100:+.1f}%" if prima_van_ex > 0 else "—"
                              for p in primas_comp],
    })
    st.dataframe(df_comp_ex, use_container_width=True, hide_index=True)