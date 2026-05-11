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

# --- Estilos globales para métricas destacadas ---
math_style = "font-family: 'Times New Roman', Times, serif; font-style: italic; font-weight: normal; padding: 0 2px;"
css_titulo = "font-size: 20px; opacity: 0.85; font-weight: 500;"
css_valor = "font-size: 28px; font-weight: bold;"
css_contenedor = "display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 12px 0;"
css_paso = "text-align: center; font-size: 22px; font-weight: bold; padding: 4px 0; margin: 0;"

# Variante para métricas secundarias
css_contenedor_sm = "display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 6px 0;"
css_titulo_sm = "font-size: 16px; opacity: 0.85; font-weight: 500;"
css_valor_sm = "font-size: 22px; font-weight: bold;"

page_header(
    titulo="12. Derivados Exóticos",
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
    "Estrategias con Exoticos",
    "Subyacentes Exóticos en Vivo",
])

tab_gap, tab_bin, tab_bar, tab_asi, tab_look, tab_comp, tab_int, tab_est_ex, tab_real = tabs


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


# =============================================================================
# HELPER: gráfica de perfil de pago al vencimiento
# =============================================================================
def _chart_payoff_exotico(titulo, S0, x_vals, series: dict,
                           x_label="Precio al Vencimiento (S_T, $)"):
    """
    Genera una gráfica interactiva del perfil de pago al vencimiento para
    opciones exóticas.

    Parameters
    ----------
    titulo : str
    S0 : float  — precio spot actual (dibuja línea vertical de referencia)
    x_vals : np.ndarray  — rango del eje x
    series : dict  — {nombre: (y_array, dash_style, width)}
              El último elemento se considera la curva principal y recibe relleno.
    x_label : str
    """
    c_th = get_current_theme()
    palette = [c_th["primary"], c_th["accent"], c_th["success"],
               c_th["danger"], c_th.get("secondary", c_th["primary"])]

    fig = go.Figure()
    items = list(series.items())

    for idx, (nombre, (y, dash, lw)) in enumerate(items):
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y, mode="lines", name=nombre,
            line=dict(color=color, width=lw, dash=dash),
        ))

    # Relleno verde/rojo sobre la última curva (payoff neto)
    y_main = items[-1][1][0]
    fig.add_trace(go.Scatter(
        x=x_vals, y=np.where(y_main >= 0, y_main, 0),
        fill="tozeroy", fillcolor="rgba(40,167,69,0.15)",
        mode="none", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=np.where(y_main < 0, y_main, 0),
        fill="tozeroy", fillcolor="rgba(220,53,69,0.15)",
        mode="none", showlegend=False,
    ))

    fig.add_hline(y=0, line_dash="dash",
                  line_color=c_th.get("text_muted", "#64748B"), line_width=1)
    fig.add_vline(x=S0, line_dash="dot",
                  line_color=c_th.get("accent", "#3B82F6"),
                  annotation_text="S₀", annotation_position="top right")
    fig.update_layout(
        title=titulo,
        xaxis_title=x_label,
        yaxis_title="Payoff al Vencimiento ($)",
        height=340,
        hovermode="x unified",
        **plotly_theme(),
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                   xanchor="right", x=1))
    return fig


# =============================================================================
# HELPER: payoff terminal de una pata exótica
# =============================================================================
def _payoff_leg_exotica(tipo: str, posicion: int,
                         S_T: np.ndarray, params: dict, prima: float) -> np.ndarray:
    """
    Calcula el flujo neto al vencimiento de una pata exótica.

    tipo admite: gap_call | gap_put | con_call | con_put |
                 aon_call | aon_put | dno_call | dno_put
    posicion: +1 Long, -1 Short
    params : dict con claves según el tipo (K, K1, K2, Q, H)
    prima  : prima ya pagada/cobrada (positiva = desembolso)
    """
    K   = params.get("K",  100.0)
    K1  = params.get("K1", K)
    K2  = params.get("K2", K)
    Q   = params.get("Q",  100.0)
    H   = params.get("H",  K * 0.85)

    if tipo == "gap_call":
        payoff = np.where(S_T > K1, S_T - K2, 0.0)
    elif tipo == "gap_put":
        payoff = np.where(S_T < K1, K2 - S_T, 0.0)
    elif tipo == "con_call":
        payoff = np.where(S_T > K, Q, 0.0)
    elif tipo == "con_put":
        payoff = np.where(S_T < K, Q, 0.0)
    elif tipo == "aon_call":
        payoff = np.where(S_T > K, S_T, 0.0)
    elif tipo == "aon_put":
        payoff = np.where(S_T < K, S_T, 0.0)
    elif tipo == "dno_call":
        payoff = np.where(S_T > H, np.maximum(S_T - K, 0.0), 0.0)
    elif tipo == "dno_put":
        payoff = np.where(S_T > H, np.maximum(K - S_T, 0.0), 0.0)
    else:
        payoff = np.zeros_like(S_T)

    return posicion * payoff - posicion * prima


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPCIONES GAP
# ═════════════════════════════════════════════════════════════════════════════
with tab_gap:
    st.markdown("### Opciones Gap")
    themed_info(
        "Las opciones Gap condicionan su pago a un strike de activación (<span style='font-family: serif; font-style: italic;'>K<sub>1</sub></span>) "
        "distinto al strike de liquidación o pago (<span style='font-family: serif; font-style: italic;'>K<sub>2</sub></span>). Esto genera una discontinuidad (salto) "
        "en la función de pagos al vencimiento. <br><br>"
        "<b>Payoff (Call):</b> Se recibe <span style='font-family: serif; font-style: italic;'>S<sub>T</sub> − K<sub>2</sub></span> "
        "únicamente si <span style='font-family: serif; font-style: italic;'>S<sub>T</sub> > K<sub>1</sub></span>."
    )

    c1, c2 = st.columns(2)
    with c1:
        S_g, _, r_g, sig_g, T_g, q_g, es_call_g = _inputs_bsm_base("gap")
        K1_g = st.number_input("Strike de activación ($K_1$)", min_value=0.01,
                                value=100.0, step=1.0, key="K1_gap")
        K2_g = st.number_input("Strike de pago ($K_2$)", min_value=0.01,
                                value=90.0, step=1.0, key="K2_gap")

    with c2:
        tipo_gap = "call" if es_call_g else "put"
        prima_gap = engine.opciones_gap(S_g, K2_g, K1_g, T_g, r_g, sig_g, q_g, tipo_gap)
        tipo_txt  = "Call" if es_call_g else "Put"

        if es_call_g:
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Gap {tipo_txt} (<span style='{math_style}'>c<sub>gap</sub></span>)</span>"
                f"<span style='{css_valor}'>${prima_gap:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Gap {tipo_txt} (<span style='{math_style}'>p<sub>gap</sub></span>)</span>"
                f"<span style='{css_valor}'>${prima_gap:,.4f}</span>"
                f"</div>"
            )

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
            themed_success(f"<div style='{css_paso}'><span style='{math_style}'>c<sub>gap</sub></span> = ${prima_gap:,.4f}</div>")
        else:
            st.latex(r"p_{gap} = K_2 e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)")
            st.latex(rf"p_{{gap}} = {K2_g:.2f} e^{{-{r_g:.4f}({T_g:.4f})}} N({-d2_g:.6f}) - {S_g:.2f} e^{{-{q_g:.4f}({T_g:.4f})}} N({-d1_g:.6f})")
            t1 = K2_g*np.exp(-r_g*T_g)*_norm.cdf(-d2_g)
            t2 = S_g*np.exp(-q_g*T_g)*_norm.cdf(-d1_g)
            st.latex(rf"p_{{gap}} = {K2_g*np.exp(-r_g*T_g):.4f}({_norm.cdf(-d2_g):.6f}) - {S_g*np.exp(-q_g*T_g):.4f}({_norm.cdf(-d1_g):.6f})")
            st.latex(rf"p_{{gap}} = {t1:.4f} - {t2:.4f}")
            themed_error(f"<div style='{css_paso}'><span style='{math_style}'>p<sub>gap</sub></span> = ${prima_gap:,.4f}</div>")

    # ── Perfil de Pago al Vencimiento ────────────────────────────────────────
    separador()
    st.markdown("#### Perfil de Pago al Vencimiento")
    _c_gap = get_current_theme()
    _ST_g  = np.linspace(S_g * 0.4, S_g * 1.6, 900)
    # Vanilla reference
    _van_g = (np.maximum(_ST_g - K1_g, 0) if es_call_g
              else np.maximum(K1_g - _ST_g, 0))
    # Gap payoff (puede ser negativo entre K1 y K2 en calls)
    _gap_pf = (np.where(_ST_g > K1_g, _ST_g - K2_g, 0.0) if es_call_g
               else np.where(_ST_g < K1_g, K2_g - _ST_g, 0.0))
    _lbl_g = "Gap Call (S_T > K1 → S_T − K2)" if es_call_g else "Gap Put (S_T < K1 → K2 − S_T)"
    _fig_gap_pf = _chart_payoff_exotico(
        f"Perfil de Pago — {_lbl_g}",
        S_g, _ST_g,
        {
            "Vanilla referencia (K1)": (_van_g,  "dot", 1.5),
            _lbl_g:                    (_gap_pf, "solid", 2.5),
        },
    )
    # Marcar K1 y K2
    _fig_gap_pf.add_vline(x=K1_g, line_dash="dot",
                           line_color=_c_gap["success"],
                           annotation_text="K1 (activación)",
                           annotation_position="bottom right")
    _fig_gap_pf.add_vline(x=K2_g, line_dash="dot",
                           line_color=_c_gap["danger"],
                           annotation_text="K2 (pago)",
                           annotation_position="bottom left")
    st.plotly_chart(_fig_gap_pf, use_container_width=True)
    themed_info(
        "La discontinuidad (salto) entre <span style='font-family:serif;font-style:italic;'>K<sub>1</sub></span> "
        "y <span style='font-family:serif;font-style:italic;'>K<sub>2</sub></span> "
        "es el rasgo diferencial del Gap: el payoff puede ser negativo "
        "cuando <span style='font-family:serif;font-style:italic;'>K<sub>2</sub> > K<sub>1</sub></span> "
        "en la zona inmediatamente ITM."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPCIONES BINARIAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_bin:
    st.markdown("### Opciones Binarias (Digitales)")
    themed_info(
        "Las opciones Binarias ofrecen una estructura de pagos discontinua (todo o nada). "
        "Son los bloques atómicos con los que se construyen derivados más complejos. <br><br>"
        "• <b>Cash-or-Nothing:</b> Paga un monto fijo en efectivo (<span style='font-family: serif; font-style: italic;'>Q</span>) si la opción termina In-The-Money. <br>"
        "• <b>Asset-or-Nothing:</b> Paga exactamente una unidad del activo subyacente (<span style='font-family: serif; font-style: italic;'>S<sub>T</sub></span>) si termina In-The-Money."
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
            var_b = "c_{CoN}" if es_call_b else "p_{CoN}"
        else:
            prima_bin = engine.opciones_asset_or_nothing(S_b, K_b, T_b, r_b, sig_b, q_b, tipo_b)
            lbl = "Asset-or-Nothing Call" if es_call_b else "Asset-or-Nothing Put"
            formula_b = r"c_{AoN} = S_0 e^{-qT} N(d_1)" if es_call_b else r"p_{AoN} = S_0 e^{-qT} N(-d_1)"
            var_b = "c_{AoN}" if es_call_b else "p_{AoN}"

        if es_call_b:
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl}</span>"
                f"<span style='{css_valor}'>${prima_bin:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl}</span>"
                f"<span style='{css_valor}'>${prima_bin:,.4f}</span>"
                f"</div>"
            )

        st.latex(formula_b)

    with paso_a_paso():
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
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
            themed_success(f"<div style='{css_paso}'><span style='{math_style}'>{var_b}</span> = ${prima_bin:,.4f}</div>")
        else:
            themed_error(f"<div style='{css_paso}'><span style='{math_style}'>{var_b}</span> = ${prima_bin:,.4f}</div>")

    # ── Perfil de Pago al Vencimiento ────────────────────────────────────────
    separador()
    st.markdown("#### Perfil de Pago al Vencimiento")
    _ST_b  = np.linspace(S_b * 0.4, S_b * 1.6, 900)
    if subtipo_bin == "Cash-or-Nothing":
        _pf_b = (np.where(_ST_b > K_b, Q_b, 0.0) if es_call_b
                 else np.where(_ST_b < K_b, Q_b, 0.0))
        _lbl_b = f"Cash-or-Nothing {'Call' if es_call_b else 'Put'} — paga Q={Q_b:.2f}"
    else:
        _pf_b = (np.where(_ST_b > K_b, _ST_b, 0.0) if es_call_b
                 else np.where(_ST_b < K_b, _ST_b, 0.0))
        _lbl_b = f"Asset-or-Nothing {'Call' if es_call_b else 'Put'} — entrega S_T"
    _fig_bin_pf = _chart_payoff_exotico(
        f"Perfil de Pago — {_lbl_b}",
        S_b, _ST_b,
        {_lbl_b: (_pf_b, "solid", 2.5)},
    )
    _fig_bin_pf.add_vline(x=K_b, line_dash="dot",
                           line_color=get_current_theme()["success"],
                           annotation_text="K", annotation_position="top right")
    st.plotly_chart(_fig_bin_pf, use_container_width=True)
    themed_info(
        "La función de pago es completamente discontinua: "
        "pasa de <b>0 a Q</b> (Cash-or-Nothing) o de <b>0 a S<sub>T</sub></b> "
        "(Asset-or-Nothing) exactamente en <span style='font-family:serif;"
        "font-style:italic;'>K</span>. No existe región de transición gradual."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPCIONES DE BARRERA
# ═════════════════════════════════════════════════════════════════════════════
with tab_bar:
    st.markdown("### Opciones de Barrera (Down-and-Out / Down-and-In)")
    themed_info(
        "Las opciones de Barrera condicionan su existencia a que el precio del subyacente cruce o no un "
        "nivel específico (<span style='font-family: serif; font-style: italic;'>H</span>) durante la vigencia del contrato. "
        "Son dependientes de la trayectoria (Path-Dependent). <br><br>"
        "• <b>Down-and-Out:</b> Se desactiva y su valor cae a <span style='font-family: serif; font-style: italic;'>0</span> si el precio toca <span style='font-family: serif; font-style: italic;'>H</span>. <br>"
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
            themed_warning("La barrera H debe ser menor que el precio spot S0 para barreras Down.")

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
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl_bar}</span>"
                f"<span style='{css_valor}'>${prima_bar:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl_bar}</span>"
                f"<span style='{css_valor}'>${prima_bar:,.4f}</span>"
                f"</div>"
            )

        separador()
        col_ba1, col_ba2, col_ba3 = st.columns(3)
        col_ba1.metric("Vanilla BSM (referencia)", f"${prima_vanilla:,.4f}")
        col_ba2.metric("Down-and-Out ($c_{KO}$)",  f"${prima_ko:,.4f}")
        col_ba3.metric("Down-and-In ($c_{KI}$)", f"${prima_ki:,.4f}")

    with paso_a_paso():
        mu_b = (r_ba - q_ba - (sig_ba**2)/2) / (sig_ba**2)
        st.latex(rf"\mu = \frac{{r - q - \sigma^2/2}}{{\sigma^2}} = \frac{{{r_ba:.4f} - {q_ba:.4f} - {(sig_ba**2)/2:.6f}}}{{{sig_ba**2:.6f}}} = {mu_b:.4f}")
        st.latex(rf"\lambda = \left(\frac{{H}}{{S_0}}\right)^{{2\mu}} = \left(\frac{{{H_ba:.2f}}}{{{S_ba:.2f}}}\right)^{{2({mu_b:.4f})}}")
        st.write("---")
        st.latex(r"c_{vanilla} = c_{KO} + c_{KI}")
        st.latex(r"c_{KI} = c_{vanilla} - c_{KO}")
        st.latex(rf"c_{{KO}} = {prima_ko:.6f}")
        st.latex(rf"c_{{KI}} = {prima_vanilla:.6f} - {prima_ko:.6f} = {prima_ki:.6f}")
        
        var_bar = "c_{KO}" if es_out else "c_{KI}"
        if es_call_ba:
            themed_success(f"<div style='{css_paso}'><span style='{math_style}'>{var_bar}</span> = ${prima_bar:,.4f}</div>")
        else:
            themed_error(f"<div style='{css_paso}'><span style='{math_style}'>{var_bar}</span> = ${prima_bar:,.4f}</div>")

    # ── Perfil de Pago al Vencimiento ────────────────────────────────────────
    separador()
    st.markdown("#### Perfil de Pago al Vencimiento")
    _c_ba  = get_current_theme()
    _ST_ba = np.linspace(S_ba * 0.4, S_ba * 1.6, 900)
    _van_ba = (np.maximum(_ST_ba - K_ba, 0.0) if es_call_ba
               else np.maximum(K_ba - _ST_ba, 0.0))
    # Down-and-Out: payoff solo si S_T > H (asumimos que si el precio termina
    # por encima de H no tocó la barrera; aproximación pedagógica)
    _dno_ba = np.where(_ST_ba > H_ba, _van_ba, 0.0)
    _dni_ba = np.where(_ST_ba <= H_ba, _van_ba, 0.0)

    if es_out:
        _pf_ba_main = _dno_ba
        _lbl_ba = f"Down-and-Out {'Call' if es_call_ba else 'Put'}"
    else:
        _pf_ba_main = _dni_ba
        _lbl_ba = f"Down-and-In {'Call' if es_call_ba else 'Put'}"

    _fig_bar_pf = _chart_payoff_exotico(
        f"Perfil de Pago — {_lbl_ba}",
        S_ba, _ST_ba,
        {
            "Vanilla referencia": (_van_ba,    "dot",   1.5),
            _lbl_ba:              (_pf_ba_main, "solid", 2.5),
        },
    )
    # Zona de barrera sombreada
    _fig_bar_pf.add_vrect(
        x0=_ST_ba[0], x1=H_ba,
        fillcolor="rgba(220,53,69,0.08)",
        layer="below", line_width=0,
        annotation_text="Zona de activación (H)",
        annotation_position="top left",
    )
    _fig_bar_pf.add_vline(x=H_ba, line_dash="dash",
                           line_color=_c_ba["danger"],
                           annotation_text=f"H = {H_ba:.2f}",
                           annotation_position="top right")
    st.plotly_chart(_fig_bar_pf, use_container_width=True)
    themed_info(
        "<b>Nota:</b> el diagrama asume que el precio termina en <span "
        "style='font-family:serif;font-style:italic;'>S_T</span> sin haber tocado "
        "<span style='font-family:serif;font-style:italic;'>H</span> durante la vigencia "
        "(Down-and-Out) o habiéndolo tocado (Down-and-In). "
        "La dependencia de trayectoria real se refleja en la prima, no sólo en el payoff terminal."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — OPCIONES ASIÁTICAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_asi:
    st.markdown("### Opciones Asiáticas")
    themed_info(
        "Las opciones Asiáticas derivan su valor del **precio promedio** del subyacente (<span style='font-family: serif; font-style: italic;'>S<sub>avg</sub></span>) "
        "durante un periodo de observación determinado. Esta promediación mitiga la volatilidad efectiva del activo, "
        "haciéndolas más baratas que las opciones vanilla y previniendo la manipulación del precio al vencimiento."
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
        else:
            prima_asi = engine.opciones_asiaticas_aritmeticas(
                S_as, K_as, T_as, r_as, sig_as, q_as, tipo_as
            )
            lbl_as = "Asiatica Aritmetica Call" if es_call_as else "Asiatica Aritmetica Put"
            
        if es_call_as:
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl_as}</span>"
                f"<span style='{css_valor}'>${prima_asi:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl_as}</span>"
                f"<span style='{css_valor}'>${prima_asi:,.4f}</span>"
                f"</div>"
            )

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
            st.latex(rf"q^* = r - b^* = {r_as:.4f} - {b_star:.6f} = {q_star:.6f}")
            d1_as = (np.log(S_as/K_as) + (r_as - q_star + sig_star**2/2)*T_as) / (sig_star*np.sqrt(T_as))
            d2_as = d1_as - sig_star*np.sqrt(T_as)
            st.latex(rf"d_1 = \frac{{\ln({S_as:.2f}/{K_as:.2f}) + ({r_as:.4f} - {q_star:.6f} + \frac{{{sig_star:.6f}^2}}{{2}}){T_as:.4f}}}{{{sig_star:.6f}\sqrt{{{T_as:.4f}}}}} = {d1_as:.6f}")
            st.latex(rf"d_2 = d_1 - \sigma^*\sqrt{{T}} = {d1_as:.6f} - {sig_star*np.sqrt(T_as):.6f} = {d2_as:.6f}")
            st.write("---")
            if es_call_as:
                st.latex(rf"c_{{asi}} = {S_as:.2f} e^{{-{q_star:.6f}({T_as:.4f})}} N({d1_as:.4f}) - {K_as:.2f} e^{{-{r_as:.4f}({T_as:.4f})}} N({d2_as:.4f}) = {prima_asi:.4f}")
            else:
                st.latex(rf"p_{{asi}} = {K_as:.2f} e^{{-{r_as:.4f}({T_as:.4f})}} N({-d2_as:.4f}) - {S_as:.2f} e^{{-{q_star:.6f}({T_as:.4f})}} N({-d1_as:.4f}) = {prima_asi:.4f}")
        else:
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
            
            st.latex(r"M_1 = \mathbb{E}[A_T]")
            st.latex(r"M_2 = \mathbb{E}[A_T^2]")
            st.latex(rf"M_1 = {M1:.6f}, \quad M_2 = {M2:.6f}")
            st.latex(rf"\sigma_{{TW}} = \sqrt{{\frac{{\ln(M_2 / M_1^2)}}{{T}}}} = {sig_tw:.6f}")
            st.write("---")
            st.latex(rf"\text{{Prima}} = \text{{BSM}}(S_0^*={M1:.4f}, K={K_as:.2f}, \sigma^*={sig_tw:.6f}) = {prima_asi:.4f}")
            
        var_asi = "c_{asi}" if es_call_as else "p_{asi}"
        if es_call_as:
            themed_success(f"<div style='{css_paso}'><span style='{math_style}'>{var_asi}</span> = ${prima_asi:,.4f}</div>")
        else:
            themed_error(f"<div style='{css_paso}'><span style='{math_style}'>{var_asi}</span> = ${prima_asi:,.4f}</div>")

    # ── Perfil de Pago al Vencimiento ────────────────────────────────────────
    separador()
    st.markdown("#### Perfil de Pago al Vencimiento")
    themed_info(
        "El eje horizontal representa el <b>precio promedio A<sub>T</sub></b> "
        "del subyacente durante la vida de la opción (no el precio terminal). "
        "El perfil es idéntico al vanilla pero se aplica sobre la media, lo que "
        "reduce la volatilidad efectiva y por lo tanto la prima."
    )
    _AT_as = np.linspace(S_as * 0.4, S_as * 1.6, 900)
    _pf_van_as = (np.maximum(_AT_as - K_as, 0.0) if es_call_as
                  else np.maximum(K_as - _AT_as, 0.0))
    # El payoff asiático es max(A_T - K, 0), idéntico en forma pero con
    # volatilidad reducida (σ* = σ/√3 para geométrica)
    _lbl_as = ("Asiatica Call — max(A_T − K, 0)" if es_call_as
               else "Asiatica Put — max(K − A_T, 0)")
    _lbl_van_as = "Vanilla (referencia, sobre S_T)"
    _fig_asi_pf = _chart_payoff_exotico(
        f"Perfil de Pago vs Precio Promedio — {_lbl_as}",
        S_as, _AT_as,
        {
            _lbl_van_as: (_pf_van_as, "dot", 1.5),
            _lbl_as:     (_pf_van_as, "solid", 2.5),
        },
        x_label="Precio Promedio al Vencimiento (A_T, $)",
    )
    _fig_asi_pf.add_vline(x=K_as, line_dash="dot",
                           line_color=get_current_theme()["success"],
                           annotation_text="K", annotation_position="top right")
    st.plotly_chart(_fig_asi_pf, use_container_width=True)
    # Mostrar comparativa de primas en métricas
    _col_as1, _col_as2, _col_as3 = st.columns(3)
    _col_as1.metric("Prima Asiatica", f"${prima_asi:,.4f}")
    _col_as2.metric("Prima Vanilla BSM", f"${prima_van_as:,.4f}")
    _col_as3.metric("Descuento por Promediacion", f"{descuento_as:.2f}%",
                    delta=f"-{descuento_as:.2f}%", delta_color="inverse")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — OPCIONES LOOKBACK
# ═════════════════════════════════════════════════════════════════════════════
with tab_look:
    st.markdown("### Opciones Lookback (Precio Flotante)")
    themed_info(
        "Las opciones Lookback otorgan el derecho retrospectivo de ejercer al precio más favorable "
        "alcanzado durante la vida del contrato (<span style='font-family: serif; font-style: italic;'>S<sub>min</sub></span> para Calls, <span style='font-family: serif; font-style: italic;'>S<sub>max</sub></span> para Puts). "
        "Al garantizar siempre el mejor precio posible, estadísticamente son los derivados vanilla más caros del mercado."
    )

    c1, c2 = st.columns(2)
    with c1:
        S_lk, _, r_lk, sig_lk, T_lk, q_lk, es_call_lk = _inputs_bsm_base(
            "look", mostrar_tipo=True
        )
        S_min_max = st.number_input(
            "Extremo observado (S_min Call / S_max Put)",
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
        lbl_lk = "Lookback Call (Mínimo Flotante)" if es_call_lk else "Lookback Put (Máximo Flotante)"

        if es_call_lk:
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl_lk}</span>"
                f"<span style='{css_valor}'>${prima_lk:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>{lbl_lk}</span>"
                f"<span style='{css_valor}'>${prima_lk:,.4f}</span>"
                f"</div>"
            )

        prima_van_lk = engine.black_scholes(S_lk, S_min_max, r_lk, sig_lk, T_lk,
                                             es_call_lk, q_lk)
        st.metric("Vanilla BSM (referencia, K = extremo)", f"${prima_van_lk:,.4f}")

    with paso_a_paso():
        if es_call_lk:
            a1 = (np.log(S_lk/S_min_max) + (r_lk - q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            a2 = a1 - sig_lk*np.sqrt(T_lk)
            a3 = (np.log(S_lk/S_min_max) + (-r_lk + q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            st.latex(rf"a_1 = \frac{{\ln(S_0/S_{{min}}) + (r-q+\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a1:.6f}")
            st.latex(rf"a_2 = a_1 - \sigma\sqrt{{T}} = {a2:.6f}")
            st.latex(rf"a_3 = \frac{{\ln(S_0/S_{{min}}) - (r-q-\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a3:.6f}")
            st.write("---")
            st.latex(r"c_{LB} = S_0 e^{-qT} N(a_1) - S_{min} e^{-rT} N(a_2) + S_0 e^{-rT} \frac{\sigma^2}{2(r-q)} \left[ \left(\frac{S_0}{S_{min}}\right)^{-\frac{2(r-q)}{\sigma^2}} N(-a_3) - e^{qT} N(-a_1) \right]")
            themed_success(f"<div style='{css_paso}'><span style='{math_style}'>c<sub>LB</sub></span> = ${prima_lk:,.4f}</div>")
        else:
            a1 = (np.log(S_min_max/S_lk) + (-r_lk + q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            a2 = a1 - sig_lk*np.sqrt(T_lk)
            a3 = (np.log(S_min_max/S_lk) + (r_lk - q_lk + sig_lk**2/2)*T_lk) / (sig_lk*np.sqrt(T_lk))
            st.latex(rf"a_1 = \frac{{\ln(S_{{max}}/S_0) + (-r+q+\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a1:.6f}")
            st.latex(rf"a_2 = a_1 - \sigma\sqrt{{T}} = {a2:.6f}")
            st.latex(rf"a_3 = \frac{{\ln(S_{{max}}/S_0) + (r-q+\sigma^2/2)T}}{{\sigma\sqrt{{T}}}} = {a3:.6f}")
            st.write("---")
            st.latex(r"p_{LB} = S_{max} e^{-rT} N(a_1) - S_0 e^{-qT} N(a_2) + S_0 e^{-rT} \frac{\sigma^2}{2(r-q)} \left[ e^{qT} N(a_1) - \left(\frac{S_0}{S_{max}}\right)^{\frac{2(r-q)}{\sigma^2}} N(a_3) \right]")
            themed_error(f"<div style='{css_paso}'><span style='{math_style}'>p<sub>LB</sub></span> = ${prima_lk:,.4f}</div>")

    # ── Perfil de Pago al Vencimiento ────────────────────────────────────────
    separador()
    st.markdown("#### Perfil de Pago al Vencimiento")
    themed_info(
        "El payoff de la Lookback es lineal en <span style='font-family:serif;"
        "font-style:italic;'>S<sub>T</sub></span> para un extremo observado fijo. "
        "El <b>extremo</b> (<span style='font-family:serif;font-style:italic;"
        ">S<sub>min</sub></span> para calls, <span style='font-family:serif;"
        "font-style:italic;'>S<sub>max</sub></span> para puts) queda fijo en el "
        "valor ingresado arriba."
    )
    _ST_lk = np.linspace(S_lk * 0.4, S_lk * 1.6, 900)
    _van_lk = (np.maximum(_ST_lk - S_min_max, 0.0) if es_call_lk
               else np.maximum(S_min_max - _ST_lk, 0.0))
    if es_call_lk:
        # Lookback call: S_T - S_min (siempre >= 0 cuando S_T >= S_min)
        _pf_lk = np.maximum(_ST_lk - S_min_max, 0.0)
        _ext_lbl = f"S_min = {S_min_max:.2f}"
        _ext_color = get_current_theme()["danger"]
    else:
        # Lookback put: S_max - S_T (siempre >= 0 cuando S_T <= S_max)
        _pf_lk = np.maximum(S_min_max - _ST_lk, 0.0)
        _ext_lbl = f"S_max = {S_min_max:.2f}"
        _ext_color = get_current_theme()["success"]

    _lbl_lk = ("Lookback Call — S_T − S_min" if es_call_lk
               else "Lookback Put — S_max − S_T")
    _fig_lk_pf = _chart_payoff_exotico(
        f"Perfil de Pago — {_lbl_lk}",
        S_lk, _ST_lk,
        {
            "Vanilla referencia (K = extremo)": (_van_lk, "dot", 1.5),
            _lbl_lk:                            (_pf_lk,  "solid", 2.5),
        },
    )
    _fig_lk_pf.add_vline(x=S_min_max, line_dash="dash",
                          line_color=_ext_color,
                          annotation_text=_ext_lbl,
                          annotation_position="top right")
    st.plotly_chart(_fig_lk_pf, use_container_width=True)
    _col_lk1, _col_lk2, _col_lk3 = st.columns(3)
    _col_lk1.metric("Prima Lookback", f"${prima_lk:,.4f}")
    _col_lk2.metric("Prima Vanilla (K = extremo)", f"${prima_van_lk:,.4f}")
    _sobrecosto = (prima_lk / prima_van_lk - 1) * 100 if prima_van_lk > 0 else 0.0
    _col_lk3.metric("Sobrecosto vs Vanilla", f"+{_sobrecosto:.2f}%",
                    delta=f"+{_sobrecosto:.2f}%", delta_color="normal")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — OPCIONES COMPUESTAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_comp:
    st.markdown("### Opciones Compuestas (Opción sobre Opción)")
    themed_info(
        "Las opciones Compuestas otorgan el derecho a comprar o vender otra opción en una fecha "
        "futura (<span style='font-family: serif; font-style: italic;'>T<sub>1</sub></span>) pagando un precio de ejercicio primario (<span style='font-family: serif; font-style: italic;'>K<sub>out</sub></span>). "
        "Requieren el uso de la distribución normal bivariada para calcular la probabilidad conjunta de ejercicio."
    )

    subtipo_comp = st.selectbox(
        "Tipo de opción compuesta:",
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
        st.markdown("**Opción exterior (contrato inicial)**")
        S_cp2   = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                                   value=100.0, step=1.0, key="comp_S")
        K_out   = st.number_input("Strike exterior ($K_{out}$)",
                                   min_value=0.01, value=5.0, step=0.5, key="comp_Kout")
        T_out   = st.number_input("Vencimiento exterior ($T_1$) años",
                                   min_value=0.001, value=0.5, step=0.25, key="comp_T1")
        st.markdown("**Opción interior (activo subyacente)**")
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
                themed_success(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>{subtipo_comp}</span>"
                    f"<span style='{css_valor}'>${prima_comp:,.4f}</span>"
                    f"</div>"
                )
            else:
                themed_error(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>{subtipo_comp}</span>"
                    f"<span style='{css_valor}'>${prima_comp:,.4f}</span>"
                    f"</div>"
                )

            prima_inner = engine.black_scholes(S_cp2, K_in, r_cp2, sig_cp2, T_in,
                                                es_call_inner, q_cp2)
            st.metric("Prima de la opción interior aislada (BSM)", f"${prima_inner:,.4f}")
        except Exception as e:
            themed_error(f"Error en el cálculo: {e}")

    with paso_a_paso():
        rho = np.sqrt(T_out / T_in)
        st.latex(r"\rho = \sqrt{\frac{T_1}{T_2}}")
        st.latex(rf"\rho = \sqrt{{\frac{{{T_out:.4f}}}{{{T_in:.4f}}}}} = {rho:.6f}")
        st.write("---")
        st.latex(r"S^* \leftarrow \text{Precio crítico que resuelve } \text{BSM}(S^*, T_2-T_1) = K_{out}")
        st.latex(r"a_{1,2} = \frac{\ln(S_0/S^*) + (r-q \pm \sigma^2/2)T_1}{\sigma\sqrt{T_1}}")
        st.latex(r"b_{1,2} = \frac{\ln(S_0/K_{in}) + (r-q \pm \sigma^2/2)T_2}{\sigma\sqrt{T_2}}")
        st.write("---")
        
        var_comp = "c_{cc}" if es_call_outer and es_call_inner else "c_{cp}" if es_call_outer else "p_{cc}" if es_call_inner else "p_{cp}"
        if es_call_outer and es_call_inner:
            st.latex(r"c_{cc} = S_0 e^{-qT_2} M(a_1, b_1; \rho) - K_{in} e^{-rT_2} M(a_2, b_2; \rho) - K_{out} e^{-rT_1} N(a_2)")
        elif es_call_outer and not es_call_inner:
            st.latex(r"c_{cp} = K_{in} e^{-rT_2} M(-a_2, -b_2; \rho) - S_0 e^{-qT_2} M(-a_1, -b_1; \rho) - K_{out} e^{-rT_1} N(-a_2)")
        elif not es_call_outer and es_call_inner:
            st.latex(r"p_{cc} = K_{out} e^{-rT_1} N(-a_2) - S_0 e^{-qT_2} M(-a_1, b_1; -\rho) + K_{in} e^{-rT_2} M(-a_2, b_2; -\rho)")
        else:
            st.latex(r"p_{cp} = K_{out} e^{-rT_1} N(a_2) - K_{in} e^{-rT_2} M(a_2, -b_2; -\rho) + S_0 e^{-qT_2} M(a_1, -b_1; -\rho)")
            
        if es_call_outer:
            themed_success(f"<div style='{css_paso}'><span style='{math_style}'>{var_comp}</span> = ${prima_comp:,.4f}</div>")
        else:
            themed_error(f"<div style='{css_paso}'><span style='{math_style}'>{var_comp}</span> = ${prima_comp:,.4f}</div>")

    # ── Perfil de Pago al Vencimiento ────────────────────────────────────────
    separador()
    st.markdown("#### Perfil de Pago al Vencimiento (en T1 — fecha de ejercicio exterior)")
    themed_info(
        "En <span style='font-family:serif;font-style:italic;'>T<sub>1</sub></span> el tenedor decide "
        "si pagar <span style='font-family:serif;font-style:italic;'>K<sub>out</sub></span> para adquirir "
        "la opción interior. El payoff del comprador es "
        "<span style='font-family:serif;font-style:italic;'>max(V<sub>inner</sub>(S,T<sub>1</sub>) − K<sub>out</sub>, 0)</span>. "
        "El eje X muestra el rango de precios spot en <span style='font-family:serif;font-style:italic;'>T<sub>1</sub></span>."
    )
    _ST_cp = np.linspace(S_cp2 * 0.4, S_cp2 * 1.6, 300)
    _inner_vals = np.array([
        engine.black_scholes(s, K_in, r_cp2, sig_cp2, T_in - T_out, es_call_inner, q_cp2)
        for s in _ST_cp
    ])
    _outer_pf = (np.maximum(_inner_vals - K_out, 0.0) if es_call_outer
                 else np.maximum(K_out - _inner_vals, 0.0))
    _lbl_cp = subtipo_comp
    _fig_cp_pf = _chart_payoff_exotico(
        f"Perfil de Pago en T1 — {_lbl_cp}",
        S_cp2, _ST_cp,
        {
            f"Valor opcion interior BSM(S, T2-T1)": (_inner_vals, "dot", 1.5),
            f"Payoff {_lbl_cp} — max(inner - K_out, 0)": (_outer_pf, "solid", 2.5),
        },
    )
    _fig_cp_pf.add_hline(y=K_out, line_dash="dash",
                          line_color=get_current_theme()["danger"],
                          annotation_text=f"K_out = {K_out:.2f}",
                          annotation_position="right")
    st.plotly_chart(_fig_cp_pf, use_container_width=True)
    themed_info(
        "El apalancamiento es doble: primero, el payoff de la opción exterior es no lineal en la "
        "opción interior; segundo, la opción interior también es no lineal en el subyacente. "
        "Esto produce una curva de pago con <b>mayor convexidad</b> que cualquier opción simple."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — OPCIONES DE INTERCAMBIO (MARGRABE)
# ═════════════════════════════════════════════════════════════════════════════
with tab_int:
    st.markdown("### Opciones de Intercambio (Fórmula de Margrabe)")
    themed_info(
        "Otorga el derecho de ceder un activo (<span style='font-family: serif; font-style: italic;'>S<sub>1</sub></span>) a cambio de recibir otro activo (<span style='font-family: serif; font-style: italic;'>S<sub>2</sub></span>), "
        "eliminando el efectivo como strike. Su valor depende intrínsecamente de la **correlación cruzada** (<span style='font-family: serif; font-style: italic;'>&rho;</span>) entre ambos activos."
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
        S1_int = st.number_input("Spot S1 (Activo a Entregar)", min_value=0.01, value=100.0, step=1.0, key="int_S1")
        sig1_in = st.number_input("Volatilidad sigma1 %", min_value=0.01, value=20.0, step=0.5, key="int_sig1") / 100
        q1_int = st.number_input("Dividendo q1 %", value=3.0, step=0.1, key="int_q1") / 100

        S2_int = st.number_input("Spot S2 (Activo a Recibir)", min_value=0.01, value=110.0, step=1.0, key="int_S2")
        sig2_in = st.number_input("Volatilidad sigma2 %", min_value=0.01, value=25.0, step=0.5, key="int_sig2") / 100
        q2_int = st.number_input("Dividendo q2 %", value=2.0, step=0.1, key="int_q2") / 100

        rho_int = st.slider("Coeficiente de Correlación (rho):", min_value=-1.0, max_value=1.0, value=0.5, step=0.01, key="int_rho")
        T_int = st.number_input("Tiempo al Vencimiento (T) años", min_value=0.001, value=1.0, step=0.25, key="int_T")

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

        themed_success(
            f"<div style='{css_contenedor}'>"
            f"<span style='{css_titulo}'>Intercambio (Recibir S2, Entregar S1)</span>"
            f"<span style='{css_valor}'>${prima_int:,.4f}</span>"
            f"</div>"
        )
        c1r, c2r, c3r = st.columns(3)
        c1r.metric("Vector Efectivo U (n1 * S1)", f"${U_eff:,.4f}")
        c2r.metric("Vector Efectivo V (n2 * S2)", f"${V_eff:,.4f}")
        c3r.metric("Volatilidad Conjunta (σ*)", f"{sig_comb*100:.4f}%")

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
        themed_success(f"<div style='{css_paso}'><span style='{math_style}'>c</span> = ${prima_int:,.4f}</div>")

    # ── Perfil de Pago al Vencimiento ────────────────────────────────────────
    separador()
    st.markdown("#### Perfil de Pago al Vencimiento")
    themed_info(
        "Se mantiene <span style='font-family:serif;font-style:italic;'>S<sub>2</sub></span> fijo en "
        "su valor actual y se varía <span style='font-family:serif;font-style:italic;'>S<sub>1</sub></span> "
        "en el eje X. El payoff es "
        "<span style='font-family:serif;font-style:italic;'>max(n<sub>2</sub>·S<sub>2</sub> − n<sub>1</sub>·S<sub>1,T</sub>, 0)</span>."
    )
    _S1T_int  = np.linspace(S1_int * 0.3, S1_int * 1.7, 900)
    _U_int    = n1_ex * _S1T_int
    _V_int    = n2_ex * S2_int
    _pf_int   = np.maximum(_V_int - _U_int, 0.0)
    _breakeven = _V_int / n1_ex  # n1*S1 = n2*S2  →  S1 = n2*S2/n1
    _fig_int_pf = _chart_payoff_exotico(
        "Perfil de Pago — Intercambio (Margrabe): max(n2·S2 − n1·S1,T, 0)",
        S1_int, _S1T_int,
        {
            "Payoff bruto (sin prima)": (_pf_int, "solid", 2.5),
        },
        x_label="Precio de S1 al Vencimiento ($)",
    )
    _fig_int_pf.add_vline(x=_breakeven, line_dash="dash",
                           line_color=get_current_theme()["success"],
                           annotation_text=f"Breakeven S1 = {_breakeven:.2f}",
                           annotation_position="top right")
    _fig_int_pf.add_hline(y=_V_int, line_dash="dot",
                           line_color=get_current_theme()["accent"],
                           annotation_text=f"n2·S2 = {_V_int:.2f}",
                           annotation_position="right")
    st.plotly_chart(_fig_int_pf, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — ESTRATEGIAS CON DERIVADOS EXOTICOS
# ═════════════════════════════════════════════════════════════════════════════
with tab_est_ex:
    st.markdown("### Estrategias con Derivados Exóticos")
    themed_info(
        "Combina opciones exóticas y vainilla para construir perfiles de pago a medida. "
        "Selecciona una estrategia predefinida o arma tu propia estructura en el modo manual. "
        "El diagrama muestra el <b>P&amp;L neto al vencimiento</b> incluyendo primas pagadas/cobradas."
    )

    # ── Parámetros globales de mercado ────────────────────────────────────────
    separador()
    st.markdown("#### Parámetros de Mercado (comunes a todas las patas)")
    _ec1, _ec2, _ec3, _ec4 = st.columns(4)
    S_est   = _ec1.number_input("Spot ($S_0$)",               min_value=0.01, value=100.0, step=1.0,  key="est_S")
    r_est   = _ec2.number_input("Tasa libre ($r$) %",         value=5.0,  step=0.1,  key="est_r")  / 100
    sig_est = _ec3.number_input("Volatilidad ($\\sigma$) %",  min_value=0.01, value=20.0, step=0.5,  key="est_sig") / 100
    T_est   = _ec4.number_input("Vencimiento ($T$) años",     min_value=0.01, value=0.5,  step=0.25, key="est_T")
    q_est   = 0.0

    separador()
    modo_est = st.radio(
        "Modo de construcción:",
        ["Estrategia Predefinida", "Manual (hasta 4 patas)"],
        horizontal=True, key="est_modo",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODO A — ESTRATEGIAS PREDEFINIDAS
    # ─────────────────────────────────────────────────────────────────────────
    if modo_est == "Estrategia Predefinida":
        ESTRATEGIAS_EX = {
            "Collar Digital": (
                "Compra una Call Cash-or-Nothing (cobra Q si S_T > K2) y vende una "
                "Put Cash-or-Nothing (paga Q si S_T < K1). Limita la exposición a "
                "la banda [K1, K2] con payoffs discontinuos en ambos extremos."
            ),
            "Strangle Binario": (
                "Compra una Call y una Put Cash-or-Nothing fuera del dinero. "
                "Paga Q si el precio sale del rango [K_put, K_call] en cualquier dirección. "
                "Rentable en entornos de alta volatilidad sin importar la dirección."
            ),
            "Gap Risk Reversal": (
                "Compra una Gap Call y vende una Gap Put con los mismos K1 y K2. "
                "Perfil alcista asimétrico con salto de pago entre K1 y K2; equivale "
                "a un Risk Reversal clásico pero con discontinuidad en el payoff."
            ),
            "Spread de Barrera (Bull)": (
                "Compra una Vanilla Call y vende una Down-and-Out Call con la misma K. "
                "Reduce el costo de la call a cambio de perder protección si el subyacente "
                "cae por debajo de la barrera H durante la vigencia del contrato."
            ),
            "Straddle Asiatico": (
                "Compra una Call Asiática y una Put Asiática con el mismo strike K. "
                "Beneficia cuando la media del activo se aleja de K, con prima inferior "
                "a un straddle vanilla porque la volatilidad efectiva se reduce."
            ),
        }

        nombre_est = st.selectbox("Estrategia:", list(ESTRATEGIAS_EX.keys()), key="est_nombre")
        themed_info(ESTRATEGIAS_EX[nombre_est])
        separador()

        _pe1, _pe2, _pe3 = st.columns(3)

        if nombre_est == "Collar Digital":
            K1_cd = _pe1.number_input("K1 — Strike Put (piso)",   min_value=0.01, value=90.0,  step=1.0, key="cd_K1")
            K2_cd = _pe2.number_input("K2 — Strike Call (techo)", min_value=0.01, value=110.0, step=1.0, key="cd_K2")
            Q_cd  = _pe3.number_input("Pago binario (Q)",          min_value=0.01, value=10.0,  step=1.0, key="cd_Q")
            pc    = engine.opciones_cash_or_nothing(S_est, K2_cd, Q_cd, T_est, r_est, sig_est, q_est, "call")
            pp    = engine.opciones_cash_or_nothing(S_est, K1_cd, Q_cd, T_est, r_est, sig_est, q_est, "put")
            costo_neto = pc - pp
            _ST = np.linspace(S_est * 0.4, S_est * 1.6, 900)
            y1  =  np.where(_ST > K2_cd, Q_cd, 0.0) - pc
            y2  = -(np.where(_ST < K1_cd, Q_cd, 0.0) - pp)
            ynet = y1 + y2
            _series = {"Long CoN Call": (y1,"dot",1.5), "Short CoN Put": (y2,"dash",1.5), "P&L Neto": (ynet,"solid",2.5)}
            _vlines = [(K1_cd,"K1 piso"), (K2_cd,"K2 techo")]

        elif nombre_est == "Strangle Binario":
            K_sc = _pe1.number_input("K Call (barrera superior)", min_value=0.01, value=110.0, step=1.0, key="sc_Kc")
            K_sp = _pe2.number_input("K Put (barrera inferior)",  min_value=0.01, value=90.0,  step=1.0, key="sc_Kp")
            Q_sc = _pe3.number_input("Pago binario (Q)",          min_value=0.01, value=15.0,  step=1.0, key="sc_Q")
            pc   = engine.opciones_cash_or_nothing(S_est, K_sc, Q_sc, T_est, r_est, sig_est, q_est, "call")
            pp   = engine.opciones_cash_or_nothing(S_est, K_sp, Q_sc, T_est, r_est, sig_est, q_est, "put")
            costo_neto = pc + pp
            _ST = np.linspace(S_est * 0.4, S_est * 1.6, 900)
            y1   = np.where(_ST > K_sc, Q_sc, 0.0) - pc
            y2   = np.where(_ST < K_sp, Q_sc, 0.0) - pp
            ynet = y1 + y2
            _series = {"Long CoN Call": (y1,"dot",1.5), "Long CoN Put": (y2,"dash",1.5), "P&L Neto": (ynet,"solid",2.5)}
            _vlines = [(K_sp,"K Put"), (K_sc,"K Call")]

        elif nombre_est == "Gap Risk Reversal":
            K1_gr = _pe1.number_input("K1 (activación)", min_value=0.01, value=100.0, step=1.0, key="gr_K1")
            K2_gr = _pe2.number_input("K2 (pago)",       min_value=0.01, value=105.0, step=1.0, key="gr_K2")
            pgc   = engine.opciones_gap(S_est, K1_gr, K2_gr, T_est, r_est, sig_est, q_est, "call")
            pgp   = engine.opciones_gap(S_est, K1_gr, K2_gr, T_est, r_est, sig_est, q_est, "put")
            costo_neto = pgc - pgp
            _ST = np.linspace(S_est * 0.4, S_est * 1.6, 900)
            y1   =  np.where(_ST > K1_gr, _ST - K2_gr, 0.0) - pgc
            y2   = -(np.where(_ST < K1_gr, K2_gr - _ST, 0.0) - pgp)
            ynet = y1 + y2
            _series = {"Long Gap Call": (y1,"dot",1.5), "Short Gap Put": (y2,"dash",1.5), "P&L Neto": (ynet,"solid",2.5)}
            _vlines = [(K1_gr,"K1 activación"), (K2_gr,"K2 pago")]

        elif nombre_est == "Spread de Barrera (Bull)":
            K_sb  = _pe1.number_input("Strike ($K$)",    min_value=0.01, value=100.0, step=1.0, key="sb_K")
            H_sb  = _pe2.number_input("Barrera ($H$)",   min_value=0.01, value=85.0,  step=1.0, key="sb_H")
            pvan  = engine.black_scholes(S_est, K_sb, r_est, sig_est, T_est, True, q_est)
            pdno  = engine.barrera_down_and_out(S_est, K_sb, H_sb, T_est, r_est, sig_est, q_est, "call")
            costo_neto = pvan - pdno
            _ST = np.linspace(S_est * 0.4, S_est * 1.6, 900)
            y1   =  np.maximum(_ST - K_sb, 0.0) - pvan
            y2   = -(np.where(_ST > H_sb, np.maximum(_ST - K_sb, 0.0), 0.0) - pdno)
            ynet = y1 + y2
            _series = {"Long Vanilla Call": (y1,"dot",1.5), "Short DNO Call": (y2,"dash",1.5), "P&L Neto": (ynet,"solid",2.5)}
            _vlines = [(H_sb,"H barrera"), (K_sb,"K strike")]

        else:  # Straddle Asiatico
            K_av = _pe1.number_input("Strike ($K$)", min_value=0.01, value=100.0, step=1.0, key="av_K")
            pac  = engine.opciones_asiaticas_geometricas(S_est, K_av, T_est, r_est, sig_est, q_est, "call")
            pap  = engine.opciones_asiaticas_geometricas(S_est, K_av, T_est, r_est, sig_est, q_est, "put")
            costo_neto = pac + pap
            _ST  = np.linspace(S_est * 0.4, S_est * 1.6, 900)
            y1   = np.maximum(_ST - K_av, 0.0) - pac
            y2   = np.maximum(K_av - _ST, 0.0) - pap
            ynet = y1 + y2
            _series = {"Long Asian Call": (y1,"dot",1.5), "Long Asian Put": (y2,"dash",1.5), "P&L Neto": (ynet,"solid",2.5)}
            _vlines = [(K_av,"K")]

        # ── Métricas ──────────────────────────────────────────────────────────
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("Prima Neta",
                    f"${costo_neto:,.4f}",
                    delta="Débito (pagado)" if costo_neto > 0 else "Crédito (cobrado)",
                    delta_color="inverse" if costo_neto > 0 else "normal")
        _max_y = float(np.max(list(_series.values())[-1][0]))
        _min_y = float(np.min(list(_series.values())[-1][0]))
        _mc2.metric("Ganancia máxima", f"${_max_y:,.2f}")
        _mc3.metric("Perdida máxima",  f"${_min_y:,.2f}")

        # ── Gráfica ───────────────────────────────────────────────────────────
        separador()
        st.markdown("#### Perfil de P&L al Vencimiento")
        _c_est   = get_current_theme()
        _pal_est = [_c_est["primary"], _c_est["success"], _c_est["accent"], _c_est["danger"]]
        _fig_est = go.Figure()
        for idx, (lbl, (y, dash, lw)) in enumerate(_series.items()):
            _fig_est.add_trace(go.Scatter(
                x=_ST, y=y, mode="lines", name=lbl,
                line=dict(color=_pal_est[idx % len(_pal_est)], width=lw, dash=dash),
            ))
        _ynet = list(_series.values())[-1][0]
        _fig_est.add_trace(go.Scatter(x=_ST, y=np.where(_ynet >= 0, _ynet, 0),
            fill="tozeroy", fillcolor="rgba(40,167,69,0.12)", mode="none", showlegend=False))
        _fig_est.add_trace(go.Scatter(x=_ST, y=np.where(_ynet < 0, _ynet, 0),
            fill="tozeroy", fillcolor="rgba(220,53,69,0.12)", mode="none", showlegend=False))
        _fig_est.add_hline(y=0, line_dash="dash",
                            line_color=_c_est.get("text_muted","#64748B"), line_width=1)
        _fig_est.add_vline(x=S_est, line_dash="dot", line_color=_c_est["accent"],
                            annotation_text="S0", annotation_position="top right")
        for _xv, _lv in _vlines:
            _fig_est.add_vline(x=_xv, line_dash="dot", line_color=_c_est["success"],
                                annotation_text=_lv, annotation_position="bottom right")
        _fig_est.update_layout(
            title=nombre_est,
            xaxis_title="Precio al Vencimiento ($)",
            yaxis_title="P&L al Vencimiento ($)",
            height=400, hovermode="x unified",
            **plotly_theme(),
        )
        _fig_est.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(_fig_est, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODO B — MANUAL (hasta 4 patas)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        TIPOS_PATA = [
            "Vanilla Call", "Vanilla Put",
            "Gap Call", "Gap Put",
            "Cash-or-Nothing Call", "Cash-or-Nothing Put",
            "Asset-or-Nothing Call", "Asset-or-Nothing Put",
            "Down-and-Out Call", "Down-and-Out Put",
        ]
        n_patas = st.number_input("Numero de patas:", min_value=1, max_value=4,
                                   value=2, step=1, key="est_npatas")
        separador()
        patas_data = []

        for i in range(int(n_patas)):
            st.markdown(f"**Pata {i+1}**")
            _pc1, _pc2, _pc3, _pc4, _pc5 = st.columns(5)
            _tipo_p  = _pc1.selectbox("Tipo",     TIPOS_PATA,                    key=f"p{i}_tipo")
            _pos_lbl = _pc2.radio("Posición", ["Long (+1)", "Short (−1)"],
                                   horizontal=True,                               key=f"p{i}_pos")
            _pos_int = 1 if "Long" in _pos_lbl else -1
            _K_p     = _pc3.number_input("K (Strike)", min_value=0.01, value=100.0,
                                          step=1.0,                               key=f"p{i}_K")
            _K2_p, _Q_p, _H_p = None, None, None

            if "Gap" in _tipo_p:
                _K2_p = _pc4.number_input("K2 (pago)", min_value=0.01, value=105.0, step=1.0, key=f"p{i}_K2")
            elif "Cash-or-Nothing" in _tipo_p:
                _Q_p  = _pc4.number_input("Q (pago fijo)", min_value=0.01, value=10.0, step=1.0, key=f"p{i}_Q")
            elif "Down-and-Out" in _tipo_p:
                _H_p  = _pc4.number_input("H (barrera)", min_value=0.01, value=85.0, step=1.0, key=f"p{i}_H")

            try:
                _es_c = "Call" in _tipo_p
                if   "Vanilla"          in _tipo_p: _prima_p = engine.black_scholes(S_est, _K_p, r_est, sig_est, T_est, _es_c, q_est)
                elif "Gap"              in _tipo_p: _prima_p = engine.opciones_gap(S_est, _K_p, _K2_p, T_est, r_est, sig_est, q_est, "call" if _es_c else "put")
                elif "Cash-or-Nothing"  in _tipo_p: _prima_p = engine.opciones_cash_or_nothing(S_est, _K_p, _Q_p, T_est, r_est, sig_est, q_est, "call" if _es_c else "put")
                elif "Asset-or-Nothing" in _tipo_p: _prima_p = engine.opciones_asset_or_nothing(S_est, _K_p, T_est, r_est, sig_est, q_est, "call" if _es_c else "put")
                elif "Down-and-Out"     in _tipo_p: _prima_p = engine.barrera_down_and_out(S_est, _K_p, _H_p, T_est, r_est, sig_est, q_est, "call" if _es_c else "put")
                else: _prima_p = 0.0
                _pc5.metric("Prima calculada", f"${_prima_p:,.4f}")
            except Exception as _e:
                _prima_p = 0.0
                _pc5.error(str(_e))

            patas_data.append({"tipo": _tipo_p, "pos": _pos_int,
                                "K": _K_p, "K2": _K2_p, "Q": _Q_p, "H": _H_p,
                                "prima": _prima_p})

        # ── P&L neto ──────────────────────────────────────────────────────────
        separador()
        st.markdown("#### Perfil de P&L al Vencimiento")
        _ST_m   = np.linspace(S_est * 0.3, S_est * 1.7, 900)
        _pf_tot = np.zeros_like(_ST_m)
        _c_m    = get_current_theme()
        _pal_m  = [_c_m["primary"], _c_m["success"], _c_m["accent"], _c_m["danger"]]
        _fig_m  = go.Figure()

        for idx, _p in enumerate(patas_data):
            _es_c = "Call" in _p["tipo"]
            _pos  = _p["pos"]
            if   "Vanilla"          in _p["tipo"]: _pi = np.maximum(_ST_m - _p["K"], 0.) if _es_c else np.maximum(_p["K"] - _ST_m, 0.)
            elif "Gap"              in _p["tipo"]: _pi = (np.where(_ST_m > _p["K"], _ST_m - _p["K2"], 0.) if _es_c else np.where(_ST_m < _p["K"], _p["K2"] - _ST_m, 0.))
            elif "Cash-or-Nothing"  in _p["tipo"]: _pi = (np.where(_ST_m > _p["K"], _p["Q"], 0.) if _es_c else np.where(_ST_m < _p["K"], _p["Q"], 0.))
            elif "Asset-or-Nothing" in _p["tipo"]: _pi = (np.where(_ST_m > _p["K"], _ST_m, 0.) if _es_c else np.where(_ST_m < _p["K"], _ST_m, 0.))
            elif "Down-and-Out"     in _p["tipo"]:
                _van = np.maximum(_ST_m - _p["K"], 0.) if _es_c else np.maximum(_p["K"] - _ST_m, 0.)
                _pi  = np.where(_ST_m > _p["H"], _van, 0.)
            else: _pi = np.zeros_like(_ST_m)

            _leg = _pos * _pi - _pos * _p["prima"]
            _pf_tot += _leg
            _fig_m.add_trace(go.Scatter(
                x=_ST_m, y=_leg, mode="lines",
                name=f"Pata {idx+1}: {'Long' if _pos>0 else 'Short'} {_p['tipo']}",
                line=dict(color=_pal_m[idx % len(_pal_m)], width=1.5, dash="dot"),
            ))

        _fig_m.add_trace(go.Scatter(x=_ST_m, y=_pf_tot, mode="lines",
            name="P&L Neto", line=dict(color=_c_m["primary"], width=3)))
        _fig_m.add_trace(go.Scatter(x=_ST_m, y=np.where(_pf_tot >= 0, _pf_tot, 0),
            fill="tozeroy", fillcolor="rgba(40,167,69,0.12)", mode="none", showlegend=False))
        _fig_m.add_trace(go.Scatter(x=_ST_m, y=np.where(_pf_tot < 0, _pf_tot, 0),
            fill="tozeroy", fillcolor="rgba(220,53,69,0.12)", mode="none", showlegend=False))
        _fig_m.add_hline(y=0, line_dash="dash",
                          line_color=_c_m.get("text_muted","#64748B"), line_width=1)
        _fig_m.add_vline(x=S_est, line_dash="dot", line_color=_c_m["accent"],
                          annotation_text="S0", annotation_position="top right")
        _fig_m.update_layout(
            title="Estrategia Manual — P&L Neto al Vencimiento",
            xaxis_title="Precio al Vencimiento ($)",
            yaxis_title="P&L al Vencimiento ($)",
            height=420, hovermode="x unified",
            **plotly_theme(),
        )
        _fig_m.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(_fig_m, use_container_width=True)

        _costo_m = sum(_p["pos"] * _p["prima"] for _p in patas_data)
        _mca, _mcb, _mcc = st.columns(3)
        _mca.metric("Prima Neta",
                    f"${_costo_m:,.4f}",
                    delta="Débito" if _costo_m > 0 else "Crédito",
                    delta_color="inverse" if _costo_m > 0 else "normal")
        _mcb.metric("Ganancia máxima", f"${float(np.max(_pf_tot)):,.2f}")
        _mcc.metric("Perdida máxima",  f"${float(np.min(_pf_tot)):,.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 9 — ACTIVOS REALES (YAHOO FINANCE)
# Valúa los 7 tipos de exóticos con datos reales de mercado
# ═════════════════════════════════════════════════════════════════════════════
with tab_real:
    st.markdown("### Valuación Exótica sobre Subyacente")
    themed_info(
        "Extrae el precio spot y la volatilidad histórica de subyacentes en vivo mediante Yahoo Finance para "
        "revaluar dinámicamente cualquier estructura de derivado exótico empleando parámetros empíricos reales."
    )
    separador()

    # ── BUSCADOR ─────────────────────────────────────────────────────────────
    st.markdown("#### Paso 1 — Alimentación de Datos de Mercado")
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        ticker_ex = st.text_input(
            "Ticker (Símbolo en Bolsa):",
            value="AAPL",
            key="ex_ticker",
            help="Ejemplos: AAPL, TSLA, MSFT, SPY, GLD, CEMEXCPO.MX",
            placeholder="Ej. AAPL, TSLA, SPY...",
        ).strip().upper()
    with col_t2:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_ex = st.button("Descargar Historial de Yahoo Finance", use_container_width=True, key="btn_ex")

    if btn_ex:
        with st.spinner(f"Extrayendo y procesando serie de tiempo para {ticker_ex}..."):
            spot_ex, vol_ex = engine.obtener_datos_subyacente(ticker_ex)
            if spot_ex is not None:
                st.session_state["ex_spot"] = float(spot_ex)
                st.session_state["ex_vol"]  = float(vol_ex * 100)
                st.session_state["ex_ticker_ok"] = ticker_ex
                st.session_state["ex_S"]   = float(spot_ex)
                st.session_state["ex_K"]   = float(spot_ex)   
                st.session_state["ex_sig"] = float(vol_ex * 100)
                themed_success(
                    f"**Extracción exitosa para {ticker_ex}.** \n"
                    f"Precio de Cierre (Spot) = **${spot_ex:,.2f}** · "
                    f"Volatilidad Anualizada = **{vol_ex*100:.2f}%**"
                )
                st.rerun()
            else:
                st.session_state.pop("ex_spot", None)
                themed_error(f"No se localizó el símbolo **{ticker_ex}**. Verifica tu red o el ticker ingresado.")

    if "ex_spot" not in st.session_state: st.session_state["ex_spot"] = 100.0
    if "ex_vol"  not in st.session_state: st.session_state["ex_vol"]  = 20.0
    if "ex_ticker_ok" not in st.session_state: st.session_state["ex_ticker_ok"] = "ACTIVO"
    if "ex_S"   not in st.session_state: st.session_state["ex_S"]   = st.session_state["ex_spot"]
    if "ex_K"   not in st.session_state: st.session_state["ex_K"]   = st.session_state["ex_spot"]
    if "ex_sig" not in st.session_state: st.session_state["ex_sig"] = st.session_state["ex_vol"]

    separador()

    # ── PARÁMETROS BASE ───────────────────────────────────────────────────────
    st.markdown("#### Paso 2 — Configuración del Contrato Base")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        S_ex  = st.number_input("Precio Spot ($S_0$)", min_value=0.01, step=1.0, key="ex_S")
        K_ex  = st.number_input("Strike ($K$)", min_value=0.01, step=1.0, key="ex_K")
        tipo_ex = st.radio("Posición Base:", ["Call", "Put"], horizontal=True, key="ex_tipo")
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
            themed_success(
                f"<div style='{css_contenedor_sm}'>"
                f"<span style='{css_titulo_sm}'>Vanilla Call</span>"
                f"<span style='{css_valor_sm}'>${prima_van_ex:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor_sm}'>"
                f"<span style='{css_titulo_sm}'>Vanilla Put</span>"
                f"<span style='{css_valor_sm}'>${prima_van_ex:,.4f}</span>"
                f"</div>"
            )
            
        moneyness_ex = ((S_ex - K_ex) / K_ex) * 100
        if (moneyness_ex > 1 and es_call_ex) or (moneyness_ex < -1 and not es_call_ex):
            themed_success(f"**ITM (Con Valor)** — Distancia: {moneyness_ex:+.2f}%")
        elif (moneyness_ex < -1 and es_call_ex) or (moneyness_ex > 1 and not es_call_ex):
            themed_warning(f"**OTM (Sin Valor)** — Distancia: {moneyness_ex:+.2f}%")
        else:
            themed_info(f"**ATM (Equilibrio)** — Distancia: {moneyness_ex:+.2f}%")

    separador()

    # ── SELECTOR DE TIPO EXÓTICO ──────────────────────────────────────────────
    st.markdown("#### Paso 3 — Inyección Exótica")
    tipo_exotico = st.selectbox(
        "Fórmula de derivado a evaluar:",
        [
            "Gap — Discontinuidad por doble strike",
            "Binaria Cash-or-Nothing — Pago fijo en efectivo",
            "Binaria Asset-or-Nothing — Entrega del activo físico",
            "Barrera Down-and-Out / Down-and-In",
            "Asiática Geométrica — Mitigación por promedio cerrado",
            "Asiática Aritmética — Mitigación por método Turnbull-Wakeman",
            "Lookback Flotante — Derecho retrospectivo extremo",
            "Compuesta — Derivado sobre derivado",
            "Intercambio (Margrabe) — Permuta de activos",
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
                themed_success(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>Gap Call</span>"
                    f"<span style='{css_valor}'>${prima_gap_ex:,.4f}</span>"
                    f"</div>"
                )
            else:
                themed_error(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>Gap Put</span>"
                    f"<span style='{css_valor}'>${prima_gap_ex:,.4f}</span>"
                    f"</div>"
                )
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
                themed_success(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>Cash-o-N Call</span>"
                    f"<span style='{css_valor}'>${prima_bin_ex:,.4f}</span>"
                    f"</div>"
                )
            else:
                themed_error(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>Cash-o-N Put</span>"
                    f"<span style='{css_valor}'>${prima_bin_ex:,.4f}</span>"
                    f"</div>"
                )
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
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Asset-o-N Call</span>"
                f"<span style='{css_valor}'>${prima_aon_ex:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Asset-o-N Put</span>"
                f"<span style='{css_valor}'>${prima_aon_ex:,.4f}</span>"
                f"</div>"
            )
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
                themed_success(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>{tipo_bar_ex} Call</span>"
                    f"<span style='{css_valor}'>${prima_bar_ex:,.4f}</span>"
                    f"</div>"
                )
            else:
                themed_error(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>{tipo_bar_ex} Put</span>"
                    f"<span style='{css_valor}'>${prima_bar_ex:,.4f}</span>"
                    f"</div>"
                )
                
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
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Asiática Geo Call</span>"
                f"<span style='{css_valor}'>${prima_asi_g:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Asiática Geo Put</span>"
                f"<span style='{css_valor}'>${prima_asi_g:,.4f}</span>"
                f"</div>"
            )
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
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Asiática Aritm Call</span>"
                f"<span style='{css_valor}'>${prima_asi_a:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Asiática Aritm Put</span>"
                f"<span style='{css_valor}'>${prima_asi_a:,.4f}</span>"
                f"</div>"
            )
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
            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Lookback Call</span>"
                f"<span style='{css_valor}'>${prima_lk_ex:,.4f}</span>"
                f"</div>"
            )
        else:
            themed_error(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Lookback Put</span>"
                f"<span style='{css_valor}'>${prima_lk_ex:,.4f}</span>"
                f"</div>"
            )
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
                themed_success(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>Compuesta ({subtipo_comp_ex})</span>"
                    f"<span style='{css_valor}'>${prima_comp_ex:,.4f}</span>"
                    f"</div>"
                )
            else:
                themed_error(
                    f"<div style='{css_contenedor}'>"
                    f"<span style='{css_titulo}'>Compuesta ({subtipo_comp_ex})</span>"
                    f"<span style='{css_valor}'>${prima_comp_ex:,.4f}</span>"
                    f"</div>"
                )
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
            st.markdown(f"**S1 — Activo a Entregar ({ticker_lbl})**")
            q1_ex = st.number_input("Dividendo q1 %", value=q_ex*100, step=0.1, key="ex_q1") / 100

            st.markdown("**S2 — Activo a Recibir**")
            ticker_ex2 = st.text_input("Símbolo en Bolsa Activo 2:", value="MSFT", key="ex_ticker2").strip().upper()
            btn_ex2 = st.button("Descargar S2 y Calcular Correlación de Mercado", key="btn_ex2", use_container_width=True)

            if btn_ex2:
                with st.spinner(f"Alineando series temporales de {ticker_lbl} y {ticker_ex2}..."):
                    try:
                        import yfinance as _yf
                        import datetime as _dtt
                        _hoy = _dtt.date.today()
                        _ini = _hoy - _dtt.timedelta(days=365)
                        _h1 = _yf.download(ticker_lbl, start=_ini, end=_hoy, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
                        _h2 = _yf.download(ticker_ex2, start=_ini, end=_hoy, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
                        _s2v, _v2v = engine.obtener_datos_subyacente(ticker_ex2)

                        if _s2v is None or len(_h1) < 20 or len(_h2) < 20:
                            themed_error(f"Fallo en la sincronización del ticker {ticker_ex2} o liquidez insuficiente.")
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
                            themed_success(f"**{ticker_ex2} Extraído.** S2=${_s2v:,.4f} | Volatilidad={_v2v*100:.2f}% | **Correlación ρ = {_rho_calc:.4f}**")
                            st.rerun()
                    except Exception as _e:
                        themed_error(f"Error de Integración: {_e}")

            if "ex_S2" not in st.session_state: st.session_state["ex_S2"] = S_ex * 1.1
            if "ex_sig2" not in st.session_state: st.session_state["ex_sig2"] = sig_ex * 100
            if "ex_ticker2_ok" not in st.session_state: st.session_state["ex_ticker2_ok"] = "ACTIVO2"
            if "ex_rho_real" not in st.session_state: st.session_state["ex_rho_real"] = 0.5

            S2_ex   = st.number_input("Spot S2", min_value=0.001, value=float(st.session_state["ex_S2"]), step=1.0, key="ex_S2_inp")
            sig2_ex = st.number_input("Volatilidad sigma2 %", min_value=0.01, value=float(st.session_state["ex_sig2"]), step=0.5, key="ex_sig2_inp") / 100
            q2_ex   = st.number_input("Dividendo q2 %", value=0.0, step=0.1, key="ex_q2") / 100
            rho_mode = st.radio("Cálculo de Correlación rho:", ["Automática (Market Data)", "Inserción Manual"], horizontal=True, key="ex_rho_mode")
            if rho_mode.startswith("Auto"):
                rho_ex = st.session_state["ex_rho_real"]
                st.metric("Vector Correlacional ρ", f"{rho_ex:.4f}")
            else:
                rho_ex = st.slider("Ajuste rho manual:", min_value=-1.0, max_value=1.0, value=float(st.session_state["ex_rho_real"]), step=0.01, key="ex_rho_slider")

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

            themed_success(
                f"<div style='{css_contenedor}'>"
                f"<span style='{css_titulo}'>Intercambio (Recibir S2, Entregar S1)</span>"
                f"<span style='{css_valor}'>${prima_int_ex:,.4f}</span>"
                f"</div>"
            )
            c1r, c2r, c3r = st.columns(3)
            c1r.metric("Vector U (n1 × S1)", f"${U_eff:,.4f}")
            c2r.metric("Vector V (n2 × S2)", f"${V_eff:,.4f}")
            c3r.metric("Volatilidad Cruzada σ*", f"{sig_comb*100:.4f}%")
            
            if "ex_b100_1" in st.session_state and "ex_b100_2" in st.session_state:
                import plotly.graph_objects as _go_i
                _b1 = st.session_state["ex_b100_1"]
                _b2 = st.session_state["ex_b100_2"]
                _fig_b = _go_i.Figure()
                _fig_b.add_trace(_go_i.Scatter(x=_b1.index.astype(str), y=_b1.values, name=ticker_lbl, mode="lines", line=dict(color=c_th["primary"], width=1.5)))
                _fig_b.add_trace(_go_i.Scatter(x=_b2.index.astype(str), y=_b2.values, name=ticker2_lbl, mode="lines", line=dict(color=c_th["accent"], width=1.5)))
                _fig_b.update_layout(title=f"Convergencia de Activos (Base 100) | ρ = {rho_ex:.4f}", xaxis_title="Fecha de Cierre", yaxis_title="Índice Normalizado", height=300, **plotly_theme())
                st.plotly_chart(_fig_b, use_container_width=True)

        with paso_a_paso():
            st.latex(rf"\sigma^* = \sqrt{{{sig_ex:.4f}^2 + {sig2_ex:.4f}^2 - 2({rho_ex:.2f})({sig_ex:.4f})({sig2_ex:.4f})}} = {sig_comb:.6f}")
            st.latex(rf"d_1 = \frac{{\ln({V_eff:.2f}/{U_eff:.2f}) + ({q1_ex:.4f} - {q2_ex:.4f} + \frac{{{sig_comb:.6f}^2}}{{2}}){T_ex:.4f}}}{{{sig_comb:.6f}\sqrt{{{T_ex:.4f}}}}}")
            st.latex(rf"d_1 = {d1_int:.6f}, \quad d_2 = {d2_int:.6f}")
            st.latex(rf"c = {V_eff:.2f} e^{{-{q2_ex:.4f}({T_ex:.4f})}} N({d1_int:.4f}) - {U_eff:.2f} e^{{-{q1_ex:.4f}({T_ex:.4f})}} N({d2_int:.4f}) = {prima_int_ex:.4f}")

    # ── GRÁFICA COMPARATIVA DE TODOS LOS EXÓTICOS ─────────────────────────────
    separador()
    st.markdown("#### Comparativa Global de Primas Estructurales")

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
        title=f"Arquitecturas de Derivados — {tipo_ex} sobre {ticker_lbl} (S={S_ex:.2f}, K={K_ex:.2f}, σ={sig_ex*100:.1f}%, T={T_ex})",
        xaxis_title="Costo de la Prima ($)",
        height=420,
        margin=dict(l=180),
        **plotly_theme(),
    )
    st.plotly_chart(fig_comp_ex, use_container_width=True)

    df_comp_ex = pd.DataFrame({
        "Estructura Derivada":   nombres_comp,
        "Prima Neta ($)":         [f"${p:,.4f}" for p in primas_comp],
        "Discrepancia vs Vanilla":        [f"{(p/prima_van_ex - 1)*100:+.1f}%" if prima_van_ex > 0 else "—"
                              for p in primas_comp],
    })
    st.dataframe(df_comp_ex, use_container_width=True, hide_index=True)