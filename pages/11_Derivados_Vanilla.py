"""
pages/11_Derivados_Vanilla.py
-----------------------------
Módulo 11: Valuación de Derivados Vanilla.
Orden pedagógico: primero Árbol Binomial CRR (intuición discreta),
luego Black-Scholes-Merton (límite continuo).
Cubre:
  - Árbol Binomial CRR: Call/Put europeas y americanas, árbol visual
  - Black-Scholes-Merton: 6 variantes (estándar, dividendo continuo,
    futuros, divisas, acción con dividendos discretos, perpetua)
  - Griegas: Delta, Gamma, Theta, Vega, Rho (BSM)
  - Comparativa BSM vs Binomial
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm as norm

from utils import get_engine, page_header, paso_a_paso, separador, alerta_metodo_numerico, themed_info, themed_success, themed_warning, themed_error, apply_plotly_theme, plotly_theme, plotly_colors, get_current_theme

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Derivados Vanilla · Calculadora Financiera",
    page_icon="⚙️",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="10. Valuación de Derivados Vanilla",
    subtitulo="Árbol Binomial CRR · Black-Scholes-Merton · Griegas"
)

# =============================================================================
# PESTAÑAS — orden pedagógico
# =============================================================================
tab_crr, tab_bsm, tab_griegas, tab_comp, tab_est, tab_real, tab_vol = st.tabs([
    "Arbol Binomial (CRR)",
    "Black-Scholes-Merton",
    "Griegas (BSM)",
    "Comparativa BSM vs CRR",
    "Estrategias con Opciones",
    "Activos Reales (Yahoo Finance)",
    "Volatilidad Implicita",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — ÁRBOL BINOMIAL CRR
# ═════════════════════════════════════════════════════════════════════════════
with tab_crr:
    st.markdown("### Árbol Binomial Cox-Ross-Rubinstein (CRR)")
    themed_info(
        "El árbol binomial discretiza el movimiento del precio en pasos de tiempo &Delta;<span style='font-family: serif; font-style: italic;'>t</span>. "
        "En cada nodo, el precio sube por <span style='font-family: serif; font-style: italic;'>u</span> o baja por <span style='font-family: serif; font-style: italic;'>d = 1/u</span>. "
        "La probabilidad neutral al riesgo <span style='font-family: serif; font-style: italic;'>p</span> garantiza la ausencia de arbitraje. "
        "Con suficientes pasos converge al precio de Black-Scholes."
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Parámetros del subyacente**")
        S_crr   = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                                   value=100.0, step=1.0, key="crr_S")
        K_crr   = st.number_input("Precio de Ejercicio ($K$)", min_value=0.01,
                                   value=100.0, step=1.0, key="crr_K")
        sig_crr = st.number_input("Volatilidad ($\\sigma$) %",
                                   min_value=0.01, value=20.0,
                                   step=0.5, key="crr_sig") / 100
        q_crr   = st.number_input("Dividendo continuo ($q$) %",
                                   value=0.0, step=0.1, key="crr_q") / 100

    with c2:
        st.markdown("**Condiciones del contrato**")
        r_crr   = st.number_input("Tasa libre de riesgo ($r$) %",
                                   value=5.0, step=0.1, key="crr_r") / 100
        T_crr   = st.number_input("Tiempo al vencimiento ($T$) años",
                                   min_value=0.01, value=1.0,
                                   step=0.25, key="crr_T")
        N_crr   = st.number_input("Número de pasos ($N$)",
                                   min_value=1, max_value=500,
                                   value=4, step=1, key="crr_N")

    with c3:
        st.markdown("**Tipo de opción**")
        tipo_op_crr  = st.radio("Opción:", ["Call", "Put"],
                                 horizontal=True, key="crr_tipo")
        estilo_crr   = st.radio("Estilo:", ["Europea", "Americana"],
                                 horizontal=True, key="crr_estilo")
        es_call_crr  = (tipo_op_crr == "Call")
        es_amer_crr  = (estilo_crr  == "Americana")

    separador()

    # ── Cálculo ───────────────────────────────────────────────────────────────
    dt_crr = T_crr / N_crr
    u_crr  = np.exp(sig_crr * np.sqrt(dt_crr))
    d_crr  = 1.0 / u_crr
    a_crr  = np.exp((r_crr - q_crr) * dt_crr)
    p_crr  = (a_crr - d_crr) / (u_crr - d_crr)

    precio_crr, arbol_precios, arbol_opcion = engine.arbol_binomial_crr(
        S_crr, K_crr, r_crr, sig_crr, T_crr,
        int(N_crr), es_call_crr, es_amer_crr, q_crr
    )

    # Métricas principales
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        themed_success(f"<h3 style='margin:0; color:inherit;'>Prima {tipo_op_crr}: ${precio_crr:,.4f}</h3>")
    col_r2.metric("Factor sube ($u$)",           f"{u_crr:.6f}")
    col_r3.metric("Factor baja ($d$)",           f"{d_crr:.6f}")
    col_r4.metric("Prob. neutral al riesgo ($p$)", f"{p_crr:.4f}")

    separador()

    # ── Paso a paso de parámetros CRR ─────────────────────────────────────────
    with paso_a_paso():
        st.latex(r"\Delta t = \frac{T}{N}")
        st.latex(rf"\Delta t = \frac{{{T_crr:.4f}}}{{{int(N_crr)}}} = {dt_crr:.6f}")
        st.write("---")
        
        st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = e^{-\sigma\sqrt{\Delta t}}")
        st.latex(rf"u = e^{{{sig_crr:.4f} \times \sqrt{{{dt_crr:.6f}}}}} = e^{{{sig_crr*np.sqrt(dt_crr):.6f}}} = {u_crr:.6f}")
        st.latex(rf"d = e^{{-{sig_crr:.4f} \times \sqrt{{{dt_crr:.6f}}}}} = {d_crr:.6f}")
        st.write("---")

        st.latex(r"a = e^{(r-q)\Delta t}")
        st.latex(rf"a = e^{{({r_crr:.4f} - {q_crr:.4f}) \times {dt_crr:.6f}}} = e^{{(r_crr-q_crr)*dt_crr:.6f}} = {a_crr:.6f}")
        st.write("---")

        st.latex(r"p = \frac{a - d}{u - d}")
        st.latex(rf"p = \frac{{{a_crr:.6f} - {d_crr:.6f}}}{{{u_crr:.6f} - {d_crr:.6f}}} = \frac{{{a_crr - d_crr:.6f}}}{{{u_crr - d_crr:.6f}}} = {p_crr:.6f}")
        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima = ${precio_crr:,.4f}</h4>")

    # ── Árbol visual (solo si N ≤ 10) ─────────────────────────────────────────
    separador()
    if int(N_crr) <= 10:
        st.markdown("#### Árbol de Precios del Subyacente y Valores de la Opción")

        tab_arbol_s, tab_arbol_o = st.tabs([
            "🌿 Árbol del Subyacente",
            "💵 Árbol de la Opción",
        ])

        def _renderizar_arbol(matriz, titulo, fmt="$.4f"):
            n = int(N_crr)
            fig = go.Figure()
            node_x, node_y, node_text = [], [], []
            edge_x, edge_y = [], []

            for col in range(n + 1):
                for row in range(col + 1):
                    x_pos = col
                    y_pos = col - 2 * row   # nodo [row, col]: up=col-row, dn=row
                    val   = matriz[col][row]
                    node_x.append(x_pos)
                    node_y.append(y_pos)
                    node_text.append(
                        format(val, ".4f") if fmt == "$.4f"
                        else f"${val:,.4f}"
                    )
                    # Aristas hacia nodos del paso siguiente
                    if col < n:
                        # Arista arriba (u)
                        edge_x += [x_pos, col + 1, None]
                        edge_y += [y_pos, y_pos + 1, None]
                        # Arista abajo (d)
                        edge_x += [x_pos, col + 1, None]
                        edge_y += [y_pos, y_pos - 1, None]

            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(color="#CBD5E1", width=1),
                hoverinfo="none",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                marker=dict(size=36, color="#1E3A8A", opacity=0.85),
                text=node_text,
                textfont=dict(color="white", size=11),
                textposition="middle center",
                hoverinfo="text",
                showlegend=False,
            ))
            fig.update_layout(
                title=titulo,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=max(350, 80 * (n + 1)),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            return fig

        with tab_arbol_s:
            st.plotly_chart(
                _renderizar_arbol(arbol_precios, "Precios del Subyacente"),
                use_container_width=True,
            )
        with tab_arbol_o:
            st.plotly_chart(
                _renderizar_arbol(arbol_opcion, f"Valores de la {tipo_op_crr} ({estilo_crr})"),
                use_container_width=True,
            )

    else:
        themed_info(
            f"Con <span style='font-family: serif; font-style: italic;'>N = {int(N_crr)}</span> pasos el árbol es demasiado grande para visualizar. "
            "Reduce <span style='font-family: serif; font-style: italic;'>N &le; 10</span> para ver el árbol gráfico."
        )

    # ── Convergencia a BSM ────────────────────────────────────────────────────
    separador()
    with st.expander("📈 Convergencia del Árbol Binomial a Black-Scholes"):
        st.markdown(
            "A medida que aumentas $N$, el precio binomial converge al precio de BSM. "
            "Solo se muestra para opciones europeas."
        )

        if not es_amer_crr:
            pasos_conv = [1, 2, 5, 10, 20, 50, 100, 200]
            precios_conv = []
            for n_c in pasos_conv:
                p_c, _, _ = engine.arbol_binomial_crr(
                    S_crr, K_crr, r_crr, sig_crr, T_crr,
                    n_c, es_call_crr, False, q_crr
                )
                precios_conv.append(p_c)

            # Precio BSM de referencia
            bsm_ref = engine.black_scholes(
                S_crr, K_crr, r_crr, sig_crr, T_crr, es_call_crr, q_crr
            )

            df_conv = pd.DataFrame({
                "Pasos (N)":            pasos_conv,
                "Precio Binomial":      precios_conv,
                "Precio BSM (límite)":  [bsm_ref] * len(pasos_conv),
            })

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=df_conv["Pasos (N)"], y=df_conv["Precio Binomial"],
                mode="lines+markers", name="CRR Binomial",
                line=dict(color="#1E3A8A"),
            ))
            fig_conv.add_hline(
                y=bsm_ref, line_dash="dash", line_color="#FF6B6B",
                annotation_text=f"BSM = {bsm_ref:.4f}",
                annotation_position="right",
            )
            fig_conv.update_layout(
                xaxis_title="Número de pasos (N)",
                yaxis_title="Prima ($)",
                height=350,
            )
            fig_conv = apply_plotly_theme(fig_conv)
            fig_conv.update_layout(**plotly_theme())
            st.plotly_chart(fig_conv, use_container_width=True)
            st.dataframe(df_conv.style.format({
                "Precio Binomial":     "${:.4f}",
                "Precio BSM (límite)": "${:.4f}",
            }), use_container_width=True, hide_index=True)
        else:
            themed_info(
                "La convergencia solo se grafica para opciones europeas, "
                "ya que BSM no tiene fórmula cerrada para opciones americanas."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BLACK-SCHOLES-MERTON
# ═════════════════════════════════════════════════════════════════════════════
with tab_bsm:
    st.markdown("### Black-Scholes-Merton (BSM)")
    themed_info(
        "BSM es el límite continuo del árbol binomial cuando <span style='font-family: serif; font-style: italic;'>N &rarr; &infin;</span>. "
        "Asume que el precio del subyacente sigue un movimiento browniano geométrico "
        "con volatilidad constante. Solo es válido para opciones **europeas**."
    )

    # ── Selección de variante ─────────────────────────────────────────────────
    variante_bsm = st.selectbox(
        "Variante del modelo:",
        [
            "BSM Estándar (sin dividendos)",
            "BSM con dividendo continuo (Merton, 1973)",
            "BSM para Futuros (Black, 1976)",
            "BSM para Divisas (Garman-Kohlhagen, 1983)",
            "BSM con dividendos discretos",
            "Opción Perpetua (T → ∞)",
        ],
        key="bsm_variante",
    )
    separador()

    c1, c2 = st.columns(2)

    # ── Inputs comunes a todas las variantes ──────────────────────────────────
    with c1:
        st.markdown("**Parámetros comunes**")
        S_bsm   = st.number_input("Precio Spot / Subyacente ($S_0$)", min_value=0.01,
                                   value=100.0, step=1.0, key="bsm_S")
        K_bsm   = st.number_input("Precio de Ejercicio ($K$)", min_value=0.01,
                                   value=100.0, step=1.0, key="bsm_K")
        r_bsm   = st.number_input("Tasa libre de riesgo ($r$) %",
                                   value=5.0, step=0.1, key="bsm_r") / 100
        sig_bsm = st.number_input("Volatilidad ($\\sigma$) %",
                                   min_value=0.01, value=20.0,
                                   step=0.5, key="bsm_sig") / 100

        if variante_bsm != "Opción Perpetua (T → ∞)":
            T_bsm = st.number_input("Tiempo al vencimiento ($T$) años",
                                     min_value=0.001, value=1.0,
                                     step=0.25, key="bsm_T")
        else:
            T_bsm = None

        tipo_bsm = st.radio("Tipo:", ["Call", "Put"],
                             horizontal=True, key="bsm_tipo")
        es_call_bsm = (tipo_bsm == "Call")

    # ── Inputs específicos + cálculo ──────────────────────────────────────────
    with c1:
        prima_bsm = None
        d1_v = d2_v = None   # para el paso a paso

        if variante_bsm == "BSM Estándar (sin dividendos)":
            prima_bsm = engine.black_scholes(S_bsm, K_bsm, r_bsm, sig_bsm, T_bsm, es_call_bsm)
            formula_c = (r"c = S_0 N(d_1) - K e^{-rT} N(d_2)")
            formula_p = (r"p = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
            formula_d = (
                r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, "
                r"\quad d_2 = d_1 - \sigma\sqrt{T}"
            )
            d1_v = (np.log(S_bsm/K_bsm) + (r_bsm + sig_bsm**2/2)*T_bsm) / (sig_bsm*np.sqrt(T_bsm))
            d2_v = d1_v - sig_bsm * np.sqrt(T_bsm)

        elif variante_bsm == "BSM con dividendo continuo (Merton, 1973)":
            q_bsm     = st.number_input("Dividendo continuo ($q$) %",
                                         value=2.0, step=0.1, key="bsm_q") / 100
            prima_bsm = engine.black_scholes(S_bsm, K_bsm, r_bsm, sig_bsm, T_bsm,
                                              es_call_bsm, q_bsm)
            formula_c = r"c = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)"
            formula_p = r"p = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)"
            formula_d = (
                r"d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}"
            )
            d1_v = (np.log(S_bsm/K_bsm) + (r_bsm - q_bsm + sig_bsm**2/2)*T_bsm) / (sig_bsm*np.sqrt(T_bsm))
            d2_v = d1_v - sig_bsm * np.sqrt(T_bsm)

        elif variante_bsm == "BSM para Futuros (Black, 1976)":
            F_bsm     = st.number_input("Precio del Futuro ($F_0$)", min_value=0.01,
                                         value=100.0, step=1.0, key="bsm_F")
            prima_bsm = engine.black_76(F_bsm, K_bsm, r_bsm, sig_bsm, T_bsm, es_call_bsm)
            formula_c = r"c = e^{-rT}[F_0 N(d_1) - K N(d_2)]"
            formula_p = r"p = e^{-rT}[K N(-d_2) - F_0 N(-d_1)]"
            formula_d = (
                r"d_1 = \frac{\ln(F_0/K) + (\sigma^2/2) T}{\sigma\sqrt{T}}"
            )
            d1_v = (np.log(F_bsm/K_bsm) + sig_bsm**2*T_bsm/2) / (sig_bsm*np.sqrt(T_bsm))
            d2_v = d1_v - sig_bsm * np.sqrt(T_bsm)

        elif variante_bsm == "BSM para Divisas (Garman-Kohlhagen, 1983)":
            rf_bsm    = st.number_input("Tasa extranjera ($r_f$) %",
                                         value=3.0, step=0.1, key="bsm_rf") / 100
            prima_bsm = engine.black_scholes(S_bsm, K_bsm, r_bsm, sig_bsm, T_bsm,
                                              es_call_bsm, rf_bsm)
            formula_c = r"c = S_0 e^{-r_f T} N(d_1) - K e^{-r_d T} N(d_2)"
            formula_p = r"p = K e^{-r_d T} N(-d_2) - S_0 e^{-r_f T} N(-d_1)"
            formula_d = (
                r"d_1 = \frac{\ln(S_0/K) + (r_d - r_f + \sigma^2/2)T}{\sigma\sqrt{T}}"
            )
            d1_v = (np.log(S_bsm/K_bsm) + (r_bsm - rf_bsm + sig_bsm**2/2)*T_bsm) / (sig_bsm*np.sqrt(T_bsm))
            d2_v = d1_v - sig_bsm * np.sqrt(T_bsm)

        elif variante_bsm == "BSM con dividendos discretos":
            n_div_bsm = st.number_input("Número de dividendos", min_value=1,
                                         max_value=6, value=1, step=1, key="bsm_ndiv")
            divs_bsm  = []
            for i in range(int(n_div_bsm)):
                cd1, cd2 = st.columns(2)
                with cd1:
                    d_i = st.number_input(f"Dividendo {i+1} ($D_{i+1}$)",
                                           min_value=0.0, value=3.0,
                                           step=0.5, key=f"bsm_d{i}")
                with cd2:
                    t_i = st.number_input(f"Tiempo {i+1} ($t$, años)",
                                           min_value=0.0, max_value=T_bsm,
                                           value=round(T_bsm * (i+1)/(n_div_bsm+1), 2),
                                           step=0.25, key=f"bsm_td{i}")
                divs_bsm.append((d_i, t_i))

            I_bsm     = sum(d * np.exp(-r_bsm * t) for d, t in divs_bsm)
            S_adj     = S_bsm - I_bsm
            prima_bsm = engine.black_scholes(S_adj, K_bsm, r_bsm, sig_bsm, T_bsm, es_call_bsm)
            formula_c = r"c_{disc} = BSM(S_0 - I, K, r, \sigma, T)"
            formula_p = r"p_{disc} = BSM(S_0 - I, K, r, \sigma, T)"
            formula_d = r"I = \sum_i D_i e^{-r t_i}"
            d1_v = (np.log(S_adj/K_bsm) + (r_bsm + sig_bsm**2/2)*T_bsm) / (sig_bsm*np.sqrt(T_bsm))
            d2_v = d1_v - sig_bsm * np.sqrt(T_bsm)

        else:  # Perpetua
            prima_bsm = engine.opcion_perpetua(S_bsm, K_bsm, r_bsm, sig_bsm, es_call_bsm)
            formula_c = r"C^* = \frac{K}{h-1}\left(\frac{(h-1)S_0}{hK}\right)^h"
            formula_p = r"P^* = \frac{K}{1-h^*}\left(\frac{(1-h^*)S_0}{h^* K}\right)^{h^*}"
            formula_d = r"h = \frac{1}{2} + \sqrt{\frac{1}{4} + \frac{2r}{\sigma^2}}"

    with c2:
        if prima_bsm is not None:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Prima {tipo_bsm}: ${prima_bsm:,.4f}</h3>")
            st.latex(formula_c if es_call_bsm else formula_p)
            st.latex(formula_d)

            # Valor intrínseco y valor temporal
            if T_bsm is not None:
                vi = max(S_bsm - K_bsm, 0) if es_call_bsm else max(K_bsm - S_bsm, 0)
                vt = prima_bsm - vi
                separador()
                col_vi1, col_vi2 = st.columns(2)
                col_vi1.metric("Valor Intrínseco", f"${vi:,.4f}")
                col_vi2.metric("Valor Temporal",   f"${vt:,.4f}")

            # ── Griegas evaluadas para la misma opcion ──────────────────
            if T_bsm is not None:
                separador()
                st.markdown("**Griegas (BSM)**")
                _q_gr_bsm = locals().get("q_bsm", locals().get("rf_bsm", 0.0))
                _S_gr_bsm = locals().get("F_bsm", S_bsm)
                _gr = engine.calcular_griegas(
                    _S_gr_bsm, K_bsm, r_bsm, sig_bsm, T_bsm, es_call_bsm, _q_gr_bsm
                )
                gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
                gcol1.metric("Delta", f"{_gr['delta']:+.5f}",
                             help="+$1 en subyacente → cambio en prima")
                gcol2.metric("Gamma", f"{_gr['gamma']:.5f}",
                             help="Cambio en Delta ante +$1")
                gcol3.metric("Theta (diario)", f"{_gr['theta']:+.5f}",
                             help="Perdida de valor por dia")
                gcol4.metric("Vega", f"{_gr['vega']:.5f}",
                             help="Cambio ante +1% de volatilidad")
                gcol5.metric("Rho", f"{_gr['rho']:+.5f}",
                             help="Cambio ante +1% en tasa libre de riesgo")

    # ── Paso a paso BSM ───────────────────────────────────────────────────────
    if prima_bsm is not None and variante_bsm != "Opción Perpetua (T → ∞)":

        Nd1 = norm.cdf(d1_v)
        Nd2 = norm.cdf(d2_v)
        nNd1 = norm.cdf(-d1_v)
        nNd2 = norm.cdf(-d2_v)

        with paso_a_paso():
            st.latex(formula_d)
            
            # --- Sustitución de d1 y d2 según variante ---
            if "Estándar" in variante_bsm:
                st.latex(rf"d_1 = \frac{{\ln({S_bsm:.2f}/{K_bsm:.2f}) + ({r_bsm:.4f} + \frac{{{sig_bsm:.4f}^2}}{{2}}){T_bsm:.4f}}}{{{sig_bsm:.4f}\sqrt{{{T_bsm:.4f}}}}}")
                st.latex(rf"d_1 = \frac{{{np.log(S_bsm/K_bsm):.6f} + {(r_bsm + sig_bsm**2/2)*T_bsm:.6f}}}{{{sig_bsm*np.sqrt(T_bsm):.6f}}} = {d1_v:.6f}")
                t1 = S_bsm
                t2 = K_bsm * np.exp(-r_bsm * T_bsm)
                
            elif "dividendo continuo" in variante_bsm:
                st.latex(rf"d_1 = \frac{{\ln({S_bsm:.2f}/{K_bsm:.2f}) + ({r_bsm:.4f} - {q_bsm:.4f} + \frac{{{sig_bsm:.4f}^2}}{{2}}){T_bsm:.4f}}}{{{sig_bsm:.4f}\sqrt{{{T_bsm:.4f}}}}}")
                st.latex(rf"d_1 = \frac{{{np.log(S_bsm/K_bsm):.6f} + {(r_bsm - q_bsm + sig_bsm**2/2)*T_bsm:.6f}}}{{{sig_bsm*np.sqrt(T_bsm):.6f}}} = {d1_v:.6f}")
                t1 = S_bsm * np.exp(-q_bsm * T_bsm)
                t2 = K_bsm * np.exp(-r_bsm * T_bsm)
                
            elif "Futuros" in variante_bsm:
                st.latex(rf"d_1 = \frac{{\ln({F_bsm:.2f}/{K_bsm:.2f}) + (\frac{{{sig_bsm:.4f}^2}}{{2}}){T_bsm:.4f}}}{{{sig_bsm:.4f}\sqrt{{{T_bsm:.4f}}}}}")
                st.latex(rf"d_1 = \frac{{{np.log(F_bsm/K_bsm):.6f} + {(sig_bsm**2/2)*T_bsm:.6f}}}{{{sig_bsm*np.sqrt(T_bsm):.6f}}} = {d1_v:.6f}")
                t1 = F_bsm * np.exp(-r_bsm * T_bsm)
                t2 = K_bsm * np.exp(-r_bsm * T_bsm)
                
            elif "Divisas" in variante_bsm:
                st.latex(rf"d_1 = \frac{{\ln({S_bsm:.2f}/{K_bsm:.2f}) + ({r_bsm:.4f} - {rf_bsm:.4f} + \frac{{{sig_bsm:.4f}^2}}{{2}}){T_bsm:.4f}}}{{{sig_bsm:.4f}\sqrt{{{T_bsm:.4f}}}}}")
                st.latex(rf"d_1 = \frac{{{np.log(S_bsm/K_bsm):.6f} + {(r_bsm - rf_bsm + sig_bsm**2/2)*T_bsm:.6f}}}{{{sig_bsm*np.sqrt(T_bsm):.6f}}} = {d1_v:.6f}")
                t1 = S_bsm * np.exp(-rf_bsm * T_bsm)
                t2 = K_bsm * np.exp(-r_bsm * T_bsm)
                
            elif "discretos" in variante_bsm:
                st.latex(rf"I = \sum D_i e^{{-r t_i}} = {I_bsm:.6f} \implies S_0 - I = {S_adj:.6f}")
                st.latex(rf"d_1 = \frac{{\ln({S_adj:.6f}/{K_bsm:.2f}) + ({r_bsm:.4f} + \frac{{{sig_bsm:.4f}^2}}{{2}}){T_bsm:.4f}}}{{{sig_bsm:.4f}\sqrt{{{T_bsm:.4f}}}}}")
                st.latex(rf"d_1 = \frac{{{np.log(S_adj/K_bsm):.6f} + {(r_bsm + sig_bsm**2/2)*T_bsm:.6f}}}{{{sig_bsm*np.sqrt(T_bsm):.6f}}} = {d1_v:.6f}")
                t1 = S_adj
                t2 = K_bsm * np.exp(-r_bsm * T_bsm)
            
            st.latex(rf"d_2 = d_1 - \sigma\sqrt{{T}} = {d1_v:.6f} - {sig_bsm*np.sqrt(T_bsm):.6f} = {d2_v:.6f}")
            st.write("---")

            # --- Probabilidades y Sustitución Final ---
            if es_call_bsm:
                st.latex(rf"N(d_1) = N({d1_v:.6f}) = {Nd1:.6f}")
                st.latex(rf"N(d_2) = N({d2_v:.6f}) = {Nd2:.6f}")
                st.write("---")
                st.latex(formula_c)
                
                if "Estándar" in variante_bsm or "discretos" in variante_bsm:
                    st.latex(rf"c = {t1:.4f}({Nd1:.6f}) - {K_bsm:.2f} e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}({Nd2:.6f})")
                elif "Futuros" in variante_bsm:
                    st.latex(rf"c = e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}[{F_bsm:.2f}({Nd1:.6f}) - {K_bsm:.2f}({Nd2:.6f})]")
                elif "Divisas" in variante_bsm:
                    st.latex(rf"c = {S_bsm:.2f} e^{{-{rf_bsm:.4f} \times {T_bsm:.4f}}}({Nd1:.6f}) - {K_bsm:.2f} e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}({Nd2:.6f})")
                elif "continuo" in variante_bsm:
                    st.latex(rf"c = {S_bsm:.2f} e^{{-{q_bsm:.4f} \times {T_bsm:.4f}}}({Nd1:.6f}) - {K_bsm:.2f} e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}({Nd2:.6f})")

                if "Futuros" in variante_bsm:
                    st.latex(rf"c = {np.exp(-r_bsm*T_bsm):.6f} [{F_bsm*Nd1:.6f} - {K_bsm*Nd2:.6f}] = {prima_bsm:.4f}")
                else:
                    st.latex(rf"c = {t1*Nd1:.6f} - {t2*Nd2:.6f} = {prima_bsm:.4f}")

            else:
                st.latex(rf"N(-d_1) = N({-d1_v:.6f}) = {nNd1:.6f}")
                st.latex(rf"N(-d_2) = N({-d2_v:.6f}) = {nNd2:.6f}")
                st.write("---")
                st.latex(formula_p)

                if "Estándar" in variante_bsm or "discretos" in variante_bsm:
                    st.latex(rf"p = {K_bsm:.2f} e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}({nNd2:.6f}) - {t1:.4f}({nNd1:.6f})")
                elif "Futuros" in variante_bsm:
                    st.latex(rf"p = e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}[{K_bsm:.2f}({nNd2:.6f}) - {F_bsm:.2f}({nNd1:.6f})]")
                elif "Divisas" in variante_bsm:
                    st.latex(rf"p = {K_bsm:.2f} e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}({nNd2:.6f}) - {S_bsm:.2f} e^{{-{rf_bsm:.4f} \times {T_bsm:.4f}}}({nNd1:.6f})")
                elif "continuo" in variante_bsm:
                    st.latex(rf"p = {K_bsm:.2f} e^{{-{r_bsm:.4f} \times {T_bsm:.4f}}}({nNd2:.6f}) - {S_bsm:.2f} e^{{-{q_bsm:.4f} \times {T_bsm:.4f}}}({nNd1:.6f})")

                if "Futuros" in variante_bsm:
                    st.latex(rf"p = {np.exp(-r_bsm*T_bsm):.6f} [{K_bsm*nNd2:.6f} - {F_bsm*nNd1:.6f}] = {prima_bsm:.4f}")
                else:
                    st.latex(rf"p = {t2*nNd2:.6f} - {t1*nNd1:.6f} = {prima_bsm:.4f}")

            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima {tipo_bsm} = ${prima_bsm:,.4f}</h4>")
            
    elif variante_bsm == "Opción Perpetua (T → ∞)":
        with paso_a_paso():
            st.latex(formula_d)
            st.latex(rf"h = \frac{{1}}{{2}} + \sqrt{{\frac{{1}}{{4}} + \frac{{2({r_bsm:.4f})}}{{{sig_bsm:.4f}^2}}}}")
            h_val = 0.5 + np.sqrt(0.25 + 2*r_bsm/(sig_bsm**2))
            st.latex(rf"h = 0.5 + \sqrt{{0.25 + {2*r_bsm/(sig_bsm**2):.6f}}} = {h_val:.6f}")

            st.write("---")

            if es_call_bsm:
                st.latex(formula_c)
                st.latex(rf"C^* = \frac{{{K_bsm:.2f}}}{{{h_val:.6f}-1}}\left(\frac{{({h_val:.6f}-1){S_bsm:.2f}}}{{{h_val:.6f}({K_bsm:.2f})}}\right)^{{{h_val:.6f}}}")
                st.latex(rf"C^* = {K_bsm/(h_val-1):.6f} \left({(h_val-1)*S_bsm / (h_val*K_bsm):.6f}\right)^{{{h_val:.6f}}} = {prima_bsm:.4f}")
            else:
                st.latex(r"h^* = \frac{1}{2} - \sqrt{\frac{1}{4} + \frac{2r}{\sigma^2}}")
                h_star = 0.5 - np.sqrt(0.25 + 2*r_bsm/(sig_bsm**2))
                st.latex(rf"h^* = 0.5 - \sqrt{{0.25 + {2*r_bsm/(sig_bsm**2):.6f}}} = {h_star:.6f}")
                st.latex(formula_p)
                st.latex(rf"P^* = \frac{{{K_bsm:.2f}}}{{1 - ({h_star:.6f})}}\left(\frac{{(1 - ({h_star:.6f})){S_bsm:.2f}}}{{{h_star:.6f}({K_bsm:.2f})}}\right)^{{{h_star:.6f}}}")
                st.latex(rf"P^* = {prima_bsm:.4f}")
                
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>Prima {tipo_bsm} = ${prima_bsm:,.4f}</h4>")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRIEGAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_griegas:
    st.markdown("### Letras Griegas (Sensibilidades de BSM)")
    themed_info(
        "Las Griegas miden cómo cambia el precio de la opción ante pequeñas "
        "variaciones en cada parámetro de mercado."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Parámetros**")
        S_gr   = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                                  value=100.0, step=1.0, key="gr_S")
        K_gr   = st.number_input("Strike ($K$)", min_value=0.01,
                                  value=100.0, step=1.0, key="gr_K")
        r_gr   = st.number_input("Tasa libre de riesgo ($r$) %",
                                  value=5.0, step=0.1, key="gr_r") / 100
        sig_gr = st.number_input("Volatilidad ($\\sigma$) %",
                                  min_value=0.01, value=20.0,
                                  step=0.5, key="gr_sig") / 100
        T_gr   = st.number_input("Tiempo al vencimiento ($T$) años",
                                  min_value=0.001, value=1.0,
                                  step=0.25, key="gr_T")
        q_gr   = st.number_input("Dividendo continuo ($q$) %",
                                  value=0.0, step=0.1, key="gr_q") / 100
        tipo_gr = st.radio("Tipo:", ["Call", "Put"],
                            horizontal=True, key="gr_tipo")
        es_call_gr = (tipo_gr == "Call")

    with c2:
        griegas = engine.calcular_griegas(S_gr, K_gr, r_gr, sig_gr, T_gr, es_call_gr, q_gr)

        delta_g = griegas["delta"]
        gamma_g = griegas["gamma"]
        theta_g = griegas["theta"]
        vega_g  = griegas["vega"]
        rho_g   = griegas["rho"]

        st.metric("Δ Delta",  f"{delta_g:+.6f}",
                  help="Variación del precio de la opción ante +$1 en el subyacente.")
        st.metric("Γ Gamma",  f"{gamma_g:.6f}",
                  help="Variación del Delta ante +$1 en el subyacente. Igual para Call y Put.")
        st.metric("Θ Theta",  f"{theta_g:+.6f} $/día",
                  help="Decaimiento del valor de la opción por cada día que pasa.")
        st.metric("𝒱 Vega",   f"{vega_g:.6f} $/1%σ",
                  help="Variación del precio ante +1% de volatilidad.")
        st.metric("ρ Rho",    f"{rho_g:+.6f} $/1%r",
                  help="Variación del precio ante +1% en la tasa libre de riesgo.")

    separador()

    # ── Fórmulas de las griegas ───────────────────────────────────────────────
    with st.expander("📐 Fórmulas de las Griegas (BSM con dividendo q)"):
        st.latex(r"\Delta_{call} = e^{-qT} N(d_1), \quad \Delta_{put} = -e^{-qT} N(-d_1)")
        st.latex(r"\Gamma = \frac{e^{-qT} n(d_1)}{S_0 \sigma \sqrt{T}}")
        st.latex(r"\Theta_{call} = -\frac{S_0 n(d_1) \sigma e^{-qT}}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qS_0e^{-qT}N(d_1)")
        st.latex(r"\mathcal{V} = S_0 \sqrt{T} e^{-qT} n(d_1) \quad (\text{por 1\% de }\sigma)")
        st.latex(r"\rho_{call} = KTe^{-rT}N(d_2), \quad \rho_{put} = -KTe^{-rT}N(-d_2)")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARATIVA BSM vs CRR
# ═════════════════════════════════════════════════════════════════════════════
with tab_comp:
    st.markdown("### Comparativa: BSM vs Árbol Binomial CRR")
    st.caption(
        "Usa los mismos parámetros para comparar directamente el precio "
        "de una opción europea calculada con ambos métodos."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        S_cp   = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                                  value=100.0, step=1.0, key="cp_S")
        K_cp   = st.number_input("Strike ($K$)", min_value=0.01,
                                  value=105.0, step=1.0, key="cp_K")
    with c2:
        r_cp   = st.number_input("Tasa libre de riesgo ($r$) %",
                                  value=5.0, step=0.1, key="cp_r") / 100
        sig_cp = st.number_input("Volatilidad ($\\sigma$) %",
                                  min_value=0.01, value=20.0,
                                  step=0.5, key="cp_sig") / 100
    with c3:
        T_cp    = st.number_input("Tiempo al vencimiento ($T$) años",
                                   min_value=0.01, value=1.0,
                                   step=0.25, key="cp_T")
        q_cp    = st.number_input("Dividendo continuo ($q$) %",
                                   value=0.0, step=0.1, key="cp_q") / 100
        tipo_cp = st.radio("Tipo:", ["Call", "Put"],
                            horizontal=True, key="cp_tipo")
        es_call_cp = (tipo_cp == "Call")

    separador()

    # Precio BSM
    bsm_comp = engine.black_scholes(S_cp, K_cp, r_cp, sig_cp, T_cp, es_call_cp, q_cp)

    # Precios CRR para distintos N
    pasos_lista = [1, 2, 5, 10, 25, 50, 100, 200, 500]
    precios_crr_comp = []
    for n_c in pasos_lista:
        p_c, _, _ = engine.arbol_binomial_crr(
            S_cp, K_cp, r_cp, sig_cp, T_cp,
            n_c, es_call_cp, False, q_cp
        )
        precios_crr_comp.append(p_c)

    df_comp = pd.DataFrame({
        "Pasos (N)":        pasos_lista,
        "CRR Binomial ($)": precios_crr_comp,
        "BSM ($)":          [bsm_comp] * len(pasos_lista),
        "Diferencia ($)":   [abs(p - bsm_comp) for p in precios_crr_comp],
    })

    # Gráfica
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=df_comp["Pasos (N)"], y=df_comp["CRR Binomial ($)"],
        mode="lines+markers", name="CRR Binomial",
        line=dict(color="#1E3A8A", width=2),
        marker=dict(size=7),
    ))
    fig_comp.add_hline(
        y=bsm_comp, line_dash="dash", line_color="#FF6B6B",
        annotation_text=f"BSM = ${bsm_comp:.4f}",
        annotation_position="right",
    )
    fig_comp.update_layout(
        xaxis_title="Número de pasos (N)",
        yaxis_title="Prima ($)",
        title=f"Convergencia CRR → BSM  |  {tipo_cp}  |  S={S_cp}, K={K_cp}, σ={sig_cp*100:.0f}%, T={T_cp}",
        height=420,
    )
    fig_comp = apply_plotly_theme(fig_comp)
    fig_comp.update_layout(**plotly_theme())
    st.plotly_chart(fig_comp, use_container_width=True)

    # Tabla
    st.dataframe(
        df_comp.style.format({
            "CRR Binomial ($)": "${:.4f}",
            "BSM ($)":          "${:.4f}",
            "Diferencia ($)":   "${:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    col_bsm_m, col_crr_m = st.columns(2)
    col_bsm_m.metric("Precio BSM (referencia)",     f"${bsm_comp:.4f}")
    col_crr_m.metric("Precio CRR (N=500)",          f"${precios_crr_comp[-1]:.4f}",
                     delta=f"Δ = ${abs(precios_crr_comp[-1]-bsm_comp):.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — ESTRATEGIAS CON OPCIONES
# ═════════════════════════════════════════════════════════════════════════════
with tab_est:
    st.markdown("### Estrategias con Opciones (Perfil de Riesgo/Rendimiento)")
    themed_info(
        "Combina múltiples calls y puts para construir cualquier estrategia. "
        "El perfil neto muestra la ganancia/pérdida al vencimiento en función del "
        "precio del subyacente."
    )

    # ── Selección de estrategia predefinida ──────────────────────────────────
    estrategia_sel = st.selectbox(
        "Selecciona una estrategia predefinida (o configura manualmente):",
        [
            "Manual (configura tú mismo)",
            "Bull Call Spread",
            "Bear Put Spread",
            "Long Straddle",
            "Short Straddle",
            "Long Strangle",
            "Butterfly (Long)",
            "Covered Call",
            "Protective Put",
            "Risk Reversal",
        ],
        key="est_sel",
    )
    separador()

    # ── Parámetros del subyacente ─────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        S_est   = st.number_input("Precio Spot actual ($S_0$)", min_value=0.01,
                                   value=100.0, step=1.0, key="est_S")
        r_est   = st.number_input("Tasa libre de riesgo ($r$) %",
                                   value=5.0, step=0.1, key="est_r") / 100
        sig_est = st.number_input("Volatilidad ($\\sigma$) %",
                                   min_value=0.01, value=20.0,
                                   step=0.5, key="est_sig") / 100
        T_est   = st.number_input("Tiempo al vencimiento ($T$) años",
                                   min_value=0.01, value=1.0,
                                   step=0.25, key="est_T")
        q_est   = st.number_input("Dividendo continuo ($q$) %",
                                   value=0.0, step=0.1, key="est_q") / 100

    # ── Configuración de patas según estrategia ───────────────────────────────
    with c2:
        st.markdown("**Definición de las patas**")

        # Calcular prima BSM para los strikes comunes
        def _prima(K, es_call):
            return engine.black_scholes(S_est, K, r_est, sig_est, T_est, es_call, q_est)

        if estrategia_sel == "Manual (configura tú mismo)":
            n_patas = st.number_input("Número de patas", min_value=1,
                                       max_value=6, value=2, step=1, key="est_n_patas")
            patas = []
            for i in range(int(n_patas)):
                st.markdown(f"**Pata {i+1}**")
                cols_p = st.columns(4)
                tipo_p   = cols_p[0].selectbox("Tipo", ["call", "put"],
                                                key=f"est_tipo_{i}")
                pos_p    = cols_p[1].selectbox("Posición", [1, -1],
                                                format_func=lambda x: "Long" if x == 1 else "Short",
                                                key=f"est_pos_{i}")
                K_p      = cols_p[2].number_input("Strike ($K$)", min_value=0.01,
                                                   value=float(S_est), step=1.0,
                                                   key=f"est_K_{i}")
                prima_p  = cols_p[3].number_input("Prima ($)",
                                                   min_value=0.0,
                                                   value=round(_prima(K_p, tipo_p == "call"), 4),
                                                   step=0.1, key=f"est_prima_{i}")
                patas.append({"tipo": tipo_p, "posicion": pos_p, "K": K_p, "prima": prima_p})

        else:
            # Estrategias predefinidas
            K1_e = st.number_input("Strike bajo ($K_1$)", min_value=0.01,
                                    value=float(S_est) * 0.95, step=1.0, key="est_K1")
            K2_e = st.number_input("Strike alto ($K_2$)", min_value=0.01,
                                    value=float(S_est) * 1.05, step=1.0, key="est_K2")

            c_K1 = _prima(K1_e, True);  p_K1 = _prima(K1_e, False)
            c_K2 = _prima(K2_e, True);  p_K2 = _prima(K2_e, False)

            if estrategia_sel == "Bull Call Spread":
                patas = [
                    {"tipo": "call", "posicion":  1, "K": K1_e, "prima": c_K1},
                    {"tipo": "call", "posicion": -1, "K": K2_e, "prima": c_K2},
                ]
                themed_info("**Long Call** K1 + **Short Call** K2  →  Beneficio si sube moderadamente.")

            elif estrategia_sel == "Bear Put Spread":
                patas = [
                    {"tipo": "put", "posicion":  1, "K": K2_e, "prima": p_K2},
                    {"tipo": "put", "posicion": -1, "K": K1_e, "prima": p_K1},
                ]
                themed_info("**Long Put** K2 + **Short Put** K1  →  Beneficio si baja moderadamente.")

            elif estrategia_sel == "Long Straddle":
                K_mid = (K1_e + K2_e) / 2
                c_mid = _prima(K_mid, True); p_mid = _prima(K_mid, False)
                patas = [
                    {"tipo": "call", "posicion": 1, "K": K_mid, "prima": c_mid},
                    {"tipo": "put",  "posicion": 1, "K": K_mid, "prima": p_mid},
                ]
                themed_info("**Long Call + Long Put** mismo strike  →  Beneficio con movimiento grande (sin importar dirección).")

            elif estrategia_sel == "Short Straddle":
                K_mid = (K1_e + K2_e) / 2
                c_mid = _prima(K_mid, True); p_mid = _prima(K_mid, False)
                patas = [
                    {"tipo": "call", "posicion": -1, "K": K_mid, "prima": c_mid},
                    {"tipo": "put",  "posicion": -1, "K": K_mid, "prima": p_mid},
                ]
                themed_info("**Short Call + Short Put** mismo strike  →  Beneficio si el precio se queda quieto.")

            elif estrategia_sel == "Long Strangle":
                patas = [
                    {"tipo": "call", "posicion": 1, "K": K2_e, "prima": c_K2},
                    {"tipo": "put",  "posicion": 1, "K": K1_e, "prima": p_K1},
                ]
                themed_info("**Long Call** K2 + **Long Put** K1  →  Más barato que Straddle, requiere mayor movimiento.")

            elif estrategia_sel == "Butterfly (Long)":
                K_mid = (K1_e + K2_e) / 2
                c_mid = _prima(K_mid, True)
                patas = [
                    {"tipo": "call", "posicion":  1, "K": K1_e,  "prima": c_K1},
                    {"tipo": "call", "posicion": -2, "K": K_mid,  "prima": c_mid},
                    {"tipo": "call", "posicion":  1, "K": K2_e,  "prima": c_K2},
                ]
                themed_info("**Long 1 Call K1 + Short 2 Calls K_mid + Long 1 Call K2** →  Beneficio máximo si el precio termina en K_mid.")

            elif estrategia_sel == "Covered Call":
                patas = [
                    {"tipo": "call", "posicion": -1, "K": K2_e, "prima": c_K2},
                ]
                themed_info("**Short Call** K2 (sobre posición larga en el subyacente)  →  Genera ingreso, limita la ganancia.")
                themed_warning("La pata larga en el subyacente no se muestra en el perfil (se asume comprado a <span style='font-family: serif; font-style: italic;'>S<sub>0</sub></span>).")

            elif estrategia_sel == "Protective Put":
                patas = [
                    {"tipo": "put", "posicion": 1, "K": K1_e, "prima": p_K1},
                ]
                themed_info("**Long Put** K1 (sobre posición larga en el subyacente)  →  Seguro contra caídas.")
                themed_warning("La pata larga en el subyacente no se muestra en el perfil.")

            elif estrategia_sel == "Risk Reversal":
                patas = [
                    {"tipo": "call", "posicion":  1, "K": K2_e, "prima": c_K2},
                    {"tipo": "put",  "posicion": -1, "K": K1_e, "prima": p_K1},
                ]
                themed_info("**Long Call** K2 + **Short Put** K1  →  Direccional alcista, se financia con la venta del put.")

    separador()

    # ── Tabla de patas y primas ───────────────────────────────────────────────
    st.markdown("#### Resumen de Patas (primas calculadas con BSM)")
    df_patas = pd.DataFrame([{
        "Pata":     i + 1,
        "Tipo":     p["tipo"].capitalize(),
        "Posición": "Long" if p["posicion"] > 0 else "Short",
        "Strike ($K$)": f"${p['K']:,.2f}",
        "Prima BSM ($)": f"${p['prima']:,.4f}",
        "Costo Neto ($)": f"${p['posicion'] * p['prima']:+,.4f}",
    } for i, p in enumerate(patas)])
    st.dataframe(df_patas, use_container_width=True, hide_index=True)

    costo_total = sum(p["posicion"] * p["prima"] for p in patas)
    if costo_total < 0:
        themed_success(f"**Prima neta recibida:** ${abs(costo_total):,.4f}  (estrategia con ingreso inicial)")
    else:
        themed_info(f"**Prima neta pagada:** ${costo_total:,.4f}  (costo de la estrategia)")

    separador()

    # ── Gráfica de perfil ──────────────────────────────────────────────────────
    fig_est = engine.graficar_estrategia(estrategia_sel, S_est, patas)
    fig_est = apply_plotly_theme(fig_est)
    fig_est.update_layout(**plotly_theme())
    st.plotly_chart(fig_est, use_container_width=True)

    separador()

    # ── Análisis de puntos clave ──────────────────────────────────────────────
    with st.expander("📐 Análisis de puntos clave (Breakeven aproximado)"):
        S_T_rng = np.linspace(S_est * 0.5, S_est * 1.5, 5000)
        payoff_tot = np.zeros_like(S_T_rng)
        for p in patas:
            pp = engine.calcular_payoff_leg(p["tipo"], p["posicion"], S_T_rng, p["K"], p["prima"])
            payoff_tot += pp

        ganancia_max = payoff_tot.max()
        perdida_max  = payoff_tot.min()
        be_idx = np.where(np.diff(np.sign(payoff_tot)))[0]
        breakevens = [S_T_rng[i] for i in be_idx]

        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Ganancia Máxima",
                      f"${ganancia_max:,.4f}" if ganancia_max < 1e8 else "Ilimitada")
        col_k2.metric("Pérdida Máxima",
                      f"${abs(perdida_max):,.4f}" if perdida_max > -1e8 else "Ilimitada")
        if breakevens:
            col_k3.metric("Punto(s) de Equilibrio",
                          " | ".join([f"${be:.2f}" for be in breakevens]))
        else:
            col_k3.metric("Punto(s) de Equilibrio", "N/A")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — ACTIVOS REALES (YAHOO FINANCE)
# ═════════════════════════════════════════════════════════════════════════════
with tab_real:
    st.markdown("### Valuación de Opciones sobre Activos Reales")
    themed_info(
        "Ingresa el ticker de cualquier activo de Yahoo Finance para obtener el **precio spot actual** "
        "y la **volatilidad histórica anualizada** automáticamente. "
        "Después puedes calcular primas BSM y CRR, ver todas las Griegas y el perfil de payoff "
        "con datos reales del mercado."
    )

    separador()

    # ── PASO 1: Buscador de ticker ────────────────────────────────────────────
    st.markdown("#### Paso 1 — Cargar datos de mercado")
    col_t1, col_t2, col_t3 = st.columns([2, 1, 1])
    with col_t1:
        ticker_real = st.text_input(
            "Ticker de Yahoo Finance:",
            value="AAPL",
            key="real_ticker",
            help="Ejemplos: AAPL, TSLA, MSFT, GOOGL, NVDA, SPY, GLD, CEMEXCPO.MX",
            placeholder="Ej. AAPL, TSLA, SPY..."
        ).strip().upper()
    with col_t2:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_buscar_real = st.button("Buscar en Yahoo Finance", use_container_width=True, key="btn_real")
    with col_t3:
        periodo_vol = st.selectbox(
            "Ventana de volatilidad:",
            ["1 año (252 días)", "6 meses (~126 días)", "3 meses (~63 días)"],
            key="real_periodo"
        )

    if btn_buscar_real:
        with st.spinner(f"Consultando datos para {ticker_real}..."):
            spot_yf, vol_yf = engine.obtener_datos_subyacente(ticker_real)
            if spot_yf is not None:
                st.session_state["real_spot"] = float(spot_yf)
                st.session_state["real_vol"]  = float(vol_yf * 100)
                st.session_state["real_ticker_ok"] = ticker_real
                # Poblar widgets directamente — esto hace que se actualicen al instante
                st.session_state["real_S"]   = float(spot_yf)
                st.session_state["real_K"]   = float(spot_yf)   # ATM (K = S)
                st.session_state["real_sig"] = float(vol_yf * 100)
                themed_success(
                    f"**{ticker_real}** cargado correctamente.  \n"
                    f"S₀ = **${spot_yf:,.2f}** · K (ATM) = **${spot_yf:,.2f}** · "
                    f"σ = **{vol_yf*100:.2f}%** — campos actualizados automaticamente."
                )
                st.rerun()
            else:
                st.session_state.pop("real_spot", None)
                st.session_state.pop("real_vol",  None)
                themed_error(
                    f"No se encontró **{ticker_real}** o no tiene suficiente historial "
                    "(se requiere mínimo 20 días de datos). Verifica el símbolo."
                )

    # Inicializar valores por defecto en session_state
    if "real_spot" not in st.session_state:
        st.session_state["real_spot"] = 100.0
    if "real_vol" not in st.session_state:
        st.session_state["real_vol"]  = 20.0
    if "real_ticker_ok" not in st.session_state:
        st.session_state["real_ticker_ok"] = "ACTIVO"
    # Inicializar widgets solo si no han sido seteados por el buscador
    if "real_S"   not in st.session_state:
        st.session_state["real_S"]   = st.session_state["real_spot"]
    if "real_K"   not in st.session_state:
        st.session_state["real_K"]   = st.session_state["real_spot"]
    if "real_sig" not in st.session_state:
        st.session_state["real_sig"] = st.session_state["real_vol"]

    separador()

    # ── PASO 2: Parámetros del contrato ──────────────────────────────────────
    st.markdown("#### Paso 2 — Parámetros del contrato")
    themed_info(
        "El Precio Spot y la Volatilidad se auto-completan con los datos de mercado. "
        "Puedes ajustarlos manualmente antes de calcular."
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        S_real  = st.number_input(
            "Precio Spot ($S_0$)",
            min_value=0.01, step=1.0,
            key="real_S",
        )
        K_real  = st.number_input(
            "Precio de Ejercicio ($K$)",
            min_value=0.01, step=1.0,
            key="real_K",
        )
        tipo_real = st.radio("Tipo:", ["Call", "Put"], horizontal=True, key="real_tipo")
        estilo_real = st.radio("Estilo:", ["Europea", "Americana"], horizontal=True, key="real_estilo")
        es_call_real = (tipo_real == "Call")
        es_amer_real = (estilo_real == "Americana")

    with col_p2:
        sig_real = st.number_input(
            "Volatilidad ($\\sigma$) %",
            min_value=0.01, step=0.5,
            key="real_sig",
        ) / 100
        r_real   = st.number_input(
            "Tasa libre de riesgo ($r$) %",
            value=5.0, step=0.1, key="real_r",
        ) / 100
        q_real   = st.number_input(
            "Dividendo continuo ($q$) %",
            value=0.0, step=0.1, key="real_q",
        ) / 100
        T_real   = st.number_input(
            "Tiempo al vencimiento ($T$) años",
            min_value=0.01, value=0.5, step=0.25, key="real_T",
        )

    with col_p3:
        N_real = st.number_input(
            "Pasos árbol binomial ($N$)",
            min_value=1, max_value=500, value=100, step=10, key="real_N",
        )
        moneyness = ((S_real - K_real) / K_real) * 100
        if moneyness > 1:
            estado_op = "In-The-Money (ITM)"
            color_estado = "success"
        elif moneyness < -1:
            estado_op = "Out-of-The-Money (OTM)"
            color_estado = "warning"
        else:
            estado_op = "At-The-Money (ATM)"
            color_estado = "info"

        st.markdown("<br>", unsafe_allow_html=True)
        if color_estado == "success":
            themed_success(f"**Estado:** {estado_op}  \n**Moneyness:** {moneyness:+.2f}%")
        elif color_estado == "warning":
            themed_warning(f"**Estado:** {estado_op}  \n**Moneyness:** {moneyness:+.2f}%")
        else:
            themed_info(f"**Estado:** {estado_op}  \n**Moneyness:** {moneyness:+.2f}%")

    separador()

    # ── PASO 3: Cálculo y resultados ──────────────────────────────────────────
    st.markdown("#### Paso 3 — Resultados de la Valuación")

    # Precios BSM y CRR
    prima_bsm_real  = engine.black_scholes(S_real, K_real, r_real, sig_real, T_real, es_call_real, q_real)
    prima_crr_real, _, _ = engine.arbol_binomial_crr(
        S_real, K_real, r_real, sig_real, T_real, int(N_real), es_call_real, es_amer_real, q_real
    )
    griegas_real = engine.calcular_griegas(S_real, K_real, r_real, sig_real, T_real, es_call_real, q_real)

    ticker_label = st.session_state.get("real_ticker_ok", "ACTIVO")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"##### Prima sobre **{ticker_label}** — {tipo_real} {estilo_real}")
        c = get_current_theme()
        
        # Uso los themed_info/success según la opción (Call es verde, Put es rojo en tu convención normal)
        if es_call_real:
            themed_success(f"<h4 style='margin:0; color:inherit;'>BSM — {tipo_real} Europea: ${prima_bsm_real:,.4f}</h4>")
            themed_success(f"<h4 style='margin:0; color:inherit;'>CRR (N={N_real}) — {tipo_real} {estilo_real}: ${prima_crr_real:,.4f}</h4>")
        else:
            themed_error(f"<h4 style='margin:0; color:inherit;'>BSM — {tipo_real} Europea: ${prima_bsm_real:,.4f}</h4>")
            themed_error(f"<h4 style='margin:0; color:inherit;'>CRR (N={N_real}) — {tipo_real} {estilo_real}: ${prima_crr_real:,.4f}</h4>")

        diff_pct = abs(prima_bsm_real - prima_crr_real) / prima_bsm_real * 100 if prima_bsm_real > 0 else 0
        st.metric("Diferencia BSM vs CRR", f"${abs(prima_bsm_real - prima_crr_real):,.4f}",
                  help=f"Diferencia relativa: {diff_pct:.3f}%")

        # Valor intrínseco y temporal
        vi_real = max(S_real - K_real, 0) if es_call_real else max(K_real - S_real, 0)
        vt_real = prima_bsm_real - vi_real
        c_vi1, c_vi2 = st.columns(2)
        c_vi1.metric("Valor Intrínseco", f"${vi_real:,.4f}")
        c_vi2.metric("Valor Temporal",   f"${vt_real:,.4f}")

    with col_r2:
        st.markdown("##### Griegas del contrato")
        g1, g2, g3 = st.columns(3)
        g1.metric("Δ Delta", f"{griegas_real['delta']:+.5f}",
                  help="+$1 en el subyacente → cambio en prima")
        g2.metric("Γ Gamma", f"{griegas_real['gamma']:.5f}",
                  help="Cambio en Delta ante +$1")
        g3.metric("Θ Theta", f"{griegas_real['theta']:+.5f}",
                  help="Pérdida de valor por día")
        g4, g5, _ = st.columns(3)
        g4.metric("𝒱 Vega",  f"{griegas_real['vega']:.5f}",
                  help="Cambio ante +1% volatilidad")
        g5.metric("ρ Rho",   f"{griegas_real['rho']:+.5f}",
                  help="Cambio ante +1% en tasa r")

        separador()
        # Interpretación rápida
        themed_info(
            f"**Delta {griegas_real['delta']:+.4f}:** Por cada $1 que suba **{ticker_label}**, "
            f"esta opción cambia **${griegas_real['delta']:+.4f}**.  \n"
            f"**Theta {griegas_real['theta']:+.4f}:** Pierde **${abs(griegas_real['theta']):.4f}** "
            f"de valor cada día que pasa."
        )

    separador()

    # ── Gráfica payoff al vencimiento ─────────────────────────────────────────
    st.markdown("#### Perfil de Payoff al Vencimiento")

    S_range   = np.linspace(S_real * 0.5, S_real * 1.5, 300)
    if es_call_real:
        payoff_r  = np.maximum(S_range - K_real, 0) - prima_bsm_real
        label_pay = f"Call K={K_real:.2f} (neto)"
    else:
        payoff_r  = np.maximum(K_real - S_range, 0) - prima_bsm_real
        label_pay = f"Put K={K_real:.2f} (neto)"

    c_theme = get_current_theme()
    fig_pay = go.Figure()
    fig_pay.add_trace(go.Scatter(
        x=S_range, y=payoff_r,
        mode="lines", name=label_pay,
        line=dict(color=c_theme["accent"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({int(c_theme['accent'][1:3],16)},{int(c_theme['accent'][3:5],16)},{int(c_theme['accent'][5:7],16)},0.12)",
    ))
    fig_pay.add_hline(y=0, line_dash="dash", line_color=c_theme["border"], line_width=1)
    fig_pay.add_vline(
        x=K_real, line_dash="dot", line_color=c_theme["primary"],
        annotation_text=f"K = {K_real:.2f}", annotation_position="top right",
    )
    fig_pay.add_vline(
        x=S_real, line_dash="solid", line_color=c_theme["success"],
        annotation_text=f"S₀ = {S_real:.2f}", annotation_position="top left",
    )
    fig_pay.update_layout(
        title=f"Payoff neto — {tipo_real} {estilo_real} sobre {ticker_label}",
        xaxis_title="Precio del subyacente al vencimiento ($S_T$)",
        yaxis_title="Ganancia / Pérdida ($)",
        height=400,
        **plotly_theme(),
    )
    st.plotly_chart(fig_pay, use_container_width=True)

    separador()

    # ── Tabla de primas para distintos strikes ────────────────────────────────
    st.markdown("#### Tabla de Primas para Distintos Strikes")
    themed_info(
        "Calcula automáticamente el precio de la opción para una cadena de strikes "
        "alrededor del precio spot actual."
    )

    strikes_pct  = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    strikes_vals = [round(S_real * (1 + p / 100), 2) for p in strikes_pct]

    filas_tabla = []
    for K_i in strikes_vals:
        prima_c = engine.black_scholes(S_real, K_i, r_real, sig_real, T_real, True,  q_real)
        prima_p = engine.black_scholes(S_real, K_i, r_real, sig_real, T_real, False, q_real)
        gr_i    = engine.calcular_griegas(S_real, K_i, r_real, sig_real, T_real, es_call_real, q_real)
        mon_i   = ((S_real - K_i) / K_i) * 100
        filas_tabla.append({
            "Strike ($K$)":   f"${K_i:,.2f}",
            "Moneyness":      f"{mon_i:+.1f}%",
            "Prima Call":     f"${prima_c:,.4f}",
            "Prima Put":      f"${prima_p:,.4f}",
            "Delta":          f"{gr_i['delta']:+.4f}",
            "Gamma":          f"{gr_i['gamma']:.4f}",
            "Theta ($/día)":  f"{gr_i['theta']:+.4f}",
            "Vega":           f"{gr_i['vega']:.4f}",
        })

    df_cadena = pd.DataFrame(filas_tabla)
    st.dataframe(df_cadena, use_container_width=True, hide_index=True)

    separador()

    # ── Simulación de escenarios ──────────────────────────────────────────────
    st.markdown("#### Análisis de Sensibilidad")
    themed_info(
        "Explora cómo cambia la prima si la volatilidad o el tiempo varían, "
        "manteniendo los demás parámetros fijos."
    )

    tab_sens_vol, tab_sens_t = st.tabs(["Por Volatilidad", "Por Tiempo"])

    with tab_sens_vol:
        vol_range = np.linspace(max(sig_real * 0.3, 0.01), sig_real * 2.5, 40)
        primas_vol = [
            engine.black_scholes(S_real, K_real, r_real, v, T_real, es_call_real, q_real)
            for v in vol_range
        ]
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_range * 100, y=primas_vol,
            mode="lines+markers",
            line=dict(color=c_theme["primary"], width=2),
            marker=dict(size=5),
            name="Prima BSM",
        ))
        fig_vol.add_vline(
            x=sig_real * 100, line_dash="dot", line_color=c_theme["accent"],
            annotation_text=f"σ actual = {sig_real*100:.1f}%",
        )
        fig_vol.update_layout(
            title="Prima vs Volatilidad",
            xaxis_title="Volatilidad σ (%)",
            yaxis_title="Prima ($)",
            height=360,
            **plotly_theme(),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab_sens_t:
        T_range = np.linspace(0.02, 2.0, 40)
        primas_t = [
            engine.black_scholes(S_real, K_real, r_real, sig_real, t, es_call_real, q_real)
            for t in T_range
        ]
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=T_range, y=primas_t,
            mode="lines+markers",
            line=dict(color=c_theme["success"], width=2),
            marker=dict(size=5),
            name="Prima BSM",
        ))
        fig_t.add_vline(
            x=T_real, line_dash="dot", line_color=c_theme["accent"],
            annotation_text=f"T actual = {T_real:.2f}",
        )
        fig_t.update_layout(
            title="Prima vs Tiempo al Vencimiento",
            xaxis_title="Tiempo al vencimiento T (años)",
            yaxis_title="Prima ($)",
            height=360,
            **plotly_theme(),
        )
        st.plotly_chart(fig_t, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — VOLATILIDAD IMPLÍCITA Y SONRISA
# ═════════════════════════════════════════════════════════════════════════════
with tab_vol:
    import datetime as _dt
    from scipy.optimize import brentq as _brentq

    st.markdown("### Volatilidad Implícita y Sonrisa de Volatilidad")
    themed_info(
        "La **volatilidad implícita** (<span style='font-family: serif; font-style: italic;'>&sigma;<sub>impl</sub></span>) es la volatilidad que hace que el precio de Black-Scholes iguale "
        "el precio de mercado actual de la opción. Inversión numérica vía bisección. "
        "La **sonrisa de volatilidad** grafica <span style='font-family: serif; font-style: italic;'>&sigma;<sub>impl</sub></span> vs. moneyness (<span style='font-family: serif; font-style: italic;'>K/S</span>) para múltiples strikes, "
        "revelando las primas de riesgo de cola que el modelo BSM clásico no logra capturar."
    )
    separador()

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.markdown("**Parámetros del mercado**")
        S_vol   = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                                   value=st.session_state.get("vol_S_key", 100.0), step=1.0, key="vol_S")
        r_vol   = st.number_input("Tasa libre de riesgo ($r$) %",
                                   value=5.0, step=0.1, key="vol_r") / 100
        q_vol   = st.number_input("Dividendo continuo ($q$) %",
                                   value=0.0, step=0.1, key="vol_q") / 100
        T_vol   = st.number_input("Vencimiento ($T$) años",
                                   min_value=0.01, value=1.0, step=0.25, key="vol_T")
        tipo_vol = st.radio("Tipo:", ["Call", "Put"], horizontal=True, key="vol_tipo")
        es_call_vol = (tipo_vol == "Call")

    with col_v2:
        st.markdown("**Cálculo de σ implícita — una sola opción**")
        K_vol      = st.number_input("Strike ($K$)", min_value=0.01,
                                      value=st.session_state.get("vol_K_key", 100.0), step=1.0, key="vol_K")
        precio_mkt = st.number_input("Precio de mercado de la opción ($)",
                                      min_value=0.001, value=st.session_state.get("vol_precio_key", 10.0),
                                      step=0.5, key="vol_precio")
        # Calcular vol implícita
        def _bsm_price(sigma):
            return engine.black_scholes(S_vol, K_vol, r_vol, sigma, T_vol,
                                         es_call_vol, q_vol) - precio_mkt
        try:
            sig_impl = _brentq(_bsm_price, 1e-6, 10.0, xtol=1e-8, maxiter=200)
            c_th_v = get_current_theme()
            themed_success(
                f"<h3 style='margin:0; color:inherit;'>Volatilidad Implícita (σ_impl): {sig_impl*100:.4f}%</h3>"
            )
            # Intrinsic value
            vi_v = max(S_vol-K_vol, 0) if es_call_vol else max(K_vol-S_vol, 0)
            vt_v = precio_mkt - vi_v
            st.metric("Valor intrínseco", f"${vi_v:.4f}")
            st.metric("Valor temporal",   f"${vt_v:.4f}")
            st.metric("Moneyness (K/S)",  f"{K_vol/S_vol:.4f}")
            
            with paso_a_paso():
                st.latex(r"f(\sigma_{impl}) = \text{BSM}(\sigma_{impl}) - P_{mercado} = 0")
                st.latex(rf"f(\sigma_{{impl}}) = \text{{BSM}}({S_vol:,.2f}, {K_vol:,.2f}, {r_vol:.4f}, \sigma_{{impl}}, {T_vol:.4f}) - {precio_mkt:,.4f} = 0")
                alerta_metodo_numerico()
                st.latex(rf"\sigma_{{impl}} \approx {sig_impl:.6f}")
                themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>\sigma_{{impl}} = {sig_impl*100:.4f}\%</h4>")
                
        except ValueError:
            themed_error(
                "No se encontró σ_impl. El precio de mercado puede estar fuera del rango "
                "teórico (mayor que el subyacente o menor que el valor intrínseco)."
            )
            sig_impl = None

    with col_v3:
        st.markdown("**Cargar datos reales (Yahoo Finance)**")
        ticker_vol = st.text_input("Ticker:", value="AAPL", key="vol_tick").strip().upper()
        btn_vol_yf = st.button("Obtener S₀ y simular Precio de Mercado", key="btn_vol_yf")
        if btn_vol_yf:
            with st.spinner("Descargando..."):
                sv, vv = engine.obtener_datos_subyacente(ticker_vol)
                if sv is not None:
                    # Configurar variables en memoria
                    st.session_state["vol_S_key"] = float(sv)
                    st.session_state["vol_K_key"] = float(sv) # Strike ATM
                    
                    # MAGIA: Simulamos el precio de mercado inyectando una prima de riesgo del +5% de volatilidad
                    precio_mercado_simulado = engine.black_scholes(sv, sv, r_vol, vv + 0.05, T_vol, es_call_vol, q_vol)
                    st.session_state["vol_precio_key"] = float(precio_mercado_simulado)
                    
                    themed_success(
                        f"**{ticker_vol}** cargado con éxito. \n"
                        f"Spot ($S_0$) y Strike ($K$) fijados en **${sv:,.2f}**.\n\n"
                        f"**NOTA:** Se calculó un Precio de Mercado simulado asumiendo un 5% de prima de riesgo sobre la Vol. Histórica ({vv*100:.2f}%)."
                    )
                    st.rerun()
                else:
                    themed_error(f"No se encontró {ticker_vol}.")

    separador()

    # ── SONRISA DE VOLATILIDAD ─────────────────────────────────────────────────
    st.markdown("#### Sonrisa de Volatilidad — σ implícita vs. Strike")
    themed_info(
        "Ingresa precios de mercado para una cadena de strikes y visualiza la sonrisa. "
        "En el mundo real, la <span style='font-family: serif; font-style: italic;'>&sigma;<sub>impl</sub></span> varía según el strike (y no es constante como asume BSM), "
        "revelando el **sesgo de volatilidad** y la **prima de riesgo de cola**."
    )

    c_th_v2 = get_current_theme()

    # Tabla de precios de mercado por strike
    strikes_pct_def = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20]
    strikes_def = [round(S_vol * (1 + p/100), 2) for p in strikes_pct_def]

    # Generate default prices using a vol smile (higher for OTM puts)
    # If the user used the button, we anchor the smile to their newly generated implied volatility
    sig_atm = sig_impl if sig_impl else 0.20
    def _smile_vol(K):
        m = np.log(K / S_vol)
        return sig_atm + 0.04 * m**2 - 0.01 * m  # synthetic smile

    def_prices = [round(engine.black_scholes(S_vol, K, r_vol, _smile_vol(K), T_vol,
                                              es_call_vol, q_vol), 4)
                  for K in strikes_def]

    df_chain = pd.DataFrame({
        "Strike ($K$)": strikes_def,
        "Precio de mercado ($)": def_prices,
    })
    st.caption("Edita los **Precios de mercado** para usar cotizaciones reales:")
    ed_chain = st.data_editor(
        df_chain, hide_index=True, use_container_width=True,
        column_config={
            "Strike ($K$)":        st.column_config.NumberColumn("Strike ($K$)", format="%.2f", step=1.0),
            "Precio de mercado ($)": st.column_config.NumberColumn("Precio Mkt ($)", format="%.4f", step=0.01, min_value=0.0001),
        }
    )

    # Calcular σ implícita para cada fila
    impl_vols = []
    for _, row in ed_chain.iterrows():
        K_r = float(row["Strike ($K$)"])
        P_r = float(row["Precio de mercado ($)"])
        try:
            def _f(sigma): return engine.black_scholes(S_vol, K_r, r_vol, sigma, T_vol,
                                                        es_call_vol, q_vol) - P_r
            sv_r = _brentq(_f, 1e-6, 10.0, xtol=1e-8, maxiter=200)
        except Exception:
            sv_r = np.nan
        impl_vols.append(sv_r)

    moneyness = [K/S_vol for K in ed_chain["Strike ($K$)"].values]

    # Gráficas
    col_smile1, col_smile2 = st.columns(2)

    with col_smile1:
        fig_smile = go.Figure()
        valid = [(m, v) for m, v in zip(moneyness, impl_vols) if not np.isnan(v)]
        if valid:
            ms, vs = zip(*valid)
            fig_smile.add_trace(go.Scatter(
                x=list(ms), y=[v*100 for v in vs],
                mode="lines+markers", name="σ implícita",
                line=dict(color=c_th_v2["accent"], width=2.5),
                marker=dict(size=8, color=c_th_v2["primary"]),
            ))
            fig_smile.add_hline(y=sig_atm*100, line_dash="dot",
                                 line_color=c_th_v2["text_muted"],
                                 annotation_text=f"σ ATM={sig_atm*100:.1f}%")
            fig_smile.add_vline(x=1.0, line_dash="dash",
                                 line_color=c_th_v2["success"],
                                 annotation_text="ATM (K=S)")
        fig_smile.update_layout(
            title="Sonrisa de Volatilidad",
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Volatilidad Implícita (%)",
            height=400, **plotly_theme(),
        )
        st.plotly_chart(fig_smile, use_container_width=True)

    with col_smile2:
        # Tabla de resultados
        df_res_vol = pd.DataFrame({
            "Strike":       ed_chain["Strike ($K$)"].values,
            "Moneyness":    [f"{m:.4f}" for m in moneyness],
            "Precio Mkt":   [f"${p:.4f}" for p in ed_chain["Precio de mercado ($)"].values],
            "σ_impl (%)":   [f"{v*100:.4f}%" if not np.isnan(v) else "—" for v in impl_vols],
            "Estado":       ["ITM" if (K > S_vol and es_call_vol) or (K < S_vol and not es_call_vol)
                              else "ATM" if abs(K/S_vol - 1) < 0.02 else "OTM"
                              for K in ed_chain["Strike ($K$)"].values],
        })
        st.dataframe(df_res_vol, hide_index=True, use_container_width=True, height=370)

    # Superficie de volatilidad
    separador()
    st.markdown("#### Superficie de Volatilidad vs. Strike y Vencimiento")
    themed_info(
        "Extensión de la sonrisa a múltiples vencimientos. "
        "Basada en la <span style='font-family: serif; font-style: italic;'>&sigma;<sub>impl</sub></span> calculada con el vencimiento actual — ajustada "
        "heurísticamente por estructura de plazos (solo con fines ilustrativos)."
    )

    T_vals  = [0.083, 0.25, 0.5, 1.0, 2.0]  # 1m, 3m, 6m, 1y, 2y
    T_labs  = ["1m","3m","6m","1y","2y"]
    K_vals  = [S_vol*(1+p/100) for p in range(-25, 30, 5)]

    surf_z = []
    for T_s in T_vals:
        row_s = []
        for K_s in K_vals:
            # Synthetic smile that changes with T (term structure)
            m_s = np.log(K_s/S_vol) / np.sqrt(T_s)
            sigma_s = sig_atm + 0.03*m_s**2 - 0.008*m_s + 0.005/T_s
            row_s.append(sigma_s*100)
        surf_z.append(row_s)

    fig_surf = go.Figure(data=go.Surface(
        z=surf_z, x=K_vals, y=T_labs,
        colorscale="Viridis",
        colorbar=dict(title="σ_impl (%)"),
    ))
    fig_surf.update_layout(
        title="Superficie de Volatilidad (ilustrativa)",
        scene=dict(
            xaxis_title="Strike ($K$)",
            yaxis_title="Vencimiento",
            zaxis_title="σ_impl (%)",
        ),
        height=500, **plotly_theme(),
    )
    st.plotly_chart(fig_surf, use_container_width=True)
    themed_warning(
        "La superficie ilustrativa usa una fórmula heurística. "
        "Para una superficie real, carga los precios de mercado de opciones reales "
        "para cada vencimiento usando el buscador de Yahoo Finance en el tab Activos Reales."
    )