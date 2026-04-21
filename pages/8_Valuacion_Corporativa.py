"""
pages/8_Valuacion_Corporativa.py
----------------------------------
Módulo 8: Valuación Corporativa.
Cubre:
  - DCF con WACC: proyección de FCF, valor terminal, valor empresa y del capital.
    Análisis de sensibilidad 2D (tasa de descuento vs tasa de crecimiento).
  - CAPM y Fama-French: Beta vs. mercado, SML, regresión de 3 factores.
  - Múltiplos comparables: panel de métricas P/E, EV/EBITDA, P/S, P/B
    descargadas automáticamente via yfinance para cualquier lista de tickers.
  - Historial de cálculos: los últimos 10 valuaciones guardadas en session_state.
"""

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from numpy.linalg import lstsq


try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

from utils import (
    get_engine, page_header, paso_a_paso, separador,
    themed_info, themed_success, themed_warning, themed_error,
    plotly_theme, plotly_colors, get_current_theme,
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Valuación Corporativa · Calculadora Financiera",
    page_icon="🏢",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="15. Valuación Corporativa",
    subtitulo="DCF · WACC · CAPM · Fama-French · Múltiplos Comparables"
)

# =============================================================================
# HISTORIAL GLOBAL
# =============================================================================
if "hist_valuaciones" not in st.session_state:
    st.session_state["hist_valuaciones"] = []

def _guardar_en_historial(tipo: str, resumen: dict):
    """Guarda un cálculo en el historial (máximo 10)."""
    entrada = {
        "fecha": datetime.datetime.now().strftime("%H:%M:%S"),
        "tipo": tipo,
        **resumen,
    }
    st.session_state["hist_valuaciones"].insert(0, entrada)
    st.session_state["hist_valuaciones"] = st.session_state["hist_valuaciones"][:10]

# =============================================================================
# TABS
# =============================================================================
tab_dcf, tab_capm, tab_mult, tab_hist = st.tabs([
    "DCF + WACC",
    "CAPM y Fama-French",
    "Múltiplos Comparables",
    "Historial de Cálculos",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DCF + WACC
# ══════════════════════════════════════════════════════════════════════════════
with tab_dcf:
    st.markdown("### Descuento de Flujos de Caja (DCF) con WACC")
    themed_info(
        "El **DCF** estima el valor intrínseco de una empresa descontando sus flujos de caja libre "
        "futuros (FCF) al costo promedio ponderado del capital (WACC). "
        "Valor empresa = VP(FCF proyectados) + VP(Valor terminal). "
        "Valor del capital = Valor empresa − Deuda neta."
    )
    separador()

    c1, c2, c3 = st.columns(3)

    # ── Inputs de la empresa ──────────────────────────────────────────────────
    with c1:
        st.markdown("**Empresa y FCF base**")
        nombre_empresa = st.text_input("Nombre de la empresa:", value="Mi Empresa", key="dcf_nom")
        fcf_base = st.number_input("FCF del último año ($M):", min_value=0.01,
                                    value=500.0, step=50.0, key="dcf_fcf")
        n_proy = st.number_input("Años de proyección explícita:", min_value=1,
                                  max_value=15, value=5, step=1, key="dcf_n")
        g_proy = st.number_input("Tasa de crecimiento proyectada (%):",
                                   value=8.0, step=0.5, key="dcf_g") / 100
        g_term = st.number_input("Tasa de crecimiento terminal (%):",
                                   value=2.5, step=0.25, key="dcf_gt") / 100

    with c2:
        st.markdown("**WACC**")
        ke_mode = st.radio("Calcular Ke con:", ["CAPM", "Ingresar directamente"],
                            horizontal=True, key="dcf_ke_mode")
        if ke_mode == "Ingresar directamente":
            Ke = st.number_input("Ke — Costo del capital (%):",
                                  value=12.0, step=0.5, key="dcf_ke") / 100
        else:
            rf_dcf   = st.number_input("Tasa libre de riesgo rf (%):",
                                        value=4.5, step=0.1, key="dcf_rf") / 100
            beta_dcf = st.number_input("Beta de la acción:", value=1.2,
                                        step=0.05, key="dcf_beta")
            rm_dcf   = st.number_input("Rendimiento del mercado E[Rm] (%):",
                                        value=10.0, step=0.5, key="dcf_rm") / 100
            Ke = rf_dcf + beta_dcf * (rm_dcf - rf_dcf)
            st.metric("Ke (CAPM)", f"{Ke*100:.2f}%")

        Kd    = st.number_input("Kd — Costo de la deuda antes de impuestos (%):",
                                 value=6.0, step=0.5, key="dcf_kd") / 100
        T_imp = st.number_input("Tasa impositiva (%):",
                                  value=30.0, step=1.0, key="dcf_T") / 100
        E_pct = st.number_input("Proporción capital (E/V) (%):",
                                  min_value=1.0, max_value=99.0,
                                  value=60.0, step=5.0, key="dcf_E") / 100
        D_pct = 1 - E_pct
        WACC  = Ke * E_pct + Kd * (1 - T_imp) * D_pct
        st.metric("WACC calculado", f"{WACC*100:.2f}%")

    with c3:
        st.markdown("**Balance (para valor del capital)**")
        deuda_total = st.number_input("Deuda total ($M):", min_value=0.0,
                                       value=1000.0, step=100.0, key="dcf_deu")
        caja        = st.number_input("Caja y equivalentes ($M):", min_value=0.0,
                                       value=200.0, step=50.0, key="dcf_caja")
        acciones    = st.number_input("Acciones en circulación (M):", min_value=0.01,
                                       value=100.0, step=10.0, key="dcf_acc")
        deuda_neta  = deuda_total - caja
        st.metric("Deuda neta ($M)", f"${deuda_neta:,.1f}M")

    separador()

    # ── Cálculo DCF ───────────────────────────────────────────────────────────
    if WACC <= g_term:
        themed_error("El WACC debe ser mayor que la tasa de crecimiento terminal para que el DCF converja.")
    else:
        # FCF proyectados
        fcfs     = [fcf_base * (1 + g_proy) ** t for t in range(1, int(n_proy) + 1)]
        pv_fcfs  = [f / (1 + WACC) ** t for t, f in enumerate(fcfs, 1)]

        # Valor terminal (perpetuidad creciente)
        fcf_n1   = fcfs[-1] * (1 + g_term)
        vt       = fcf_n1 / (WACC - g_term)
        pv_vt    = vt / (1 + WACC) ** int(n_proy)

        # Valor empresa y del capital
        vp_fcf_total = sum(pv_fcfs)
        v_empresa    = vp_fcf_total + pv_vt
        v_capital    = v_empresa - deuda_neta
        precio_acc   = v_capital / acciones if acciones > 0 else 0

        # Resultados
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("VP FCFs proyectados ($M)", f"${vp_fcf_total:,.1f}M")
        col_r2.metric("VP Valor terminal ($M)",   f"${pv_vt:,.1f}M",
                       help=f"Valor terminal = ${vt:,.1f}M · Peso en EV: {pv_vt/v_empresa*100:.1f}%")
        col_r3.metric("Valor empresa (EV)",       f"${v_empresa:,.1f}M")
        col_r4.metric("Valor del capital (Eq)",   f"${v_capital:,.1f}M")

        themed_success(f"<h2 style='margin:0; color:inherit; text-align:center;'>Precio Teórico por Acción: ${precio_acc:,.2f}</h2>")

        with paso_a_paso():
            # 1. WACC
            st.latex(r"WACC = K_e \left(\frac{E}{V}\right) + K_d (1-T) \left(\frac{D}{V}\right)")
            st.latex(rf"WACC = {Ke:.4f} ({E_pct:.2f}) + {Kd:.4f} (1 - {T_imp:.2f}) ({D_pct:.2f}) = {WACC:.6f}")
            
            # 2. Valor Terminal
            st.latex(r"TV_n = \frac{FCF_{n+1}}{WACC - g_{term}} = \frac{FCF_n (1+g_{term})}{WACC - g_{term}}")
            st.latex(rf"TV_{{{n_proy:g}}} = \frac{{{fcfs[-1]:,.2f} (1+{g_term:.4f})}}{{{WACC:.4f} - {g_term:.4f}}} = \frac{{{fcf_n1:,.2f}}}{{{WACC - g_term:.4f}}} = {vt:,.2f}")
            
            # 3. Enterprise Value
            st.latex(r"EV = \sum_{t=1}^{n} \frac{FCF_t}{(1+WACC)^t} + \frac{TV_n}{(1+WACC)^n}")
            st.latex(rf"EV = {vp_fcf_total:,.2f} + \frac{{{vt:,.2f}}}{{(1+{WACC:.4f})^{{{n_proy:g}}}}}")
            st.latex(rf"EV = {vp_fcf_total:,.2f} + {pv_vt:,.2f} = {v_empresa:,.2f}")
            
            # 4. Equity Value & Price
            st.latex(r"\text{Equity} = EV - \text{Deuda Neta}")
            st.latex(rf"\text{{Equity}} = {v_empresa:,.2f} - {deuda_neta:,.2f} = {v_capital:,.2f}")
            st.latex(r"P_0 = \frac{\text{Equity}}{\text{Acciones en circulación}}")
            st.latex(rf"P_0 = \frac{{{v_capital:,.2f}}}{{{acciones:,.2f}}} = {precio_acc:,.2f}")

        separador()

        # Tabla de FCFs proyectados
        st.markdown("##### Tabla de proyección FCF")
        df_proy = pd.DataFrame({
            "Año":             [f"Año {t}" for t in range(1, int(n_proy)+1)],
            "FCF ($M)":        [f"${f:,.2f}" for f in fcfs],
            "Factor descuento":[f"{1/(1+WACC)**t:.6f}" for t in range(1, int(n_proy)+1)],
            "VP FCF ($M)":     [f"${p:,.2f}" for p in pv_fcfs],
        })
        df_proy.loc[len(df_proy)] = ["Terminal", f"${vt:,.2f}", f"{1/(1+WACC)**int(n_proy):.6f}", f"${pv_vt:,.2f}"]
        st.dataframe(df_proy, hide_index=True, use_container_width=True)

        # Gráfica de FCFs
        c_th = get_current_theme()
        fig_fcf = go.Figure()
        fig_fcf.add_trace(go.Bar(
            x=[f"Año {t}" for t in range(1, int(n_proy)+1)],
            y=pv_fcfs,
            name="VP FCF", marker_color=c_th["primary"],
        ))
        fig_fcf.add_trace(go.Bar(
            x=["Valor Terminal"], y=[pv_vt],
            name="VP Terminal", marker_color=c_th["accent"],
        ))
        fig_fcf.update_layout(
            title="Composición del Valor Empresa (Enterprise Value)",
            xaxis_title="Período", yaxis_title="VP ($M)",
            barmode="group", height=380, **plotly_theme(),
        )
        st.plotly_chart(fig_fcf, use_container_width=True)

        separador()

        # ── Análisis de sensibilidad 2D ───────────────────────────────────────
        st.markdown("##### Análisis de sensibilidad 2D — Valor por acción ($)")
        themed_info(
            "Muestra cómo varía el **precio por acción** al cambiar el WACC "
            "y la tasa de crecimiento terminal simultáneamente."
        )
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            wacc_rng = st.slider("Rango WACC (%): ±", min_value=1.0, max_value=5.0,
                                  value=2.0, step=0.5, key="dcf_sens_wacc")
        with col_s2:
            g_rng = st.slider("Rango g terminal (%): ±", min_value=0.5, max_value=3.0,
                               value=1.0, step=0.5, key="dcf_sens_g")

        wacc_vals = np.linspace(WACC - wacc_rng/100, WACC + wacc_rng/100, 7)
        g_vals    = np.linspace(g_term - g_rng/100, max(g_term + g_rng/100, g_term + 0.005), 7)

        sens_matrix = np.zeros((len(g_vals), len(wacc_vals)))
        for i, g_v in enumerate(g_vals):
            for j, w_v in enumerate(wacc_vals):
                if w_v <= g_v or w_v <= 0:
                    sens_matrix[i, j] = np.nan
                    continue
                fcf_n1_s = fcfs[-1] * (1 + g_v)
                vt_s     = fcf_n1_s / (w_v - g_v)
                pv_vt_s  = vt_s / (1 + w_v) ** int(n_proy)
                pv_f_s   = sum(fcfs[t-1] / (1+w_v)**t for t in range(1, int(n_proy)+1))
                vc_s     = (pv_f_s + pv_vt_s) - deuda_neta
                sens_matrix[i, j] = vc_s / acciones if acciones > 0 else 0

        df_sens = pd.DataFrame(
            np.round(sens_matrix, 2),
            index=[f"g={v*100:.1f}%" for v in g_vals],
            columns=[f"WACC={v*100:.1f}%" for v in wacc_vals],
        )

        fig_sens = go.Figure(go.Heatmap(
            z=sens_matrix,
            x=[f"{v*100:.1f}%" for v in wacc_vals],
            y=[f"{v*100:.1f}%" for v in g_vals],
            colorscale="RdYlGn",
            text=np.round(sens_matrix, 2),
            texttemplate="$%{text:.2f}",
            colorbar=dict(title="Precio ($)"),
        ))
        fig_sens.update_layout(
            title="Precio por acción vs WACC y g terminal",
            xaxis_title="WACC", yaxis_title="g terminal",
            height=420, **plotly_theme(),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        # Guardar en historial
        _guardar_en_historial("DCF", {
            "empresa": nombre_empresa,
            "valor_empresa": f"${v_empresa:,.1f}M",
            "precio_accion": f"${precio_acc:,.2f}",
            "WACC": f"{WACC*100:.2f}%",
        })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAPM Y FAMA-FRENCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_capm:
    st.markdown("### CAPM y Modelos de Factor")
    themed_info(
        "Estima la **Beta** de una acción frente al mercado usando datos reales. "
        "Traza la Línea del Mercado de Valores (SML)."
    )
    separador()

    if not YFINANCE_OK:
        themed_error("yfinance no está instalado. `pip install yfinance`")
    else:
        col_ca1, col_ca2 = st.columns([2, 1])
        with col_ca1:
            ticker_capm = st.text_input("Ticker de la acción:", value="AAPL", key="capm_tick")
            index_capm  = st.selectbox("Índice de mercado:", ["^GSPC (S&P 500)", "^IXIC (NASDAQ)", "^MXX (IPC México)"],
                                        key="capm_idx")
            idx_symbol  = index_capm.split(" ")[0]
        with col_ca2:
            rf_capm  = st.number_input("Tasa libre de riesgo anual (%):", value=4.5,
                                        step=0.1, key="capm_rf") / 100
            yr_capm  = st.number_input("Años de historial:", min_value=1, max_value=10,
                                        value=3, step=1, key="capm_yr")

        btn_capm = st.button("Calcular Beta y CAPM", use_container_width=True, key="btn_capm")

        if btn_capm:
            with st.spinner("Descargando datos históricos..."):
                try:
                    hoy_c   = datetime.date.today()
                    inicio_c = hoy_c - datetime.timedelta(days=int(yr_capm*365))
                    data_a  = yf.download(ticker_capm, start=inicio_c, end=hoy_c,
                                           progress=False, auto_adjust=True)["Close"]
                    data_m  = yf.download(idx_symbol,  start=inicio_c, end=hoy_c,
                                           progress=False, auto_adjust=True)["Close"]

                    if data_a.empty or data_m.empty:
                        themed_error("No se pudieron descargar datos. Verifica el ticker.")
                    else:
                        ret_a = data_a.pct_change().dropna()
                        ret_m = data_m.pct_change().dropna()
                        df_r  = pd.concat([ret_a, ret_m], axis=1, join="inner")
                        df_r.columns = ["Accion", "Mercado"]

                        # Exceso de rendimiento diario
                        rf_daily = (1 + rf_capm) ** (1/252) - 1
                        df_r["Exc_A"] = df_r["Accion"]  - rf_daily
                        df_r["Exc_M"] = df_r["Mercado"] - rf_daily

                        # Regresión OLS: Exc_A = alpha + beta * Exc_M
                        X = np.column_stack([np.ones(len(df_r)), df_r["Exc_M"].values])
                        y = df_r["Exc_A"].values
                        coeffs, _, _, _ = lstsq(X, y, rcond=None)
                        alpha_capm, beta_capm = coeffs

                        # Métricas anualizadas
                        ret_anual_a = ((1 + df_r["Accion"].mean()) ** 252 - 1)
                        ret_anual_m = ((1 + df_r["Mercado"].mean()) ** 252 - 1)
                        vol_anual_a = df_r["Accion"].std() * np.sqrt(252)
                        vol_anual_m = df_r["Mercado"].std() * np.sqrt(252)
                        corr_am     = df_r["Accion"].corr(df_r["Mercado"])
                        ke_capm_est = rf_capm + beta_capm * (ret_anual_m - rf_capm)

                        st.session_state["capm_result"] = {
                            "ticker": ticker_capm, "beta": beta_capm,
                            "alpha": alpha_capm, "ret_a": ret_anual_a,
                            "ret_m": ret_anual_m, "vol_a": vol_anual_a,
                            "vol_m": vol_anual_m, "corr": corr_am,
                            "ke": ke_capm_est, "df_r": df_r,
                            "rf": rf_capm,
                        }
                except Exception as e:
                    themed_error(f"Error: {e}")

        if "capm_result" in st.session_state:
            res = st.session_state["capm_result"]
            c_th = get_current_theme()

            themed_success(f"<h3 style='margin:0; color:inherit;'>Rendimiento Requerido (Ke): {res['ke']*100:.2f}%</h3>")
            
            with paso_a_paso():
                st.latex(r"K_e = r_f + \beta (E[R_m] - r_f)")
                st.latex(rf"K_e = {res['rf']:.4f} + {res['beta']:.4f} ({res['ret_m']:.4f} - {res['rf']:.4f})")
                st.latex(rf"K_e = {res['rf']:.4f} + {res['beta']:.4f} ({res['ret_m']-res['rf']:.4f})")
                st.latex(rf"K_e = {res['ke']:.6f}")
                st.latex(r"\text{Alpha de Jensen } (\alpha) = R_{real} - K_e")
                st.latex(rf"\alpha = {res['ret_a']:.6f} - {res['ke']:.6f} = {res['ret_a']-res['ke']:.6f}")

            separador()

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Beta (β)",              f"{res['beta']:.4f}")
            cm2.metric("Alpha (α) anualizado",  f"{(res['ret_a']-res['ke'])*100:.2f}%")
            cm3.metric(f"Ret. anual {res['ticker']}", f"{res['ret_a']*100:.2f}%")
            cm4.metric("Correlación c/mercado", f"{res['corr']:.4f}")

            separador()

            col_sc, col_sml = st.columns(2)

            with col_sc:
                df_r = res["df_r"]
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=df_r["Exc_M"]*100, y=df_r["Exc_A"]*100,
                    mode="markers", name="Retornos diarios",
                    marker=dict(size=3, color=c_th["primary"], opacity=0.4),
                ))
                x_line = np.linspace(df_r["Exc_M"].min(), df_r["Exc_M"].max(), 50)
                y_line = res["alpha"] + res["beta"] * x_line
                fig_sc.add_trace(go.Scatter(
                    x=x_line*100, y=y_line*100,
                    mode="lines", name=f"β={res['beta']:.3f}",
                    line=dict(color=c_th["accent"], width=2.5),
                ))
                fig_sc.update_layout(
                    title=f"Regresión {res['ticker']} vs Mercado",
                    xaxis_title="Exc. retorno mercado (%)",
                    yaxis_title=f"Exc. retorno {res['ticker']} (%)",
                    height=380, **plotly_theme(),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            with col_sml:
                betas_sml = np.linspace(0, 2.5, 50)
                ke_sml    = res["rf"] + betas_sml * (res["ret_m"] - res["rf"])
                fig_sml = go.Figure()
                fig_sml.add_trace(go.Scatter(
                    x=betas_sml, y=ke_sml*100,
                    mode="lines", name="SML",
                    line=dict(color=c_th["primary"], width=2.5),
                ))
                fig_sml.add_trace(go.Scatter(
                    x=[res["beta"]], y=[res["ke"]*100],
                    mode="markers", name=f"Ke Teórico ({res['ticker']})",
                    marker=dict(size=14, color=c_th["accent"],
                                symbol="star"),
                ))
                fig_sml.add_trace(go.Scatter(
                    x=[res["beta"]], y=[res["ret_a"]*100],
                    mode="markers", name=f"Ret. Real ({res['ticker']})",
                    marker=dict(size=10, color=c_th["success"],
                                symbol="diamond"),
                ))
                fig_sml.update_layout(
                    title="Línea del Mercado de Valores (SML)",
                    xaxis_title="Beta (β)", yaxis_title="Rendimiento esperado (%)",
                    height=380, **plotly_theme(),
                )
                st.plotly_chart(fig_sml, use_container_width=True)

            _guardar_en_historial("CAPM", {
                "ticker": res["ticker"], "beta": f"{res['beta']:.4f}",
                "Ke (CAPM)": f"{res['ke']*100:.2f}%",
                "alpha anual": f"{(res['ret_a']-res['ke'])*100:.2f}%",
            })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MÚLTIPLOS COMPARABLES
# ══════════════════════════════════════════════════════════════════════════════
with tab_mult:
    st.markdown("### Valuación por Múltiplos Comparables")
    themed_info(
        "Descarga automáticamente los múltiplos de mercado de un panel de empresas comparables "
        "usando **yfinance**. Calcula el rango de valor implícito para tu empresa objetivo "
        "aplicando los múltiplos medianos del panel."
    )
    separador()

    if not YFINANCE_OK:
        themed_error("yfinance no está instalado.")
    else:
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            tickers_comp = st.text_input(
                "Empresas comparables (tickers separados por coma):",
                value="AAPL, MSFT, GOOGL, META, AMZN",
                key="mult_tickers",
            )
            ticker_obj = st.text_input(
                "Empresa objetivo (opcional — para calcular valor implícito):",
                value="", key="mult_obj", placeholder="Ej. NVDA",
            )
        with col_m2:
            st.write("")
            btn_mult = st.button("Descargar múltiplos", use_container_width=True,
                                  key="btn_mult")

        if btn_mult:
            ticks = [t.strip().upper() for t in tickers_comp.split(",") if t.strip()]
            with st.spinner(f"Descargando datos de {len(ticks)} empresas..."):
                rows = []
                for t in ticks:
                    try:
                        info = yf.Ticker(t).info
                        rows.append({
                            "Ticker":         t,
                            "Nombre":         info.get("shortName", t)[:30],
                            "P/E (trailing)": info.get("trailingPE"),
                            "P/E (forward)":  info.get("forwardPE"),
                            "EV/EBITDA":      info.get("enterpriseToEbitda"),
                            "P/S":            info.get("priceToSalesTrailing12Months"),
                            "P/B":            info.get("priceToBook"),
                            "Precio ($)":     info.get("currentPrice") or info.get("regularMarketPrice"),
                            "Cap. Mercado ($B)": (info.get("marketCap") or 0) / 1e9,
                            "EV ($B)":           (info.get("enterpriseValue") or 0) / 1e9,
                        })
                    except Exception:
                        rows.append({"Ticker": t, "Nombre": "Error al descargar"})

                df_mult = pd.DataFrame(rows)
                st.session_state["df_mult"] = df_mult

            # Descargar datos de la empresa objetivo si se ingresó
            if ticker_obj.strip():
                try:
                    info_obj = yf.Ticker(ticker_obj.strip().upper()).info
                    st.session_state["mult_obj_info"] = info_obj
                    st.session_state["mult_obj_tick"] = ticker_obj.strip().upper()
                except Exception as e:
                    themed_warning(f"No se pudo obtener datos de {ticker_obj}: {e}")

        if "df_mult" in st.session_state:
            df_m = st.session_state["df_mult"]
            c_th = get_current_theme()

            # Panel de comparables
            st.markdown("##### Panel de múltiplos de las empresas comparables")
            numeric_cols = ["P/E (trailing)","P/E (forward)","EV/EBITDA","P/S","P/B"]
            df_display = df_m.copy()
            for col in numeric_cols:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:.2f}x" if pd.notna(x) and x is not None else "—"
                    )
            df_display["Cap. Mercado ($B)"] = df_display["Cap. Mercado ($B)"].apply(
                lambda x: f"${x:.1f}B" if pd.notna(x) else "—")
            df_display["Precio ($)"] = df_display["Precio ($)"].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "—")
            st.dataframe(df_display, hide_index=True, use_container_width=True)

            separador()

            # Estadísticas del panel
            st.markdown("##### Estadísticas del panel")
            stats_rows = []
            for col in numeric_cols:
                if col not in df_m.columns:
                    continue
                vals_c = pd.to_numeric(df_m[col], errors="coerce").dropna()
                vals_c = vals_c[(vals_c > 0) & (vals_c < 500)]  # filtrar outliers extremos
                if len(vals_c) > 0:
                    stats_rows.append({
                        "Múltiplo": col,
                        "Mínimo":  f"{vals_c.min():.2f}x",
                        "Mediana": f"{vals_c.median():.2f}x",
                        "Media":   f"{vals_c.mean():.2f}x",
                        "Máximo":  f"{vals_c.max():.2f}x",
                        "n":       int(len(vals_c)),
                    })
            df_stats = pd.DataFrame(stats_rows)
            st.dataframe(df_stats, hide_index=True, use_container_width=True)

            # Valuación implícita de la empresa objetivo
            if "mult_obj_info" in st.session_state:
                separador()
                st.markdown(f"##### Valuación implícita de **{st.session_state['mult_obj_tick']}**")
                info_o = st.session_state["mult_obj_info"]

                metricas_obj = {
                    "P/E (trailing)": info_o.get("trailingEps"),
                    "P/E (forward)":  info_o.get("forwardEps"),
                    "EV/EBITDA":      (info_o.get("ebitda") or 0) / 1e9,
                    "P/S":            (info_o.get("totalRevenue") or 0) / (info_o.get("sharesOutstanding") or 1),
                    "P/B":            info_o.get("bookValue"),
                }
                precio_actual = info_o.get("currentPrice") or info_o.get("regularMarketPrice")

                impl_rows = []
                for col, met_val in metricas_obj.items():
                    if col not in df_stats["Múltiplo"].values or met_val is None:
                        continue
                    med_mult = float(df_stats[df_stats["Múltiplo"]==col]["Mediana"].iloc[0].replace("x",""))
                    precio_imp = met_val * med_mult
                    if precio_imp > 0:
                        updown = (precio_imp / precio_actual - 1) * 100 if precio_actual else 0
                        impl_rows.append({
                            "Múltiplo usado": col,
                            "Métrica base": f"{met_val:.2f}",
                            "Múltiplo mediana panel": f"{med_mult:.2f}x",
                            "Precio implícito ($)": f"${precio_imp:.2f}",
                            "vs Precio actual": f"{updown:+.1f}%",
                        })

                if impl_rows:
                    themed_success(f"<h3 style='margin:0; color:inherit;'>Precio Real de Mercado: ${precio_actual:.2f}</h3>")
                    
                    with paso_a_paso():
                        st.latex(r"P_{implícito} = \text{Métrica}_{\text{Empresa}} \times \text{Múltiplo}_{\text{Mediana del Panel}}")
                        st.latex(rf"P_{{P/E}} = {metricas_obj.get('P/E (trailing)', 0):.2f} \times {float(df_stats[df_stats['Múltiplo']=='P/E (trailing)']['Mediana'].iloc[0].replace('x','')):.2f} = {metricas_obj.get('P/E (trailing)', 0)*float(df_stats[df_stats['Múltiplo']=='P/E (trailing)']['Mediana'].iloc[0].replace('x','')):.2f}")

                    st.dataframe(pd.DataFrame(impl_rows), hide_index=True, use_container_width=True)

            _guardar_en_historial("Múltiplos", {
                "panel": tickers_comp[:40],
                "empresas": str(len(df_m)),
            })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORIAL DE CÁLCULOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("### Historial de Cálculos")
    themed_info(
        "Se guardan automáticamente los últimos **10 cálculos** realizados en esta sesión. "
        "Útil para comparar resultados con distintos supuestos sin necesidad de reescribir parámetros."
    )
    separador()

    hist = st.session_state.get("hist_valuaciones", [])
    if not hist:
        themed_warning("Aún no hay cálculos en el historial. Realiza una valuación para verla aquí.")
    else:
        col_h1, col_h2 = st.columns([4, 1])
        with col_h2:
            if st.button("Limpiar historial", key="limpiar_hist"):
                st.session_state["hist_valuaciones"] = []
                st.rerun()
        with col_h1:
            for i, entrada in enumerate(hist):
                badge = {
                    "DCF": "🏢",
                    "CAPM": "📐",
                    "Múltiplos": "📊",
                }.get(entrada.get("tipo",""), "📌")
                with st.expander(
                    f"{badge} [{entrada['fecha']}] {entrada['tipo']} — "
                    + " · ".join(f"{k}={v}" for k, v in entrada.items()
                                  if k not in ("fecha","tipo"))
                ):
                    df_e = pd.DataFrame([{k: v for k, v in entrada.items()
                                          if k not in ("fecha","tipo","df_r")}])
                    st.dataframe(df_e, hide_index=True, use_container_width=True)