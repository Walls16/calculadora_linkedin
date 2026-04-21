"""
pages/8_Riesgo.py
-----------------
Módulo 8: Análisis de Riesgo.
Tab 1: Riesgo de Portafolios de Acciones — VaR / CVaR paramétrico y Monte Carlo.
Tab 2: CreditMetrics — Riesgo de crédito en portafolios de bonos corporativos.
"""

import datetime
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from credit_engine import (
    RATINGS_EMIT as RATINGS,
    RATING_IDX,
    DEFAULT_TM,
    DEFAULT_SPREADS,
    DEFAULT_TREASURY,
    _TM_RAW_17x19,
    build_transition_matrix,
    bond_values_per_rating,
    var_cvar_parametric,
    var_cvar_from_simulations,
    scale_var_cvar,
    gaussian_copula_simulation,
    thresholds_per_bond,
    expected_value_and_sigma,
    N_EMIT as N_R,
    TRADING_DAYS,
)
from utils import (
    get_engine, page_header, separador,
    themed_info, themed_success, themed_warning, themed_error,
    plotly_theme, plotly_colors, get_current_theme,
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Riesgo · Calculadora Financiera",
    page_icon="📉",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="8. Análisis de Riesgo",
    subtitulo="Portafolios de Acciones (VaR/CVaR) · CreditMetrics para Bonos Corporativos"
)

# =============================================================================
# TABS PRINCIPALES
# =============================================================================
tab_port, tab_cm = st.tabs([
    "Riesgo de Portafolios (Acciones)",
    "CreditMetrics — Riesgo de Bonos",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RIESGO DE PORTAFOLIOS (original)
# ══════════════════════════════════════════════════════════════════════════════
with tab_port:

    with st.expander("Seleccionar Activos y Periodo", expanded=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            tickers_str = st.text_input(
                "Símbolos (separados por coma):", value="AAPL, MSFT, META",
                help="Ejemplos: AAPL, CEMEXCPO.MX, SPY, AMZN",
            )
        with c2:
            hoy = datetime.date.today()
            fecha_inicio = st.date_input(
                "Fecha de Inicio (Histórico)",
                value=hoy - datetime.timedelta(days=365 * 3),
            )

    tickers_list = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    separador()

    st.markdown("#### Asignar Ponderaciones Manuales")
    st.caption("Haz clic en las celdas de la columna **Peso (%)** para ajustar el porcentaje.")

    if len(tickers_list) > 0:
        peso_eq = round(100.0 / len(tickers_list), 4)
        df_pesos_ini = pd.DataFrame({
            "Ticker":   tickers_list,
            "Peso (%)": [peso_eq] * len(tickers_list),
        })
        df_pesos_editado = st.data_editor(
            df_pesos_ini, hide_index=True, use_container_width=True,
            column_config={
                "Ticker":   st.column_config.TextColumn("Ticker", disabled=True),
                "Peso (%)": st.column_config.NumberColumn(
                    "Peso (%)", min_value=0.0, max_value=100.0,
                    step=0.01, format="%.4f%%"),
            },
        )
        suma_pesos = df_pesos_editado["Peso (%)"].sum()
        if abs(suma_pesos - 100.0) > 0.1:
            themed_warning(
                f"Tus porcentajes suman **{suma_pesos:.1f}%**. "
                "El modelo los ajustará proporcionalmente al 100%."
            )
    else:
        themed_error("Ingresa al menos un ticker válido.")
        st.stop()

    separador()
    st.markdown("#### Configurar Escenario de Riesgo")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        val_portafolio = st.number_input(
            "Capital Total Invertido ($)", min_value=100.0,
            value=100_000.0, step=10_000.0, key="riesgo_capital"
        )
    with col_r2:
        conf_str  = st.selectbox("Nivel de Confianza", ["95%", "99%"], index=1, key="riesgo_conf_sel")
        confianza = 0.95 if conf_str == "95%" else 0.99
    with col_r3:
        horizonte_str = st.selectbox(
            "Horizonte de Tiempo",
            ["1 Día", "10 Días", "21 Días (1 Mes)"],
            index=1,
        )
        dias_h = int(horizonte_str.split()[0])

    ejecutar_riesgo = st.button(
        "Calcular Métricas de Riesgo", use_container_width=True, key="btn_riesgo",
    )
    separador()

    if ejecutar_riesgo:
        dict_pesos = dict(zip(
            df_pesos_editado["Ticker"],
            df_pesos_editado["Peso (%)"] / 100.0,
        ))
        with st.spinner("Descargando precios y simulando Monte Carlo..."):
            try:
                data, rend_p, vol_p, pesos_reales, cols_reales = engine.evaluar_portafolio_personalizado(
                    tickers_list, dict_pesos, fecha_inicio, hoy
                )
                st.session_state.update({
                    "riesgo_data": data, "riesgo_rend": rend_p, "riesgo_vol": vol_p,
                    "riesgo_pesos": pesos_reales, "riesgo_cols": list(cols_reales),
                    "riesgo_capital_val": val_portafolio, "riesgo_confianza": confianza,
                    "riesgo_horizonte": horizonte_str, "riesgo_dias": dias_h,
                    "riesgo_hoy": hoy,
                })
                themed_success("Análisis completado.")
            except Exception as e:
                themed_error(f"Error en el cálculo: {e}")

    if "riesgo_rend" in st.session_state:
        rend_p       = st.session_state["riesgo_rend"]
        vol_p        = st.session_state["riesgo_vol"]
        pesos_reales = st.session_state["riesgo_pesos"]
        cols_reales  = st.session_state["riesgo_cols"]
        capital      = st.session_state["riesgo_capital_val"]
        conf         = st.session_state["riesgo_confianza"]
        h_str        = st.session_state["riesgo_horizonte"]
        dias         = st.session_state["riesgo_dias"]
        data         = st.session_state["riesgo_data"]
        hoy_g        = st.session_state["riesgo_hoy"]

        st.markdown("### Perfil del Portafolio")
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Rendimiento Esperado Anual", f"{rend_p*100:.2f}%")
        c_m2.metric("Volatilidad Anual (σ)",       f"{vol_p*100:.2f}%")
        sharpe_ref = (rend_p - 0.05) / vol_p if vol_p > 0 else 0.0
        c_m3.metric("Ratio de Sharpe", f"{sharpe_ref:.4f}", help="Tasa libre de riesgo: 5% anual")
        separador()

        var_p,  _, _, _ = engine.calcular_var_parametrico(rend_p, vol_p, capital, conf, dias)
        var_mc, cvar_mc = engine.calcular_var_cvar_montecarlo(rend_p, vol_p, capital, conf, dias)

        st.markdown(f"### VaR — Horizonte: **{h_str}** | Confianza: **{conf*100:.0f}%**")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("VaR Paramétrico",            f"${var_p:,.2f}")
        col_res2.metric("VaR Monte Carlo",             f"${var_mc:,.2f}")
        col_res3.metric("CVaR (Expected Shortfall)",   f"${cvar_mc:,.2f}")

        tab_comp_p, tab_hist_p, tab_dl_p = st.tabs([
            "Composición Efectiva", "Precios Históricos", "Exportar Datos"
        ])
        with tab_comp_p:
            df_pie = pd.DataFrame({"Activo": cols_reales, "Peso": pesos_reales})
            fig_pie = px.pie(df_pie, values="Peso", names="Activo", hole=0.4,
                             color_discrete_sequence=plotly_colors())
            fig_pie.update_layout(height=380, **plotly_theme())
            st.plotly_chart(fig_pie, use_container_width=True)
        with tab_hist_p:
            precios_norm = (data / data.iloc[0]) * 100
            fig_hist = px.line(precios_norm, x=precios_norm.index, y=precios_norm.columns,
                               labels={"value": "Valor ($)", "Date": "Fecha"})
            fig_hist.update_layout(hovermode="x unified", **plotly_theme())
            st.plotly_chart(fig_hist, use_container_width=True)
        with tab_dl_p:
            st.download_button(
                "Precios Históricos (.csv)",
                data=data.to_csv().encode("utf-8"),
                file_name=f"precios_{hoy_g}.csv", mime="text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CREDITMETRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_cm:
    st.markdown("### CreditMetrics — Riesgo de Crédito en Portafolios de Bonos")
    themed_info(
        "**CreditMetrics** (J.P. Morgan, 1997) cuantifica el riesgo de crédito de un portafolio "
        "de bonos corporativos. Soporta **1 a 10 bonos**, caso **independiente** (VaR Paramétrico) "
        "y **correlacionado** (Cópula Gaussiana via Monte Carlo)."
    )
    separador()

    # ──────────────────────────────────────────────────────────────────────────
    # CONSTANTES
    # ──────────────────────────────────────────────────────────────────────────
    CONF_LEVELS = [0.90, 0.95, 0.99, 0.999]
    CONF_LABELS = ["90%", "95%", "99%", "99.9%"]

    # ──────────────────────────────────────────────────────────────────────────
    # SUB-TABS
    # ──────────────────────────────────────────────────────────────────────────
    from credit_engine import NR_METHODS
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        nr_mode = st.selectbox(
            "Tratamiento de NR (Not Rated):",
            list(NR_METHODS.keys()),
            format_func=lambda x: NR_METHODS[x],
            key="cm_nr_mode",
            help="Controla si se normaliza la columna NR o se usa la matriz cruda clásica.",
        )
    with col_opt2:
        _mode_info = {
            "redistribute": "Filas suman 1.0. NR redistribuido proporcionalmente (uso profesional).",
            "simple_normalize": "Filas suman 1.0. NR descartado; cols AAA..D escaladas proporcionalmente.",
            "raw_no_d_nr": "Filas NO suman 1.0 — matriz clásica de libro. D y NR ignorados completamente.",
            "raw_with_d": "Excel clásico — S&P crudas (AAA..D), filas NO suman 1 (NR excluido)",
        }
        themed_info(_mode_info[nr_mode])

    if (st.session_state.get("cm_last_mode") != nr_mode or "cm_tm" not in st.session_state):
        from credit_engine import _TM_RAW_17x19
        st.session_state["cm_tm"] = build_transition_matrix(_TM_RAW_17x19.copy(), nr_mode)
        st.session_state["cm_last_mode"] = nr_mode
        _raw_d = np.zeros((1, 19)); _raw_d[0, 17] = 1.0
        st.session_state["cm_tm_raw"] = np.vstack([_TM_RAW_17x19.copy(), _raw_d])

    _N_STATES = st.session_state["cm_tm"].shape[0]
    _RATINGS_WORK = RATINGS[:_N_STATES]
    separador()

    st1, st2, st3, st4, st5, st6 = st.tabs([
        "Bonos del Portafolio",
        "Matriz de Transición",
        "Curva de Tasas",
        "Caso Independiente",
        "Caso Correlacionado",
        "Exportar a Excel",
    ])

    # ── ST1: PARÁMETROS DE BONOS ──────────────────────────────────────────────
    with st1:
        st.markdown("#### Configurar bonos del portafolio (1 a 10 bonos)")
        themed_info(
            "Ingresa los parámetros de cada bono: nombre, calificación actual (S&P), "
            "valor nominal, cupón anual, vencimiento, pagos por año y tasa de recuperación "
            "esperada en caso de default."
        )
        n_bonds = st.number_input("Número de bonos:", min_value=1, max_value=10,
                                   value=2, step=1, key="cm_n")
        _defs = [
            {"n":"Nvidia",  "r":"AA-", "vn":100., "c":5.0,  "t":5, "p":1, "rec":43.18},
            {"n":"Henkel",  "r":"A",   "vn":100., "c":5.0,  "t":5, "p":1, "rec":43.18},
            {"n":"Bono C",  "r":"BBB", "vn":100., "c":6.0,  "t":3, "p":2, "rec":40.0},
            {"n":"Bono D",  "r":"BB",  "vn":100., "c":7.0,  "t":3, "p":2, "rec":35.0},
            {"n":"Bono E",  "r":"B+",  "vn":100., "c":8.0,  "t":3, "p":2, "rec":30.0},
            {"n":"Bono F",  "r":"BBB+","vn":100., "c":5.5,  "t":4, "p":2, "rec":40.0},
            {"n":"Bono G",  "r":"A+",  "vn":100., "c":4.5,  "t":4, "p":1, "rec":43.0},
            {"n":"Bono H",  "r":"AA",  "vn":100., "c":4.0,  "t":3, "p":1, "rec":43.0},
            {"n":"Bono I",  "r":"BBB-","vn":100., "c":6.5,  "t":5, "p":2, "rec":40.0},
            {"n":"Bono J",  "r":"B",   "vn":100., "c":9.0,  "t":3, "p":4, "rec":35.0},
        ]
        bond_params = []
        for i in range(int(n_bonds)):
            d = _defs[i]
            st.markdown(f"##### Bono {i+1}")
            co1, co2, co3, co4 = st.columns(4)
            with co1:
                nom = st.text_input("Nombre", value=d["n"], key=f"cm_nom_{i}")
                rat = st.selectbox("Calificación", RATINGS[:-1],
                                   index=RATINGS.index(d["r"]) if d["r"] in RATINGS else 0,
                                   key=f"cm_r_{i}")
            with co2:
                vn = st.number_input("Nominal ($)", min_value=1., value=d["vn"],
                                      step=10., key=f"cm_vn_{i}")
                cp = st.number_input("Cupón anual (%)", min_value=0., value=d["c"],
                                      step=0.25, key=f"cm_c_{i}") / 100
            with co3:
                T  = st.number_input("Vencimiento (años)", min_value=1, max_value=10,
                                      value=d["t"], step=1, key=f"cm_T_{i}")
                pg = st.selectbox("Pagos/año", [1,2,4,12],
                                   index=[1,2,4,12].index(d["p"]), key=f"cm_p_{i}",
                                   format_func=lambda x: {1:"Anual",2:"Semestral",
                                                           4:"Trimestral",12:"Mensual"}[x])
            with co4:
                rc = st.number_input("Recuperación D (%)", min_value=0., max_value=100.,
                                      value=d["rec"], step=1., key=f"cm_rc_{i}") / 100
            bond_params.append({
                "nombre": nom, "rating": rat, "rating_idx": RATING_IDX[rat],
                "VN": vn, "cupon_pct": cp, "T": int(T), "pagos": pg, "recov": rc,
            })
            if i < int(n_bonds) - 1: st.markdown("---")
        st.session_state["cm_bparams"] = bond_params
        themed_success(f"**{int(n_bonds)} bono(s)** configurados.")

    # ── ST2: MATRIZ DE TRANSICIÓN ─────────────────────────────────────────────
    with st2:
        st.markdown("#### Matriz de Transición de Calificaciones S&P (editable)")
        themed_info(
            "Probabilidades de migración de calificación en 1 año. "
            "Fuente: S&P Global, 1981–2021. Las **filas** son la calificación actual "
            "y las **columnas** la calificación destino."
        )

        if "cm_tm_raw" not in st.session_state:
            from credit_engine import _TM_RAW_17x19
            _raw = _TM_RAW_17x19.copy()
            _d_row = np.zeros((1, 19)); _d_row[0, 17] = 1.0
            st.session_state["cm_tm_raw"] = np.vstack([_raw, _d_row])
        if "cm_tm" not in st.session_state:
            st.session_state["cm_tm"] = DEFAULT_TM.copy()

        col_tm1, col_tm2 = st.columns([4, 1])
        with col_tm2:
            if st.button("Restaurar S&P", key="cm_rst_tm"):
                from credit_engine import _TM_RAW_17x19
                _raw = _TM_RAW_17x19.copy()
                _d = np.zeros((1,19)); _d[0,17] = 1.0
                st.session_state["cm_tm_raw"] = np.vstack([_raw, _d])
                st.session_state["cm_tm"] = DEFAULT_TM.copy()
                st.rerun()
            row_sums = st.session_state["cm_tm_raw"].sum(axis=1)
            df_sums = pd.DataFrame({
                "Rating": RATINGS,
                "Suma": [f"{s:.4f}" for s in row_sums],
                "": ["✓" if abs(s-1)<0.002 else "⚠" for s in row_sums]
            })
            st.caption("Verificación de filas:")
            st.dataframe(df_sums, hide_index=True, use_container_width=True, height=460)

        with col_tm1:
            from credit_engine import RATINGS_EMIT as _RALL
            _DEST_LABELS = _RALL[:17] + ["D", "NR"]
            df_tm_edit = pd.DataFrame(
                st.session_state["cm_tm_raw"],
                index=RATINGS,
                columns=_DEST_LABELS,
            )
            df_tm_edit.index.name = "From / To"
            ed = st.data_editor(
                df_tm_edit.round(6), use_container_width=True, height=570, num_rows="fixed",
                column_config={c: st.column_config.NumberColumn(
                    c, format="%.4f", step=0.0001, min_value=0., max_value=1.)
                    for c in _DEST_LABELS},
            )
            raw_arr = ed.values.astype(float)
            st.session_state["cm_tm_raw"] = raw_arr
            _active_mode = st.session_state.get("cm_nr_mode", "redistribute")
            st.session_state["cm_tm"] = build_transition_matrix(raw_arr[:17], _active_mode)

        c_th = get_current_theme()
        _tm_plot = st.session_state.get('cm_tm', DEFAULT_TM)
        fig_hm = go.Figure(go.Heatmap(
            z=_tm_plot[:min(_N_STATES, N_R-1), :min(_N_STATES, N_R-1)],
            x=list(_RATINGS_WORK[:min(_N_STATES, N_R-1)]),
            y=list(_RATINGS_WORK[:min(_N_STATES, N_R-1)]),
            colorscale=[[0,"#FFFFFF"],[1, c_th["primary"]]],
            text=np.round(_tm_plot[:min(_N_STATES, N_R-1), :min(_N_STATES, N_R-1)]*100, 3),
            texttemplate="%{text:.2f}%", showscale=True,
        ))
        fig_hm.update_layout(
            title="Mapa de calor — Probabilidades de transición (%)",
            xaxis_title="Rating destino", yaxis_title="Rating origen",
            height=550, **plotly_theme(),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── ST3: CURVA DE TASAS ───────────────────────────────────────────────────
    with st3:
        st.markdown("#### Curva de Tasas del Tesoro y Tasas por Calificación")
        themed_info(
            "Tasas **todo-incluido** (Treasury + spread) usadas para descontar los flujos del bono "
            "según la **nueva calificación** al final del año."
        )

        _bond_ps = st.session_state.get("cm_bparams", [])
        _max_T   = min(max((b["T"] for b in _bond_ps), default=5), 10)
        _prev_maxT = st.session_state.get("cm_max_T_prev", 0)
        _n_spr_rows = min(17, _N_STATES)

        def _make_tsy(max_t):
            v = np.full(max_t, DEFAULT_TREASURY[-1])
            v[:min(max_t, len(DEFAULT_TREASURY))] = DEFAULT_TREASURY[:min(max_t, len(DEFAULT_TREASURY))]
            return v

        def _make_spr(max_t):
            s = np.zeros((DEFAULT_SPREADS.shape[0], max_t))
            c = min(max_t, DEFAULT_SPREADS.shape[1])
            s[:, :c] = DEFAULT_SPREADS[:, :c]
            if max_t > DEFAULT_SPREADS.shape[1]:
                for _ci in range(DEFAULT_SPREADS.shape[1], max_t):
                    s[:, _ci] = DEFAULT_SPREADS[:, -1]
            return s

        if "cm_tsy" not in st.session_state:
            st.session_state["cm_tsy"] = _make_tsy(_max_T)
        elif _prev_maxT != _max_T:
            old_v = st.session_state["cm_tsy"]
            new_v = _make_tsy(_max_T)
            new_v[:min(_max_T, len(old_v))] = old_v[:min(_max_T, len(old_v))]
            st.session_state["cm_tsy"] = new_v

        if "cm_spr" not in st.session_state:
            st.session_state["cm_spr"] = _make_spr(_max_T)
        elif _prev_maxT != _max_T:
            old_s = st.session_state["cm_spr"]
            new_s = _make_spr(_max_T)
            oc = min(old_s.shape[1], _max_T)
            new_s[:, :oc] = old_s[:, :oc]
            st.session_state["cm_spr"] = new_s

        st.session_state["cm_max_T_prev"] = _max_T
        _year_cols = [f"Año {i+1}" for i in range(_max_T)]

        col3a, col3b = st.columns([1, 3])
        with col3a:
            st.markdown(f"**Tesoro — {_max_T} año(s)**")
            df_t = pd.DataFrame({
                "Año": list(range(1, _max_T+1)),
                "Yield (%)": (st.session_state["cm_tsy"]*100).round(4),
            })
            ed_t = st.data_editor(
                df_t, hide_index=True, use_container_width=True,
                column_config={
                    "Año": st.column_config.NumberColumn(disabled=True),
                    "Yield (%)": st.column_config.NumberColumn(format="%.4f", step=0.001),
                }
            )
            st.session_state["cm_tsy"] = ed_t["Yield (%)"].values / 100
            if st.button("Restaurar tasas por defecto", key="cm_rst_spr"):
                st.session_state.pop("cm_tsy", None)
                st.session_state.pop("cm_spr", None)
                st.session_state.pop("cm_max_T_prev", None)
                st.rerun()

        with col3b:
            st.markdown(f"**Tasas todo-incluido por calificación — {_max_T} año(s)**")
            df_s = pd.DataFrame(
                st.session_state["cm_spr"][:_n_spr_rows] * 100,
                index=RATINGS[:_n_spr_rows],
                columns=_year_cols,
            )
            df_s.index.name = "Rating"
            ed_s = st.data_editor(
                df_s.round(4), use_container_width=True,
                height=min(30 + _n_spr_rows * 35, 560),
                column_config={c: st.column_config.NumberColumn(
                    c, format="%.4f", step=0.001) for c in _year_cols}
            )
            nspr = st.session_state["cm_spr"].copy()
            nspr[:_n_spr_rows] = ed_s.values / 100
            st.session_state["cm_spr"] = nspr

        fig_yc = go.Figure()
        fig_yc.add_trace(go.Scatter(
            x=list(range(1,_max_T+1)), y=(st.session_state["cm_tsy"]*100).tolist(),
            name="Tesoro (rf)", mode="lines+markers",
            line=dict(color=c_th["primary"], width=3, dash="dash"),
        ))
        for r_name, r_row in zip(RATINGS[:_n_spr_rows], st.session_state["cm_spr"][:_n_spr_rows]):
            if r_name in ["AAA","AA","A","BBB","BB","B","CCC/C"]:
                fig_yc.add_trace(go.Scatter(
                    x=list(range(1,_max_T+1)), y=(r_row[:_max_T]*100).tolist(),
                    name=r_name, mode="lines", opacity=0.75,
                ))
        fig_yc.update_layout(
            title="Curvas de rendimiento por calificación",
            xaxis_title="Plazo (años)", yaxis_title="Tasa (%)",
            height=380, **plotly_theme(),
        )
        st.plotly_chart(fig_yc, use_container_width=True)

    def _get_bond_data():
        params  = st.session_state.get("cm_bparams", [])
        spr     = st.session_state.get("cm_spr", DEFAULT_SPREADS)
        mode    = st.session_state.get("cm_nr_mode", "redistribute")
        inc_d   = (mode != "raw_no_d_nr")
        out = []
        for bp in params:
            vals = bond_values_per_rating(bp["VN"], bp["cupon_pct"], bp["T"],
                                          bp["pagos"], bp["recov"], spr,
                                          include_d=inc_d)
            out.append({**bp, "values": vals})
        return out

    def _build_metrics_table(scaled: dict, conf_levels, conf_labels, ev: float) -> pd.DataFrame:
        rows = []
        for cf, lb in zip(conf_levels, conf_labels):
            r = scaled[cf]
            rows.append({
                "Confianza":         lb,
                "VaR 1 año ($)":     f"${r['VaR_1y']:,.4f}",
                "CVaR 1 año ($)":    f"${r['CVaR_1y']:,.4f}",
                "VaR 1 día ($)":     f"${r['VaR_1d']:,.4f}",
                "CVaR 1 día ($)":    f"${r['CVaR_1d']:,.4f}",
                "VaR 10 días ($)":   f"${r['VaR_10d']:,.4f}",
                "CVaR 10 días ($)":  f"${r['CVaR_10d']:,.4f}",
                "Capital (3×VaR10d)":f"${r['Capital']:,.4f}",
                "VaR 1y % E[V]":     f"{r['VaR_1y']/ev*100:.2f}%" if ev > 0 else "—",
            })
        return pd.DataFrame(rows)

    # ── ST4: CASO INDEPENDIENTE (VaR Paramétrico) ─────────────────────────
    with st4:
        st.markdown("#### CreditMetrics — Caso Independiente (VaR Paramétrico)")
        themed_info(
            "Bajo el supuesto de **independencia**, calculamos la media (E[V]) y "
            "la desviación estándar (σ) del portafolio sumando las varianzas individuales. "
            "Luego aplicamos el **VaR Paramétrico** asumiendo una distribución Normal.\n\n"
            f"Los VaR/CVaR anuales se escalan a **1 día** y **10 días** usando la raíz cuadrada "
            f"del tiempo (√T), con {TRADING_DAYS} días de trading anuales."
        )

        if st.button("Calcular VaR Paramétrico", use_container_width=True, key="btn_ind"):
            with st.spinner("Calculando métricas analíticas..."):
                bd = _get_bond_data()
                if not bd:
                    themed_error("Configura los bonos primero.")
                else:
                    tm = st.session_state.get("cm_tm", DEFAULT_TM)

                    # 1. Calcular E[V] y σ analíticamente (ultra rápido, sin convolución)
                    per_bond, port_stats = expected_value_and_sigma(bd, tm)
                    ev_port = port_stats["EV_port"]
                    sigma_port = port_stats["sigma_port"]

                    # 2. Calcular VaR Paramétrico
                    vr_p = var_cvar_parametric(ev_port, sigma_port, CONF_LEVELS)
                    scaled_p = scale_var_cvar(vr_p, CONF_LEVELS)

                    st.session_state.update({
                        "cm_ivars_param":  vr_p,
                        "cm_iscaled_param":scaled_p,
                        "cm_ibonds":       bd,
                        "cm_ev":           ev_port,
                        "cm_sigma":        sigma_port,
                        "cm_per_bond":     per_bond,
                    })
                    themed_success("VaR Paramétrico calculado al instante.")

        if "cm_ivars_param" in st.session_state:
            vr_p    = st.session_state["cm_ivars_param"]
            sc_p    = st.session_state["cm_iscaled_param"]
            ev      = st.session_state["cm_ev"]
            sigma   = st.session_state["cm_sigma"]
            c_th    = get_current_theme()

            col_ev1, col_ev2 = st.columns(2)
            col_ev1.metric("E[V] Portafolio", f"${ev:,.4f}")
            col_ev2.metric("σ Portafolio",    f"${sigma:,.4f}")
            separador()

            st.markdown("##### ✅ VaR Paramétrico — 1 año · 1 día · 10 días · Capital")
            df_metrics_p = _build_metrics_table(sc_p, CONF_LEVELS, CONF_LABELS, ev)
            st.dataframe(df_metrics_p, hide_index=True, use_container_width=True)

            r99p = sc_p[0.99]
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("VaR 1 día (99%)",   f"${r99p['VaR_1d']:,.2f}")
            col_p2.metric("CVaR 1 día (99%)",  f"${r99p['CVaR_1d']:,.2f}")
            col_p3.metric("VaR 10 días (99%)", f"${r99p['VaR_10d']:,.2f}")
            col_p4.metric("Capital requerido", f"${r99p['Capital']:,.2f}",
                          help="3 × VaR 10 días — multiplicador Basilea II/III")
            separador()

            # ── Gráfico de la Distribución Asumida (Normal) ───────────
            st.markdown("##### Distribución Asumida (Normal)")
            x_vals = np.linspace(ev - 3.5*sigma, ev + 3.5*sigma, 200)
            y_vals = norm.pdf(x_vals, ev, sigma)

            fig_n = go.Figure()
            fig_n.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines',
                name='Normal(E[V], σ)', line=dict(color=c_th["primary"], width=3)
            ))
            fig_n.add_vline(x=ev, line_color=c_th["success"], line_width=2, annotation_text=f"E[V]={ev:,.2f}")
            
            # Plotear el cuantil de 99%
            q99 = ev - vr_p[0.99]["VaR"]
            fig_n.add_vline(x=q99, line_dash="dot", line_color=c_th["danger"], annotation_text="VaR 99%")

            fig_n.update_layout(
                title="Distribución Paramétrica del Valor del Portafolio",
                xaxis_title="Valor ($)", yaxis_title="Densidad",
                height=420, **plotly_theme()
            )
            st.plotly_chart(fig_n, use_container_width=True)


    # ── ST5: CASO CORRELACIONADO ──────────────────────────────────────────────
    with st5:
        st.markdown("#### CreditMetrics — Caso Correlacionado (Cópula Gaussiana)")
        themed_info(
            "Simulación Monte Carlo con **Cópula Gaussiana**: se simulan variables normales "
            "correlacionadas, se mapean a ratings usando los umbrales N⁻¹(P acumulada) de la "
            "matriz de transición, y se calcula el valor del portafolio.\n\n"
            f"Los VaR/CVaR anuales se escalan a **1 día** y **10 días** usando √T "
            f"({TRADING_DAYS} días/año). **Capital = 3 × VaR 10 días**."
        )

        bd_c = _get_bond_data()
        n_bc = len(bd_c)

        col5a, col5b = st.columns([2, 1])
        with col5a:
            st.markdown("##### Matriz de correlación entre bonos")
            if ("cm_corrm" not in st.session_state or
                    st.session_state["cm_corrm"].shape[0] != n_bc):
                st.session_state["cm_corrm"] = np.eye(n_bc)
            noms_bc = [b["nombre"] for b in bd_c]
            df_cr = pd.DataFrame(st.session_state["cm_corrm"], index=noms_bc, columns=noms_bc)
            df_cr.index.name = "Bono"
            ed_cr = st.data_editor(
                df_cr.round(4), use_container_width=True,
                column_config={c: st.column_config.NumberColumn(
                    c, format="%.4f", step=0.01, min_value=-1., max_value=1.)
                    for c in noms_bc},
            )
            cm_arr = ed_cr.values.astype(float)
            np.fill_diagonal(cm_arr, 1.0)
            st.session_state["cm_corrm"] = cm_arr

        with col5b:
            n_sims5 = st.select_slider(
                "Simulaciones:", options=[10_000,50_000,100_000,200_000],
                value=50_000, key="cm_sims5",
            )
            seed5 = st.number_input("Semilla:", min_value=0, value=42, key="cm_seed5")

            st.markdown("##### Umbrales N⁻¹ por bono")
            tm5 = st.session_state.get("cm_tm", DEFAULT_TM)
            th_rows = []
            for b in bd_c:
                cum_p = np.cumsum(tm5[b["rating_idx"]])
                with np.errstate(all='ignore'):
                    thresh = norm.ppf(np.clip(cum_p, 1e-15, 1-1e-15))
                thresh[-1] = np.inf
                th_rows.append([b["nombre"], b["rating"]] +
                               [f"{t:.3f}" if np.isfinite(t) else "∞" for t in thresh[:8]])
            df_th = pd.DataFrame(th_rows,
                                 columns=["Bono","Rating"]+[f"z({r})" for r in list(_RATINGS_WORK[:8])])
            st.dataframe(df_th, hide_index=True, use_container_width=True, height=280)

        separador()

        if st.button("Ejecutar simulación correlacionada", use_container_width=True, key="btn_corr5"):
            with st.spinner(f"Simulando {n_sims5:,} escenarios..."):
                sims5 = gaussian_copula_simulation(
                    bd_c, st.session_state.get("cm_tm", DEFAULT_TM),
                    st.session_state["cm_corrm"],
                    n_sims=n_sims5, seed=int(seed5),
                )
                vc5    = var_cvar_from_simulations(sims5, CONF_LEVELS)
                sc5    = scale_var_cvar(vc5, CONF_LEVELS)
                st.session_state["cm_csims"]   = sims5
                st.session_state["cm_cvars"]   = vc5
                st.session_state["cm_cscaled"] = sc5
            themed_success(f"Simulación completada: {n_sims5:,} caminos.")

        if "cm_cvars" in st.session_state:
            vc5    = st.session_state["cm_cvars"]
            sc5    = st.session_state.get("cm_cscaled", scale_var_cvar(vc5, CONF_LEVELS))
            s5     = st.session_state["cm_csims"]
            ev5    = vc5[0.95]["EV"]
            c_th   = get_current_theme()

            st.markdown(f"**E[V] simulado = ${ev5:,.4f}**")

            st.markdown("##### Métricas de Riesgo — 1 año · 1 día · 10 días · Capital")
            df_metrics5 = _build_metrics_table(sc5, CONF_LEVELS, CONF_LABELS, ev5)
            st.dataframe(df_metrics5, hide_index=True, use_container_width=True)

            r99c = sc5[0.99]
            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            col_c1.metric("VaR 1 día (99%)",   f"${r99c['VaR_1d']:,.2f}")
            col_c2.metric("CVaR 1 día (99%)",  f"${r99c['CVaR_1d']:,.2f}")
            col_c3.metric("VaR 10 días (99%)", f"${r99c['VaR_10d']:,.2f}")
            col_c4.metric("Capital requerido", f"${r99c['Capital']:,.2f}",
                          help="3 × VaR 10 días — multiplicador Basilea II/III")
            separador()

            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(
                x=s5, nbinsx=120, name="Distribución simulada",
                marker_color=c_th["primary"], opacity=0.7,
                histnorm="probability density",
            ))
            fig5.add_vline(x=ev5, line_color=c_th["success"], line_width=2,
                           annotation_text=f"E[V]={ev5:.2f}")
            for cf, lb in zip([0.95, 0.99, 0.999], ["95%","99%","99.9%"]):
                q = vc5[cf]["q"]
                fig5.add_vline(x=q, line_dash="dot", line_color=c_th["danger"],
                               annotation_text=f"VaR {lb}")
            fig5.update_layout(
                title=f"Distribución simulada del portafolio ({n_sims5:,} escenarios)",
                xaxis_title="Valor del portafolio ($)", yaxis_title="Densidad",
                height=420, **plotly_theme(),
            )
            st.plotly_chart(fig5, use_container_width=True)

    # ── ST6: EXPORTAR A EXCEL ─────────────────────────────────────────────────
    with st6:
        st.markdown("#### Exportar modelo completo a Excel")
        themed_info(
            "Descarga un libro Excel con **7 hojas**: "
            "(1) Parámetros de los bonos, "
            "(2) Matriz de transición, "
            "(3) Curva de tasas y spreads, "
            "(4) Distribuciones individuales por bono, "
            "(5) Distribución Conjunta (Removida por optimización Paramétrica), "
            "(6) Resultados VaR/CVaR 1 año, "
            "(7) Métricas escaladas: 1 día · 10 días · Capital."
        )

        if st.button("Generar Excel", use_container_width=True, key="btn_xls"):
            try:
                import openpyxl
                from openpyxl.styles import Font, PatternFill, Alignment
                from openpyxl.utils import get_column_letter

                wb = openpyxl.Workbook()
                HDR  = Font(bold=True, color="FFFFFF", size=10)
                FIL1 = PatternFill("solid", start_color="203F9A")
                FIL2 = PatternFill("solid", start_color="E84797")
                FIL3 = PatternFill("solid", start_color="F2C8D8")
                CTR  = Alignment(horizontal="center")

                def hdr(ws, r, c, v, fill=FIL1):
                    cell = ws.cell(row=r, column=c, value=v)
                    cell.font = HDR; cell.fill = fill; cell.alignment = CTR

                # Sheet 1: Bond params
                ws1 = wb.active; ws1.title = "1 Parametros"
                for j, h in enumerate(["Nombre","Rating","VN","Cupon%","T","Pagos/año","Recup%"], 1):
                    hdr(ws1, 1, j, h); ws1.column_dimensions[get_column_letter(j)].width = 14
                for i, b in enumerate(st.session_state.get("cm_bparams",[]), 2):
                    for j, v in enumerate([b["nombre"],b["rating"],b["VN"],
                                           b["cupon_pct"]*100,b["T"],b["pagos"],b["recov"]*100], 1):
                        ws1.cell(row=i, column=j, value=v).fill = FIL3

                # Sheet 2: Transition matrix
                ws2 = wb.create_sheet("2 Matriz Transicion")
                hdr(ws2, 1, 1, "From\\To", FIL2)
                tm_x = st.session_state.get("cm_tm", DEFAULT_TM)
                for j, r in enumerate(RATINGS, 2):
                    hdr(ws2, 1, j, r); ws2.column_dimensions[get_column_letter(j)].width = 8
                ws2.column_dimensions["A"].width = 8
                for i, rf in enumerate(RATINGS):
                    ws2.cell(row=i+2, column=1, value=rf).font = Font(bold=True)
                    for j in range(N_R):
                        c = ws2.cell(row=i+2, column=j+2, value=float(tm_x[i,j]))
                        c.number_format = "0.0000%"; c.alignment = CTR
                        if i == j: c.fill = FIL3

                # Sheet 3: Rates
                ws3 = wb.create_sheet("3 Tasas")
                ws3.cell(row=1, column=1, value="Treasury Yield").font = Font(bold=True)
                for j, yr in enumerate([1,2,3,4,5], 2):
                    hdr(ws3, 1, j, f"Año {yr}"); ws3.column_dimensions[get_column_letter(j)].width = 12
                tsy_x = st.session_state.get("cm_tsy", DEFAULT_TREASURY)
                ws3.cell(row=2, column=1, value="rf").font = Font(bold=True)
                for j, v in enumerate(tsy_x, 2):
                    ws3.cell(row=2, column=j, value=float(v)).number_format="0.0000%"
                ws3.cell(row=4, column=1, value="Spreads all-in").font = Font(bold=True)
                for j, yr in enumerate([1,2,3,4,5], 2):
                    hdr(ws3, 5, j, f"Año {yr}")
                spr_x = st.session_state.get("cm_spr", DEFAULT_SPREADS)
                for i, rn in enumerate(RATINGS[:N_R-1]):
                    ws3.cell(row=i+6, column=1, value=rn).font = Font(bold=True)
                    for j, v in enumerate(spr_x[i], 2):
                        ws3.cell(row=i+6, column=j, value=float(v)).number_format="0.0000%"

                # Sheet 4: Individual bond values
                ws4 = wb.create_sheet("4 Valores Bonos")
                bd_x = _get_bond_data()
                tm_x2 = st.session_state.get("cm_tm", DEFAULT_TM)
                col_off = 1
                for b in bd_x:
                    hdr(ws4, 1, col_off, b["nombre"], FIL2)
                    hdr(ws4, 1, col_off+1, "Prob")
                    hdr(ws4, 1, col_off+2, "Valor ($)")
                    for k in range(3):
                        ws4.column_dimensions[get_column_letter(col_off+k)].width = 14
                    pb = tm_x2[b["rating_idx"]]
                    _nv = len(b["values"])
                    _rl = list(_RATINGS_WORK[:_nv])
                    for ri, (rn, p, v) in enumerate(zip(_rl, pb[:_nv], b["values"]), 2):
                        ws4.cell(row=ri, column=col_off, value=rn)
                        ws4.cell(row=ri, column=col_off+1, value=float(p)).number_format="0.0000%"
                        ws4.cell(row=ri, column=col_off+2, value=float(v)).number_format="#,##0.0000"
                    col_off += 4

                # Sheet 5: Distribución Conjunta (Eliminada por optimización)
                ws5 = wb.create_sheet("5 Distribucion Conjunta")
                ws5.cell(row=1, column=1, value="Nota: Se removió la convolución exacta para utilizar únicamente VaR Paramétrico.")
                ws5.column_dimensions["A"].width = 80

                # Sheet 6: VaR / CVaR anual
                ws6 = wb.create_sheet("6 VaR CVaR Anual")
                hdrs6 = ["Conf","Metodo","VaR 1y ($)","CVaR 1y ($)","E[V] ($)","σ ($)"]
                for j, h in enumerate(hdrs6, 1):
                    hdr(ws6, 1, j, h); ws6.column_dimensions[get_column_letter(j)].width = 20
                rw = 2
                for cf, lb in zip(CONF_LEVELS, CONF_LABELS):
                    # Paramétrico
                    if "cm_ivars_param" in st.session_state:
                        r = st.session_state["cm_ivars_param"][cf]
                        for j, v in enumerate([lb,"Paramétrico",r["VaR"],r["CVaR"],r["EV"],r["sigma"]], 1):
                            c = ws6.cell(row=rw, column=j, value=v)
                            c.fill = FIL3
                            if j >= 3: c.number_format = "#,##0.0000"
                        rw += 1
                    # Correlacionado
                    if "cm_cvars" in st.session_state:
                        r = st.session_state["cm_cvars"][cf]
                        for j, v in enumerate([lb,"Correlacionado",r["VaR"],r["CVaR"],r["EV"],r.get("sigma",0)], 1):
                            c = ws6.cell(row=rw, column=j, value=v)
                            if j >= 3: c.number_format = "#,##0.0000"
                        rw += 1

                # Sheet 7: Métricas escaladas (1d · 10d · Capital)
                ws7 = wb.create_sheet("7 Metricas Escaladas")
                ws7.cell(row=1, column=1,
                         value=f"Escala: VaR_1d = VaR_1y ÷ √{TRADING_DAYS}  |  "
                               f"VaR_10d = VaR_1d × √10  |  Capital = 3 × VaR_10d"
                         ).font = Font(italic=True, color="555555")
                hdrs7 = ["Conf","Método",
                         "VaR 1d ($)","CVaR 1d ($)",
                         "VaR 10d ($)","CVaR 10d ($)",
                         "Capital ($)","E[V] ($)"]
                for j, h in enumerate(hdrs7, 1):
                    hdr(ws7, 2, j, h)
                    ws7.column_dimensions[get_column_letter(j)].width = 18
                rw7 = 3
                for metodo, sc_data, fill in [
                    ("Paramétrico",  st.session_state.get("cm_iscaled_param"), FIL3),
                    ("Correlacionado", st.session_state.get("cm_cscaled"), None),
                ]:
                    if sc_data is None:
                        continue
                    for cf, lb in zip(CONF_LEVELS, CONF_LABELS):
                        r = sc_data[cf]
                        vals_row = [
                            lb, metodo,
                            r["VaR_1d"],  r["CVaR_1d"],
                            r["VaR_10d"], r["CVaR_10d"],
                            r["Capital"], r["EV"],
                        ]
                        for j, v in enumerate(vals_row, 1):
                            cell = ws7.cell(row=rw7, column=j, value=v)
                            if fill: cell.fill = fill
                            if j >= 3: cell.number_format = "#,##0.0000"
                        rw7 += 1

                buf = io.BytesIO(); wb.save(buf); buf.seek(0)
                st.session_state["cm_excel"] = buf.getvalue()
                themed_success("Libro Excel generado con éxito.")
            except Exception as e:
                themed_error(f"Error generando Excel: {e}")
                import traceback; st.code(traceback.format_exc())

        if "cm_excel" in st.session_state:
            st.download_button(
                "Descargar CreditMetrics (.xlsx)",
                data=st.session_state["cm_excel"],
                file_name="CreditMetrics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )