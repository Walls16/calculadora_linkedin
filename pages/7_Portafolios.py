"""
pages/7_Portafolios.py
----------------------
Módulo 7: Teoría de Portafolios — Comparativa de 5 Estrategias.
Métodos:
  1. Markowitz — Máximo Sharpe Ratio
  2. Markowitz — Mínima Varianza Global
  3. 1/N       — Equiponderación
  4. Paridad de Riesgo (Risk Parity)
  5. MVSK      — Media-Varianza-Asimetría-Curtosis
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    get_engine, page_header, separador,
    themed_info, themed_success, themed_warning, themed_error,
    apply_plotly_theme, get_current_theme,
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Portafolios Eficientes · Calculadora Financiera",
    page_icon="📦",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="7. Teoría de Portafolios — Comparativa de Estrategias",
    subtitulo="5 métodos de ponderación · pypfopt · SciPy · Datos reales de Yahoo Finance"
)

# ─── Paleta fija por estrategia ───────────────────────────────────────────────
ESTRATEGIA_COLORES = {
    "Máx. Sharpe":       "#FF4B4B",   # Rojo
    "Mín. Varianza":     "#00E5FF",   # Cian
    "1/N Equiponderado": "#FFD700",   # Amarillo
    "Paridad de Riesgo": "#7CFC00",   # Verde lima
    "MVSK":              "#FF8C00",   # Naranja
}
ESTRATEGIA_SIMBOLOS = {
    "Máx. Sharpe":       "star",
    "Mín. Varianza":     "star",
    "1/N Equiponderado": "diamond",
    "Paridad de Riesgo": "circle",
    "MVSK":              "pentagon",
}

# =============================================================================
# PANEL DE CONFIGURACIÓN
# =============================================================================
with st.expander("Configuración del Portafolio y Mercado", expanded=True):
    themed_info(
        "Define los activos, el periodo histórico y la **Tasa Libre de Riesgo**. "
        "El sistema calculará simultáneamente las **5 estrategias de ponderación** y "
        "te mostrará por qué cada una es mejor en distintas dimensiones."
    )

    c_in1, c_in2, c_in3 = st.columns([2, 1, 1])

    with c_in1:
        tickers_input = st.text_input(
            "Símbolos (separados por coma):",
            value="AAPL, MSFT, GOOGL, NVDA, TSLA",
            help="Ejemplos: AAPL (Apple), CEMEXCPO.MX (Cemex), SPY (S&P 500 ETF)",
        )

    with c_in2:
        hoy          = datetime.date.today()
        hace_3_anios = hoy - datetime.timedelta(days=365 * 3)
        fecha_inicio = st.date_input("Fecha de Inicio", value=hace_3_anios)
        fecha_fin    = st.date_input("Fecha de Fin",    value=hoy)

    with c_in3:
        tasa_libre = st.number_input(
            "Tasa Libre de Riesgo ($r_f$) %",
            value=5.0, step=0.1, key="pf_rf"
        ) / 100
        st.write("")
        ejecutar = st.button("Optimizar Portafolio", use_container_width=True)

# =============================================================================
# FASE 1 — CÁLCULO
# =============================================================================
if ejecutar:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 2:
        themed_error("Necesitas al menos 2 activos para generar una frontera eficiente.")
    else:
        with st.spinner(f"Descargando datos y resolviendo 5 optimizaciones para {len(tickers)} activos..."):
            try:
                resultado = engine.optimizacion_portafolios(
                    tickers, fecha_inicio, fecha_fin, tasa_libre
                )
                st.session_state["datos_portafolio"]  = resultado
                st.session_state["tickers_guardados"] = tickers_input
                st.session_state["fecha_hoy_pf"]      = hoy
                themed_success("¡Optimización de las 5 estrategias completada exitosamente!")
            except Exception as e:
                themed_error(f"Ocurrió un error al procesar los datos: {e}")
                themed_info(
                    "Verifica que los símbolos sean correctos en Yahoo Finance, "
                    "que haya conexión a internet y que el rango de fechas sea válido."
                )

# =============================================================================
# FASE 2 — VISUALIZACIÓN
# =============================================================================
if "datos_portafolio" in st.session_state:

    if st.session_state.get("tickers_guardados") != tickers_input:
        themed_warning(
            "Detectamos cambios en los símbolos. "
            "Presiona **Optimizar Portafolio** para recalcular."
        )

    data, mu, S, resultados, nube = st.session_state["datos_portafolio"]
    hoy_guardado = st.session_state.get("fecha_hoy_pf", datetime.date.today())
    ret_sim, vol_sim, sharpe_sim = nube

    # ── Tabla resumen de métricas ─────────────────────────────────────────────
    separador()
    st.markdown("### Resumen Comparativo de Estrategias")
    themed_info(
        "Las **5 estrategias** tienen objetivos distintos: maximizar eficiencia, minimizar riesgo, "
        "diversificar de forma ingenua, equilibrar el riesgo real, o capturar asimetrías de cola. "
        "No existe una sola ganadora; la mejor depende del perfil de cada inversor."
    )

    filas_resumen = []
    for nombre, (ret, vol, sharpe, pesos) in resultados.items():
        filas_resumen.append({
            "Estrategia":              nombre,
            "Rendimiento Anual (E[R])": f"{ret*100:.2f}%",
            "Volatilidad Anual (σ)":    f"{vol*100:.2f}%",
            "Ratio de Sharpe":          f"{sharpe:.4f}",
        })
    df_resumen = pd.DataFrame(filas_resumen).set_index("Estrategia")
    st.dataframe(df_resumen, use_container_width=True)

    separador()

    # ── Pestañas ──────────────────────────────────────────────────────────────
    tab_front, tab_pesos, tab_hist, tab_var, tab_teoria, tab_dl = st.tabs([
        "Frontera Eficiente",
        "Composición (wᵢ)",
        "Desempeño Histórico",
        "Análisis de Riesgo (VaR)",
        "Marco Teórico",
        "Exportar Datos",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — FRONTERA EFICIENTE
    # ─────────────────────────────────────────────────────────────────────────
    with tab_front:
        st.markdown("#### Gráfica Riesgo vs. Rendimiento — Las 5 Estrategias")
        themed_info(
            "Cada **estrella o símbolo** representa una estrategia concreta. "
            "La nube de puntos muestra 2,500 portafolios aleatorios coloreados por su Ratio de Sharpe. "
            "Los puntos hacia la **izquierda y arriba** son los más eficientes."
        )

        fig_ef = go.Figure()

        # Nube Monte Carlo
        fig_ef.add_trace(go.Scatter(
            x=vol_sim, y=ret_sim,
            mode="markers",
            marker=dict(
                size=4,
                color=sharpe_sim,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
            ),
            text=[
                f"Rendimiento: {r*100:.2f}%<br>Riesgo: {v*100:.2f}%<br>Sharpe: {s:.2f}"
                for r, v, s in zip(ret_sim, vol_sim, sharpe_sim)
            ],
            hoverinfo="text",
            name="Portafolios Posibles (Monte Carlo)",
        ))

        # Puntos de estrategias
        for nombre, (ret, vol, sharpe, _) in resultados.items():
            fig_ef.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode="markers+text",
                marker=dict(
                    symbol=ESTRATEGIA_SIMBOLOS[nombre],
                    size=20,
                    color=ESTRATEGIA_COLORES[nombre],
                    line=dict(width=1.5, color="black"),
                ),
                text=[nombre],
                textposition="top center",
                textfont=dict(size=11, color=ESTRATEGIA_COLORES[nombre]),
                name=f"{nombre} (σ={vol*100:.1f}%, E[R]={ret*100:.1f}%)",
                hovertemplate=(
                    f"<b>{nombre}</b><br>"
                    f"Rendimiento: {ret*100:.2f}%<br>"
                    f"Volatilidad: {vol*100:.2f}%<br>"
                    f"Sharpe: {sharpe:.4f}<extra></extra>"
                ),
            ))

        fig_ef.update_layout(
            xaxis_title="Riesgo / Volatilidad Anualizada (σ)",
            yaxis_title="Rendimiento Esperado Anualizado E[R]",
            template="none",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=580,
        )
        fig_ef = apply_plotly_theme(fig_ef)
        st.plotly_chart(fig_ef, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — COMPOSICIÓN DE PESOS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_pesos:
        st.markdown("#### Distribución de Capital por Estrategia")
        themed_success(
            "Cada barra muestra cómo se distribuye el **100% del capital** entre los activos "
            "según cada estrategia. Nota cómo la equiponderación (1/N) asigna exactamente el "
            "mismo peso a todos, mientras que Markowitz puede concentrarse en pocos activos."
        )

        # Construir DataFrame de pesos
        df_pesos_dict = {}
        for nombre, (_, _, _, pesos) in resultados.items():
            df_pesos_dict[nombre] = pesos
        df_pesos_tab = pd.DataFrame(df_pesos_dict).fillna(0)

        df_melted = (
            df_pesos_tab.reset_index()
            .melt(id_vars="index", var_name="Estrategia", value_name="Peso")
            .rename(columns={"index": "Activo"})
        )
        df_melted = df_melted[df_melted["Peso"] > 0.001]

        fig_pesos = px.bar(
            df_melted,
            y="Estrategia", x="Peso", color="Activo",
            orientation="h",
            text_auto=".1%",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Distribución del Portafolio por Estrategia",
        )
        fig_pesos.update_layout(
            xaxis_title="Porcentaje de Inversión",
            yaxis_title="",
            xaxis_tickformat=".0%",
            template="none",
            height=420,
            legend_title="Símbolos",
            barmode="stack",
        )
        fig_pesos.update_traces(
            textfont_size=12, textangle=0,
            textposition="inside", cliponaxis=False,
        )
        fig_pesos = apply_plotly_theme(fig_pesos)
        st.plotly_chart(fig_pesos, use_container_width=True)

        separador()
        st.markdown("##### Pesos numéricos exactos")
        st.dataframe(
            df_pesos_tab.style.format("{:.4%}"),
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — DESEMPEÑO HISTÓRICO
    # ─────────────────────────────────────────────────────────────────────────
    with tab_hist:
        st.markdown("#### Desempeño Histórico Comparativo (Base 100)")
        themed_info(
            "Se simulan las **5 estrategias aplicadas al mismo periodo histórico**, "
            "comparadas contra las acciones individuales. Cada portafolio arranca en $100. "
            "Esto muestra *por qué* una estrategia domina en rendimiento total o en estabilidad."
        )

        # Retornos diarios
        retornos = data.pct_change().dropna()

        # Calcular valor de cada portafolio en el tiempo
        fig_hist = go.Figure()

        # Primero: líneas individuales (punteadas, más tenues)
        precios_norm = (data / data.iloc[0]) * 100
        for col in precios_norm.columns:
            fig_hist.add_trace(go.Scatter(
                x=precios_norm.index,
                y=precios_norm[col],
                mode="lines",
                name=col,
                line=dict(width=1, dash="dot"),
                opacity=0.45,
            ))

        # Luego: líneas de portafolios (sólidas, gruesas)
        for nombre, (_, _, _, pesos) in resultados.items():
            w_vec = np.array([pesos.get(t, 0.0) for t in data.columns])
            ret_port = retornos.values @ w_vec          # retorno diario del portafolio
            valor_port = 100 * np.cumprod(1 + ret_port)
            valor_series = pd.Series(valor_port, index=retornos.index)

            fig_hist.add_trace(go.Scatter(
                x=valor_series.index,
                y=valor_series.values,
                mode="lines",
                name=nombre,
                line=dict(
                    color=ESTRATEGIA_COLORES[nombre],
                    width=2.8,
                ),
                hovertemplate=f"<b>{nombre}</b><br>Fecha: %{{x|%Y-%m-%d}}<br>Valor: $%{{y:.2f}}<extra></extra>",
            ))

        fig_hist.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Valor de la Inversión (Base 100 = $100)",
            template="none",
            hovermode="x unified",
            legend=dict(groupclick="toggleitem"),
            height=560,
        )
        fig_hist = apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Tabla de retorno total
        separador()
        st.markdown("##### Retorno Total del Periodo")
        filas_rt = []
        for nombre, (_, _, _, pesos) in resultados.items():
            w_vec = np.array([pesos.get(t, 0.0) for t in data.columns])
            ret_port = retornos.values @ w_vec
            valor_final = 100 * np.prod(1 + ret_port)
            retorno_total = (valor_final / 100 - 1)
            filas_rt.append({"Estrategia": nombre, "Valor Final ($)": f"${valor_final:.2f}", "Retorno Total": f"{retorno_total*100:.2f}%"})
        st.dataframe(pd.DataFrame(filas_rt).set_index("Estrategia"), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — VaR
    # ─────────────────────────────────────────────────────────────────────────
    with tab_var:
        st.markdown("#### Análisis de Riesgo Extremo: VaR y CVaR")
        themed_warning(
            "El **Valor en Riesgo (VaR)** proyecta la pérdida máxima esperada en escenarios negativos. "
            "Aquí se comparan las 5 estrategias para el mismo capital invertido."
        )

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            val_port = st.number_input(
                "Capital Total Invertido ($)",
                min_value=100.0, value=100_000.0,
                step=10_000.0, key="pf_var_capital"
            )
        with col_v2:
            conf_str  = st.selectbox("Nivel de Confianza", ["95%", "99%"], key="pf_conf")
            confianza = 0.95 if conf_str == "95%" else 0.99

        separador()

        horizontes  = [1, 10, 21]
        nombres_hor = ["1 Día", "10 Días (Basilea)", "21 Días (1 Mes)"]

        estrat_sel = st.selectbox(
            "Ver análisis detallado de estrategia:",
            list(resultados.keys()),
            key="var_estrat_sel",
        )

        ret_sel, vol_sel, _, _ = resultados[estrat_sel]

        def _tabla_var(rend, vol, capital, conf):
            filas = []
            for h, nom in zip(horizontes, nombres_hor):
                var_p, _, _, _  = engine.calcular_var_parametrico(rend, vol, capital, conf, h)
                var_mc, cvar_mc = engine.calcular_var_cvar_montecarlo(rend, vol, capital, conf, h)
                filas.append({
                    "Horizonte":        nom,
                    "VaR Paramétrico":  f"${var_p:,.2f}",
                    "VaR Monte Carlo":  f"${var_mc:,.2f}",
                    "CVaR (ES)":        f"${cvar_mc:,.2f}",
                })
            return pd.DataFrame(filas).set_index("Horizonte")

        themed_success(
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-size:18px;font-weight:bold;'>{estrat_sel}</span>"
            f"<span style='font-size:16px;'>Rendimiento Anual: {ret_sel*100:.2f}% · Volatilidad: {vol_sel*100:.2f}%</span>"
            f"</div>"
        )
        st.dataframe(_tabla_var(ret_sel, vol_sel, val_port, confianza), use_container_width=True)

        separador()
        with st.expander("Conceptos Clave de Administración de Riesgos"):
            themed_info(
                "**VaR Paramétrico:** Asume distribución normal. "
                r"Fórmula: $VaR = V_0(Z_\alpha \sigma\sqrt{t} - \mu t)$"
            )
            themed_success(
                "**VaR Monte Carlo:** Genera miles de escenarios aleatorios y extrae "
                "la pérdida en el percentil deseado. Captura mejor las colas gordas."
            )
            themed_warning(
                "**CVaR (Expected Shortfall):** Promedio de pérdidas *una vez que el VaR fue superado*. "
                "Responde: *'Si las cosas salen muy mal, ¿qué tan mal serán?'*"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — MARCO TEÓRICO
    # ─────────────────────────────────────────────────────────────────────────
    with tab_teoria:
        st.markdown("#### Marco Teórico de las 5 Estrategias")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            themed_success("**1. Máximo Sharpe Ratio (Markowitz)**")
            st.write("Maximiza el rendimiento excedente por unidad de riesgo. Requiere estimación de retornos esperados.")
            st.latex(r"\max_{\mathbf{w}} \; S = \frac{E[R_p] - r_f}{\sigma_p} \quad \text{s.a.} \; \sum w_i = 1,\; w_i \ge 0")

            themed_info("**2. Mínima Varianza Global (Markowitz)**")
            st.write("Solo optimiza la covarianza; no requiere pronósticos de retornos (más robusto al error de estimación).")
            st.latex(r"\min_{\mathbf{w}} \; \sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w} \quad \text{s.a.} \; \sum w_i = 1,\; w_i \ge 0")

            themed_warning("**3. 1/N Equiponderación**")
            st.write("Asigna el mismo peso a cada activo. Extremadamente robusto; estudios empíricos (DeMiguel et al. 2009) muestran que supera a Markowitz fuera de muestra en muchos casos.")
            st.latex(r"w_i = \frac{1}{N} \quad \forall \; i")

        with col_t2:
            themed_success("**4. Paridad de Riesgo (Risk Parity)**")
            st.write("Cada activo contribuye *igual* al riesgo total del portafolio, sin importar el tamaño de la posición. Popularizado por el fondo Bridgewater All Weather.")
            st.latex(r"RC_i = w_i \cdot \frac{(\Sigma \mathbf{w})_i}{\sigma_p} = \frac{\sigma_p}{N} \quad \forall \; i")

            themed_info("**5. MVSK — Momentos de Orden Superior**")
            st.write("Extiende Markowitz para incorporar asimetría (skewness) y curtosis (kurtosis), capturando eventos de cola y distribuciones no normales. Maximiza una función de utilidad:")
            st.latex(r"U = E[R_p] - \lambda_2 \sigma_p^2 + \lambda_3 \text{Skew}_p - \lambda_4 \text{Kurt}_p")
            st.caption("Pesos estándar usados: λ₂=1, λ₃=0.5, λ₄=0.5 (Harvey et al. 2010)")

        separador()
        themed_info(
            "**¿Cuándo usar cada estrategia?** "
            "Sharpe es ideal con pronósticos confiables de retornos. "
            "Mínima Varianza para inversores conservadores. "
            "1/N cuando hay incertidumbre sobre el modelo. "
            "Paridad de Riesgo para carteras multi-activo diversificadas por riesgo. "
            "MVSK cuando la distribución de retornos es asimétrica o con colas gruesas."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 — EXPORTAR
    # ─────────────────────────────────────────────────────────────────────────
    with tab_dl:
        st.markdown("#### Descarga de Datos para Réplica")
        themed_info(
            "Exporta los precios históricos y los vectores de pesos de las 5 estrategias."
        )

        col_d1, col_d2 = st.columns(2)

        csv_precios = data.to_csv().encode("utf-8")

        # Construir DF de pesos con las 5 estrategias
        df_pesos_export = pd.DataFrame(
            {nombre: {t: pesos.get(t, 0.0) for t in data.columns}
             for nombre, (_, _, _, pesos) in resultados.items()}
        )
        df_pesos_export.index.name = "Ticker"
        csv_pesos = df_pesos_export.to_csv().encode("utf-8")

        with col_d1:
            st.download_button(
                label="⬇️ Descargar Precios Históricos (.csv)",
                data=csv_precios,
                file_name=f"precios_historicos_{hoy_guardado}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Precios de cierre ajustados limpios de NAs.")

        with col_d2:
            st.download_button(
                label="⬇️ Descargar Pesos Óptimos 5 Estrategias (.csv)",
                data=csv_pesos,
                file_name=f"pesos_5estrategias_{hoy_guardado}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Vectores $w_i$ de las 5 estrategias de optimización.")