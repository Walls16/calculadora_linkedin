"""
pages/7_Portafolios.py
----------------------
Módulo 7: Teoría de Portafolios (Frontera Eficiente).
Optimización matemática exacta con pypfopt y datos reales de Yahoo Finance.
Cubre:
  - Portafolio de Máximo Sharpe Ratio (EfficientFrontier.max_sharpe)
  - Portafolio de Mínima Varianza Global (EfficientFrontier.min_volatility)
  - Frontera eficiente (nube Monte Carlo de 2500 pesos aleatorios)
  - Pestaña de Pesos Óptimos comparados (Sharpe vs Mínima Varianza)
  - Pestaña de Precios Históricos normalizados
  - Pestaña de VaR/CVaR paramétrico y Monte Carlo
  - Pestaña de Exportar CSV (precios históricos y pesos óptimos)
"""

import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import get_engine, page_header, separador, themed_info, themed_success, themed_warning, themed_error, apply_plotly_theme

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
    titulo="7. Teoría de Portafolios (Frontera Eficiente)",
    subtitulo="Optimización matemática exacta · pypfopt · Datos reales de Yahoo Finance"
)

# =============================================================================
# PANEL DE CONFIGURACIÓN
# =============================================================================
with st.expander("Configuración del Portafolio y Mercado", expanded=True):
    themed_info(
        "Define los activos que formarán tu portafolio, el periodo histórico de análisis y la **Tasa Libre de Riesgo** "
        "(el rendimiento seguro que pagaría un bono del gobierno). El sistema descargará los precios reales de la bolsa "
        "y calculará la combinación de pesos matemáticamente perfecta para tu inversión."
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
        st.write("") # Espaciador visual
        ejecutar = st.button("Optimizar Portafolio", use_container_width=True)

# =============================================================================
# FASE 1 — CÁLCULO (solo al presionar el botón)
# =============================================================================
if ejecutar:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 2:
        themed_error("Necesitas al menos 2 activos para generar una frontera de Markowitz.")
    else:
        with st.spinner(f"Descargando datos históricos de {len(tickers)} activos y resolviendo matrices de covarianza..."):
            try:
                resultados = engine.optimizacion_markowitz(
                    tickers, fecha_inicio, fecha_fin, tasa_libre
                )
                st.session_state["datos_portafolio"]  = resultados
                st.session_state["tickers_guardados"] = tickers_input
                st.session_state["fecha_hoy_pf"]      = hoy
                themed_success("¡Optimización matemática completada exitosamente!")
            except Exception as e:
                themed_error(f"Ocurrió un error al procesar los datos: {e}")
                themed_info(
                    "Verifica que los símbolos sean correctos en Yahoo Finance, "
                    "que haya conexión a internet y que el rango de fechas sea válido."
                )

# =============================================================================
# FASE 2 — VISUALIZACIÓN (siempre que haya datos en session_state)
# =============================================================================
if "datos_portafolio" in st.session_state:

    if st.session_state.get("tickers_guardados") != tickers_input:
        themed_warning(
            "Detectamos cambios en los símbolos. "
            "Presiona **Optimizar Portafolio** para recalcular con los nuevos activos."
        )

    data, mu, S, res_sharpe, res_min, nube = st.session_state["datos_portafolio"]
    hoy_guardado = st.session_state.get("fecha_hoy_pf", datetime.date.today())

    rend_s, vol_s, sharpe_s, pesos_sharpe = res_sharpe
    rend_m, vol_m, sharpe_m, pesos_min    = res_min
    ret_sim, vol_sim, sharpe_sim          = nube

    # ── Métricas del portafolio óptimo ────────────────────────────────────────
    separador()
    st.markdown("### Portafolio Óptimo — Máximo Ratio de Sharpe")
    themed_success(
        "El **Portafolio de Máximo Sharpe** es la combinación matemática exacta de activos que te ofrece "
        "el mayor rendimiento esperado posible por cada punto de riesgo (volatilidad) que estás asumiendo."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rendimiento Esperado Anual ($E[R]$)", f"{rend_s*100:.2f}%")
    c2.metric("Volatilidad Anual ($\sigma$)",       f"{vol_s*100:.2f}%")
    c3.metric("Ratio de Sharpe ($S$)",             f"{sharpe_s:.4f}")

    separador()

    # ── Pestañas ──────────────────────────────────────────────────────────────
    tab_front, tab_pesos, tab_hist, tab_var, tab_dl = st.tabs([
        "Frontera Eficiente",
        "Composición Óptima (wᵢ)",
        "Desempeño Histórico",
        "Análisis de Riesgo (VaR)",
        "Exportar Datos",
    ])

    # ── TAB 1: FRONTERA ───────────────────────────────────────────────────────
    with tab_front:
        st.markdown("#### Gráfica Riesgo vs. Rendimiento")
        themed_info(
            "La **Frontera Eficiente** es la curva (o límite superior de la nube) que agrupa los mejores portafolios posibles. "
            "Cada punto en la gráfica es una distribución distinta de tu dinero. Los puntos más altos "
            "y situados hacia la izquierda representan las combinaciones más eficientes."
        )

        fig_ef = go.Figure()

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

        fig_ef.add_trace(go.Scatter(
            x=[vol_s], y=[rend_s],
            mode="markers",
            marker=dict(symbol="star", size=18, color="#FF4B4B",
                        line=dict(width=1.5, color="black")),
            name="Máximo Sharpe (Óptimo)",
        ))

        fig_ef.add_trace(go.Scatter(
            x=[vol_m], y=[rend_m],
            mode="markers",
            marker=dict(symbol="star", size=18, color="#00E5FF",
                        line=dict(width=1.5, color="black")),
            name="Mínima Varianza Global",
        ))

        fig_ef.update_layout(
            xaxis_title="Riesgo / Volatilidad Anualizada (σ)",
            yaxis_title="Rendimiento Esperado Anualizado E[R]",
            template="none",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=550,
        )
        fig_ef = apply_plotly_theme(fig_ef)
        st.plotly_chart(fig_ef, use_container_width=True)

        separador()
        
        # --- EXPLICACIÓN MATEMÁTICA ---
        with st.expander("¿Cómo funciona la Teoría Moderna de Portafolios (Markowitz)?"):
            st.markdown("A diferencia de una anualidad clásica, la optimización de portafolios no es un despeje simple, sino un **problema de programación cuadrática**. La computadora prueba miles de combinaciones para encontrar los pesos exactos ($w_i$) que cumplen con estos principios:")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                themed_info("**1. Rendimiento Esperado ($E[R_p]$)**")
                st.write("Es el promedio ponderado de los rendimientos históricos de cada activo ($\mu_i$). Matemáticamente, el producto punto del vector de pesos y el vector de retornos.")
                st.latex(r"E[R_p] = \sum_{i=1}^n w_i \mu_i = \mathbf{w}^T \boldsymbol{\mu}")
                
                themed_info("**3. Ratio de Sharpe ($S$)**")
                st.write("Mide la rentabilidad excedente por cada unidad de riesgo. El portafolio óptimo (la estrella roja en la gráfica) es el que maximiza esta ecuación.")
                st.latex(r"S = \frac{E[R_p] - r_f}{\sigma_p}")

            with col_m2:
                themed_info("**2. Riesgo del Portafolio (Varianza $\sigma_p^2$)**")
                st.write("Aquí radica la magia de la **diversificación**. El riesgo total depende de la Matriz de Covarianza ($\Sigma$), es decir, de qué tan correlacionados están los activos entre sí.")
                st.latex(r"\sigma_p^2 = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij} = \mathbf{w}^T \Sigma \mathbf{w}")
                
                themed_info("**4. El Algoritmo de Optimización**")
                st.write("Para encontrar la frontera eficiente, el sistema resuelve iterativamente el vector $\mathbf{w}$ para maximizar $S$, sujeto a las restricciones del mundo real:")
                st.latex(r"\sum_{i=1}^n w_i = 1 \quad \text{y} \quad w_i \ge 0")

    # ── TAB 2: PESOS ──────────────────────────────────────────────────────────
    with tab_pesos:
        st.markdown("#### Comparativa de Asignación de Capital (<span style='font-family: serif; font-style: italic;'>w<sub>i</sub></span>)", unsafe_allow_html=True)
        themed_success(
            "La **Asignación de Capital** muestra el porcentaje exacto de tu dinero que debes "
            "destinar a cada acción para replicar la estrategia seleccionada (ya sea buscar la máxima eficiencia con Sharpe "
            "o buscar la máxima estabilidad con Mínima Varianza)."
        )

        df_pesos_tab = pd.DataFrame({
            "Máximo Sharpe (Rendimiento)": pesos_sharpe,
            "Mínima Varianza (Seguridad)": pesos_min,
        }).fillna(0)

        df_melted = (
            df_pesos_tab.reset_index()
            .melt(id_vars="index", var_name="Estrategia", value_name="Peso")
            .rename(columns={"index": "Activo"})
        )
        df_melted = df_melted[df_melted["Peso"] > 0.001] # Filtrar pesos muy cercanos a 0

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
            height=400,
            legend_title="Símbolos",
            barmode="stack",
        )
        fig_pesos.update_traces(
            textfont_size=13, textangle=0,
            textposition="inside", cliponaxis=False,
        )
        fig_pesos = apply_plotly_theme(fig_pesos)
        st.plotly_chart(fig_pesos, use_container_width=True)

        separador()
        st.markdown("##### Pesos numéricos exactos")
        st.dataframe(
            df_pesos_tab.style.format("{:.4%}").set_properties(**{
                "background-color": "#F8FAFC",
                "color": "#0F172A",
                "text-align": "center"
            }),
            use_container_width=True,
        )

    # ── TAB 3: HISTÓRICO ──────────────────────────────────────────────────────
    with tab_hist:
        st.markdown("#### Precios Históricos Normalizados (Base 100)")
        themed_info(
            "La **Normalización Base 100** iguala matemáticamente el precio de todas tus acciones a $100 al inicio del "
            "periodo analizado. Esto te permite comparar visualmente el crecimiento y la volatilidad real entre ellas, "
            "sin que importe si una acción vale en la bolsa $10 y otra $500."
        )

        precios_norm = (data / data.iloc[0]) * 100
        fig_hist = px.line(
            precios_norm,
            x=precios_norm.index,
            y=precios_norm.columns,
            labels={"value": "Valor de Inversión ($)", "Date": "Fecha"},
            template="none",
        )
        fig_hist.update_layout(hovermode="x unified", legend_title="Activos")
        fig_hist = apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── TAB 4: VaR ────────────────────────────────────────────────────────────
    with tab_var:
        st.markdown("#### Análisis de Riesgo Extremo: VaR y CVaR")
        themed_warning(
            "El **Valor en Riesgo (VaR)** proyecta la pérdida de capital máxima esperada que tu portafolio podría sufrir "
            "en escenarios negativos de mercado, basándose en el nivel de confianza y la volatilidad histórica de los activos."
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
        nombres_hor = [
            "1 Día (Mesa de Trading)",
            "10 Días (Basilea / Regulatorio)",
            "21 Días (1 Mes Financiero)",
        ]

        def _tabla_var(rend, vol, capital, conf):
            filas = []
            for h, nom in zip(horizontes, nombres_hor):
                var_p, _, _, _  = engine.calcular_var_parametrico(rend, vol, capital, conf, h)
                var_mc, cvar_mc = engine.calcular_var_cvar_montecarlo(rend, vol, capital, conf, h)
                filas.append({
                    "Horizonte":       nom,
                    "VaR Paramétrico": f"${var_p:,.2f}",
                    "VaR Monte Carlo": f"${var_mc:,.2f}",
                    "CVaR (ES)":       f"${cvar_mc:,.2f}",
                })
            return pd.DataFrame(filas).set_index("Horizonte")

        themed_success(f"<h4 style='margin:0; color:inherit;'>Portafolio Máximo Sharpe (Rendimiento Anual: {rend_s*100:.2f}%)</h4>")
        st.dataframe(_tabla_var(rend_s, vol_s, val_port, confianza), use_container_width=True)

        st.write("") # Espaciador

        themed_info(f"<h4 style='margin:0; color:inherit;'>Portafolio Mínima Varianza Global (Rendimiento Anual: {rend_m*100:.2f}%)</h4>")
        st.dataframe(_tabla_var(rend_m, vol_m, val_port, confianza), use_container_width=True)

        separador()
        
        with st.expander("Conceptos Clave de Administración de Riesgos"):
            themed_info(
                "**Value at Risk (VaR) Paramétrico:** Asume que los retornos siguen una campana de Gauss (distribución normal). "
                r"Fórmula clásica: $VaR = V_0(Z_\alpha \sigma\sqrt{t} - \mu t)$"
            )
            themed_success(
                "**VaR por Simulación Monte Carlo:** Genera miles de escenarios aleatorios basados en la historia y extrae "
                "la pérdida exacta en el percentil deseado. Es más realista porque captura las 'colas gordas' (caídas abruptas del mercado)."
            )
            themed_warning(
                "**Conditional VaR (CVaR o Expected Shortfall):** Mide el promedio de las pérdidas *una vez que el VaR ha sido superado*. "
                "En otras palabras, responde a la pregunta: *'Si las cosas salen realmente mal, ¿qué tan mal se van a poner?'*"
            )

    # ── TAB 5: EXPORTAR ───────────────────────────────────────────────────────
    with tab_dl:
        st.markdown("#### Descarga de Datos para Réplica")
        themed_info(
            "Exporta la base de datos completa de precios históricos y los vectores de pesos "
            "óptimos generados por el algoritmo, listos para importarlos a Excel o Python."
        )

        col_d1, col_d2 = st.columns(2)

        csv_precios = data.to_csv().encode("utf-8")

        df_pesos_export             = pd.DataFrame({
            "Máximo Sharpe":   pesos_sharpe,
            "Mínima Varianza": pesos_min,
        })
        df_pesos_export.index.name = "Ticker"
        csv_pesos                   = df_pesos_export.to_csv().encode("utf-8")

        with col_d1:
            st.download_button(
                label="⬇️ Descargar Precios Históricos (.csv)",
                data=csv_precios,
                file_name=f"precios_historicos_{hoy_guardado}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Incluye precios de cierre ajustados (Adjusted Close) y limpios de NAs.")

        with col_d2:
            st.download_button(
                label="⬇️ Descargar Pesos Óptimos (.csv)",
                data=csv_pesos,
                file_name=f"pesos_optimos_{hoy_guardado}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Incluye los vectores $w_i$ resultantes de la optimización.")