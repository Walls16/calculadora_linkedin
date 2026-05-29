"""
pages/12_Formulario.py
----------------------
Módulo 12: Formulario Oficial de Matemáticas Financieras.
Tres pestañas descargables en HTML con MathJax:
  - Matemáticas Financieras (Tasas, TVM, Rentas, Amortización)
  - Acciones y Bonos (Gordon-Shapiro, YTM, Múltiplos)
  - Derivados Financieros (Forwards, BSM 6 variantes)
"""

import streamlit as st
from utils import get_engine, page_header, separador, themed_info, themed_success, themed_warning, themed_error

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Formulario · Calculadora Financiera",
    page_icon="📋",
    layout="wide",
)

get_engine()   # Mantiene la caché activa aunque no se use el motor aquí

page_header(
    titulo="12. Formulario Oficial",
    subtitulo="Cheat-sheet completo · Tres secciones descargables en HTML con renderizado LaTeX"
)

st.write(
    "Explora las fórmulas por categoría. "
    "Al final de cada pestaña encontrarás un botón para descargar esa sección "
    "como archivo HTML interactivo con todas las ecuaciones renderizadas."
)

separador()

# =============================================================================
# GENERADOR DE HTML
# =============================================================================
def _html(titulo: str, cuerpo: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>{titulo}</title>
  <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$','$'],['\\\\(','\\\\)']]
      }}
    }};
  </script>
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    body {{
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      line-height: 1.6; max-width: 1050px;
      margin: 0 auto; padding: 24px; color: #333;
    }}
    h1 {{
      text-align: center; color: #1E3A8A;
      border-bottom: 2px solid #1E3A8A; padding-bottom: 10px;
    }}
    h2 {{
      color: #2563EB; margin-top: 32px;
      border-bottom: 1px solid #E2E8F0; padding-bottom: 5px;
    }}
    table {{
      width: 100%; border-collapse: collapse;
      margin: 14px 0 22px; font-size: 0.94em;
    }}
    th, td {{
      border: 1px solid #CBD5E1; padding: 11px; text-align: left;
      vertical-align: middle;
    }}
    th {{ background: #F8FAFC; color: #0F172A; font-weight: bold; }}
    td:nth-child(2) {{ text-align: center; background: #fff; }}
    .footer {{
      text-align: center; margin-top: 52px;
      font-size: 0.8em; color: #64748B;
    }}
  </style>
</head>
<body>
  <h1>{titulo}</h1>
  {cuerpo}
  <div class="footer">
    Generado automáticamente por la Calculadora Actuarial de Owen — ¡Mucho éxito!
  </div>
</body>
</html>"""


# =============================================================================
# PESTAÑAS
# =============================================================================
tab_mf, tab_ab, tab_der = st.tabs([
    "Matemáticas Financieras",
    "Acciones y Bonos",
    "Derivados Financieros",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MATEMÁTICAS FINANCIERAS
# ─────────────────────────────────────────────────────────────────────────────
with tab_mf:

    st.subheader("1. Tasas de Interés")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **Triple igualdad** | $1 + i = \left(1 + \frac{i^{(m)}}{m}\right)^m = e^\delta$ | Relaciona la equivalencia entre las tres tasas: efectiva, nominal e instantánea. |
| **Tasa Efectiva Anual ($i$)** | $i = \left(1 + \frac{i^{(m)}}{m}\right)^m - 1$ | Calcula el rendimiento real anual a partir de una tasa nominal capitalizable $m$ veces. |
| **Tasa Instantánea ($\delta$)** | $\delta = m \ln\!\left(1 + \frac{i^{(m)}}{m}\right)$ | Calcula la fuerza de interés (tasa de capitalización continua). |
| **Instantánea → Efectiva** | $i = e^\delta - 1$ | Convierte tasa continua a efectiva anual. |
| **Instantánea → Nominal** | $i^{(m)} = m \left(e^{\delta/m} - 1\right)$ | Convierte tasa continua a nominal anual. |
| **Nominal → Nominal** | $i^{(p)} = p\left[\left(1 + \frac{i^{(m)}}{m}\right)^{m/p} - 1\right]$ | Transforma una tasa nominal de $m$ a $p$ capitalizaciones por año. |
    """)

    st.subheader("2. Valor del Dinero en el Tiempo (TVM)")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **VF Efectiva** | $VF = C_0 (1+i)^n$ | Monto acumulado con tasa efectiva. |
| **VF Nominal** | $VF = C_0 \left(1+\frac{i^{(m)}}{m}\right)^{nm}$ | Monto acumulado con tasa nominal periódica. |
| **VF Instantánea** | $VF = C_0 e^{\delta n}$ | Monto acumulado con capitalización continua. |
| **VP Efectiva** | $VP = C_n (1+i)^{-n}$ | Descuento de capital a tasa efectiva. |
| **VP Nominal** | $VP = C_n \left(1+\frac{i^{(m)}}{m}\right)^{-nm}$ | Descuento con tasa nominal periódica. |
| **VP Instantánea** | $VP = C_n e^{-\delta n}$ | Descuento bajo capitalización continua. |
| **Periodos ($n$)** | $n = \dfrac{\ln(C_n/C_0)}{\ln(1+i)}$ | Tiempo necesario para que un capital alcance un valor futuro. |
| **Tasa ($i$)** | $i = \left(\dfrac{C_n}{C_0}\right)^{1/n} - 1$ | Tasa de interés implícita de una inversión a un solo pago. |
    """)

    st.subheader("3. Valor Futuro de Rentas")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **Vencidas** | $VF = R \, s_{\overline{nm}\,\vert\,i_m} = R \left[\dfrac{(1+i_m)^{nm}-1}{i_m}\right]$ | Depósitos fijos al final de cada periodo. |
| **Anticipadas** | $VF = R \, \ddot{s}_{\overline{nm}\,\vert\,i_m} = R \left[\dfrac{(1+i_m)^{nm}-1}{i_m}\right](1+i_m)$ | Depósitos fijos al inicio de cada periodo. |
| **Pagaderas $p$ veces** | $VF = R \left[\dfrac{(1+i_p)^{np}-1}{i_p}\right]$ | Frecuencia de pago distinta a la capitalización. |
| **Continuas ($\delta$)** | $VF = \bar{R}\left[\dfrac{e^{\delta n}-1}{\delta}\right]$ | Flujo de caja continuo a tasa instantánea. |
| **Continuas ($i$)** | $VF = \bar{R}\left[\dfrac{(1+i)^n-1}{\ln(1+i)}\right]$ | Flujo continuo expresado con tasa efectiva anual. |
| **Geométrica ($i_m \neq q_m$)** | $VF = R_1\left[\dfrac{(1+i_m)^{nm}-(1+q_m)^{nm}}{i_m-q_m}\right]$ | Pagos que crecen a tasa porcentual constante $q_m$. |
| **Geométrica ($i_m = q_m$)** | $VF = nm \cdot R_1(1+i_m)^{nm-1}$ | Caso especial: tasa de interés igual a tasa de crecimiento. |
| **Aritmética** | $VF = R_1\, s_{\overline{nm}\,\vert\,i_m} + \dfrac{G}{i_m}\!\left(s_{\overline{nm}\,\vert\,i_m} - nm\right)$ | Pagos que crecen sumando una cantidad fija $G$. |
    """)

    st.subheader("4. Valor Presente de Rentas")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **Vencidas** | $VP = R \, a_{\overline{nm}\,\vert\,i_m} = R \left[\dfrac{1-(1+i_m)^{-nm}}{i_m}\right]$ | Valor actual de pagos fijos al final de cada periodo. |
| **Anticipadas** | $VP = R \, \ddot{a}_{\overline{nm}\,\vert\,i_m} = R \left[\dfrac{1-(1+i_m)^{-nm}}{i_m}\right](1+i_m)$ | Valor actual de pagos fijos al inicio de cada periodo. |
| **Perpetuas** | $VP = \dfrac{R}{i_m}$ | Pago fijo a recibir para siempre. |
| **Pagaderas $p$ veces** | $VP = R \left[\dfrac{1-(1+i_p)^{-np}}{i_p}\right]$ | Frecuencia de pago distinta a la capitalización. |
| **Continuas ($\delta$)** | $VP = \bar{R}\left[\dfrac{1-e^{-\delta n}}{\delta}\right]$ | Flujo continuo a tasa instantánea. |
| **Continuas ($i$)** | $VP = \bar{R}\left[\dfrac{1-(1+i)^{-n}}{\ln(1+i)}\right]$ | Flujo continuo con tasa efectiva. |
| **Geométrica ($i_m \neq q_m$)** | $VP = R_1\left[\dfrac{1-\left(\frac{1+q_m}{1+i_m}\right)^{nm}}{i_m-q_m}\right]$ | Valor actual de pagos con crecimiento porcentual $q_m$. |
| **Geométrica ($i_m = q_m$)** | $VP = \dfrac{nm \cdot R_1}{1+i_m}$ | Caso especial geométrico. |
| **Aritmética** | $VP = R_1\, a_{\overline{nm}\,\vert\,i_m} + \dfrac{G}{i_m}\!\left(a_{\overline{nm}\,\vert\,i_m} - nm\cdot v^{nm}\right)$ | Valor actual con crecimiento aritmético $G$. |
    """)

    st.subheader("5. Número de Periodos y Amortización")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **$nm$ desde VF** | $nm = \dfrac{\ln\!\left(\frac{VF \cdot i_m}{R}+1\right)}{\ln(1+i_m)}$ | Tiempo para alcanzar meta de ahorro $VF$ con pagos $R$. |
| **$nm$ desde VP** | $nm = \dfrac{-\ln\!\left(1-\frac{VP \cdot i_m}{R}\right)}{\ln(1+i_m)}$ | Tiempo para liquidar préstamo $VP$ con cuotas $R$. |
| **Pago fijo ($R$)** | $R = VP \left[\dfrac{i_m}{1-(1+i_m)^{-nm}}\right]$ | Cuota periódica para amortizar un préstamo. |
| **Préstamo ($VP$)** | $VP = R \left[\dfrac{1-(1+i_m)^{-nm}}{i_m}\right]$ | Monto máximo de crédito pagando cuota $R$. |
    """)

    separador()

    # HTML descargable (Note que aquí también cambiamos los | de la notación por \vert)
    html_mf = _html("Formulario: Matemáticas Financieras", """
  <h2>1. Tasas de Interés</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>Triple Igualdad</td><td>$$1 + i = \\left(1 + \\frac{i^{(m)}}{m}\\right)^m = e^\\delta$$</td><td>Equivalencia entre las tres tasas.</td></tr>
    <tr><td>Efectiva Anual ($i$)</td><td>$$i = \\left(1 + \\frac{i^{(m)}}{m}\\right)^m - 1$$</td><td>Nominal a Efectiva.</td></tr>
    <tr><td>Instantánea ($\\delta$)</td><td>$$\\delta = m \\ln\\!\\left(1 + \\frac{i^{(m)}}{m}\\right)$$</td><td>Nominal a Continua.</td></tr>
    <tr><td>Continua a Efectiva</td><td>$$i = e^\\delta - 1$$</td><td>Instantánea a Efectiva.</td></tr>
    <tr><td>Continua a Nominal</td><td>$$i^{(m)} = m\\left(e^{\\delta/m}-1\\right)$$</td><td>Instantánea a Nominal.</td></tr>
    <tr><td>Nominal a Nominal</td><td>$$i^{(p)} = p\\left[\\left(1+\\frac{i^{(m)}}{m}\\right)^{m/p}-1\\right]$$</td><td>Cambio de capitalización.</td></tr>
  </table>

  <h2>2. Valor del Dinero en el Tiempo</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>VF (Efectiva / Nominal / Continua)</td>
        <td>$$VF = C_0(1+i)^n \\quad|\\quad VF = C_0\\!\\left(1+\\frac{i^{(m)}}{m}\\right)^{nm} \\quad|\\quad VF = C_0 e^{\\delta n}$$</td>
        <td>Monto acumulado.</td></tr>
    <tr><td>VP (Efectiva / Nominal / Continua)</td>
        <td>$$VP = C_n(1+i)^{-n} \\quad|\\quad VP = C_n\\!\\left(1+\\frac{i^{(m)}}{m}\\right)^{-nm} \\quad|\\quad VP = C_n e^{-\\delta n}$$</td>
        <td>Descuento de capital.</td></tr>
    <tr><td>Periodos y Tasa</td>
        <td>$$n = \\frac{\\ln(C_n/C_0)}{\\ln(1+i)} \\qquad i = \\left(\\frac{C_n}{C_0}\\right)^{1/n}-1$$</td>
        <td>Tiempo y rendimiento implícito.</td></tr>
  </table>

  <h2>3. Valor Futuro de Rentas</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>Vencidas</td><td>$$VF = R\\left[\\frac{(1+i_m)^{nm}-1}{i_m}\\right]$$</td><td>Depósitos al final del periodo.</td></tr>
    <tr><td>Anticipadas</td><td>$$VF = R\\left[\\frac{(1+i_m)^{nm}-1}{i_m}\\right](1+i_m)$$</td><td>Depósitos al inicio del periodo.</td></tr>
    <tr><td>Continuas ($\\delta$)</td><td>$$VF = \\bar{R}\\left[\\frac{e^{\\delta n}-1}{\\delta}\\right]$$</td><td>Flujo ininterrumpido.</td></tr>
    <tr><td>Geométrica ($i_m\\neq q_m$)</td><td>$$VF = R_1\\left[\\frac{(1+i_m)^{nm}-(1+q_m)^{nm}}{i_m-q_m}\\right]$$</td><td>Crecimiento porcentual $q_m$.</td></tr>
    <tr><td>Geométrica ($i_m = q_m$)</td><td>$$VF = nm\\cdot R_1(1+i_m)^{nm-1}$$</td><td>Caso especial geométrico.</td></tr>
    <tr><td>Aritmética</td><td>$$VF = R_1\\,s_{\\overline{nm}\\,\\vert\\,i_m} + \\frac{G}{i_m}\\!\\left(s_{\\overline{nm}\\,\\vert\\,i_m}-nm\\right)$$</td><td>Gradiente aritmético $G$.</td></tr>
  </table>

  <h2>4. Valor Presente de Rentas</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>Vencidas</td><td>$$VP = R\\left[\\frac{1-(1+i_m)^{-nm}}{i_m}\\right]$$</td><td>Pagos al final del periodo.</td></tr>
    <tr><td>Anticipadas</td><td>$$VP = R\\left[\\frac{1-(1+i_m)^{-nm}}{i_m}\\right](1+i_m)$$</td><td>Pagos al inicio del periodo.</td></tr>
    <tr><td>Perpetua</td><td>$$VP = \\frac{R}{i_m}$$</td><td>Pago fijo a perpetuidad.</td></tr>
    <tr><td>Continua ($\\delta$)</td><td>$$VP = \\bar{R}\\left[\\frac{1-e^{-\\delta n}}{\\delta}\\right]$$</td><td>Flujo ininterrumpido.</td></tr>
    <tr><td>Geométrica ($i_m\\neq q_m$)</td><td>$$VP = R_1\\left[\\frac{1-\\left(\\frac{1+q_m}{1+i_m}\\right)^{nm}}{i_m-q_m}\\right]$$</td><td>Crecimiento porcentual $q_m$.</td></tr>
    <tr><td>Geométrica ($i_m = q_m$)</td><td>$$VP = \\frac{nm\\cdot R_1}{1+i_m}$$</td><td>Caso especial geométrico.</td></tr>
    <tr><td>Aritmética</td><td>$$VP = R_1\\,a_{\\overline{nm}\\,\\vert\\,i_m} + \\frac{G}{i_m}\\!\\left(a_{\\overline{nm}\\,\\vert\\,i_m}-nm\\cdot v^{nm}\\right)$$</td><td>Gradiente aritmético $G$.</td></tr>
  </table>

  <h2>5. Número de Periodos y Amortización</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>$nm$ desde VF</td><td>$$nm = \\frac{\\ln\\!\\left(\\frac{VF\\cdot i_m}{R}+1\\right)}{\\ln(1+i_m)}$$</td><td>Periodos para alcanzar meta de ahorro.</td></tr>
    <tr><td>$nm$ desde VP</td><td>$$nm = \\frac{-\\ln\\!\\left(1-\\frac{VP\\cdot i_m}{R}\\right)}{\\ln(1+i_m)}$$</td><td>Periodos para liquidar préstamo.</td></tr>
    <tr><td>Pago fijo ($R$)</td><td>$$R = VP\\left[\\frac{i_m}{1-(1+i_m)^{-nm}}\\right]$$</td><td>Cuota de amortización.</td></tr>
    <tr><td>Préstamo ($VP$)</td><td>$$VP = R\\left[\\frac{1-(1+i_m)^{-nm}}{i_m}\\right]$$</td><td>Monto máximo de crédito.</td></tr>
  </table>
""")

    st.download_button(
        "⬇️ Descargar Formulario: Matemáticas Financieras (HTML)",
        data=html_mf,
        file_name="Form_MatematicasFinancieras.html",
        mime="text/html",
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ACCIONES Y BONOS
# ─────────────────────────────────────────────────────────────────────────────
with tab_ab:

    st.subheader("1. Valuación de Bonos")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **Precio (notación actuarial)** | $P = Fr \cdot a_{\overline{nm}\,\vert\,i_m} + C(1+i_m)^{-nm}$ | Precio teórico = VP cupones + VP principal. |
| **Precio (forma explícita)** | $P = Fr \left[\dfrac{1-(1+i_m)^{-nm}}{i_m}\right] + C(1+i_m)^{-nm}$ | Valuación usando la tasa de rendimiento del mercado $i_m$. |
| **Yield to Maturity (YTM)** | $P_{mdo} = Fr \left[\dfrac{1-(1+i_m)^{-nm}}{i_m}\right] + C(1+i_m)^{-nm}$ | Despejar $i_m$ iterativamente dado el precio de mercado. |
| **Duración de Macaulay** | $D_{Mac} = \dfrac{\sum_{t=1}^{nm} t \cdot VP(CF_t)}{P} \div m$ | Tiempo promedio ponderado para recuperar la inversión (años). |
| **Duración Modificada** | $D_{Mod} = \dfrac{D_{Mac}}{1+i_m}$ | Sensibilidad lineal del precio ante +1% en tasas. |
| **Convexidad** | $C = \dfrac{\sum_{t=1}^{nm} t(t+1) \cdot VP(CF_t)}{P \cdot m^2 \cdot (1+i_m)^2}$ | Curvatura de la relación precio-tasa. |
| **Cambio en precio** | $\dfrac{\Delta P}{P} \approx -D_{Mod}(\Delta y) + \tfrac{1}{2}C(\Delta y)^2$ | Estimación del impacto de un movimiento de tasas. |
    """)

    st.subheader("2. Valuación de Acciones")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **Gordon-Shapiro ($P_0$)** | $P_0 = \dfrac{D_1}{k-g}$ | Precio teórico con dividendos crecientes a perpetuidad. |
| **Rendimiento requerido ($k$)** | $k = \dfrac{D_1}{P_0} + g$ | Costo de capital accionario implícito en el precio de mercado. |
| **P/E Ratio** | $P_0 = \text{UPA} \times \left(\dfrac{P}{E}\right)$ | Valuación relativa por utilidad por acción. |
| **P/S Ratio** | $P_0 = \text{VPA} \times \left(\dfrac{P}{S}\right)$ | Valuación relativa por ventas por acción. |
| **EV/EBITDA** | $EV = \text{EBITDA} \times \left(\dfrac{EV}{\text{EBITDA}}\right)$ | Valor de la empresa completa. |
| **P/B Ratio** | $P_0 = \text{VLA} \times \left(\dfrac{P}{B}\right)$ | Valuación relativa por valor en libros. |
    """)

    separador()

    html_ab = _html("Formulario: Acciones y Bonos", """
  <h2>1. Valuación de Bonos</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>Precio del Bono</td>
        <td>$$P = Fr\\left[\\frac{1-(1+i_m)^{-nm}}{i_m}\\right]+C(1+i_m)^{-nm}$$</td>
        <td>VP cupones + VP principal.</td></tr>
    <tr><td>YTM</td>
        <td>$$P_{mdo} = Fr\\left[\\frac{1-(1+i_m)^{-nm}}{i_m}\\right]+C(1+i_m)^{-nm}$$</td>
        <td>Tasa intrínseca dado el precio.</td></tr>
    <tr><td>Duración Macaulay</td>
        <td>$$D_{Mac} = \\frac{\\sum_{t=1}^{nm} t\\cdot VP(CF_t)}{P}\\div m$$</td>
        <td>Plazo promedio ponderado.</td></tr>
    <tr><td>Duración Modificada</td>
        <td>$$D_{Mod} = \\frac{D_{Mac}}{1+i_m}$$</td>
        <td>Sensibilidad lineal precio-tasa.</td></tr>
    <tr><td>Convexidad</td>
        <td>$$C = \\frac{\\sum_{t=1}^{nm}t(t+1)\\cdot VP(CF_t)}{P\\cdot m^2\\cdot(1+i_m)^2}$$</td>
        <td>Curvatura precio-tasa.</td></tr>
    <tr><td>Cambio en precio</td>
        <td>$$\\frac{\\Delta P}{P}\\approx -D_{Mod}(\\Delta y)+\\tfrac{1}{2}C(\\Delta y)^2$$</td>
        <td>Impacto de movimiento de tasas.</td></tr>
  </table>

  <h2>2. Valuación de Acciones</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>Gordon-Shapiro</td>
        <td>$$P_0 = \\frac{D_1}{k-g} \\qquad k = \\frac{D_1}{P_0}+g$$</td>
        <td>Dividendos crecientes a perpetuidad.</td></tr>
    <tr><td>Múltiplos</td>
        <td>$$P_0 = \\text{UPA}\\times\\!\\left(\\frac{P}{E}\\right) \\quad|\\quad EV = \\text{EBITDA}\\times\\!\\left(\\frac{EV}{\\text{EBITDA}}\\right)$$</td>
        <td>Valuación relativa.</td></tr>
  </table>
""")

    st.download_button(
        "⬇️ Descargar Formulario: Acciones y Bonos (HTML)",
        data=html_ab,
        file_name="Form_AccionesBonos.html",
        mime="text/html",
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DERIVADOS FINANCIEROS
# ─────────────────────────────────────────────────────────────────────────────
with tab_der:

    st.subheader("1. Contratos Forward")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **Activo simple** | $F_0 = S_0 e^{rT}$ | Precio de entrega teórico sin ingresos ni costos. |
| **Con dividendo continuo / tasa extranjera** | $F_0 = S_0 e^{(r-q)T}$ | Activos con yield continuo $q$ o divisas con $r_f$. |
| **Con dividendos discretos** | $F_0 = (S_0 - I)e^{rT}$ | Ajuste por VP de dividendos discretos $I$. |
| **Commodity (costo almacenaje)** | $F_0 = S_0 e^{(r+u)T}$ | Commodities con costo continuo de almacenaje $u$. |
| **Divisas (PTCI)** | $F_0 = S_0 e^{(r_d - r_f)T}$ | Paridad cubierta de tasas de interés. |
| **Valor en vida (posición larga)** | $f_t = S_t e^{-q(T-t)} - F_0 e^{-r(T-t)}$ | Valor de mercado del contrato antes del vencimiento. |
| **Tasa forward implícita (FRA)** | $R_F = \dfrac{r_2 t_2 - r_1 t_1}{t_2 - t_1}$ | Tasa de interés pactada para un periodo futuro. |
| **Valor del FRA** | $V_{FRA} = N(R_F - R_K)(t_2 - t_1)e^{-r_2 t_2}$ | Valor de mercado de un FRA antes del vencimiento. |
    """)

    st.subheader("2. Árbol Binomial CRR")
    st.markdown(r"""
| Concepto | Fórmula | Para qué se usa |
| :--- | :---: | :--- |
| **Paso de tiempo** | $\Delta t = T/N$ | Divide el horizonte en $N$ periodos iguales. |
| **Factores de movimiento** | $u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u$ | Tamaños del salto hacia arriba y abajo. |
| **Probabilidad neutral al riesgo** | $p = \dfrac{e^{(r-q)\Delta t} - d}{u - d}$ | Probabilidad que elimina el arbitraje en cada nodo. |
| **Retroinducción (europea)** | $f_t = e^{-r\Delta t}[p \cdot f_u + (1-p) \cdot f_d]$ | Precio en cada nodo descontando hacia atrás. |
| **Retroinducción (americana)** | $f_t = \max\!\left(\text{ejercicio},\, e^{-r\Delta t}[p \cdot f_u + (1-p) \cdot f_d]\right)$ | Permite ejercicio anticipado en cada nodo. |
    """)

    st.subheader("3. Black-Scholes-Merton (BSM)")
    st.markdown(r"""
| Modelo | Call ($c$) y Put ($p$) | $d_1$ |
| :--- | :---: | :---: |
| **1. Simple** | $c = S_0 N(d_1) - Ke^{-rT}N(d_2)$ / $p = Ke^{-rT}N(-d_2) - S_0 N(-d_1)$ | $d_1 = \dfrac{\ln(S_0/K)+(r+\sigma^2/2)T}{\sigma\sqrt{T}}$ |
| **2. Dividendo continuo** | $c = S_0 e^{-qT}N(d_1) - Ke^{-rT}N(d_2)$ / $p = Ke^{-rT}N(-d_2) - S_0 e^{-qT}N(-d_1)$ | $d_1 = \dfrac{\ln(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}}$ |
| **3. Futuros (Black 76)** | $c = e^{-rT}[F_0 N(d_1)-KN(d_2)]$ / $p = e^{-rT}[KN(-d_2)-F_0 N(-d_1)]$ | $d_1 = \dfrac{\ln(F_0/K)+(\sigma^2/2)T}{\sigma\sqrt{T}}$ |
| **4. Divisas (G-K)** | $c = S_0 e^{-r_f T}N(d_1) - Ke^{-rT}N(d_2)$ / $p = Ke^{-rT}N(-d_2)-S_0 e^{-r_f T}N(-d_1)$ | $d_1 = \dfrac{\ln(S_0/K)+(r-r_f+\sigma^2/2)T}{\sigma\sqrt{T}}$ |
| **Común a todos** | $d_2 = d_1 - \sigma\sqrt{T}$ | — |
    """)

    st.subheader("4. Griegas (BSM con dividendo $q$)")
    st.markdown(r"""
| Griega | Fórmula | Interpretación |
| :--- | :---: | :--- |
| **Δ Delta** | $\Delta_c = e^{-qT}N(d_1)$, $\quad\Delta_p = -e^{-qT}N(-d_1)$ | Cambio en prima ante +$1 en el subyacente. |
| **Γ Gamma** | $\Gamma = \dfrac{e^{-qT}n(d_1)}{S_0\sigma\sqrt{T}}$ | Cambio en Delta ante +$1 (igual para call y put). |
| **Θ Theta (call)** | $\Theta_c = -\dfrac{S_0 n(d_1)\sigma e^{-qT}}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qS_0 e^{-qT}N(d_1)$ | Decaimiento temporal ($/día). |
| **𝒱 Vega** | $\mathcal{V} = S_0\sqrt{T}\,e^{-qT}n(d_1)$ | Cambio en prima ante +1% de volatilidad. |
| **ρ Rho (call)** | $\rho_c = KTe^{-rT}N(d_2)$ | Cambio en prima ante +1% en tasa libre de riesgo. |
    """)

    st.subheader("5. Derivados Exóticos")
    st.markdown(r"""
| Tipo | Fórmula clave | Nota |
| :--- | :---: | :--- |
| **Gap Call** | $c_{gap} = S_0 e^{-qT}N(d_1) - K_2 e^{-rT}N(d_2)$ con $d_1$ basado en $K_1$ | $K_1$ activa, $K_2$ paga. |
| **Cash-or-Nothing Call** | $c_{CoN} = Qe^{-rT}N(d_2)$ | Paga $Q$ fijo si $S_T > K$. |
| **Asset-or-Nothing Call** | $c_{AoN} = S_0 e^{-qT}N(d_1)$ | Paga $S_T$ si $S_T > K$. |
| **Paridad Call BSM** | $c = c_{AoN} - Ke^{-rT}\cdot c_{CoN}(Q=1)$ | BSM = combinación de binarias. |
| **Paridad Barrera** | $c_{KO} + c_{KI} = c_{vanilla}$ | Knock-out + Knock-in = Vanilla. |
| **Asiática Geométrica** | $\sigma^* = \sigma/\sqrt{3}$, $b^* = \tfrac{1}{2}(r-q-\sigma^2/6)$; usar BSM con $\sigma^*$, $b^*$ | Media geométrica = fórmula cerrada. |
| **Lookback flotante** | $c_{LB} = S_0 e^{-qT}N(a_1) - S_{min}e^{-rT}N(a_2) + \cdots$ | Goldman-Sosin-Gatto (1979). |
| **Compuesta Call-sobre-Call** | $c_{cc} = S_0 e^{-qT_2}M(a_1,b_1;\rho) - K_{in}e^{-rT_2}M(a_2,b_2;\rho) - K_{out}e^{-rT_1}N(a_2)$ | Normal bivariada; $\rho=\sqrt{T_1/T_2}$. |
| **Intercambio (Margrabe)** | $c = S_2 e^{-q_2 T}N(d_1) - S_1 e^{-q_1 T}N(d_2)$; $\sigma^*=\sqrt{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}$ | Entregar $S_1$, recibir $S_2$. |
    """)

    separador()

    html_der = _html("Formulario: Derivados Financieros", """
  <h2>1. Contratos Forward</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>Precio teórico (base)</td><td>$$F_0 = S_0 e^{rT}$$</td><td>Sin ingresos ni costos.</td></tr>
    <tr><td>Con dividendo continuo / tasa extranjera</td><td>$$F_0 = S_0 e^{(r-q)T}$$</td><td>Yield continuo $q$ o divisas.</td></tr>
    <tr><td>Con dividendos discretos</td><td>$$F_0 = (S_0-I)e^{rT}$$</td><td>Ajuste por VP dividendos $I$.</td></tr>
    <tr><td>Commodity</td><td>$$F_0 = S_0 e^{(r+u)T}$$</td><td>Costo continuo de almacenaje $u$.</td></tr>
    <tr><td>Divisas (PTCI)</td><td>$$F_0 = S_0 e^{(r_d-r_f)T}$$</td><td>Paridad cubierta.</td></tr>
    <tr><td>Valor en vida (larga)</td><td>$$f_t = S_t e^{-q(T-t)}-F_0 e^{-r(T-t)}$$</td><td>Valor de mercado en $t$.</td></tr>
    <tr><td>Tasa forward (FRA)</td><td>$$R_F = \\frac{r_2 t_2-r_1 t_1}{t_2-t_1}$$</td><td>Tasa pactada para periodo futuro.</td></tr>
    <tr><td>Valor del FRA</td><td>$$V_{FRA} = N(R_F-R_K)(t_2-t_1)e^{-r_2 t_2}$$</td><td>Valor de mercado del FRA.</td></tr>
  </table>

  <h2>2. Árbol Binomial CRR</h2>
  <table>
    <tr><th>Concepto</th><th>Fórmula</th><th>Uso</th></tr>
    <tr><td>Parámetros CRR</td>
        <td>$$\\Delta t=T/N \\quad u=e^{\\sigma\\sqrt{\\Delta t}} \\quad d=1/u \\quad p=\\frac{e^{(r-q)\\Delta t}-d}{u-d}$$</td>
        <td>Parámetros neutros al riesgo.</td></tr>
    <tr><td>Retroinducción</td>
        <td>Europea: $$f=e^{-r\\Delta t}[p\\cdot f_u+(1-p)\\cdot f_d]$$  Americana: $$f=\\max\\!\\left(\\text{ejercicio},\\,e^{-r\\Delta t}[p\\cdot f_u+(1-p)f_d]\\right)$$</td>
        <td>Valoración hacia atrás.</td></tr>
  </table>

  <h2>3. Black-Scholes-Merton (6 variantes)</h2>
  <table>
    <tr><th>Modelo</th><th>Call ($c$) / Put ($p$)</th><th>$d_1$</th></tr>
    <tr><td>1. Simple</td>
        <td>$$c=S_0 N(d_1)-Ke^{-rT}N(d_2)$$ $$p=Ke^{-rT}N(-d_2)-S_0 N(-d_1)$$</td>
        <td>$$d_1=\\frac{\\ln(S_0/K)+(r+\\sigma^2/2)T}{\\sigma\\sqrt{T}}$$</td></tr>
    <tr><td>2. Dividendo continuo ($q$)</td>
        <td>$$c=S_0 e^{-qT}N(d_1)-Ke^{-rT}N(d_2)$$ $$p=Ke^{-rT}N(-d_2)-S_0 e^{-qT}N(-d_1)$$</td>
        <td>$$d_1=\\frac{\\ln(S_0/K)+(r-q+\\sigma^2/2)T}{\\sigma\\sqrt{T}}$$</td></tr>
    <tr><td>3. Futuros (Black 76)</td>
        <td>$$c=e^{-rT}[F_0 N(d_1)-KN(d_2)]$$ $$p=e^{-rT}[KN(-d_2)-F_0 N(-d_1)]$$</td>
        <td>$$d_1=\\frac{\\ln(F_0/K)+(\\sigma^2/2)T}{\\sigma\\sqrt{T}}$$</td></tr>
    <tr><td>4. Divisas (G-K)</td>
        <td>$$c=S_0 e^{-r_f T}N(d_1)-Ke^{-rT}N(d_2)$$ $$p=Ke^{-rT}N(-d_2)-S_0 e^{-r_f T}N(-d_1)$$</td>
        <td>$$d_1=\\frac{\\ln(S_0/K)+(r-r_f+\\sigma^2/2)T}{\\sigma\\sqrt{T}}$$</td></tr>
    <tr><td colspan="3" style="text-align:center;">Para todos los modelos: $$d_2 = d_1 - \\sigma\\sqrt{T}$$</td></tr>
  </table>

  <h2>4. Griegas (BSM con dividendo $q$)</h2>
  <table>
    <tr><th>Griega</th><th>Fórmula</th><th>Interpretación</th></tr>
    <tr><td>Delta</td><td>$$\\Delta_c = e^{-qT}N(d_1) \\quad \\Delta_p = -e^{-qT}N(-d_1)$$</td><td>+$1 en subyacente.</td></tr>
    <tr><td>Gamma</td><td>$$\\Gamma = \\frac{e^{-qT}n(d_1)}{S_0\\sigma\\sqrt{T}}$$</td><td>Cambio en Delta.</td></tr>
    <tr><td>Theta (call)</td><td>$$\\Theta_c = -\\frac{S_0 n(d_1)\\sigma e^{-qT}}{2\\sqrt{T}}-rKe^{-rT}N(d_2)+qS_0 e^{-qT}N(d_1)$$</td><td>Decaimiento temporal.</td></tr>
    <tr><td>Vega</td><td>$$\\mathcal{V} = S_0\\sqrt{T}\\,e^{-qT}n(d_1)$$</td><td>+1% de volatilidad.</td></tr>
    <tr><td>Rho (call)</td><td>$$\\rho_c = KTe^{-rT}N(d_2)$$</td><td>+1% en tasa $r$.</td></tr>
  </table>
""")

    st.download_button(
        "⬇️ Descargar Formulario: Derivados Financieros (HTML)",
        data=html_der,
        file_name="Form_Derivados.html",
        mime="text/html",
        use_container_width=True,
    )