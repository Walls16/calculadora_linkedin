"""
pages/10_Forwards.py
-------------------
Módulo 10: Contratos Forward.
Cubre:
  - Precio teórico del forward (con y sin dividendos / costo de almacenamiento)
  - Valuación de un forward en vida (valor de mercado en t < T)
  - Forward sobre divisas (Paridad Cubierta de Tasas de Interés)
  - Forward Rate Agreement (FRA)
"""

import numpy as np
import streamlit as st

from utils import get_engine, page_header, paso_a_paso, separador, themed_info, themed_success, themed_warning, themed_error, apply_plotly_theme

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Forwards · Calculadora Financiera",
    page_icon="📃",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="9. Contratos Forward",
    subtitulo="Precio teórico · Valuación en vida · Divisas (PTCI) · FRA"
)

# =============================================================================
# PESTAÑAS
# =============================================================================
tab_precio, tab_valor, tab_divisa, tab_fra = st.tabs([
    " Precio Teórico (F₀)",
    " Valuación en Vida (f_t)",
    " Forward sobre Divisas",
    " FRA (Forward Rate Agreement)",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRECIO TEÓRICO DEL FORWARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_precio:
    st.markdown("### Precio Teórico del Contrato Forward ($F_0$)")
    themed_info(
        "El precio forward es el precio justo que evita el arbitraje. "
        "Se calcula llevando el precio spot al futuro con la tasa libre de riesgo, "
        "ajustando por cualquier beneficio o costo de mantener el activo subyacente."
    )

    col_cap1, col_cap2 = st.columns([1, 1])
    with col_cap1:
        tipo_capitalizacion = st.radio(
            "Tipo de capitalización:",
            ["Continua ($e^{rT}$)", "Discreta ($(1+r)^T$)"],
            horizontal=True,
            key="fwd_cap"
        )
    es_cont = "Continua" in tipo_capitalizacion
    separador()

    tipo_subyacente = st.radio(
        "Tipo de subyacente:",
        [
            "Activo sin rendimientos (acciones sin dividendo, oro sin almacenamiento)",
            "Activo con dividendo continuo o tasa extranjera (q)",
            "Activo con dividendos discretos (D₁, D₂, …)",
            "Commodity con costo almacenamiento continuo (u)",
            "Commodity con costos discretos (C₁, C₂, …)",
        ],
        key="fwd_tipo_sub",
    )
    separador()

    c1, c2 = st.columns(2)

    # ── Inputs comunes ────────────────────────────────────────────────────────
    with c1:
        S0_fwd = st.number_input("Precio Spot ($S_0$)", min_value=0.01,
                                  value=100.0, step=1.0, key="fwd_s0")
        r_fwd  = st.number_input("Tasa libre de riesgo ($r$) %",
                                  value=5.0, step=0.1, key="fwd_r") / 100
        T_fwd  = st.number_input("Tiempo al vencimiento ($T$) en años",
                                  min_value=0.01, value=1.0,
                                  step=0.25, key="fwd_T")

        # Tasa equivalente para el motor si es discreta (truco matemático)
        r_calc = r_fwd if es_cont else np.log(1 + r_fwd)

    # ── Inputs específicos por tipo ───────────────────────────────────────────
    F0_res    = None
    formula   = ""
    extra_val = {}

    with c1:
        if tipo_subyacente.startswith("Activo sin"):
            F0_res  = engine.precio_forward(S0_fwd, r_calc, T_fwd)
            formula = r"F_0 = S_0 e^{rT}" if es_cont else r"F_0 = S_0 (1+r)^T"

        elif tipo_subyacente.startswith("Activo con dividendo continuo"):
            q_fwd   = st.number_input("Dividendo / tasa extranjera ($q$) %",
                                       value=2.0, step=0.1, key="fwd_q") / 100
            q_calc  = q_fwd if es_cont else np.log(1 + q_fwd)
            F0_res  = engine.precio_forward_dividendo_continuo(S0_fwd, r_calc, q_calc, T_fwd)
            formula = r"F_0 = S_0 e^{(r-q)T}" if es_cont else r"F_0 = S_0 \frac{(1+r)^T}{(1+q)^T}"
            extra_val["q"] = q_fwd

        elif tipo_subyacente.startswith("Activo con dividendos discretos"):
            st.markdown("**Dividendos discretos (monto y tiempo):**")
            n_div = st.number_input("Número de dividendos", min_value=1,
                                     max_value=6, value=2, step=1, key="fwd_ndiv")
            divs  = []
            for i in range(int(n_div)):
                cd1, cd2 = st.columns(2)
                with cd1:
                    d_i = st.number_input(f"Dividendo {i+1} ($D_{i+1}$)",
                                           min_value=0.0, value=2.0, step=0.5, key=f"fwd_d{i}")
                with cd2:
                    t_i = st.number_input(f"Tiempo {i+1} ($t_{i+1}$, años)",
                                           min_value=0.0, max_value=float(T_fwd),
                                           value=round(T_fwd * (i+1) / (n_div+1), 2), step=0.25, key=f"fwd_td{i}")
                divs.append((d_i, t_i))
            
            # Valor presente de los dividendos (según capitalización)
            if es_cont:
                I_fwd = sum(d * np.exp(-r_fwd * t) for d, t in divs)
            else:
                I_fwd = sum(d * (1 + r_fwd)**(-t) for d, t in divs)

            F0_res  = engine.precio_forward_dividendos_discretos(S0_fwd, r_calc, T_fwd, I_fwd)
            formula = r"F_0 = (S_0 - I) e^{rT}" if es_cont else r"F_0 = (S_0 - I)(1+r)^T"
            extra_val["I"] = I_fwd
            extra_val["divs"] = divs

        elif tipo_subyacente.startswith("Commodity con costo almacenamiento continuo"):
            u_fwd   = st.number_input("Costo de almacenamiento continuo ($u$) %",
                                       value=1.5, step=0.1, key="fwd_u") / 100
            u_calc  = u_fwd if es_cont else np.log(1 + u_fwd)
            F0_res  = engine.precio_forward_commodity(S0_fwd, r_calc, u_calc, T_fwd)
            formula = r"F_0 = S_0 e^{(r+u)T}" if es_cont else r"F_0 = S_0 (1+r)^T (1+u)^T"
            extra_val["u"] = u_fwd

        else: # Costos discretos
            st.markdown("**Costos discretos (monto y tiempo):**")
            n_cost = st.number_input("Número de costos", min_value=1,
                                     max_value=6, value=2, step=1, key="fwd_ncost")
            costos = []
            for i in range(int(n_cost)):
                cc1, cc2 = st.columns(2)
                with cc1:
                    c_i = st.number_input(f"Costo {i+1} ($C_{i+1}$)",
                                           min_value=0.0, value=5.0, step=0.5, key=f"fwd_c{i}")
                with cc2:
                    t_i = st.number_input(f"Tiempo {i+1} ($t_{i+1}$, años)",
                                           min_value=0.0, max_value=float(T_fwd),
                                           value=round(T_fwd * (i+1) / (n_cost+1), 2), step=0.25, key=f"fwd_tc{i}")
                costos.append((c_i, t_i))
            
            # Valor presente de los costos (según capitalización)
            if es_cont:
                VP_C = sum(c * np.exp(-r_fwd * t) for c, t in costos)
            else:
                VP_C = sum(c * (1 + r_fwd)**(-t) for c, t in costos)

            # Enviamos al motor como dividendo negativo para que se SUME a S0
            F0_res  = engine.precio_forward_dividendos_discretos(S0_fwd, r_calc, T_fwd, -VP_C)
            formula = r"F_0 = (S_0 + C) e^{rT}" if es_cont else r"F_0 = (S_0 + C)(1+r)^T"
            extra_val["C"] = VP_C
            extra_val["costos"] = costos

    with c2:
        if F0_res is not None:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Precio Forward Teórico (F₀): ${F0_res:,.4f}</h3>")
            st.latex(formula)

            separador()

            with paso_a_paso():
                st.latex(formula)
                if tipo_subyacente.startswith("Activo sin"):
                    if es_cont:
                        st.latex(rf"F_0 = {S0_fwd:,.2f} e^{{{r_fwd:.4f}({T_fwd:.4f})}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} e^{{{r_fwd*T_fwd:.6f}}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} ({np.exp(r_fwd*T_fwd):.6f})")
                    else:
                        st.latex(rf"F_0 = {S0_fwd:,.2f} (1 + {r_fwd:.4f})^{{{T_fwd:.4f}}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} ({(1+r_fwd)**T_fwd:.6f})")

                elif tipo_subyacente.startswith("Activo con dividendo continuo"):
                    q_v = extra_val["q"]
                    if es_cont:
                        st.latex(rf"F_0 = {S0_fwd:,.2f} e^{{({r_fwd:.4f} - {q_v:.4f})({T_fwd:.4f})}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} e^{{{r_fwd-q_v:.6f}({T_fwd:.4f})}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} ({np.exp((r_fwd-q_v)*T_fwd):.6f})")
                    else:
                        st.latex(rf"F_0 = {S0_fwd:,.2f} \frac{{(1 + {r_fwd:.4f})^{{{T_fwd:.4f}}}}}{{(1 + {q_v:.4f})^{{{T_fwd:.4f}}}}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} \frac{{{(1+r_fwd)**T_fwd:.6f}}}{{{(1+q_v)**T_fwd:.6f}}}")

                elif tipo_subyacente.startswith("Activo con dividendos discretos"):
                    I_v    = extra_val["I"]
                    divs_v = extra_val["divs"]
                    st.latex(r"I = \sum_{i=1}^n D_i " + (r"e^{-r t_i}" if es_cont else r"(1+r)^{-t_i}"))
                    for idx, (d_i, t_i) in enumerate(divs_v, 1):
                        if es_cont:
                            st.latex(rf"VP(D_{idx}) = {d_i:.2f} e^{{-{r_fwd:.4f}({t_i:.4f})}} = {d_i*np.exp(-r_fwd*t_i):.6f}")
                        else:
                            st.latex(rf"VP(D_{idx}) = {d_i:.2f} (1 + {r_fwd:.4f})^{{-{t_i:.4f}}} = {d_i*(1+r_fwd)**(-t_i):.6f}")
                    st.write("---")
                    st.latex(rf"I = {I_v:.6f}")
                    st.latex(rf"S_0 - I = {S0_fwd:,.2f} - {I_v:.6f} = {S0_fwd - I_v:.6f}")
                    st.write("---")
                    st.latex(formula)
                    if es_cont:
                        st.latex(rf"F_0 = ({S0_fwd - I_v:.6f}) e^{{{r_fwd:.4f}({T_fwd:.4f})}}")
                        st.latex(rf"F_0 = ({S0_fwd - I_v:.6f}) ({np.exp(r_fwd*T_fwd):.6f})")
                    else:
                        st.latex(rf"F_0 = ({S0_fwd - I_v:.6f}) (1 + {r_fwd:.4f})^{{{T_fwd:.4f}}}")
                        st.latex(rf"F_0 = ({S0_fwd - I_v:.6f}) ({(1+r_fwd)**T_fwd:.6f})")

                elif tipo_subyacente.startswith("Commodity con costo almacenamiento continuo"):
                    u_v = extra_val["u"]
                    if es_cont:
                        st.latex(rf"F_0 = {S0_fwd:,.2f} e^{{({r_fwd:.4f} + {u_v:.4f})({T_fwd:.4f})}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} e^{{{r_fwd+u_v:.6f}({T_fwd:.4f})}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} ({np.exp((r_fwd+u_v)*T_fwd):.6f})")
                    else:
                        st.latex(rf"F_0 = {S0_fwd:,.2f} (1 + {r_fwd:.4f})^{{{T_fwd:.4f}}} (1 + {u_v:.4f})^{{{T_fwd:.4f}}}")
                        st.latex(rf"F_0 = {S0_fwd:,.2f} ({(1+r_fwd)**T_fwd:.6f}) ({(1+u_v)**T_fwd:.6f})")

                else: # Costos Discretos
                    C_v      = extra_val["C"]
                    costos_v = extra_val["costos"]
                    st.latex(r"C = \sum_{i=1}^n C_i " + (r"e^{-r t_i}" if es_cont else r"(1+r)^{-t_i}"))
                    for idx, (c_i, t_i) in enumerate(costos_v, 1):
                        if es_cont:
                            st.latex(rf"VP(C_{idx}) = {c_i:.2f} e^{{-{r_fwd:.4f}({t_i:.4f})}} = {c_i*np.exp(-r_fwd*t_i):.6f}")
                        else:
                            st.latex(rf"VP(C_{idx}) = {c_i:.2f} (1 + {r_fwd:.4f})^{{-{t_i:.4f}}} = {c_i*(1+r_fwd)**(-t_i):.6f}")
                    st.write("---")
                    st.latex(rf"C = {C_v:.6f}")
                    st.latex(rf"S_0 + C = {S0_fwd:,.2f} + {C_v:.6f} = {S0_fwd + C_v:.6f}")
                    st.write("---")
                    st.latex(formula)
                    if es_cont:
                        st.latex(rf"F_0 = ({S0_fwd + C_v:.6f}) e^{{{r_fwd:.4f}({T_fwd:.4f})}}")
                        st.latex(rf"F_0 = ({S0_fwd + C_v:.6f}) ({np.exp(r_fwd*T_fwd):.6f})")
                    else:
                        st.latex(rf"F_0 = ({S0_fwd + C_v:.6f}) (1 + {r_fwd:.4f})^{{{T_fwd:.4f}}}")
                        st.latex(rf"F_0 = ({S0_fwd + C_v:.6f}) ({(1+r_fwd)**T_fwd:.6f})")

                themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>F_0 = {F0_res:,.4f}</h4>")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VALUACIÓN EN VIDA
# ─────────────────────────────────────────────────────────────────────────────
with tab_valor:
    st.markdown("### Valor de Mercado del Forward en $t$ ($f_t$)")
    themed_info(
        "Una vez firmado el contrato, su valor puede ser positivo o negativo "
        "dependiendo de cómo haya evolucionado el precio spot. "
        "Esta fórmula es la base de la contabilidad de derivados (IFRS 9 / Hedge Accounting)."
    )

    col_cap1, col_cap2 = st.columns([1, 1])
    with col_cap1:
        tipo_cap_val = st.radio(
            "Tipo de capitalización:",
            ["Continua ($e^{rT}$)", "Discreta ($(1+r)^T$)"],
            horizontal=True,
            key="val_cap"
        )
    es_cont_val = "Continua" in tipo_cap_val
    separador()

    c1, c2 = st.columns(2)

    with c1:
        St_val  = st.number_input("Precio Spot actual ($S_t$)", min_value=0.01,
                                   value=105.0, step=1.0, key="val_St")
        F0_val  = st.number_input("Precio Forward pactado originalmente ($F_0$)", min_value=0.01,
                                   value=102.0, step=1.0, key="val_F0")
        r_val   = st.number_input("Tasa libre de riesgo ($r$) %",
                                   value=5.0, step=0.1, key="val_r") / 100
        tau_val = st.number_input("Tiempo restante ($T - t$) en años",
                                   min_value=0.01, value=0.5,
                                   step=0.25, key="val_tau")
        q_val   = st.number_input("Dividendo continuo / tasa extranjera ($q$) %  "
                                   "(0 si no aplica)",
                                   value=0.0, step=0.1, key="val_q") / 100
        
        # Truco del motor
        r_calc_val = r_val if es_cont_val else np.log(1 + r_val)
        q_calc_val = q_val if es_cont_val else np.log(1 + q_val)

    with c2:
        ft_res = engine.valor_forward_en_vida(St_val, F0_val, r_calc_val, q_calc_val, tau_val)

        themed_info(f"<h3 style='margin:0; color:inherit;'>Valor del Forward (f_t): ${ft_res:,.4f}</h3>")
        
        if ft_res > 0:
            themed_success(f"La posición **LARGA (long)** gana: ${abs(ft_res):,.4f}")
        elif ft_res < 0:
            themed_error(f"La posición **LARGA (long)** pierde: ${abs(ft_res):,.4f}")
        else:
            themed_warning("El contrato está a la par (Valor = 0)")

        formula_v = r"f_t = S_t e^{-q(T-t)} - F_0 e^{-r(T-t)}" if es_cont_val else r"f_t = \frac{S_t}{(1+q)^{T-t}} - \frac{F_0}{(1+r)^{T-t}}"
        st.latex(formula_v)

    with paso_a_paso():
        st.latex(formula_v)
        if es_cont_val:
            disc_S  = St_val  * np.exp(-q_val  * tau_val)
            disc_F0 = F0_val  * np.exp(-r_val  * tau_val)
            st.latex(rf"f_t = {St_val:,.2f} e^{{-{q_val:.4f}({tau_val:.4f})}} - {F0_val:,.2f} e^{{-{r_val:.4f}({tau_val:.4f})}}")
            st.latex(rf"f_t = {St_val:,.2f} ({np.exp(-q_val * tau_val):.6f}) - {F0_val:,.2f} ({np.exp(-r_val * tau_val):.6f})")
        else:
            disc_S  = St_val  * (1 + q_val)**(-tau_val)
            disc_F0 = F0_val  * (1 + r_val)**(-tau_val)
            st.latex(rf"f_t = \frac{{{St_val:,.2f}}}{{(1 + {q_val:.4f})^{{{tau_val:.4f}}}}} - \frac{{{F0_val:,.2f}}}{{(1 + {r_val:.4f})^{{{tau_val:.4f}}}}}")
            st.latex(rf"f_t = {St_val:,.2f} ({(1+q_val)**(-tau_val):.6f}) - {F0_val:,.2f} ({(1+r_val)**(-tau_val):.6f})")
        
        st.latex(rf"f_t = {disc_S:.6f} - {disc_F0:.6f}")
        themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>f_t = {ft_res:,.4f}</h4>")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FORWARD SOBRE DIVISAS (PTCI)
# ─────────────────────────────────────────────────────────────────────────────
with tab_divisa:
    st.markdown("### Forward sobre Divisas — Paridad Cubierta de Tasas de Interés (PTCI)")
    themed_info(
        "El precio forward de una divisa se determina por la diferencia entre "
        "las tasas de interés de los dos países. Si no se cumpliera esta relación, "
        "existiría arbitraje libre de riesgo entre los mercados de dinero y divisas."
    )

    col_cap1, col_cap2 = st.columns([1, 1])
    with col_cap1:
        tipo_cap_div = st.radio(
            "Tipo de capitalización:",
            ["Continua ($e^{rT}$)", "Discreta ($(1+r)^T$)"],
            horizontal=True,
            key="div_cap"
        )
    es_cont_div = "Continua" in tipo_cap_div
    separador()

    c1, c2 = st.columns(2)

    with c1:
        S0_fx   = st.number_input("Tipo de cambio Spot ($S_0$ doméstico/extranjero)",
                                   min_value=0.0001, value=17.50, step=0.1,
                                   key="fx_s0",
                                   help="Ejemplo: MXN por USD → 17.50")
        r_d     = st.number_input("Tasa libre de riesgo doméstica ($r_d$) %",
                                   value=11.25, step=0.1, key="fx_rd") / 100
        r_f_fx  = st.number_input("Tasa libre de riesgo extranjera ($r_f$) %",
                                   value=5.25, step=0.1, key="fx_rf") / 100
        T_fx    = st.number_input("Tiempo al vencimiento ($T$) en años",
                                   min_value=0.01, value=1.0,
                                   step=0.25, key="fx_T")
        
        # Truco del motor
        r_d_calc = r_d if es_cont_div else np.log(1 + r_d)
        r_f_calc = r_f_fx if es_cont_div else np.log(1 + r_f_fx)

    with c2:
        F0_fx = engine.precio_forward_divisa(S0_fx, r_d_calc, r_f_calc, T_fx)

        prima_pct   = (F0_fx / S0_fx - 1) * 100

        themed_success(f"<h3 style='margin:0; color:inherit;'>Tipo de Cambio Forward (F₀): {F0_fx:,.4f}</h3>")
        
        st.metric("Prima / Descuento Forward vs Spot",
                  f"{prima_pct:+.2f}%",
                  help="Positivo = prima (divisa extranjera más cara a futuro), "
                       "Negativo = descuento")
        
        formula_div = r"F_0 = S_0 e^{(r_d - r_f)T}" if es_cont_div else r"F_0 = S_0 \left[ \frac{1+r_d}{1+r_f} \right]^T"
        st.latex(formula_div)

    with paso_a_paso():
        st.latex(formula_div)
        if es_cont_div:
            diferencial = r_d - r_f_fx
            st.latex(rf"F_0 = {S0_fx:.4f} e^{{({r_d:.4f} - {r_f_fx:.4f})({T_fx:.4f})}}")
            st.latex(rf"F_0 = {S0_fx:.4f} e^{{{diferencial:.6f}({T_fx:.4f})}}")
            st.latex(rf"F_0 = {S0_fx:.4f} ({np.exp(diferencial*T_fx):.6f})")
        else:
            st.latex(rf"F_0 = {S0_fx:.4f} \left[ \frac{{1 + {r_d:.4f}}}{{1 + {r_f_fx:.4f}}} \right]^{{{T_fx:.4f}}}")
            st.latex(rf"F_0 = {S0_fx:.4f} \left[ \frac{{{1+r_d:.4f}}}{{{1+r_f_fx:.4f}}} \right]^{{{T_fx:.4f}}}")
            st.latex(rf"F_0 = {S0_fx:.4f} ( {((1+r_d)/(1+r_f_fx))**T_fx:.6f} )")

        themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>F_0 = {F0_fx:.4f}</h4>")

    separador()
    with st.expander("Intuición económica de la PTCI"):
        st.markdown(
            "**¿Por qué la divisa con mayor tasa cotiza con descuento forward?**\n\n"
            "Si el peso (MXN) paga mayor tasa que el dólar (USD), "
            "un arbitrajista podría:\n\n"
            "1. Pedir USD prestados a tasa baja, convertirlos a MXN al tipo spot.\n"
            "2. Invertir los MXN a la tasa alta.\n"
            "3. Vender MXN a futuro para cubrir el riesgo cambiario.\n\n"
            "Para que este arbitraje no genere ganancias garantizadas, "
            "el MXN **debe depreciarse** a futuro exactamente en la diferencia "
            "de tasas. Eso es precisamente lo que fija la fórmula de la PTCI."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — FRA (FORWARD RATE AGREEMENT)
# ─────────────────────────────────────────────────────────────────────────────
with tab_fra:
    st.markdown("### Forward Rate Agreement (FRA)")
    themed_info(
        "Un FRA es un acuerdo para fijar hoy la tasa de interés de un préstamo "
        "o depósito que comenzará en el futuro. "
        "Se liquida en efectivo al inicio del periodo (descuento al inicio)."
    )

    col_cap1, col_cap2 = st.columns([1, 1])
    with col_cap1:
        tipo_cap_fra = st.radio(
            "Tipo de capitalización:",
            ["Continua ($e^{rT}$)", "Discreta ($(1+r)^T$)"],
            horizontal=True,
            key="fra_cap"
        )
    es_cont_fra = "Continua" in tipo_cap_fra
    separador()

    c1, c2 = st.columns(2)

    with c1:
        t1_fra   = st.number_input("Inicio del periodo ($t_1$, años desde hoy)",
                                    min_value=0.01, value=0.5,
                                    step=0.25, key="fra_t1")
        t2_fra   = st.number_input("Fin del periodo ($t_2$, años desde hoy)",
                                    min_value=0.02, value=1.0,
                                    step=0.25, key="fra_t2")
        r1_fra   = st.number_input("Tasa spot a $t_1$ ($r_1$) %",
                                    value=4.5, step=0.1, key="fra_r1") / 100
        r2_fra   = st.number_input("Tasa spot a $t_2$ ($r_2$) %",
                                    value=5.0, step=0.1, key="fra_r2") / 100
        Nf_fra   = st.number_input("Nocional del FRA ($N$)",
                                    min_value=1.0, value=1_000_000.0,
                                    step=100_000.0, key="fra_N")
        R_K      = st.number_input("Tasa pactada en el FRA ($R_K$) % anual",
                                    value=5.5, step=0.1, key="fra_rk") / 100

        # Mapeos al motor
        r1_calc_fra = r1_fra if es_cont_fra else np.log(1 + r1_fra)
        r2_calc_fra = r2_fra if es_cont_fra else np.log(1 + r2_fra)
        rk_calc_fra = R_K if es_cont_fra else np.log(1 + R_K)

    if t2_fra <= t1_fra:
        themed_error("$t_2$ debe ser mayor que $t_1$.")
    else:
        with c2:
            tau_fra = t2_fra - t1_fra
            R_F_cont, val_fra = engine.fra(r1_calc_fra, r2_calc_fra, t1_fra, t2_fra, Nf_fra, rk_calc_fra)
            
            # Reconvertir al formato de visualización si es discreta
            R_F_disp = R_F_cont if es_cont_fra else (np.exp(R_F_cont) - 1)

            themed_info(f"<h3 style='margin:0; color:inherit;'>Tasa Forward Implícita (R_F): {R_F_disp*100:.4f}%</h3>")
            
            if val_fra > 0:
                themed_success(f"<h3 style='margin:0; color:inherit;'>Valor FRA (Largo): ${val_fra:,.2f}</h3>")
                st.caption("Ganancia para el receptor de tasa fija")
            elif val_fra < 0:
                themed_error(f"<h3 style='margin:0; color:inherit;'>Valor FRA (Largo): ${val_fra:,.2f}</h3>")
                st.caption("Pérdida para el receptor de tasa fija")
            else:
                themed_warning(f"<h3 style='margin:0; color:inherit;'>Valor FRA (Largo): ${val_fra:,.2f}</h3>")
                st.caption("El contrato está a la par")

            formula_rf = r"R_F = \frac{r_2 t_2 - r_1 t_1}{t_2 - t_1}" if es_cont_fra else r"R_F = \left[ \frac{(1+r_2)^{t_2}}{(1+r_1)^{t_1}} \right]^{\frac{1}{t_2 - t_1}} - 1"
            formula_vf = r"V_{FRA} = N (R_F - R_K)(t_2 - t_1) e^{-r_2 t_2}" if es_cont_fra else r"V_{FRA} = N (R_F - R_K)(t_2 - t_1) (1+r_2)^{-t_2}"
            st.latex(formula_rf)
            st.latex(formula_vf)

        with paso_a_paso():
            st.latex(formula_rf)
            if es_cont_fra:
                num_rf = r2_fra * t2_fra - r1_fra * t1_fra
                st.latex(rf"R_F = \frac{{{r2_fra:.4f}({t2_fra:.4f}) - {r1_fra:.4f}({t1_fra:.4f})}}{{{t2_fra:.4f} - {t1_fra:.4f}}}")
                st.latex(rf"R_F = \frac{{{r2_fra*t2_fra:.6f} - {r1_fra*t1_fra:.6f}}}{{{tau_fra:.4f}}}")
                st.latex(rf"R_F = \frac{{{num_rf:.6f}}}{{{tau_fra:.4f}}} = {R_F_disp:.6f}")
            else:
                st.latex(rf"R_F = \left[ \frac{{(1 + {r2_fra:.4f})^{{{t2_fra:.4f}}}}}{{(1 + {r1_fra:.4f})^{{{t1_fra:.4f}}}}} \right]^{{\frac{{1}}{{{tau_fra:.4f}}}}} - 1")
                st.latex(rf"R_F = \left[ \frac{{{(1+r2_fra)**t2_fra:.6f}}}{{{(1+r1_fra)**t1_fra:.6f}}} \right]^{{{1/tau_fra:.4f}}} - 1")
                st.latex(rf"R_F = \left[ {((1+r2_fra)**t2_fra) / ((1+r1_fra)**t1_fra):.6f} \right]^{{{1/tau_fra:.4f}}} - 1 = {R_F_disp:.6f}")

            st.write("---")

            st.latex(formula_vf)
            diff_R  = R_F_disp - R_K
            if es_cont_fra:
                factor  = np.exp(-r2_fra * t2_fra)
                st.latex(rf"V_{{FRA}} = {Nf_fra:,.0f} ({R_F_disp:.6f} - {R_K:.6f}) ({tau_fra:.4f}) e^{{-{r2_fra:.4f}({t2_fra:.4f})}}")
                st.latex(rf"V_{{FRA}} = {Nf_fra:,.0f} ({diff_R:.6f}) ({tau_fra:.4f}) ({factor:.6f})")
            else:
                factor = (1 + r2_fra)**(-t2_fra)
                st.latex(rf"V_{{FRA}} = {Nf_fra:,.0f} ({R_F_disp:.6f} - {R_K:.6f}) ({tau_fra:.4f}) (1 + {r2_fra:.4f})^{{-{t2_fra:.4f}}}")
                st.latex(rf"V_{{FRA}} = {Nf_fra:,.0f} ({diff_R:.6f}) ({tau_fra:.4f}) ({factor:.6f})")

            st.latex(rf"V_{{FRA}} = {val_fra:,.2f}")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>V_{{FRA}} = ${val_fra:,.2f}</h4>")