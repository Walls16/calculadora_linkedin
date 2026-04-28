"""
pages/3_Rentas.py
-----------------
Módulo 3: Valuación de Rentas y Anualidades.
Cubre:
  - Valor Futuro: constantes, geométricas, aritméticas
  - Valor Presente: constantes, geométricas, aritméticas (+ perpetuas)
  - Número de Periodos (n): analítico y numérico
"""

import numpy as np
import streamlit as st

from utils import get_engine, page_header, paso_a_paso, separador, alerta_metodo_numerico, themed_info, themed_success, themed_warning, themed_error

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Rentas y Anualidades · Calculadora Financiera",
    page_icon="💰",
    layout="wide",
)

engine = get_engine()

page_header(
    titulo="3. Valuación de Rentas y Anualidades",
    subtitulo="Constantes · Geométricas · Aritméticas · Perpetuas · Continuas"
)

# =============================================================================
# PESTAÑAS PRINCIPALES
# =============================================================================
tab_vf, tab_vp, tab_n = st.tabs([
    "Valor Futuro de Rentas",
    "Valor Presente de Rentas",
    "Número de Periodos (n)",
])

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS (Para no repetir cajas de texto)
# ─────────────────────────────────────────────────────────────────────────────

def _inputs_tasa_efectiva(sufijo: str):
    im = st.number_input("Tasa efectiva periódica ($i_m$) %", value=1.0, step=0.1, key=f"im_eff_{sufijo}") / 100
    n  = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key=f"n_eff_{sufijo}")
    m  = st.number_input("Periodos por año ($m$)", min_value=1.0, value=12.0, step=1.0, key=f"m_eff_{sufijo}")
    return im, n * m, m

def _inputs_tasa_nominal(sufijo: str):
    i_nom = st.number_input("Tasa nominal anual ($i^{(m)}$) %", value=12.0, step=0.1, key=f"inom_{sufijo}") / 100
    n     = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key=f"n_nom_{sufijo}")
    m     = st.number_input("Periodos por año ($m$)", min_value=1.0, value=12.0, step=1.0, key=f"m_nom_{sufijo}")
    return i_nom / m, n * m, m, i_nom

def _inputs_tasa_efectiva_pq(sufijo: str):
    im = st.number_input("Tasa efec. interés ($i_m$) %", value=1.0, step=0.1, key=f"im_geo_{sufijo}") / 100
    qm = st.number_input("Tasa efec. crecimiento ($q_m$) %", value=0.5, step=0.1, key=f"qm_geo_{sufijo}") / 100
    n  = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key=f"n_geo_{sufijo}")
    m  = st.number_input("Periodos por año ($m$)", min_value=1.0, value=12.0, step=1.0, key=f"m_geo_{sufijo}")
    return im, qm, n * m, m

def _inputs_tasa_nominal_pq(sufijo: str):
    i_nom = st.number_input("Tasa nom. interés ($i^{(m)}$) %", value=12.0, step=0.1, key=f"inom_geo_{sufijo}") / 100
    q_nom = st.number_input("Tasa nom. crecimiento ($q^{(m)}$) %", value=5.0, step=0.1, key=f"qnom_geo_{sufijo}") / 100
    n     = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key=f"n_ngeo_{sufijo}")
    m     = st.number_input("Periodos por año ($m$)", min_value=1.0, value=12.0, step=1.0, key=f"m_ngeo_{sufijo}")
    return i_nom / m, q_nom / m, n * m, m


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — VALOR FUTURO
# ═════════════════════════════════════════════════════════════════════════════
with tab_vf:
    st.markdown("### Valor futuro de Rentas y Anualidades")
    themed_success(
        "El **Valor Futuro (Monto Acumulado)** determina la capitalización total al final de un plazo si "
        "realizas depósitos periódicos secuenciales (<span style='font-family: serif; font-style: italic;'>R</span>) "
        "y reinviertes los intereses generados a una tasa específica."
    )
    
    tipo_vf = st.radio("Tipo de Renta:", ["Constantes Periódicas", "Crecientes Geométricas", "Crecientes Aritméticas"], horizontal=True, key="radio_tipo_vf")
    separador()

    # ──────────────────────────────────────────────
    # VF — CONSTANTES
    # ──────────────────────────────────────────────
    if tipo_vf == "Constantes Periódicas":
        escenario = st.selectbox("Selecciona el escenario de capitalización:", [
            "Vencidas a tasa efectiva im",
            "Anticipadas a tasa efectiva im",
            "Vencidas a tasa nominal i(m)",
            "Anticipadas a tasa nominal i(m)",
            "Vencidas pagaderas p veces al año a tasa nominal i(m)",
            "Continuas a tasa instantánea δ",
        ], key="sel_const_vf")

        c1, c2 = st.columns(2)
        with c1:
            R_vf = st.number_input("Pago periódico ($R$)", min_value=0.0, value=1_000.0, step=100.0, key="R_vf_const")

            if escenario == "Vencidas a tasa efectiva im":
                im_vf, nm_vf, _ = _inputs_tasa_efectiva("vf_vec_e")
                vf_res   = engine.vf_anualidad_efectiva(R_vf, im_vf, nm_vf, anticipada=False)
                formula  = r"VF = R \left[ \frac{(1+i_m)^{nm} - 1}{i_m} \right]"

            elif escenario == "Anticipadas a tasa efectiva im":
                im_vf, nm_vf, _ = _inputs_tasa_efectiva("vf_ant_e")
                vf_res  = engine.vf_anualidad_efectiva(R_vf, im_vf, nm_vf, anticipada=True)
                formula = r"VF = R \left[ \frac{(1+i_m)^{nm} - 1}{i_m} \right](1+i_m)"

            elif escenario == "Vencidas a tasa nominal i(m)":
                im_vf, nm_vf, m_cap, i_nom_vf = _inputs_tasa_nominal("vf_vec_n")
                vf_res   = engine.vf_anualidad_efectiva(R_vf, im_vf, nm_vf, anticipada=False)
                formula  = r"VF = R \left[ \frac{\left(1+\frac{i^{(m)}}{m}\right)^{nm} - 1}{\frac{i^{(m)}}{m}} \right]"

            elif escenario == "Anticipadas a tasa nominal i(m)":
                im_vf, nm_vf, m_cap, i_nom_vf = _inputs_tasa_nominal("vf_ant_n")
                vf_res  = engine.vf_anualidad_efectiva(R_vf, im_vf, nm_vf, anticipada=True)
                formula = r"VF = R \left[ \frac{\left(1+\frac{i^{(m)}}{m}\right)^{nm} - 1}{\frac{i^{(m)}}{m}} \right]\left(1+\frac{i^{(m)}}{m}\right)"

            elif escenario == "Vencidas pagaderas p veces al año a tasa nominal i(m)":
                i_nom_vf = st.number_input("Tasa nominal ($i^{(m)}$) %", value=12.0, step=0.1, key="inom_vf_p") / 100
                m_cap    = st.number_input("Capitalizaciones por año ($m$)", min_value=1.0, value=12.0, step=1.0, key="mcap_vf")
                p_pag    = st.number_input("Pagos por año ($p$)", min_value=1.0, value=4.0,  step=1.0, key="p_vf")
                n_anios  = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key="n_vf_p")
                vf_res  = engine.vf_anualidad_nominal(R_vf, i_nom_vf, m_cap, p_pag, n_anios)
                formula = r"VF = R \left[ \frac{(1+i_p)^{np} - 1}{i_p} \right]"
                i_p     = engine.tasa_nominal_m_a_nominal_p(i_nom_vf, m_cap, p_pag) / p_pag
                im_vf, nm_vf = i_p, n_anios * p_pag

            else:  # Continuas
                R_anual = st.number_input("Flujo anual total ($\\bar{R}$)", min_value=0.0, value=12_000.0, step=1_000.0, key="R_cont_vf")
                tipo_t  = st.radio("Ingresar tasa como:", ["Tasa instantánea (δ)", "Tasa efectiva anual (i)"], horizontal=True, key="tipo_cont_vf")
                n_cont  = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key="n_cont_vf")

                if tipo_t == "Tasa instantánea (δ)":
                    delta_vf = st.number_input("δ %", value=10.0, step=0.1, key="delta_vf") / 100
                    vf_res   = engine.vf_anualidad_continua(R_anual, delta_vf, n_cont)
                    formula  = r"VF = \bar{R} \left[ \frac{e^{\delta n} - 1}{\delta} \right]"
                else:
                    i_eff_vf = st.number_input("i %", value=10.51, step=0.1, key="ieff_vf") / 100
                    delta_vf = np.log(1 + i_eff_vf)
                    vf_res   = engine.vf_anualidad_continua(R_anual, delta_vf, n_cont)
                    formula  = r"VF = \bar{R} \left[ \frac{(1+i)^n - 1}{\ln(1+i)} \right]"
                R_vf, im_vf, nm_vf = R_anual, delta_vf, n_cont

        with c2:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Valor Futuro: ${vf_res:,.4f}</h3>")
            st.latex(formula)

        with paso_a_paso():
            st.latex(formula)
            if "nominal i(m)" in escenario and "pagaderas" not in escenario:
                anticipada_str = rf"\left(1+\frac{{{i_nom_vf:.4f}}}{{{m_cap:g}}}\right)" if "Anticipadas" in escenario else ""
                st.latex(rf"VF = {R_vf:,.2f} \left[ \frac{{\left(1+\frac{{{i_nom_vf:.4f}}}{{{m_cap:g}}}\right)^{{{nm_vf:g}}} - 1}}{{\frac{{{i_nom_vf:.4f}}}{{{m_cap:g}}}}} \right]" + anticipada_str)
                anticipada_str2 = rf"(1+{im_vf:.6f})" if "Anticipadas" in escenario else ""
                st.latex(rf"VF = {R_vf:,.2f} \left[ \frac{{(1+{im_vf:.6f})^{{{nm_vf:g}}} - 1}}{{{im_vf:.6f}}} \right]" + anticipada_str2)
                
                cap_n = (1 + im_vf)**nm_vf
                factor = (cap_n - 1) / im_vf
                st.latex(rf"VF = {R_vf:,.2f} \left[ \frac{{{cap_n:.6f} - 1}}{{{im_vf:.6f}}} \right]" + anticipada_str2)
                st.latex(rf"VF = {R_vf:,.2f} [{factor:.6f}]" + anticipada_str2)

            elif escenario in ("Vencidas a tasa efectiva im", "Anticipadas a tasa efectiva im", "Vencidas pagaderas p veces al año a tasa nominal i(m)"):
                if "pagaderas" in escenario:
                    st.latex(rf"i_p = \left(1 + \frac{{{i_nom_vf:.4f}}}{{{m_cap:g}}}\right)^{{\frac{{{m_cap:g}}}{{{p_pag:g}}}}} - 1 = {i_p:.6f}")
                    st.latex(rf"np = {n_anios:g} \times {p_pag:g} = {nm_vf:g}")
                    st.write("---")
                
                cap_n = (1 + im_vf)**nm_vf
                factor = (cap_n - 1) / im_vf
                
                if "Anticipadas" in escenario:
                    st.latex(rf"VF = {R_vf:,.2f} \left[ \frac{{(1 + {im_vf:.6f})^{{{nm_vf:g}}} - 1}}{{{im_vf:.6f}}} \right] (1 + {im_vf:.6f})")
                    st.latex(rf"VF = {R_vf:,.2f} \left[ \frac{{{cap_n:.6f} - 1}}{{{im_vf:.6f}}} \right] ({1+im_vf:.6f})")
                    st.latex(rf"VF = {R_vf:,.2f} [{factor:.6f}] ({1+im_vf:.6f})")
                else:
                    st.latex(rf"VF = {R_vf:,.2f} \left[ \frac{{(1 + {im_vf:.6f})^{{{nm_vf:g}}} - 1}}{{{im_vf:.6f}}} \right]")
                    st.latex(rf"VF = {R_vf:,.2f} \left[ \frac{{{cap_n:.6f} - 1}}{{{im_vf:.6f}}} \right]")
                    st.latex(rf"VF = {R_vf:,.2f} [{factor:.6f}]")
                    
            else: # Continuas
                if tipo_t == "Tasa instantánea (δ)":
                    cap_n = np.exp(delta_vf * n_cont)
                    factor = (cap_n - 1) / delta_vf
                    st.latex(rf"VF = {R_anual:,.2f} \left[ \frac{{e^{{({delta_vf:.4f} \times {n_cont:g})}} - 1}}{{{delta_vf:.4f}}} \right]")
                    st.latex(rf"VF = {R_anual:,.2f} \left[ \frac{{{cap_n:.6f} - 1}}{{{delta_vf:.4f}}} \right]")
                    st.latex(rf"VF = {R_anual:,.2f} [{factor:.6f}]")
                else:
                    st.latex(rf"\delta = \ln(1 + {i_eff_vf:.4f}) = {delta_vf:.6f}")
                    st.write("---")
                    cap_n = (1 + i_eff_vf)**n_cont
                    factor = (cap_n - 1) / delta_vf
                    st.latex(rf"VF = {R_anual:,.2f} \left[ \frac{{(1 + {i_eff_vf:.4f})^{{{n_cont:g}}} - 1}}{{{delta_vf:.6f}}} \right]")
                    st.latex(rf"VF = {R_anual:,.2f} \left[ \frac{{{cap_n:.6f} - 1}}{{{delta_vf:.6f}}} \right]")
                    st.latex(rf"VF = {R_anual:,.2f} [{factor:.6f}]")
                    
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>VF = ${vf_res:,.4f}</h4>")

    # ──────────────────────────────────────────────
    # VF — GEOMÉTRICAS
    # ──────────────────────────────────────────────
    elif tipo_vf == "Crecientes Geométricas":
        st.markdown("#### Rentas que crecen a una tasa porcentual constante $q_m$")
        tipo_t_geo = st.radio("Ingresar tasas como:", ["Tasa efectiva periódica", "Tasa nominal anual"], horizontal=True, key="tipo_t_geo_vf")
        separador()

        c1, c2 = st.columns(2)
        with c1:
            R1_vf = st.number_input("Primer pago ($R_1$)", min_value=0.0, value=1_000.0, step=100.0, key="R1_geo_vf")

            if tipo_t_geo == "Tasa efectiva periódica":
                im_geo, qm_geo, nm_geo, m_geo = _inputs_tasa_efectiva_pq("vf_e")
                str_i, str_q = r"i_m", r"q_m"
                val_i, val_q = f"{im_geo:.4f}", f"{qm_geo:.4f}"
            else:
                im_geo, qm_geo, nm_geo, m_geo = _inputs_tasa_nominal_pq("vf_n")
                str_i, str_q = r"\frac{i^{(m)}}{m}", r"\frac{q^{(m)}}{m}"
                val_i, val_q = f"{im_geo:.6f}", f"{qm_geo:.6f}"

        vf_geo = engine.vf_gradiente_geo(R1_vf, im_geo, qm_geo, nm_geo)

        with c2:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Valor Futuro: ${vf_geo:,.4f}</h3>")
            if im_geo != qm_geo:
                st.latex(rf"VF = R_1 \left[ \frac{{(1+{str_i})^{{nm}} - (1+{str_q})^{{nm}}}}{{{str_i} - {str_q}}} \right]")
            else:
                st.latex(rf"VF = nm \cdot R_1 (1+{str_i})^{{nm-1}}")

        with paso_a_paso():
            if im_geo != qm_geo:
                st.latex(rf"VF = R_1 \left[ \frac{{(1+{str_i})^{{nm}} - (1+{str_q})^{{nm}}}}{{{str_i} - {str_q}}} \right]")
                num1 = (1 + im_geo)**nm_geo
                num2 = (1 + qm_geo)**nm_geo
                den  = im_geo - qm_geo
                st.latex(rf"VF = {R1_vf:,.2f} \left[ \frac{{(1 + {val_i})^{{{nm_geo:g}}} - (1 + {val_q})^{{{nm_geo:g}}}}}{{{val_i} - {val_q}}} \right]")
                st.latex(rf"VF = {R1_vf:,.2f} \left[ \frac{{{num1:.6f} - {num2:.6f}}}{{{den:.6f}}} \right]")
                st.latex(rf"VF = {R1_vf:,.2f} [{((num1-num2)/den):.6f}]")
            else:
                st.latex(rf"VF = nm \cdot R_1 (1+{str_i})^{{nm-1}}")
                st.latex(rf"VF = {nm_geo:g} \times {R1_vf:,.2f} \times (1 + {val_i})^{{{nm_geo:g}-1}}")
                st.latex(rf"VF = {nm_geo * R1_vf:,.2f} \times {(1+im_geo)**(nm_geo-1):.6f}")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>VF = ${vf_geo:,.4f}</h4>")

    # ──────────────────────────────────────────────
    # VF — ARITMÉTICAS
    # ──────────────────────────────────────────────
    else:
        st.markdown("#### Rentas que crecen sumando una cantidad fija $G$ por periodo")
        tipo_t_arit = st.radio("Ingresar tasa como:", ["Tasa efectiva periódica", "Tasa nominal anual"], horizontal=True, key="tipo_t_arit_vf")
        separador()

        c1, c2 = st.columns(2)
        with c1:
            R1_arit = st.number_input("Primer pago ($R_1$)", min_value=0.0, value=1_000.0, step=100.0, key="R1_arit_vf")
            G_vf    = st.number_input("Gradiente ($G$)", value=100.0, step=50.0, key="G_vf")

            if tipo_t_arit == "Tasa efectiva periódica":
                im_arit, nm_arit, _ = _inputs_tasa_efectiva("arit_vf_e")
                str_i_a = r"i_m"
                val_i_a = f"{im_arit:.4f}"
            else:
                im_arit, nm_arit, _, _ = _inputs_tasa_nominal("arit_vf_n")
                str_i_a = r"\frac{i^{(m)}}{m}"
                val_i_a = f"{im_arit:.6f}"

        vf_arit = engine.vf_gradiente_aritmetico(R1_arit, G_vf, im_arit, nm_arit)

        with c2:
            themed_success(f"<h3 style='margin:0; color:inherit;'>Valor Futuro: ${vf_arit:,.4f}</h3>")
            st.latex(rf"VF = R_1 \left[\frac{{(1+{str_i_a})^{{nm}}-1}}{{{str_i_a}}}\right] + \frac{{G}}{{{str_i_a}}}\left(\left[\frac{{(1+{str_i_a})^{{nm}}-1}}{{{str_i_a}}}\right] - nm\right)")

        with paso_a_paso():
            st.latex(rf"VF = R_1 \left[\frac{{(1+{str_i_a})^{{nm}}-1}}{{{str_i_a}}}\right] + \frac{{G}}{{{str_i_a}}}\left(\left[\frac{{(1+{str_i_a})^{{nm}}-1}}{{{str_i_a}}}\right] - nm\right)")
            sn = ((1 + im_arit)**nm_arit - 1) / im_arit
            t1_val = R1_arit * sn
            t2_val = (G_vf / im_arit) * (sn - nm_arit)
            cap_n = (1 + im_arit)**nm_arit

            st.latex(rf"VF = {R1_arit:,.2f} \left[\frac{{(1+{val_i_a})^{{{nm_arit:g}}}-1}}{{{val_i_a}}}\right] + \frac{{{G_vf:,.2f}}}{{{val_i_a}}}\left(\left[\frac{{(1+{val_i_a})^{{{nm_arit:g}}}-1}}{{{val_i_a}}}\right] - {nm_arit:g}\right)")
            st.latex(rf"VF = {R1_arit:,.2f} \left[\frac{{{cap_n:.6f}-1}}{{{val_i_a}}}\right] + {G_vf/im_arit:,.2f} \left(\left[\frac{{{cap_n:.6f}-1}}{{{val_i_a}}}\right] - {nm_arit:g}\right)")
            st.latex(rf"VF = {R1_arit:,.2f} [{sn:.6f}] + {G_vf/im_arit:,.2f} ({sn:.6f} - {nm_arit:g})")
            st.latex(rf"VF = {t1_val:,.2f} + {t2_val:,.2f}")
            themed_success(f"<h4 style='margin:0; color:inherit; text-align:center;'>VF = ${vf_arit:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALOR PRESENTE
# ═════════════════════════════════════════════════════════════════════════════
with tab_vp:
    st.markdown("### Valor presente de Rentas y Anualidades")
    themed_info(
        "El **Valor Presente (Capital)** determina matemáticamente el fondeo necesario hoy "
        "para sostener una serie de retiros periódicos (<span style='font-family: serif; font-style: italic;'>R</span>) "
        "en el futuro hasta agotar el principal, descontando el efecto temporal a una tasa de interés de mercado."
    )
    
    tipo_vp = st.radio("Tipo de Renta:", ["Constantes Periódicas", "Crecientes Geométricas", "Crecientes Aritméticas"], horizontal=True, key="radio_tipo_vp")
    separador()

    # ──────────────────────────────────────────────
    # VP — CONSTANTES
    # ──────────────────────────────────────────────
    if tipo_vp == "Constantes Periódicas":
        escenario_vp = st.selectbox("Selecciona el escenario de actualización:", [
            "Vencidas a tasa efectiva im",
            "Anticipadas a tasa efectiva im",
            "Vencidas a tasa nominal i(m)",
            "Anticipadas a tasa nominal i(m)",
            "Perpetuas a tasa efectiva im",
            "Perpetuas a tasa nominal i(m)",
            "Vencidas pagaderas p veces al año a tasa nominal i(m)",
            "Continuas a tasa instantánea δ o efectiva i",
        ], key="sel_const_vp")

        c1, c2 = st.columns(2)
        with c1:
            R_vp = st.number_input("Pago periódico ($R$)", min_value=0.0, value=1_000.0, step=100.0, key="R_vp_const")

            if escenario_vp == "Vencidas a tasa efectiva im":
                im_vp, nm_vp, _ = _inputs_tasa_efectiva("vp_vec_e")
                vp_res  = engine.vp_anualidad_efectiva(R_vp, im_vp, nm_vp, anticipada=False)
                formula_vp = r"VP = R \left[ \frac{1 - (1+i_m)^{-nm}}{i_m} \right]"

            elif escenario_vp == "Anticipadas a tasa efectiva im":
                im_vp, nm_vp, _ = _inputs_tasa_efectiva("vp_ant_e")
                vp_res  = engine.vp_anualidad_efectiva(R_vp, im_vp, nm_vp, anticipada=True)
                formula_vp = r"VP = R \left[ \frac{1 - (1+i_m)^{-nm}}{i_m} \right](1+i_m)"

            elif escenario_vp == "Vencidas a tasa nominal i(m)":
                im_vp, nm_vp, m_cap, i_nom_vp = _inputs_tasa_nominal("vp_vec_n")
                vp_res  = engine.vp_anualidad_efectiva(R_vp, im_vp, nm_vp, anticipada=False)
                formula_vp = r"VP = R \left[ \frac{1 - \left(1+\frac{i^{(m)}}{m}\right)^{-nm}}{\frac{i^{(m)}}{m}} \right]"

            elif escenario_vp == "Anticipadas a tasa nominal i(m)":
                im_vp, nm_vp, m_cap, i_nom_vp = _inputs_tasa_nominal("vp_ant_n")
                vp_res  = engine.vp_anualidad_efectiva(R_vp, im_vp, nm_vp, anticipada=True)
                formula_vp = r"VP = R \left[ \frac{1 - \left(1+\frac{i^{(m)}}{m}\right)^{-nm}}{\frac{i^{(m)}}{m}} \right]\left(1+\frac{i^{(m)}}{m}\right)"

            elif escenario_vp == "Perpetuas a tasa efectiva im":
                im_vp = st.number_input("Tasa efectiva periódica ($i_m$) %", value=1.0, step=0.1, key="im_perp_e") / 100
                nm_vp = 0
                vp_res   = engine.vp_perpetuidad(R_vp, im_vp)
                formula_vp = r"VP = \frac{R}{i_m}"

            elif escenario_vp == "Perpetuas a tasa nominal i(m)":
                i_nom_pp = st.number_input("Tasa nominal ($i^{(m)}$) %", value=12.0, step=0.1, key="inom_perp_n") / 100
                m_pp     = st.number_input("Periodos por año ($m$)", min_value=1.0, value=12.0, step=1.0, key="m_perp_n")
                im_vp, nm_vp = i_nom_pp / m_pp, 0
                vp_res   = engine.vp_perpetuidad(R_vp, im_vp)
                formula_vp = r"VP = \frac{R}{\frac{i^{(m)}}{m}}"

            elif escenario_vp == "Vencidas pagaderas p veces al año a tasa nominal i(m)":
                i_nom_vp2 = st.number_input("Tasa nominal ($i^{(m)}$) %", value=12.0, step=0.1, key="inom_vp_p") / 100
                m_cap_vp  = st.number_input("Capitalizaciones por año ($m$)", min_value=1.0, value=12.0, step=1.0, key="mcap_vp")
                p_pag_vp  = st.number_input("Pagos por año ($p$)", min_value=1.0, value=4.0,  step=1.0, key="p_vp")
                n_anios_vp = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key="n_vp_p")
                vp_res    = engine.vp_anualidad_nominal(R_vp, i_nom_vp2, m_cap_vp, p_pag_vp, n_anios_vp)
                formula_vp = r"VP = R \left[ \frac{1 - (1+i_p)^{-np}}{i_p} \right]"
                i_p_vp    = engine.tasa_nominal_m_a_nominal_p(i_nom_vp2, m_cap_vp, p_pag_vp) / p_pag_vp
                im_vp, nm_vp = i_p_vp, n_anios_vp * p_pag_vp

            else:  # Continuas
                R_anual_vp = st.number_input("Flujo anual total ($\\bar{R}$)", min_value=0.0, value=12_000.0, step=1_000.0, key="R_cont_vp")
                tipo_t_vp  = st.radio("Ingresar tasa como:", ["Tasa instantánea (δ)", "Tasa efectiva anual (i)"], horizontal=True, key="tipo_cont_vp")
                n_cont_vp  = st.number_input("Años ($n$)", min_value=0.0, value=5.0, step=1.0, key="n_cont_vp")

                if tipo_t_vp == "Tasa instantánea (δ)":
                    delta_vp = st.number_input("δ %", value=10.0, step=0.1, key="delta_vp") / 100
                    vp_res   = engine.vp_anualidad_continua(R_anual_vp, delta_vp, n_cont_vp)
                    formula_vp = r"VP = \bar{R} \left[ \frac{1 - e^{-\delta n}}{\delta} \right]"
                else:
                    i_eff_vp = st.number_input("i %", value=10.51, step=0.1, key="ieff_vp") / 100
                    delta_vp = np.log(1 + i_eff_vp)
                    vp_res   = engine.vp_anualidad_continua(R_anual_vp, delta_vp, n_cont_vp)
                    formula_vp = r"VP = \bar{R} \left[ \frac{1 - (1+i)^{-n}}{\ln(1+i)} \right]"
                R_vp, im_vp, nm_vp = R_anual_vp, delta_vp, n_cont_vp

        with c2:
            themed_info(f"<h3 style='margin:0; color:inherit;'>Valor Presente: ${vp_res:,.4f}</h3>")
            st.latex(formula_vp)

        with paso_a_paso():
            st.latex(formula_vp)

            if "nominal i(m)" in escenario_vp and "pagaderas" not in escenario_vp and "Perpetuas" not in escenario_vp:
                anticipada_str = rf"\left(1+\frac{{{i_nom_vp:.4f}}}{{{m_cap:g}}}\right)" if "Anticipadas" in escenario_vp else ""
                st.latex(rf"VP = {R_vp:,.2f} \left[ \frac{{1 - \left(1+\frac{{{i_nom_vp:.4f}}}{{{m_cap:g}}}\right)^{{-{nm_vp:g}}}}}{{\frac{{{i_nom_vp:.4f}}}{{{m_cap:g}}}}} \right]" + anticipada_str)
                anticipada_str2 = rf"(1+{im_vp:.6f})" if "Anticipadas" in escenario_vp else ""
                st.latex(rf"VP = {R_vp:,.2f} \left[ \frac{{1 - (1+{im_vp:.6f})^{{-{nm_vp:g}}}}}{{{im_vp:.6f}}} \right]" + anticipada_str2)
                
                desc_n = (1 + im_vp)**(-nm_vp)
                factor = (1 - desc_n) / im_vp
                st.latex(rf"VP = {R_vp:,.2f} \left[ \frac{{1 - {desc_n:.6f}}}{{{im_vp:.6f}}} \right]" + anticipada_str2)
                st.latex(rf"VP = {R_vp:,.2f} [{factor:.6f}]" + anticipada_str2)

            elif escenario_vp in ("Vencidas a tasa efectiva im", "Anticipadas a tasa efectiva im", "Vencidas pagaderas p veces al año a tasa nominal i(m)"):
                if "pagaderas" in escenario_vp:
                    st.latex(rf"i_p = \left(1 + \frac{{{i_nom_vp2:.4f}}}{{{m_cap_vp:g}}}\right)^{{\frac{{{m_cap_vp:g}}}{{{p_pag_vp:g}}}}} - 1 = {i_p_vp:.6f}")
                    st.latex(rf"np = {n_anios_vp:g} \times {p_pag_vp:g} = {nm_vp:g}")
                    st.write("---")

                desc_n = (1 + im_vp)**(-nm_vp)
                factor = (1 - desc_n) / im_vp

                if "Anticipadas" in escenario_vp:
                    st.latex(rf"VP = {R_vp:,.2f} \left[ \frac{{1 - (1 + {im_vp:.6f})^{{-{nm_vp:g}}}}}{{{im_vp:.6f}}} \right] (1 + {im_vp:.6f})")
                    st.latex(rf"VP = {R_vp:,.2f} \left[ \frac{{1 - {desc_n:.6f}}}{{{im_vp:.6f}}} \right] ({1+im_vp:.6f})")
                    st.latex(rf"VP = {R_vp:,.2f} [{factor:.6f}] ({1+im_vp:.6f})")
                else:
                    st.latex(rf"VP = {R_vp:,.2f} \left[ \frac{{1 - (1 + {im_vp:.6f})^{{-{nm_vp:g}}}}}{{{im_vp:.6f}}} \right]")
                    st.latex(rf"VP = {R_vp:,.2f} \left[ \frac{{1 - {desc_n:.6f}}}{{{im_vp:.6f}}} \right]")
                    st.latex(rf"VP = {R_vp:,.2f} [{factor:.6f}]")

            elif escenario_vp == "Perpetuas a tasa efectiva im":
                st.latex(rf"VP = \frac{{{R_vp:,.2f}}}{{{im_vp:.6f}}}")

            elif escenario_vp == "Perpetuas a tasa nominal i(m)":
                st.latex(rf"VP = \frac{{{R_vp:,.2f}}}{{\frac{{{i_nom_pp:.4f}}}{{{m_pp:g}}}}}")
                st.latex(rf"VP = \frac{{{R_vp:,.2f}}}{{{im_vp:.6f}}}")
            
            else: # Continuas
                if tipo_t_vp == "Tasa instantánea (δ)":
                    desc_n = np.exp(-delta_vp * n_cont_vp)
                    factor = (1 - desc_n) / delta_vp
                    st.latex(rf"VP = {R_anual_vp:,.2f} \left[ \frac{{1 - e^{{-{delta_vp:.4f} \times {n_cont_vp:g}}}}}{{{delta_vp:.4f}}} \right]")
                    st.latex(rf"VP = {R_anual_vp:,.2f} \left[ \frac{{1 - {desc_n:.6f}}}{{{delta_vp:.4f}}} \right]")
                    st.latex(rf"VP = {R_anual_vp:,.2f} [{factor:.6f}]")
                else:
                    st.latex(rf"\delta = \ln(1 + {i_eff_vp:.4f}) = {delta_vp:.6f}")
                    st.write("---")
                    desc_n = (1 + i_eff_vp)**(-n_cont_vp)
                    factor = (1 - desc_n) / delta_vp
                    st.latex(rf"VP = {R_anual_vp:,.2f} \left[ \frac{{1 - (1 + {i_eff_vp:.4f})^{{-{n_cont_vp:g}}}}}{{{delta_vp:.6f}}} \right]")
                    st.latex(rf"VP = {R_anual_vp:,.2f} \left[ \frac{{1 - {desc_n:.6f}}}{{{delta_vp:.6f}}} \right]")
                    st.latex(rf"VP = {R_anual_vp:,.2f} [{factor:.6f}]")
            
            themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>VP = ${vp_res:,.4f}</h4>")

    # ──────────────────────────────────────────────
    # VP — GEOMÉTRICAS
    # ──────────────────────────────────────────────
    elif tipo_vp == "Crecientes Geométricas":
        st.markdown("#### Rentas que crecen a una tasa porcentual constante $q_m$")
        tipo_t_geo_vp = st.radio("Ingresar tasas como:", ["Tasa efectiva periódica", "Tasa nominal anual"], horizontal=True, key="tipo_t_geo_vp")
        separador()

        c1, c2 = st.columns(2)
        with c1:
            R1_vp = st.number_input("Primer pago ($R_1$)", min_value=0.0, value=1_000.0, step=100.0, key="R1_geo_vp")

            if tipo_t_geo_vp == "Tasa efectiva periódica":
                im_geo_vp, qm_geo_vp, nm_geo_vp, _ = _inputs_tasa_efectiva_pq("vp_e")
                str_i_gvp, str_q_gvp = r"i_m", r"q_m"
                val_i_gvp, val_q_gvp = f"{im_geo_vp:.4f}", f"{qm_geo_vp:.4f}"
            else:
                im_geo_vp, qm_geo_vp, nm_geo_vp, _ = _inputs_tasa_nominal_pq("vp_n")
                str_i_gvp, str_q_gvp = r"\frac{i^{(m)}}{m}", r"\frac{q^{(m)}}{m}"
                val_i_gvp, val_q_gvp = f"{im_geo_vp:.6f}", f"{qm_geo_vp:.6f}"

        vp_geo = engine.vp_gradiente_geo(R1_vp, im_geo_vp, qm_geo_vp, nm_geo_vp)

        with c2:
            themed_info(f"<h3 style='margin:0; color:inherit;'>Valor Presente: ${vp_geo:,.4f}</h3>")
            if im_geo_vp != qm_geo_vp:
                st.latex(rf"VP = R_1 \left[ \frac{{1 - \left(\frac{{1+{str_q_gvp}}}{{1+{str_i_gvp}}}\right)^{{nm}}}}{{{str_i_gvp} - {str_q_gvp}}} \right]")
            else:
                st.latex(rf"VP = \frac{{nm \cdot R_1}}{{1+{str_i_gvp}}}")

        with paso_a_paso():
            if im_geo_vp != qm_geo_vp:
                st.latex(rf"VP = R_1 \left[ \frac{{1 - \left(\frac{{1+{str_q_gvp}}}{{1+{str_i_gvp}}}\right)^{{nm}}}}{{{str_i_gvp} - {str_q_gvp}}} \right]")
                frac = (1 + qm_geo_vp) / (1 + im_geo_vp)
                den  = im_geo_vp - qm_geo_vp
                factor_geo = (1 - frac**nm_geo_vp) / den

                st.latex(rf"VP = {R1_vp:,.2f} \left[ \frac{{1 - \left(\frac{{1+{val_q_gvp}}}{{1+{val_i_gvp}}}\right)^{{{nm_geo_vp:g}}}}}{{{val_i_gvp} - {val_q_gvp}}} \right]")
                st.latex(rf"VP = {R1_vp:,.2f} \left[ \frac{{1 - ({frac:.6f})^{{{nm_geo_vp:g}}}}}{{{den:.6f}}} \right]")
                st.latex(rf"VP = {R1_vp:,.2f} \left[ \frac{{1 - {frac**nm_geo_vp:.6f}}}{{{den:.6f}}} \right]")
                st.latex(rf"VP = {R1_vp:,.2f} [{factor_geo:.6f}]")
            else:
                st.latex(rf"VP = \frac{{nm \cdot R_1}}{{1+{str_i_gvp}}}")
                st.latex(rf"VP = \frac{{{nm_geo_vp:g} \times {R1_vp:,.2f}}}{{1 + {val_i_gvp}}}")
                st.latex(rf"VP = \frac{{{nm_geo_vp * R1_vp:,.2f}}}{{{1 + im_geo_vp:.6f}}}")
            themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>VP = ${vp_geo:,.4f}</h4>")

    # ──────────────────────────────────────────────
    # VP — ARITMÉTICAS
    # ──────────────────────────────────────────────
    else:
        st.markdown("#### Rentas que crecen sumando una cantidad fija $G$ por periodo")
        tipo_t_arit_vp = st.radio("Ingresar tasa como:", ["Tasa efectiva periódica", "Tasa nominal anual"], horizontal=True, key="tipo_t_arit_vp")
        separador()

        c1, c2 = st.columns(2)
        with c1:
            R1_arit_vp = st.number_input("Primer pago ($R_1$)", min_value=0.0, value=1_000.0, step=100.0, key="R1_arit_vp")
            G_vp       = st.number_input("Gradiente ($G$)", value=100.0, step=50.0, key="G_vp")

            if tipo_t_arit_vp == "Tasa efectiva periódica":
                im_arit_vp, nm_arit_vp, _ = _inputs_tasa_efectiva("arit_vp_e")
                str_i_av = r"i_m"
                val_i_av = f"{im_arit_vp:.4f}"
            else:
                im_arit_vp, nm_arit_vp, _, _ = _inputs_tasa_nominal("arit_vp_n")
                str_i_av = r"\frac{i^{(m)}}{m}"
                val_i_av = f"{im_arit_vp:.6f}"

        vp_arit = engine.vp_gradiente_aritmetico(R1_arit_vp, G_vp, im_arit_vp, nm_arit_vp)

        with c2:
            themed_info(f"<h3 style='margin:0; color:inherit;'>Valor Presente: ${vp_arit:,.4f}</h3>")
            st.latex(rf"VP = R_1 \left[\frac{{1-(1+{str_i_av})^{{-nm}}}}{{{str_i_av}}}\right] + \frac{{G}}{{{str_i_av}}}\left(\left[\frac{{1-(1+{str_i_av})^{{-nm}}}}{{{str_i_av}}}\right] - nm(1+{str_i_av})^{{-nm}}\right)")

        with paso_a_paso():
            st.latex(rf"VP = R_1 \left[\frac{{1-(1+{str_i_av})^{{-nm}}}}{{{str_i_av}}}\right] + \frac{{G}}{{{str_i_av}}}\left(\left[\frac{{1-(1+{str_i_av})^{{-nm}}}}{{{str_i_av}}}\right] - nm(1+{str_i_av})^{{-nm}}\right)")
            an   = (1 - (1 + im_arit_vp)**(-nm_arit_vp)) / im_arit_vp
            v_nm = nm_arit_vp * (1 + im_arit_vp)**(-nm_arit_vp)
            desc_n = (1 + im_arit_vp)**(-nm_arit_vp)
            t1v  = R1_arit_vp * an
            t2v  = (G_vp / im_arit_vp) * (an - v_nm)

            st.latex(rf"VP = {R1_arit_vp:,.2f} \left[\frac{{1-(1+{val_i_av})^{{-{nm_arit_vp:g}}}}}{{{val_i_av}}}\right] + \frac{{{G_vp:,.2f}}}{{{val_i_av}}}\left(\left[\frac{{1-(1+{val_i_av})^{{-{nm_arit_vp:g}}}}}{{{val_i_av}}}\right] - {nm_arit_vp:g}(1+{val_i_av})^{{-{nm_arit_vp:g}}}\right)")
            st.latex(rf"VP = {R1_arit_vp:,.2f} \left[\frac{{1-{desc_n:.6f}}}{{{val_i_av}}}\right] + {G_vp/im_arit_vp:,.2f}\left(\left[\frac{{1-{desc_n:.6f}}}{{{val_i_av}}}\right] - {nm_arit_vp:g}({desc_n:.6f})\right)")
            st.latex(rf"VP = {R1_arit_vp:,.2f} [{an:.6f}] + {G_vp/im_arit_vp:,.2f} ({an:.6f} - {v_nm:.6f})")
            st.latex(rf"VP = {t1v:,.2f} + {t2v:,.2f}")
            themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>VP = ${vp_arit:,.4f}</h4>")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — NÚMERO DE PERIODOS (n)
# ═════════════════════════════════════════════════════════════════════════════
with tab_n:
    st.markdown("### Determinación del número de periodos ($n$) en Rentas")
    themed_info(
        "Despeja la variable de tiempo (<span style='font-family: serif; font-style: italic;'>n</span>) "
        "en las ecuaciones de anualidades utilizando propiedades logarítmicas. En los modelos asimétricos o con gradientes "
        "donde no existe un despeje analítico posible, el motor converge el resultado iterando numéricamente."
    )

    c_top1, c_top2 = st.columns(2)
    base_n      = c_top1.selectbox("Calcular $n$ desde:", ["Valor Futuro (Monto Acumulado)", "Valor Presente (Capital Inicial)"], key="sel_base_n")
    tipo_renta_n = c_top2.selectbox("Tipo de Renta:", ["Constante Periódica", "Creciente Geométrica", "Creciente Aritmética"], key="sel_tipo_n")
    separador()

    tipo_tasa_n = st.radio("Ingresar tasa como:", ["Tasa efectiva periódica", "Tasa nominal anual"], horizontal=True, key="tipo_tasa_n")
    separador()

    c3, c4 = st.columns(2)
    with c3:
        lbl_meta = "Valor Futuro Objetivo ($VF$)" if "Futuro" in base_n else "Valor Presente ($VP$)"
        Meta = st.number_input(lbl_meta, min_value=0.01, value=50_000.0, step=1_000.0, key="n_meta")
        R_n = st.number_input("Primer pago ($R_1$)", min_value=0.01, value=1_000.0, step=100.0, key="n_pago")

        if tipo_tasa_n == "Tasa efectiva periódica":
            im_n = st.number_input("Tasa efectiva periódica ($i_m$) %", value=1.0, step=0.1, key="n_im_eff") / 100
            m_n  = st.number_input("Periodos por año ($m$)", min_value=1.0, value=12.0, step=1.0, key="n_m_eff")
            str_i_n = r"i_m"
            val_i_n = f"{im_n:.4f}"
            str_q_n = r"q_m"
        else:
            i_nom_n = st.number_input("Tasa nominal anual ($i^{(m)}$) %", value=12.0, step=0.1, key="n_inom") / 100
            m_n     = st.number_input("Periodos por año ($m$)", min_value=1.0, value=12.0, step=1.0, key="n_m_nom")
            im_n    = i_nom_n / m_n
            str_i_n = r"\frac{i^{(m)}}{m}"
            val_i_n = f"{im_n:.6f}"
            str_q_n = r"\frac{q^{(m)}}{m}"

        qm_n, G_n = None, None
        if tipo_renta_n == "Creciente Geométrica":
            if tipo_tasa_n == "Tasa efectiva periódica":
                qm_n = st.number_input("Tasa efectiva de crecimiento ($q_m$) %", value=0.5, step=0.1, key="n_qm") / 100
            else:
                q_nom_n = st.number_input("Tasa nominal crecimiento ($q^{(m)}$) %", value=5.0, step=0.1, key="n_qnom") / 100
                qm_n = q_nom_n / m_n

        if tipo_renta_n == "Creciente Aritmética":
            G_n = st.number_input("Gradiente ($G$)", value=50.0, step=10.0, key="n_g")

    with c4:
        st.markdown("#### Resultado")
        n_res_total       = np.nan
        usa_metodo_numerico = False
        es_vf = "Futuro" in base_n

        if tipo_renta_n == "Constante Periódica":
            if es_vf:
                n_res_total = engine.nper_anualidad_vf(Meta, R_n, im_n)
                formula_n   = rf"nm = \frac{{\ln\left(\frac{{VF \cdot {str_i_n}}}{{R}} + 1\right)}}{{\ln(1+{str_i_n})}}"
            else:
                n_res_total = engine.nper_anualidad_vp(Meta, R_n, im_n)
                formula_n   = rf"nm = \frac{{-\ln\left(1 - \frac{{VP \cdot {str_i_n}}}{{R}}\right)}}{{\ln(1+{str_i_n})}}"

        elif tipo_renta_n == "Creciente Geométrica":
            usa_metodo_numerico = True
            if es_vf:
                n_res_total = engine.nper_gradiente_geo_vf(Meta, R_n, im_n, qm_n)
                formula_n   = rf"VF = R_1 \left[\frac{{(1+{str_i_n})^{{nm}}-(1+{str_q_n})^{{nm}}}}{{{str_i_n}-{str_q_n}}}\right]"
            else:
                n_res_total = engine.nper_gradiente_geo_vp(Meta, R_n, im_n, qm_n)
                formula_n   = rf"VP = R_1 \left[\frac{{1-\left(\frac{{1+{str_q_n}}}{{1+{str_i_n}}}\right)^{{nm}}}}{{{str_i_n}-{str_q_n}}}\right]"

        else:
            usa_metodo_numerico = True
            if es_vf:
                n_res_total = engine.nper_gradiente_arit_vf(Meta, R_n, G_n, im_n)
                formula_n   = r"f(nm)=VF(nm)-VF_{objetivo}=0"
            else:
                n_res_total = engine.nper_gradiente_arit_vp(Meta, R_n, G_n, im_n)
                formula_n   = r"f(nm)=VP(nm)-VP_{objetivo}=0"

        if np.isnan(n_res_total):
            themed_error("El monto objetivo es inalcanzable con la configuración de rentas y tasas actuales.")
        else:
            themed_info(f"<h3 style='margin:0; color:inherit;'>Total de Periodos (nm): {n_res_total:.4f}</h3>")
            
            if usa_metodo_numerico:
                alerta_metodo_numerico()
                
            anios_decimal = n_res_total / m_n
            st.write(f"**Años totales ($n$):** {anios_decimal:.4f} años")
            st.latex(formula_n)

            with paso_a_paso():
                st.latex(formula_n)
                if not usa_metodo_numerico:
                    if es_vf:
                        num_val = (Meta * im_n / R_n) + 1
                        st.latex(rf"nm = \frac{{\ln\left(\frac{{{Meta:,.2f} \cdot {val_i_n}}}{{{R_n:,.2f}}} + 1\right)}}{{\ln(1+{val_i_n})}}")
                        st.latex(rf"nm = \frac{{\ln({(Meta * im_n / R_n):.6f} + 1)}}{{\ln({1+im_n:.6f})}}")
                        st.latex(rf"nm = \frac{{\ln({num_val:.6f})}}{{{np.log(1+im_n):.6f}}}")
                        st.latex(rf"nm = \frac{{{np.log(num_val):.6f}}}{{{np.log(1+im_n):.6f}}}")
                    else:
                        num_val = 1 - (Meta * im_n / R_n)
                        st.latex(rf"nm = \frac{{-\ln\left(1 - \frac{{{Meta:,.2f} \cdot {val_i_n}}}{{{R_n:,.2f}}}\right)}}{{\ln(1+{val_i_n})}}")
                        st.latex(rf"nm = \frac{{-\ln(1 - {(Meta * im_n / R_n):.6f})}}{{\ln({1+im_n:.6f})}}")
                        st.latex(rf"nm = \frac{{-\ln({num_val:.6f})}}{{{np.log(1+im_n):.6f}}}")
                        st.latex(rf"nm = \frac{{-{np.log(num_val):.6f}}}{{{np.log(1+im_n):.6f}}}")
                        
                    themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>nm = {n_res_total:.4f}</h4>")
                else:
                    st.latex(r"f(nm) = \text{Valor}_\text{renta}(nm) - \text{Objetivo} = 0")
                    themed_info(f"<h4 style='margin:0; color:inherit; text-align:center;'>nm \approx {n_res_total:.4f} \text{{ (Numérico)}}</h4>")

            # Desglose del tiempo (meses, días)
            st.markdown("**Desglose exacto temporal:**")
            df_n = engine.desglosar_periodos(anios_decimal)
            st.dataframe(df_n.style.set_properties(**{"background-color": "#F3F4F6", "color": "#1E3A8A", "font-weight": "bold", "text-align": "center"}), use_container_width=True, hide_index=True)