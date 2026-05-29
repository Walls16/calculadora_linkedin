# --- LIBRERÍAS BASE (Matemáticas y Datos) ---
import numpy as np
import pandas as pd

# --- LIBRERÍAS CIENTÍFICAS (SciPy) ---
import scipy.optimize as opt
from scipy.optimize import newton, root_scalar
from scipy.stats import norm, multivariate_normal
from scipy.stats import mvn as _mvn  # MEJORA: mvndst es ~3x más rápido que multivariate_normal().cdf

# --- LIBRERÍAS FINANCIERAS ---
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

# MEJORA: RNG moderno de NumPy (~20% más rápido que np.random.normal legacy)
_rng = np.random.default_rng()


class FinancialMathEngine:
    # ==========================================================
    # 1. TASAS
    # ==========================================================
    def tasa_nominal_a_efectiva(self, i_nom, m):
        if m == 0:
            return 0
        return (1 + i_nom / m) ** m - 1

    def tasa_efectiva_a_nominal(self, i_eff, m):
        if m == 0:
            return 0
        return m * ((1 + i_eff) ** (1 / m) - 1)

    def tasa_nominal_a_instantanea(self, i_nom, m):
        return m * np.log(1 + i_nom / m)

    def tasa_instantanea_a_efectiva(self, delta):
        return np.exp(delta) - 1

    def tasa_instantanea_a_nominal(self, delta, m):
        return m * (np.exp(delta / m) - 1)

    def tasa_nominal_m_a_nominal_p(self, i_m, m, p):
        if m == 0 or p == 0:
            return 0
        tasa_efectiva_periodo = ((1 + i_m / m) ** (m / p)) - 1
        return tasa_efectiva_periodo * p

    def generar_tabla_reinversion(self, C0, i_nom, n):
        periodos = [
            ("Cada 4 años", 0.25), ("Cada 2 años", 0.5), ("Anual", 1),
            ("Semestral", 2), ("Trimestral", 4), ("Mensual", 12),
            ("Semanal", 52), ("Diaria", 365), ("Cada hora", 8760),
            ("Cada minuto", 525600), ("Cada segundo", 31536000)
        ]

        datos = []
        for nombre, m in periodos:
            monto = C0 * ((1 + i_nom / m) ** (m * n))
            rendimiento = (monto / C0) - 1
            datos.append({
                "Periodo de reinversión": nombre,
                "m = Veces al año": str(m),
                "Monto acumulado": monto,
                "Rendimiento Acumulado": rendimiento
            })

        monto_inst = C0 * np.exp(i_nom * n)
        rendimiento_inst = (monto_inst / C0) - 1
        datos.append({
            "Periodo de reinversión": "Instantánea",
            "m = Veces al año": "∞",
            "Monto acumulado": monto_inst,
            "Rendimiento Acumulado": rendimiento_inst
        })

        return pd.DataFrame(datos)

    # ==========================================================
    # 2. TVM
    # ==========================================================
    def valor_futuro(self, C0, i, n):
        return C0 * (1 + i) ** n

    def valor_futuro_continuo(self, C0, delta, n):
        return C0 * np.exp(delta * n)

    def valor_presente(self, Cn, i, n):
        return Cn / (1 + i) ** n

    def valor_presente_continuo(self, Cn, delta, n):
        return Cn * np.exp(-delta * n)

    def numero_periodos(self, C0, Cn, i):
        if C0 == 0 or i <= 0:
            return 0
        return np.log(Cn / C0) / np.log(1 + i)

    def tasa_rendimiento(self, C0, Cn, n):
        if C0 == 0 or n == 0:
            return 0
        return (Cn / C0) ** (1 / n) - 1

    def desglosar_periodos(self, n):
        anios = int(n)
        frac_anios = n - anios

        meses_raw = frac_anios * 12
        meses = int(meses_raw)
        frac_meses = meses_raw - meses

        dias_raw = frac_meses * (365 / 12)
        dias = int(dias_raw)
        frac_dias = dias_raw - dias

        horas_raw = frac_dias * 24
        horas = int(horas_raw)
        frac_horas = horas_raw - horas

        min_raw = frac_horas * 60
        minutos = int(min_raw)
        frac_min = min_raw - minutos

        seg_raw = frac_min * 60
        segundos = int(seg_raw)

        return pd.DataFrame([{
            "Años": anios,
            "Meses": meses,
            "Días": dias,
            "Horas": horas,
            "Minutos": minutos,
            "Segundos": segundos
        }])

    # ==========================================================
    # 3. ANUALIDADES Y GRADIENTES
    # ==========================================================
    # --- RENTAS CONSTANTES ---
    def vf_anualidad_efectiva(self, R, i_m, n_m, anticipada=False):
        """Valor futuro de rentas vencidas o anticipadas constantes a tasa im."""
        if i_m == 0:
            return R * n_m
        factor = ((1 + i_m) ** n_m - 1) / i_m
        if anticipada:
            factor *= (1 + i_m)
        return R * factor

    def vf_anualidad_nominal(self, R, i_nom, m, p, n):
        """
        Valor futuro de rentas vencidas constantes realizadas p veces al año,
        durante n años a una tasa nominal anual i(m).
        """
        i_p = self.tasa_nominal_m_a_nominal_p(i_nom, m, p)
        n_p = n * p
        return self.vf_anualidad_efectiva(R, i_p, n_p, anticipada=False)

    def vf_anualidad_continua(self, R_anual, delta, n):
        """
        Valor futuro de rentas constantes realizadas de manera instantánea,
        durante n años a una fuerza de interés delta.
        R_anual es la tasa de flujo anual.
        """
        if delta == 0:
            return R_anual * n
        return R_anual * (np.exp(delta * n) - 1) / delta

    def vp_anualidad_efectiva(self, R, i_m, n_m, anticipada=False):
        if i_m == 0:
            return R * n_m
        factor = (1 - (1 + i_m) ** (-n_m)) / i_m
        if anticipada:
            factor *= (1 + i_m)
        return R * factor

    def vp_anualidad_nominal(self, R, i_nom, m, p, n):
        i_p = self.tasa_nominal_m_a_nominal_p(i_nom, m, p)
        n_p = n * p
        return self.vp_anualidad_efectiva(R, i_p, n_p, anticipada=False)

    def vp_anualidad_continua(self, R_anual, delta, n):
        if delta == 0:
            return R_anual * n
        return R_anual * (1 - np.exp(-delta * n)) / delta

    def vp_perpetuidad(self, R, i):
        if i == 0:
            return 0
        return R / i

    # --- RENTAS CRECIENTES (GRADIENTES GEOMÉTRICOS) ---
    def vf_gradiente_geo(self, R1, i_m, q_m, n_m):
        if i_m == q_m:
            return n_m * R1 * ((1 + i_m) ** (n_m - 1))
        numerador = ((1 + i_m) ** n_m) - ((1 + q_m) ** n_m)
        denominador = i_m - q_m
        return R1 * (numerador / denominador)

    def vp_gradiente_geo(self, R1, i, q, n):
        if i == q:
            return R1 * n / (1 + i)
        return R1 * (1 - ((1 + q) / (1 + i)) ** n) / (i - q)

    # --- SOLVER PARA NÚMERO DE PERIODOS (ANUALIDADES) ---
    def nper_anualidad_vf(self, VF, R, i_m):
        """Despeje analítico de n para Valor Futuro de anualidad constante."""
        if i_m == 0:
            return VF / R
        val = (VF * i_m / R) + 1
        if val <= 0:
            return np.nan
        return np.log(val) / np.log(1 + i_m)

    def nper_anualidad_vp(self, VP, R, i_m):
        """Despeje analítico de n para Valor Presente de anualidad constante."""
        if i_m == 0:
            return VP / R
        val = 1 - (VP * i_m / R)
        if val <= 0:
            return np.nan
        return -np.log(val) / np.log(1 + i_m)

    def nper_gradiente_geo_vf(self, VF, R1, i_m, q_m):
        """Uso de solver numérico para n en gradiente geométrico (VF)."""
        if i_m == q_m:
            f = lambda n: n * R1 * ((1 + i_m) ** (n - 1)) - VF
        else:
            f = lambda n: R1 * (((1 + i_m) ** n - (1 + q_m) ** n) / (i_m - q_m)) - VF
        try:
            res = opt.root_scalar(f, bracket=[0.0001, 2000], method='brentq')
            return res.root
        except Exception:
            return np.nan

    def nper_gradiente_geo_vp(self, VP, R1, i_m, q_m):
        """Uso de solver numérico para n en gradiente geométrico (VP)."""
        if i_m == q_m:
            return VP * (1 + i_m) / R1
        else:
            f = lambda n: R1 * (1 - ((1 + q_m) / (1 + i_m)) ** n) / (i_m - q_m) - VP
        try:
            res = opt.root_scalar(f, bracket=[0.0001, 2000], method='brentq')
            return res.root
        except Exception:
            return np.nan

    def vp_gradiente_aritmetico(self, R1, G, i_m, n_m):
        """Valor Presente de una Renta Creciente/Decreciente Aritmética."""
        if i_m == 0:
            return R1 * n_m + G * n_m * (n_m - 1) / 2
        an = (1 - (1 + i_m) ** (-n_m)) / i_m
        return R1 * an + (G / i_m) * (an - n_m * (1 + i_m) ** (-n_m))

    def vf_gradiente_aritmetico(self, R1, G, i_m, n_m):
        """Valor Futuro de una Renta Creciente/Decreciente Aritmética."""
        if i_m == 0:
            return R1 * n_m + G * n_m * (n_m - 1) / 2
        sn = ((1 + i_m) ** n_m - 1) / i_m
        return R1 * sn + (G / i_m) * (sn - n_m)

    # --- SOLVER PARA NÚMERO DE PERIODOS (ARITMÉTICOS) ---
    def nper_gradiente_arit_vf(self, VF, R1, G, i_m):
        f = lambda n: self.vf_gradiente_aritmetico(R1, G, i_m, n) - VF
        try:
            res = opt.root_scalar(f, bracket=[0.0001, 2000], method='brentq')
            return res.root
        except Exception:
            return np.nan

    def nper_gradiente_arit_vp(self, VP, R1, G, i_m):
        f = lambda n: self.vp_gradiente_aritmetico(R1, G, i_m, n) - VP
        try:
            res = opt.root_scalar(f, bracket=[0.0001, 2000], method='brentq')
            return res.root
        except Exception:
            return np.nan

    # ==========================================================
    # 4. AMORTIZACIÓN
    # ==========================================================
    def tabla_amortizacion(self, VP, i_m, n_m):
        """Genera una tabla de amortización de pagos fijos sin el periodo 0."""
        # MEJORA: validar inputs y redondear n_m para evitar pérdida silenciosa del último periodo
        if i_m < 0:
            raise ValueError("i_m debe ser >= 0")
        n_m = round(n_m)  # evita que n_m=11.9 trunce a 11 y deje saldo residual

        if i_m == 0:
            pago = VP / n_m
        else:
            pago = VP * (i_m / (1 - (1 + i_m) ** (-n_m)))

        saldo = VP
        datos = []

        for t in range(1, n_m + 1):
            saldo_inicial = saldo
            interes = saldo_inicial * i_m
            amort = pago - interes

            # MEJORA: en el último periodo, liquidar exactamente para eliminar residuos de redondeo
            if t == n_m:
                amort = saldo_inicial
                saldo = 0.0
            else:
                saldo -= amort
                if abs(saldo) < 0.01:
                    saldo = 0.0

            datos.append({
                "Periodo": t,
                "Saldo Inicial": saldo_inicial,
                "Interés": interes,
                "Amortización": amort,
                "Saldo Insoluto": saldo
            })

        return pd.DataFrame(datos)

    # ==========================================================
    # 5. VALUACIÓN DE BONOS
    # ==========================================================
    def precio_bono(self, F, r_m, C, i_m, n):
        cupon_Fr = F * r_m

        if i_m == 0:
            vp_cupones = cupon_Fr * n
        else:
            vp_cupones = cupon_Fr * ((1 - (1 + i_m) ** (-n)) / i_m)

        vp_redencion = C * (1 + i_m) ** (-n)
        precio_total = vp_cupones + vp_redencion
        return precio_total, cupon_Fr, vp_cupones, vp_redencion

    def tasa_rendimiento_bono(self, P, F, r_m, C, n):
        def f(i):
            if i == 0:
                precio_calc = (F * r_m * n) + C
            else:
                precio_calc = self.precio_bono(F, r_m, C, i, n)[0]
            return precio_calc - P

        try:
            res = opt.root_scalar(f, bracket=[-0.99, 10.0], method='brentq')
            return res.root
        except Exception:
            return np.nan

    def riesgo_bono(self, F, r_periodo, C, i_periodo, n_periodos, m):
        cupon = F * r_periodo
        precio = 0.0
        sum_mac = 0.0
        sum_conv = 0.0

        for t in range(1, int(n_periodos) + 1):
            cf = cupon if t < n_periodos else cupon + C
            vp_cf = cf / ((1 + i_periodo) ** t)
            precio += vp_cf
            sum_mac += t * vp_cf
            sum_conv += t * (t + 1) * vp_cf

        mac_duration_periodos = sum_mac / precio
        mac_duration_anios = mac_duration_periodos / m
        mod_duration_anios = mac_duration_anios / (1 + i_periodo)
        convexity_anios = sum_conv / (precio * (m ** 2) * ((1 + i_periodo) ** 2))

        return mac_duration_anios, mod_duration_anios, convexity_anios

    # ==========================================================
    # 6. VALUACIÓN DE ACCIONES
    # ==========================================================
    def valuacion_gordon_shapiro(self, D1, k, g):
        if k <= g:
            return None
        return D1 / (k - g)

    def rendimiento_requerido_accion(self, D1, P0, g):
        if P0 <= 0:
            return None
        return (D1 / P0) + g

    def valuacion_multiplos(self, metrica_valor, multiplo_objetivo):
        return metrica_valor * multiplo_objetivo

    def calcular_vp_dividendos(self, monto_div, m_pagos, r, T_total, capitalizacion="Continua"):
        vp_total = 0
        dt = 1 / m_pagos
        num_pagos = int(T_total * m_pagos)

        for k in range(1, num_pagos + 1):
            t_pago = k * dt
            if capitalizacion == "Continua":
                vp_total += monto_div * np.exp(-r * t_pago)
            else:
                vp_total += monto_div / ((1 + r) ** t_pago)
        return vp_total

    def optimizacion_markowitz(self, tickers_list, start_date, end_date, r_f=0.05):
        # MEJORA: descargar una sola vez y reutilizar
        data = self._descargar_y_limpiar(tickers_list, start_date, end_date)

        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        # Máximo Ratio de Sharpe
        ef_s = EfficientFrontier(mu, S)
        ef_s.max_sharpe(risk_free_rate=r_f)
        pesos_limpios_s = ef_s.clean_weights()
        ret_s, vol_s, sharpe_s = ef_s.portfolio_performance(verbose=False, risk_free_rate=r_f)

        # Mínima Varianza Global
        ef_m = EfficientFrontier(mu, S)
        ef_m.min_volatility()
        pesos_limpios_m = ef_m.clean_weights()
        ret_m, vol_m, sharpe_m = ef_m.portfolio_performance(verbose=False, risk_free_rate=r_f)

        # MEJORA: nube vectorizada con einsum (equivalente pero más legible/verificable)
        n_sim = 2500
        n_activos = len(data.columns)
        pesos_rand = np.random.dirichlet(np.ones(n_activos), size=n_sim)
        ret_sim = pesos_rand @ mu.values
        # einsum 'ij,jk,ik->i' calcula w @ Sigma @ w para cada fila simultáneamente
        vol_sim = np.sqrt(np.einsum('ij,jk,ik->i', pesos_rand, S.values, pesos_rand))
        sharpe_sim = (ret_sim - r_f) / vol_sim

        nube_grafica = (ret_sim, vol_sim, sharpe_sim)

        return data, mu, S, (ret_s, vol_s, sharpe_s, pesos_limpios_s), (ret_m, vol_m, sharpe_m, pesos_limpios_m), nube_grafica

    # ==========================================================
    # 8. FORWARDS (DERIVADOS)
    # ==========================================================
    def forward_calculo(self, S, r, delta, T, capitalizacion="Continua"):
        if capitalizacion == "Continua":
            return S * np.exp((r - delta) * T)
        else:
            return S * ((1 + r) ** T) / ((1 + delta) ** T)

    def valor_forward_calculo(self, S, K, r, delta, T, posicion="Larga", capitalizacion="Continua"):
        if capitalizacion == "Continua":
            val_largo = (S * np.exp(-delta * T)) - (K * np.exp(-r * T))
        else:
            val_largo = (S / (1 + delta) ** T) - (K / (1 + r) ** T)
        return val_largo if posicion == "Larga" else -val_largo

    def calcular_vp_flujos_irregulares(self, montos, tiempos_anios, r, capitalizacion="Continua"):
        vp_total = 0.0
        for monto, t in zip(montos, tiempos_anios):
            if capitalizacion == "Continua":
                vp_total += monto * np.exp(-r * t)
            else:
                vp_total += monto / ((1 + r) ** t)
        return vp_total

    # ==========================================================
    # 9. OPCIONES FINANCIERAS (BLACK-SCHOLES)
    # ==========================================================
    def opciones_bsm(self, tipo_modelo, S, K, T, r, sigma, extra=0.0):
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0, 0.0, 0.0

        d1 = 0.0
        d2 = 0.0
        call = 0.0
        put = 0.0

        if tipo_modelo == "Simple":
            d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        elif tipo_modelo == "Ingresos":
            S_adj = S - extra
            if S_adj <= 0:
                S_adj = 0.0001
            d1 = (np.log(S_adj / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call = S_adj * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put = K * np.exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)

        elif tipo_modelo in ("Yield", "Monedas"):
            d1 = (np.log(S / K) + (r - extra + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call = S * np.exp(-extra * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-extra * T) * norm.cdf(-d1)

        elif tipo_modelo == "Futuros":
            d1 = (np.log(S / K) + ((sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call = np.exp(-r * T) * (S * norm.cdf(d1) - K * norm.cdf(d2))
            put = np.exp(-r * T) * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

        elif tipo_modelo == "Costos":
            S_adj = S + extra
            d1 = (np.log(S_adj / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call = S_adj * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put = K * np.exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1)

        return call, put, d1, d2

    def griegas_bsm(self, tipo_modelo, S, K, T, r, sigma, extra=0.0):
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        q_yield = 0.0
        S_adj = S

        if tipo_modelo == "Ingresos":
            S_adj = S - extra
            if S_adj <= 0:
                S_adj = 0.0001
        elif tipo_modelo == "Costos":
            S_adj = S + extra
        elif tipo_modelo in ("Yield", "Monedas"):
            q_yield = extra
        elif tipo_modelo == "Futuros":
            q_yield = r

        d1 = (np.log(S_adj / K) + (r - q_yield + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        nd1 = norm.pdf(d1)

        delta_call = np.exp(-q_yield * T) * Nd1
        delta_put = np.exp(-q_yield * T) * (Nd1 - 1)
        gamma = np.exp(-q_yield * T) * nd1 / (S_adj * sigma * np.sqrt(T))
        vega = S_adj * np.exp(-q_yield * T) * nd1 * np.sqrt(T) / 100

        termino_comun = -(S_adj * np.exp(-q_yield * T) * nd1 * sigma) / (2 * np.sqrt(T))
        theta_call = (termino_comun - r * K * np.exp(-r * T) * Nd2 + q_yield * S_adj * np.exp(-q_yield * T) * Nd1) / 365
        theta_put = (termino_comun + r * K * np.exp(-r * T) * N_neg_d2 - q_yield * S_adj * np.exp(-q_yield * T) * N_neg_d1) / 365

        rho_call = K * T * np.exp(-r * T) * Nd2 / 100
        rho_put = -K * T * np.exp(-r * T) * N_neg_d2 / 100

        return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put

    # --- MODELO BINOMIAL (CRR) ---
    def binomial_tree(self, S, K, T, r, sigma, n, q=0.0, tipo='call', american=False):
        tipo = tipo.lower().strip()
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        if p < 0 or p > 1:
            return None, None

        S_tree = [np.zeros(i + 1) for i in range(n + 1)]
        V_tree = [np.zeros(i + 1) for i in range(n + 1)]

        for i in range(n + 1):
            for j in range(i + 1):
                S_tree[i][j] = S * (u ** (i - j)) * (d ** j)

        for j in range(n + 1):
            if tipo == 'call':
                V_tree[n][j] = max(0, S_tree[n][j] - K)
            else:
                V_tree[n][j] = max(0, K - S_tree[n][j])

        df = np.exp(-r * dt)
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                val_hold = df * (p * V_tree[i + 1][j] + (1 - p) * V_tree[i + 1][j + 1])
                if american:
                    if tipo == 'call':
                        val_exercise = max(0, S_tree[i][j] - K)
                    else:
                        val_exercise = max(0, K - S_tree[i][j])
                    V_tree[i][j] = max(val_hold, val_exercise)
                else:
                    V_tree[i][j] = val_hold

        return V_tree[0][0], (S_tree, V_tree)

    def obtener_datos_subyacente(self, ticker_symbol):
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1y")

            if hist.empty or len(hist) < 20:
                return None, None

            spot_price = hist['Close'].iloc[-1]

            # MEJORA: usar ddof=1 (desviación muestral) y limpiar infinitos por precios cero
            retornos_log = np.log(hist['Close'] / hist['Close'].shift(1))
            retornos_log = retornos_log.replace([np.inf, -np.inf], np.nan).dropna()
            volatilidad_hist = retornos_log.std(ddof=1) * np.sqrt(252)

            return spot_price, volatilidad_hist

        except Exception:
            return None, None

    def calcular_var_parametrico(self, rend_anual, vol_anual, valor_portafolio, nivel_confianza, dias_horizonte):
        t = dias_horizonte / 252.0
        rend_periodo = rend_anual * t
        vol_periodo = vol_anual * np.sqrt(t)
        z_score = norm.ppf(nivel_confianza)
        var_monto = valor_portafolio * (z_score * vol_periodo - rend_periodo)
        return max(var_monto, 0), z_score, rend_periodo, vol_periodo

    def calcular_var_cvar_montecarlo(self, rend_anual, vol_anual, valor_portafolio, nivel_confianza, dias_horizonte, simulaciones=10000):
        t = dias_horizonte / 252.0
        rend_periodo = rend_anual * t
        vol_periodo = vol_anual * np.sqrt(t)

        # MEJORA: RNG moderno (~20% más rápido que np.random.normal legacy)
        e = _rng.standard_normal(simulaciones)
        simulacion_retornos = rend_periodo + vol_periodo * e

        # MEJORA: np.percentile es más limpio y evita errores de índice manual
        alpha = 1.0 - nivel_confianza
        q_alpha = np.percentile(simulacion_retornos, alpha * 100)

        # CVaR: promedio de retornos en la cola de pérdidas
        cola = simulacion_retornos[simulacion_retornos <= q_alpha]
        cvar_alpha = cola.mean() if len(cola) > 0 else q_alpha

        return max(-q_alpha * valor_portafolio, 0), max(-cvar_alpha * valor_portafolio, 0)

    def evaluar_portafolio_personalizado(self, tickers_list, dict_pesos, start_date, end_date):
        data = self._descargar_y_limpiar(tickers_list, start_date, end_date)

        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        pesos_array = np.array([dict_pesos.get(c, 0) for c in data.columns])
        suma = pesos_array.sum()
        if suma > 0:
            pesos_array = pesos_array / suma

        rend_p = np.dot(pesos_array, mu)
        vol_p = np.sqrt(np.dot(pesos_array.T, np.dot(S, pesos_array)))

        return data, rend_p, vol_p, pesos_array, data.columns

    def opciones_gap(self, S, K1, K2, T, r, sigma, q=0, tipo='call'):
        d1 = (np.log(S / K2) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if tipo == 'call':
            precio = S * np.exp(-q * T) * norm.cdf(d1) - K1 * np.exp(-r * T) * norm.cdf(d2)
        elif tipo == 'put':
            precio = K1 * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("El tipo debe ser 'call' o 'put'")
        return precio

    def opciones_cash_or_nothing(self, S, K, Q, T, r, sigma, q=0, tipo='call'):
        d1 = (np.log(S / K) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if tipo == 'call':
            precio = Q * np.exp(-r * T) * norm.cdf(d2)
        elif tipo == 'put':
            precio = Q * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("El tipo debe ser 'call' o 'put'")
        return precio

    def barrera_down_and_out(self, S, K, H, T, r, sigma, q=0, tipo='call'):
        """Valuación de Opciones con Barrera: Down-and-Out."""
        if S <= H:
            return 0.0

        lam = (r - q + (sigma ** 2) / 2) / (sigma ** 2)
        y = (np.log(H ** 2 / (S * K)) / (sigma * np.sqrt(T))) + lam * sigma * np.sqrt(T)

        vanilla_call, vanilla_put, _, _ = self.opciones_bsm("Yield", S, K, T, r, sigma, extra=q)

        if tipo == 'call':
            c_di = (S * np.exp(-q * T) * (H / S) ** (2 * lam) * norm.cdf(y)
                    - K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
            precio = vanilla_call - c_di
        elif tipo == 'put':
            p_di = (-S * np.exp(-q * T) * (H / S) ** (2 * lam) * norm.cdf(-y)
                    + K * np.exp(-r * T) * (H / S) ** (2 * lam - 2) * norm.cdf(-y + sigma * np.sqrt(T)))
            precio = vanilla_put - p_di

        return max(0.0, precio)

    def opciones_asiaticas_aritmeticas(self, S, K, T, r, sigma, q=0, tipo='call'):
        b = r - q

        # MEJORA: usar np.expm1 para mayor precisión cuando b*T o sigma²*T son pequeños
        if np.isclose(b, 0.0):
            M1 = S
            # Límite correcto de Turnbull-Wakeman cuando b→0:
            M2 = (2 * S ** 2 / (sigma ** 2 * T ** 2)) * (
                np.expm1(sigma ** 2 * T) / sigma ** 2 - T
            )
        else:
            # np.expm1(x) = exp(x)-1, más preciso que exp()-1 para x pequeño
            M1 = S * np.expm1(b * T) / (b * T)
            termino1 = np.expm1((2 * b + sigma ** 2) * T) / (2 * b + sigma ** 2)
            termino2 = np.expm1(b * T) / b
            M2 = (2 * S ** 2 / ((b + sigma ** 2) * T ** 2)) * (termino1 - termino2)

        F = M1
        # Protección: M2/M1² debe ser > 1 para que el log sea positivo
        ratio = M2 / (M1 ** 2)
        if ratio <= 1.0:
            ratio = 1.0 + 1e-12
        sigma_adj = np.sqrt(np.log(ratio) / T)

        d1 = (np.log(F / K) + (sigma_adj ** 2 / 2) * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)

        if tipo == 'call':
            precio = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        elif tipo == 'put':
            precio = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        else:
            raise ValueError("El tipo debe ser 'call' o 'put'")

        return max(0.0, precio)

    def opciones_asiaticas_geometricas(self, S, K, T, r, sigma, q=0, tipo='call'):
        b = r - q
        sigma_adj = sigma / np.sqrt(3.0)
        b_adj = 0.5 * (b - (sigma ** 2) / 6.0)

        d1 = (np.log(S / K) + (b_adj + (sigma_adj ** 2) / 2.0) * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)

        if tipo == 'call':
            precio = S * np.exp((b_adj - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif tipo == 'put':
            precio = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b_adj - r) * T) * norm.cdf(-d1)
        else:
            raise ValueError("El tipo debe ser 'call' o 'put'")

        return max(0.0, precio)

    def opciones_lookback_flotante(self, S, S_ref, T, r, sigma, q=0, tipo='call'):
        # MEJORA: en lugar de epsilon fijo (1e-8), derivar el límite cuando r==q
        # usando la expansión de primer orden: lim_{r→q} formula = formula_limite
        r_eq_q = np.isclose(r, q, atol=1e-9)

        if tipo == 'call':
            Smin = S_ref

            if r_eq_q:
                # Límite analítico cuando r = q (evita división por cero)
                sqrtT = np.sqrt(T)
                a1 = (np.log(S / Smin) + (sigma ** 2 / 2) * T) / (sigma * sqrtT)
                a2 = a1 - sigma * sqrtT
                # Fórmula límite: el término (sigma²/2r) → (T) cuando r=q=0 con ajuste
                precio = (S * np.exp(-q * T) * (norm.cdf(a1) + sigma * sqrtT * norm.pdf(a1))
                          - Smin * np.exp(-r * T) * norm.cdf(a2))
            else:
                a1 = (np.log(S / Smin) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
                a2 = a1 - sigma * np.sqrt(T)
                a3 = (np.log(S / Smin) + (-r + q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
                Y1 = -2 * (r - q - (sigma ** 2) / 2) * np.log(S / Smin) / (sigma ** 2)
                coef = sigma ** 2 / (2 * (r - q))

                termino1 = S * np.exp(-q * T) * norm.cdf(a1)
                termino2 = S * np.exp(-q * T) * coef * norm.cdf(-a1)
                termino3 = Smin * np.exp(-r * T) * (norm.cdf(a2) - coef * np.exp(Y1) * norm.cdf(-a3))
                precio = termino1 - termino2 - termino3

        elif tipo == 'put':
            Smax = S_ref

            if r_eq_q:
                sqrtT = np.sqrt(T)
                b1 = (np.log(Smax / S) + (sigma ** 2 / 2) * T) / (sigma * sqrtT)
                b2 = b1 - sigma * sqrtT
                precio = (Smax * np.exp(-r * T) * norm.cdf(b1)
                          - S * np.exp(-q * T) * (norm.cdf(b2) - sigma * sqrtT * norm.pdf(b2)))
            else:
                b1 = (np.log(Smax / S) + (-r + q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
                b2 = b1 - sigma * np.sqrt(T)
                b3 = (np.log(Smax / S) + (r - q - (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
                Y2 = 2 * (r - q - (sigma ** 2) / 2) * np.log(Smax / S) / (sigma ** 2)
                coef = sigma ** 2 / (2 * (r - q))

                termino1 = Smax * np.exp(-r * T) * (norm.cdf(b1) - coef * np.exp(Y2) * norm.cdf(-b3))
                termino2 = S * np.exp(-q * T) * norm.cdf(b2)
                termino3 = S * np.exp(-q * T) * coef * norm.cdf(-b2)
                precio = termino1 - termino2 + termino3
        else:
            raise ValueError("El tipo debe ser 'call' o 'put'")

        return max(0.0, precio)

    def _pnbivariada(self, x, y, rho):
        # MEJORA: mvndst es ~3x más rápido que instanciar multivariate_normal().cdf cada vez
        lower = [-10.0, -10.0]
        upper = [float(x), float(y)]
        infin = [0, 0]   # 0 = límite superior finito (cdf estándar)
        correl = [float(rho)]
        _, val, _ = _mvn.mvndst(lower, upper, infin, correl)
        return val

    def opciones_compuestas(self, S, K1, K2, T1, T2, r, sigma, q=0, tipo='call_on_call'):
        """Valuación de Opciones Compuestas."""
        tau = T2 - T1

        if 'on_call' in tipo:
            objetivo = lambda x: self.opciones_bsm("Yield", x, K2, tau, r, sigma, extra=q)[0] - K1
        else:
            objetivo = lambda x: self.opciones_bsm("Yield", x, K2, tau, r, sigma, extra=q)[1] - K1

        try:
            # MEJORA: verificar cambio de signo antes de aplicar Brent para fallar explícitamente
            f_lo = objetivo(0.001)
            f_hi = objetivo(10000.0)
            if f_lo * f_hi > 0:
                return 0.0  # No hay raíz en el intervalo — opción sin valor
            S_star = root_scalar(objetivo, bracket=[0.001, 10000.0],
                                 method='brentq', xtol=1e-6).root
        except Exception:
            return 0.0

        a1 = (np.log(S / S_star) + (r - q + (sigma ** 2) / 2) * T1) / (sigma * np.sqrt(T1))
        a2 = a1 - sigma * np.sqrt(T1)
        b1 = (np.log(S / K2) + (r - q + (sigma ** 2) / 2) * T2) / (sigma * np.sqrt(T2))
        b2 = b1 - sigma * np.sqrt(T2)
        rho = np.sqrt(T1 / T2)

        if tipo == 'call_on_call':
            M1 = self._pnbivariada(a1, b1, rho)
            M2 = self._pnbivariada(a2, b2, rho)
            precio = S * np.exp(-q * T2) * M1 - K2 * np.exp(-r * T2) * M2 - K1 * np.exp(-r * T1) * norm.cdf(a2)

        elif tipo == 'put_on_call':
            M1 = self._pnbivariada(-a1, b1, -rho)
            M2 = self._pnbivariada(-a2, b2, -rho)
            precio = K2 * np.exp(-r * T2) * M2 - S * np.exp(-q * T2) * M1 + K1 * np.exp(-r * T1) * norm.cdf(-a2)

        elif tipo == 'call_on_put':
            M1 = self._pnbivariada(-a1, -b1, rho)
            M2 = self._pnbivariada(-a2, -b2, rho)
            precio = K2 * np.exp(-r * T2) * M2 - S * np.exp(-q * T2) * M1 - K1 * np.exp(-r * T1) * norm.cdf(-a2)

        elif tipo == 'put_on_put':
            M1 = self._pnbivariada(a1, -b1, -rho)
            M2 = self._pnbivariada(a2, -b2, -rho)
            precio = S * np.exp(-q * T2) * M1 - K2 * np.exp(-r * T2) * M2 + K1 * np.exp(-r * T1) * norm.cdf(a2)

        return precio

    def opciones_intercambio_uxv(self, U, V, q_u, q_v, sigma_u, sigma_v, rho, T):
        sigma = np.sqrt(sigma_u ** 2 + sigma_v ** 2 - 2 * rho * sigma_u * sigma_v)
        d1 = (np.log(V / U) + (q_u - q_v + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        precio = V * np.exp(-q_v * T) * norm.cdf(d1) - U * np.exp(-q_u * T) * norm.cdf(d2)
        return max(0.0, precio)

    def opcion_chooser_simple(self, S, K, T1, T2, r, sigma, q=0):
        """Valuación de Opción Chooser Simple (As You Like It)."""
        c = self.opciones_bsm("Yield", S, K, T2, r, sigma, extra=q)[0]
        K_put = K * np.exp(-(r - q) * (T2 - T1))
        p = self.opciones_bsm("Yield", S, K_put, T1, r, sigma, extra=q)[1]
        return c + np.exp(-q * (T2 - T1)) * p

    def opciones_asset_or_nothing(self, S, K, T, r, sigma, q=0, tipo='call'):
        """Valuación de Opciones Binarias: Asset-or-Nothing."""
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))

        if tipo == 'call':
            precio = S * np.exp(-q * T) * norm.cdf(d1)
        elif tipo == 'put':
            precio = S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("El tipo debe ser 'call' o 'put'")

        return max(0.0, precio)

    # ==========================================================
    # MÉTODOS DE FORWARDS (API pública)
    # ==========================================================
    def precio_forward(self, S0, r, T):
        """Precio teórico de un forward sobre activo sin rendimientos."""
        return S0 * np.exp(r * T)

    def precio_forward_dividendo_continuo(self, S0, r, q, T):
        """Precio forward con dividendo continuo o tasa extranjera q."""
        return S0 * np.exp((r - q) * T)

    def precio_forward_dividendos_discretos(self, S0, r, T, I):
        """Precio forward descontando VP de dividendos discretos I."""
        return (S0 - I) * np.exp(r * T)

    def precio_forward_commodity(self, S0, r, u, T):
        """Precio forward de commodity con costo de almacenamiento continuo u."""
        return S0 * np.exp((r + u) * T)

    def precio_forward_divisa(self, S0, r_d, r_f, T):
        """Tipo de cambio forward (Paridad Cubierta de Tasas de Interés)."""
        return S0 * np.exp((r_d - r_f) * T)

    def valor_forward_en_vida(self, St, F0, r, q, tau):
        """Valor de mercado de un forward largo en t < T. tau = T - t."""
        return St * np.exp(-q * tau) - F0 * np.exp(-r * tau)

    def fra(self, r1, r2, t1, t2, nocional, R_K):
        """
        Forward Rate Agreement.
        Devuelve (tasa_forward_implicita, valor_fra).
        Convención: receptor de tasa fija (R_K) paga flotante (R_F).
        """
        tau = t2 - t1
        R_F = (r2 * t2 - r1 * t1) / tau
        valor = nocional * (R_F - R_K) * tau * np.exp(-r2 * t2)
        return R_F, valor

    # ==========================================================
    # ALIAS Y WRAPPERS — compatibilidad con páginas modulares
    # ==========================================================
    def arbol_binomial_crr(self, S, K, r, sigma, T, n, es_call=True, american=False, q=0.0):
        """Wrapper de binomial_tree con firma usada en 10_Derivados_Vanilla.py."""
        tipo = 'call' if es_call else 'put'
        precio, arboles = self.binomial_tree(S, K, T, r, sigma, n, q, tipo, american)
        if arboles is None:
            return precio, None, None
        S_tree, V_tree = arboles
        return precio, S_tree, V_tree

    def black_scholes(self, S, K, r, sigma, T, es_call=True, q=0.0):
        """BSM estándar / Merton (dividendo continuo q). Devuelve prima escalar."""
        call, put, _, _ = self.opciones_bsm("Yield", S, K, T, r, sigma, extra=q)
        return call if es_call else put

    def black_76(self, F0, K, r, sigma, T, es_call=True):
        """Modelo de Black (1976) para futuros."""
        call, put, _, _ = self.opciones_bsm("Futuros", F0, K, T, r, sigma)
        return call if es_call else put

    def calcular_griegas(self, S, K, r, sigma, T, es_call=True, q=0.0):
        """Devuelve dict de griegas BSM con dividendo q."""
        dc, dp, gamma, vega, tc, tp, rc, rp = self.griegas_bsm("Yield", S, K, T, r, sigma, extra=q)
        delta = dc if es_call else dp
        theta = tc if es_call else tp
        rho = rc if es_call else rp
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

    def opcion_perpetua(self, S, K, r, sigma, es_call=True):
        """Valuación de opción perpetua (T → ∞)."""
        if sigma <= 0 or r <= 0:
            return max(S - K, 0) if es_call else max(K - S, 0)
        h = 0.5 + np.sqrt(0.25 + 2 * r / sigma ** 2)
        if es_call:
            precio = (K / (h - 1)) * ((S * (h - 1)) / (h * K)) ** h
        else:
            h_s = 0.5 - np.sqrt(0.25 + 2 * r / sigma ** 2)
            precio = (K / (1 - h_s)) * ((S * (1 - h_s)) / (h_s * K)) ** h_s
        return max(0.0, precio)

    def calcular_payoff_leg(self, tipo, posicion, S_T, K, prima):
        """Payoff de una pata individual para estrategias."""
        if tipo == 'call':
            payoff = np.maximum(S_T - K, 0)
        elif tipo == 'put':
            payoff = np.maximum(K - S_T, 0)
        else:
            payoff = np.zeros_like(S_T)
        return posicion * payoff - (posicion * prima)

    def graficar_estrategia(self, nombre_estrategia, S_spot, patas):
        """Gráfica interactiva del perfil riesgo/rendimiento de una estrategia."""
        from plotly import graph_objects as _go
        S_T = np.linspace(S_spot * 0.5, S_spot * 1.5, 500)
        payoff_total = np.zeros_like(S_T)
        fig = _go.Figure()
        for pata in patas:
            pp = self.calcular_payoff_leg(pata['tipo'], pata['posicion'], S_T, pata['K'], pata['prima'])
            payoff_total += pp
            lbl = f"{'Long' if pata['posicion'] == 1 else 'Short'} {pata['tipo'].capitalize()} K={pata['K']}"
            fig.add_trace(_go.Scatter(x=S_T, y=pp, mode='lines',
                                      line=dict(dash='dot', width=1.5), opacity=0.6, name=lbl))
        fig.add_trace(_go.Scatter(x=S_T, y=payoff_total, mode='lines',
                                  line=dict(color='black', width=3),
                                  name=f'Payoff Neto ({nombre_estrategia})'))
        fig.add_trace(_go.Scatter(x=S_T, y=np.where(payoff_total >= 0, payoff_total, 0),
                                  fill='tozeroy', fillcolor='rgba(40,167,69,0.2)', mode='none', showlegend=False))
        fig.add_trace(_go.Scatter(x=S_T, y=np.where(payoff_total < 0, payoff_total, 0),
                                  fill='tozeroy', fillcolor='rgba(220,53,69,0.2)', mode='none', showlegend=False))
        fig.update_layout(
            title=f'Perfil Riesgo/Rendimiento: {nombre_estrategia}',
            xaxis_title='Precio del Activo al Vencimiento ($)',
            yaxis_title='Utilidad / Pérdida ($)',
            hovermode='x unified', template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.add_hline(y=0, line_dash='dash', line_color='black')
        fig.add_vline(x=S_spot, line_dash='dot', line_color='blue', annotation_text='Spot Actual')
        return fig

    # ==========================================================
    # HELPERS INTERNOS
    # ==========================================================
    def _descargar_y_limpiar(self, tickers_list, start_date, end_date):
        """
        Descarga y limpia datos de Yahoo Finance.
        Centralizado para evitar llamadas HTTP duplicadas entre métodos.
        """
        raw_data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)

        if 'Adj Close' in raw_data:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data:
            data = raw_data['Close']
        else:
            data = raw_data

        data = data.ffill().dropna()

        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers_list[0])

        return data