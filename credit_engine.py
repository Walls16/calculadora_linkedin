"""
credit_engine.py
----------------
Motor de riesgo de crédito — CreditMetrics (J.P. Morgan, 1997).
Separado de financial_engine.py para mantener la modularidad.

Funcionalidades:
  - Matriz de transición S&P 1981-2021 con 19 ratings (AAA..D + NR)
  - Redistribución correcta de NR (proporcional entre estados AAA..D)
  - Valoración de bonos por calificación destino
  - Distribución por convolución exacta (caso independiente)
  - Cópula Gaussiana Monte Carlo (caso correlacionado)
  - VaR y CVaR crediticio para niveles de confianza arbitrarios
  - Escalado a 1 día, 10 días y capital regulatorio (3 × VaR 10d)
"""

import numpy as np
from scipy.stats import norm


# =============================================================================
# RATINGS Y DIMENSIONES
# =============================================================================
# 19 ratings de destino (columnas): AAA..CCC/C, D, NR
RATINGS_DEST = [
    "AAA","AA+","AA","AA-","A+","A","A-",
    "BBB+","BBB","BBB-","BB+","BB","BB-",
    "B+","B","B-","CCC/C","D","NR"
]

# 18 ratings de emisor (filas): AAA..CCC/C   (D es absorbente, no sale como emisor)
# Nota: la fila D se agrega manualmente como estado absorbente
RATINGS_EMIT = [
    "AAA","AA+","AA","AA-","A+","A","A-",
    "BBB+","BBB","BBB-","BB+","BB","BB-",
    "B+","B","B-","CCC/C","D"
]

# Para modelos de valor usamos solo 18 destinos (sin NR)
RATINGS_VAL = RATINGS_EMIT  # mismo orden
N_EMIT = len(RATINGS_EMIT)   # 18
N_DEST = len(RATINGS_DEST)   # 19  (con NR)
RATING_IDX = {r: i for i, r in enumerate(RATINGS_EMIT)}

# Días de trading anuales (usado para escalar VaR/CVaR)
TRADING_DAYS = 252


# =============================================================================
# MATRIZ RAW S&P 1981-2021  (17 filas x 19 cols — NR incluido)
# Fuente: S&P Global Ratings Research, "Default, Transition and Recovery" 2021.
# Rows: AAA..CCC/C  (17 emisores calificados; D se añade como absorbente)
# Cols: AAA..D,NR   (19 destinos)
# =============================================================================
_TM_RAW_17x19 = np.array([
    [0.8709,0.0586,0.025,0.0068,0.0016,0.0024,0.0013,0.0,0.0005,0.0,0.0003,0.0005,0.0003,0.0,0.0003,0.0,0.0005,0.0,0.031],       # AAA
    [0.0221,0.7968,0.1059,0.0338,0.0068,0.0032,0.0018,0.0005,0.0009,0.0005,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0279],               # AA+
    [0.0041,0.0131,0.8099,0.0842,0.0263,0.0112,0.0035,0.0038,0.0013,0.0008,0.0005,0.0003,0.0002,0.0002,0.0,0.0002,0.0005,0.0002,0.0399],  # AA
    [0.0004,0.001,0.0368,0.7899,0.0974,0.0218,0.0057,0.0024,0.0014,0.0006,0.0003,0.0,0.0,0.0003,0.0008,0.0,0.0,0.0003,0.0409],    # AA-
    [0.0,0.0005,0.0041,0.0423,0.7916,0.0854,0.0204,0.0058,0.0032,0.0008,0.0005,0.0008,0.0001,0.0006,0.0003,0.0,0.0,0.0005,0.0431], # A+
    [0.0003,0.0004,0.0021,0.0039,0.0519,0.7936,0.0661,0.0232,0.0081,0.0025,0.0009,0.001,0.0006,0.0008,0.0002,0.0,0.0001,0.0005,0.044],   # A
    [0.0003,0.0001,0.0005,0.0014,0.0038,0.0614,0.7887,0.0727,0.0185,0.0053,0.0012,0.0012,0.001,0.001,0.0003,0.0001,0.0003,0.0005,0.0416], # A-
    [0.0,0.0001,0.0005,0.0006,0.0019,0.0069,0.0677,0.7665,0.0798,0.015,0.0034,0.0026,0.0012,0.0014,0.0009,0.0002,0.0006,0.0009,0.0498],  # BBB+
    [0.0001,0.0001,0.0004,0.0002,0.0009,0.0028,0.0095,0.0726,0.7668,0.062,0.0128,0.0063,0.0026,0.002,0.001,0.0003,0.0005,0.0014,0.0577], # BBB
    [0.0001,0.0001,0.0002,0.0004,0.0006,0.0013,0.0023,0.0108,0.0903,0.7287,0.0557,0.0201,0.0081,0.0036,0.0021,0.0015,0.0019,0.0023,0.0699], # BBB-
    [0.0004,0.0,0.0,0.0003,0.0003,0.0008,0.0008,0.0037,0.0149,0.1078,0.661,0.077,0.0257,0.0097,0.0051,0.0022,0.0033,0.0031,0.0841], # BB+
    [0.0,0.0,0.0003,0.0001,0.0,0.0005,0.0004,0.0015,0.0056,0.0187,0.095,0.6534,0.0865,0.0236,0.0103,0.0035,0.0048,0.0046,0.091],  # BB
    [0.0,0.0,0.0,0.0001,0.0001,0.0001,0.0004,0.0009,0.0022,0.0032,0.0161,0.094,0.6389,0.086,0.0301,0.0079,0.0071,0.0092,0.1037],  # BB-
    [0.0,0.0001,0.0,0.0003,0.0,0.0003,0.0006,0.0004,0.0005,0.001,0.0029,0.0138,0.0823,0.6253,0.0947,0.0258,0.0178,0.0194,0.1147], # B+
    [0.0,0.0,0.0001,0.0001,0.0,0.0003,0.0003,0.0001,0.0005,0.0003,0.0009,0.002,0.0105,0.0726,0.616,0.1001,0.0396,0.0299,0.1266],  # B
    [0.0,0.0,0.0,0.0,0.0001,0.0003,0.0,0.0005,0.0005,0.0008,0.0007,0.0016,0.0038,0.0202,0.0965,0.5528,0.1215,0.0589,0.1417],     # B-
    [0.0,0.0,0.0,0.0,0.0002,0.0,0.0007,0.0004,0.0007,0.0004,0.0002,0.0013,0.0034,0.0085,0.0253,0.1004,0.4391,0.2655,0.1539],     # CCC/C
], dtype=float)


# Default all-in yields (Treasury + spread) por rating destino (AAA..CCC/C) x año (1..5)
DEFAULT_SPREADS = np.array([
    [0.0434,0.0426,0.0427,0.04325,0.0438],   # AAA
    [0.0442,0.0434,0.0435,0.04405,0.0446],   # AA+
    [0.0450,0.0442,0.0443,0.04485,0.0454],   # AA
    [0.0461,0.0453,0.0454,0.04595,0.0465],   # AA-
    [0.0472,0.0464,0.0465,0.04705,0.0476],   # A+
    [0.0483,0.0475,0.0476,0.04815,0.0487],   # A
    [0.0498,0.0490,0.0491,0.04965,0.0502],   # A-
    [0.0517,0.0509,0.0510,0.05155,0.0521],   # BBB+
    [0.0536,0.0528,0.0529,0.05345,0.0540],   # BBB
    [0.0566,0.0558,0.0559,0.05645,0.0570],   # BBB-
    [0.0596,0.0588,0.0589,0.05945,0.0600],   # BB+
    [0.0642,0.0634,0.0635,0.06405,0.0646],   # BB
    [0.0706,0.0698,0.0699,0.07045,0.0710],   # BB-
    [0.0770,0.0762,0.0763,0.07685,0.0774],   # B+
    [0.0851,0.0843,0.0844,0.08495,0.0855],   # B
    [0.0959,0.0951,0.0952,0.09575,0.0963],   # B-
    [0.1311,0.1303,0.1304,0.13095,0.1315],   # CCC/C
    [np.nan]*5,                               # D — usa tasa de recuperación
], dtype=float)

DEFAULT_TREASURY = np.array([0.0365, 0.0357, 0.0358, 0.03635, 0.0369])


# =============================================================================
# FUNCIONES PÚBLICAS
# =============================================================================

NR_METHODS = {
    "raw_with_d":       "Excel clásico — S&P crudas (AAA..D)",
    "redistribute":     "Redistribuir NR proporcionalmente",
    "simple_normalize": "Normalizar sin NR — escala simple (ejercicio clasico)",
    "raw_no_d_nr":      "Sin normalizar — ignorar D y NR, usar crudas AAA..CCC/C",
}


def build_transition_matrix(raw_17x19: np.ndarray = None,
                             nr_treatment: str = "redistribute") -> np.ndarray:
    """
    Construye la matriz de transición de trabajo a partir de la raw 17x19.

    nr_treatment:
      'redistribute'    → redistribuye NR proporcionalmente entre AAA..D
                          → retorna (18×18), filas suman 1 exactamente
      'simple_normalize'→ descarta NR, renormaliza los 18 cols (AAA..D) a 1
                          → retorna (18×18), filas suman 1 exactamente
      'raw_no_d_nr'     → usa AAA..CCC/C tal como aparecen en la fuente,
                          SIN normalizar, SIN columna D.
                          Las filas NO suman 1 (ejercicio de clase clásico).
                          → retorna (17×17) para que la lógica sea coherente.
                          En este modo bond_values solo calcula 17 valores.
    """
    if raw_17x19 is None:
        raw_17x19 = _TM_RAW_17x19.copy()

    raw = raw_17x19.astype(float).copy()   # (17, 19)

    # Separar columnas: 0..16=AAA..CCC/C, 17=D, 18=NR
    rated_probs = raw[:, :17]   # AAA..CCC/C  (17 cols)
    d_probs     = raw[:, 17]    # D
    nr_probs    = raw[:, 18]    # NR

    if nr_treatment == "raw_with_d":
        # Probabilidades S&P crudas, columnas AAA..D (18 destinos), SIN NR.
        # Las filas NO suman 1 — NR se excluye sin redistribuir (= método Excel).
        # Este es el modo clásico de los libros de texto / ejercicios de CreditMetrics.
        # Devuelve (18, 18): filas 0..16 suman < 1;  fila D = estado absorbente.
        raw18 = np.column_stack([rated_probs, d_probs])  # (17, 18)
        d_row = np.zeros(18); d_row[-1] = 1.0
        return np.vstack([raw18, d_row])   # (18, 18)

    elif nr_treatment == "redistribute":
        # Redistribuir NR proporcionalmente entre los 18 estados (AAA..CCC/C + D)
        rated_plus_d = np.column_stack([rated_probs, d_probs])  # (17, 18)
        sums_rated_d = rated_plus_d.sum(axis=1, keepdims=True)
        sums_rated_d[sums_rated_d == 0] = 1.0
        adjusted = rated_plus_d + nr_probs[:, None] * (rated_plus_d / sums_rated_d)
        # Normalize to sum exactly 1
        rs = adjusted.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
        adjusted = adjusted / rs   # (17, 18)
        d_row = np.zeros(18); d_row[-1] = 1.0
        return np.vstack([adjusted, d_row])   # (18, 18)

    elif nr_treatment == "simple_normalize":
        # Descartar NR, renormalizar (AAA..D) a 1 via escala simple
        rated_plus_d = np.column_stack([rated_probs, d_probs])  # (17, 18)
        sums = rated_plus_d.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        adjusted = rated_plus_d / sums   # (17, 18)
        d_row = np.zeros(18); d_row[-1] = 1.0
        return np.vstack([adjusted, d_row])   # (18, 18)

    else:  # 'raw_no_d_nr'
        # Usar AAA..CCC/C tal como son, sin D, sin NR, sin normalizar
        # Retorna (17, 17) — coherente para ejercicios de clase
        # NOTA: las filas suman < 1 porque NR y D están excluidos
        # En este modo se asume que el residuo "desaparece" (no default, no NR)
        return rated_probs.copy()   # (17, 17)


# Matriz por defecto lista para usar (18x18)
DEFAULT_TM = build_transition_matrix()
# Constante de dimension que depende del modo
_TM_SIZE_BY_MODE = {"raw_with_d": 18, "redistribute": 18, "simple_normalize": 18, "raw_no_d_nr": 17}
# Ratings visibles en modo raw (17 estados, sin D)
RATINGS_NO_D = RATINGS_EMIT[:17]  # AAA..CCC/C


def bond_values_per_rating(VN: float, cupon_pct: float, T: int,
                            pagos_ano: int, recovery_pct: float,
                            spreads: np.ndarray,
                            include_d: bool = True,
                            spread_times: np.ndarray = None) -> np.ndarray:
    """
    Calcula el valor del bono si migra a cada posible rating destino.

    Parámetros
    ----------
    VN            : Valor nominal
    cupon_pct     : Tasa cupón anual (0.05 = 5%)
    T             : Años al vencimiento (entero)
    pagos_ano     : Pagos por año (1=anual, 2=semestral, 4=trimestral…)
    recovery_pct  : Tasa de recuperación en D (0.43 = 43%)
    spreads       : np.ndarray shape (>=17, n_cols) — tasa all-in por calificación
    include_d     : True → retorna 18 valores (AAA..D); False → 17 (sin D, modo raw)
    spread_times  : np.ndarray de los tiempos en años que corresponden a las columnas
                    de spreads. Ej. [0.5,1,1.5,2,2.5,3] para semestral, [1,2,3,4,5]
                    para anual. Si None, se asume [1,2,…,n_cols] (anual).

    Devuelve
    --------
    np.ndarray de shape (18,) ó (17,)
    """
    n_rated = 17
    n_cols  = spreads.shape[1]

    # Construct time grid for the yield curve columns
    if spread_times is None:
        times = np.arange(1, n_cols + 1, dtype=float)   # [1,2,…,n_cols]
    else:
        times = np.asarray(spread_times, dtype=float)

    vals  = np.zeros(n_rated + (1 if include_d else 0))
    cupon = VN * cupon_pct / pagos_ano
    # Cash flow times
    cf_times = np.array([(t + 1) / pagos_ano for t in range(T * pagos_ano)])

    for r_idx in range(n_rated):
        y_row = spreads[r_idx]
        pv = 0.0
        for t_yr in cf_times:
            # Find the yield for this cash-flow time via nearest available tenor
            idx = int(np.argmin(np.abs(times - t_yr)))
            # If the exact time is beyond the last available, use the last
            if t_yr > times[-1]:
                idx = len(times) - 1
            y = y_row[idx]
            pv += cupon / (1.0 + y) ** t_yr
        # Principal discount at maturity T
        idx_T = int(np.argmin(np.abs(times - T)))
        if T > times[-1]:
            idx_T = len(times) - 1
        pv += VN / (1.0 + y_row[idx_T]) ** T
        vals[r_idx] = pv

    if include_d:
        vals[17] = recovery_pct * VN

    return vals


def independent_distribution(bonds_data: list, trans_mat: np.ndarray) -> list:
    """
    Distribución exacta del portafolio via convolución iterativa.
    Funciona con matrices 18x18 (modos redistribute/simple_normalize)
    y también con 17x17 (modo raw_no_d_nr).

    bonds_data : list de dicts con 'rating_idx' y 'values'
    trans_mat  : np.ndarray (N x N) donde N=18 ó 17

    Mejoras de rendimiento vs. versión original:
      - np.bincount (buffered) reemplaza np.add.at (unbuffered) → ~3-5x más rápido
      - Se filtran probabilidades insignificantes antes del producto exterior
    """
    # Redondeo a 2 decimales colapsa valores casi idénticos, reduciendo el
    # espacio de estados de K^N a ~10^3..10^4 para 5-10 bonos.
    decimals = 2

    probs0 = trans_mat[bonds_data[0]['rating_idx']]
    vals0  = np.round(bonds_data[0]['values'], decimals).astype(float)
    # SAFETY: en modo raw_no_d_nr la matriz es 17×17 pero si cm_tm fue
    # sobreescrito con una 18×18 (bug de sesión), truncamos al mínimo.
    _n0    = min(len(probs0), len(vals0))
    probs0 = probs0[:_n0]
    vals0  = vals0[:_n0]
    mask0  = probs0 > 1e-14
    cur_vals  = vals0[mask0]
    cur_probs = probs0[mask0]

    for b in bonds_data[1:]:
        pb = trans_mat[b['rating_idx']]
        vb = np.round(b['values'], decimals).astype(float)
        _nb = min(len(pb), len(vb))
        pb, vb = pb[:_nb], vb[:_nb]
        pmask = pb > 1e-14
        pb, vb = pb[pmask], vb[pmask]

        # Vectorized outer product
        joint_probs = np.outer(cur_probs, pb).ravel()
        joint_vals  = np.round((cur_vals[:, None] + vb[None, :]).ravel(), decimals)

        # --- CORRECCIÓN DE RENDIMIENTO ---
        # np.unique devuelve índices inversos; np.bincount(inv, weights) es
        # ~3-5x más rápido que np.add.at porque opera en modo buffered.
        unique_v, inv = np.unique(joint_vals, return_inverse=True)
        agg_p = np.bincount(inv, weights=joint_probs, minlength=len(unique_v))

        keep = agg_p > 1e-14
        cur_vals, cur_probs = unique_v[keep], agg_p[keep]

    return sorted(zip(cur_vals.tolist(), cur_probs.tolist()))


def var_cvar_from_distribution(sorted_dist: list,
                                conf_levels=(0.90, 0.95, 0.99, 0.999),
                                normalize: bool = True) -> dict:
    """
    VaR y CVaR a partir de una distribución discreta ordenada (ascending).

    normalize : Si True (por defecto), normaliza las probabilidades a 1.

    Devuelve dict: {conf: {'EV', 'sigma', 'VaR', 'CVaR', 'q', 'sum_probs'}}

    CORRECCIÓN: el cálculo del cuantil usaba searchsorted(...) - 1, lo cual
    desplazaba el índice hacia un valor con CDF < alpha (sub-cuantil).
    Ahora se usa directamente el índice que devuelve searchsorted, que apunta
    al primer elemento con CDF >= alpha (definición estándar del cuantil inferior).
    """
    vals  = np.array([d[0] for d in sorted_dist], dtype=float)
    probs = np.array([d[1] for d in sorted_dist], dtype=float)
    sum_p = float(probs.sum())

    if normalize:
        probs_n = probs / sum_p if sum_p > 0 else probs
    else:
        probs_n = probs

    ev    = float(np.dot(vals, probs_n))
    var2  = float(np.dot((vals - ev) ** 2, probs_n))
    sigma = float(var2 ** 0.5)
    cum   = np.cumsum(probs_n)

    out = {}
    for conf in conf_levels:
        alpha = 1.0 - conf

        # CORRECCIÓN: searchsorted devuelve el primer i donde cum[i] >= alpha.
        # Ese es el cuantil inferior correcto; NO restar 1.
        idx   = int(np.searchsorted(cum, alpha, side='left'))
        idx   = min(idx, len(vals) - 1)   # clamping defensivo
        q_val = float(vals[idx])

        var_v = max(ev - q_val, 0.0)

        # CVaR = E[V | V <= q]  →  pérdida esperada en la cola
        tail_mask = vals <= q_val
        tail_sum  = probs_n[tail_mask].sum()
        if tail_sum > 1e-15:
            cvar_v = max(
                ev - float(np.dot(vals[tail_mask], probs_n[tail_mask]) / tail_sum),
                0.0,
            )
        else:
            cvar_v = var_v

        out[conf] = {
            "EV": ev, "sigma": sigma,
            "VaR": var_v, "CVaR": cvar_v,
            "q": q_val, "sum_probs": sum_p,
        }

    return out


# =============================================================================
# ESCALADO TEMPORAL: 1 día · 10 días · Capital regulatorio
# =============================================================================

def scale_var_cvar(results: dict,
                   conf_levels=(0.90, 0.95, 0.99, 0.999)) -> dict:
    """
    Escala los VaR/CVaR anuales de CreditMetrics a horizontes regulatorios
    usando la raíz cuadrada del tiempo (sqrt-of-time rule).

    CreditMetrics es un modelo de 1 año. La conversión estándar es:
        VaR_1d  = VaR_1y  / √252
        VaR_10d = VaR_1d  × √10  =  VaR_1y × √(10/252)
        CVaR    escala igual que VaR
        Capital = 3 × VaR_10d  (multiplicador de Basilea II/III)

    Parámetros
    ----------
    results     : dict devuelto por var_cvar_from_distribution o
                  var_cvar_from_simulations (keyed by confidence level)
    conf_levels : niveles de confianza a incluir

    Devuelve
    --------
    dict: {conf: {
        'EV', 'sigma',
        'VaR_1y', 'CVaR_1y',          # horizonte original (1 año)
        'VaR_1d', 'CVaR_1d',           # 1 día de trading
        'VaR_10d', 'CVaR_10d',         # 10 días de trading
        'Capital',                      # 3 × VaR_10d
        'q', 'sum_probs'               # info auxiliar (si existe)
    }}
    """
    _k1d  = 1.0 / np.sqrt(TRADING_DAYS)          # ÷ √252
    _k10d = np.sqrt(10.0 / TRADING_DAYS)          # × √(10/252)
    CAPITAL_MULTIPLIER = 3.0                       # Basilea II/III

    out = {}
    for conf in conf_levels:
        if conf not in results:
            continue
        r = results[conf]
        var_1y  = r["VaR"]
        cvar_1y = r["CVaR"]

        var_1d   = var_1y  * _k1d
        cvar_1d  = cvar_1y * _k1d
        var_10d  = var_1y  * _k10d
        cvar_10d = cvar_1y * _k10d
        capital  = CAPITAL_MULTIPLIER * var_10d

        out[conf] = {
            "EV":      r["EV"],
            "sigma":   r["sigma"],
            "VaR_1y":  var_1y,
            "CVaR_1y": cvar_1y,
            "VaR_1d":  var_1d,
            "CVaR_1d": cvar_1d,
            "VaR_10d": var_10d,
            "CVaR_10d": cvar_10d,
            "Capital": capital,
            # campos opcionales
            "q":        r.get("q"),
            "sum_probs": r.get("sum_probs"),
        }
    return out


def gaussian_copula_simulation(bonds_data: list, trans_mat: np.ndarray,
                                corr_mat: np.ndarray,
                                n_sims: int = 50_000,
                                seed: int = None) -> np.ndarray:
    """
    Cópula Gaussiana Monte Carlo para el caso correlacionado.

    Parámetros
    ----------
    bonds_data : list de dicts con 'rating_idx' y 'values'
    trans_mat  : (18, 18) matriz de transición normalizada
    corr_mat   : (n, n) matriz de correlación entre activos (proxy accionaria)
    n_sims     : número de simulaciones
    seed       : semilla aleatoria (None = aleatorio en cada ejecución)

    Devuelve
    --------
    np.ndarray (n_sims,) — valores simulados del portafolio
    """
    rng = np.random.default_rng(seed)
    n   = len(bonds_data)

    # Umbrales N^-1 para cada bono
    thresholds = []
    for b in bonds_data:
        cum_p = np.cumsum(trans_mat[b['rating_idx']])
        with np.errstate(all='ignore'):
            thresh = norm.ppf(np.clip(cum_p, 1e-15, 1.0 - 1e-15))
        thresh[-1] = np.inf
        thresholds.append(thresh)

    # Descomposición de Cholesky (con jitter si no es definida positiva)
    C = corr_mat.copy().astype(float)
    np.fill_diagonal(C, 1.0)
    jitter = 0.0
    while True:
        try:
            L = np.linalg.cholesky(C + np.eye(n) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter = jitter * 2 + 1e-8

    # Simular variables normales correlacionadas
    Z = rng.standard_normal((n_sims, n))
    X = Z @ L.T   # (n_sims, n)

    # Mapear a calificaciones y calcular valor del portafolio
    port_vals = np.zeros(n_sims)
    for b_idx, (b, thresh) in enumerate(zip(bonds_data, thresholds)):
        n_vals = len(b['values'])
        r_sim = np.searchsorted(thresh, X[:, b_idx]).clip(0, n_vals - 1)
        port_vals += np.array(b['values'], dtype=float)[r_sim]

    return port_vals


def var_cvar_from_simulations(sim_vals: np.ndarray,
                               conf_levels=(0.90, 0.95, 0.99, 0.999)) -> dict:
    """
    VaR y CVaR a partir de valores simulados.

    Devuelve dict: {conf: {'EV', 'sigma', 'VaR', 'CVaR', 'q'}}
    """
    ev    = float(np.mean(sim_vals))
    sigma = float(np.std(sim_vals))
    out   = {}
    for conf in conf_levels:
        q    = float(np.quantile(sim_vals, 1.0 - conf))
        var  = max(ev - q, 0.0)
        tail = sim_vals[sim_vals <= q]
        cvar = max(ev - float(tail.mean()), 0.0) if len(tail) > 0 else var
        out[conf] = {"EV": ev, "sigma": sigma, "VaR": var, "CVaR": cvar, "q": q}
    return out


def var_cvar_parametric(ev: float, sigma: float,
                        conf_levels=(0.90, 0.95, 0.99, 0.999)) -> dict:
    """
    VaR y CVaR paramétrico (aproximación normal — método clásico de CreditMetrics).

    Utiliza la media y varianza ya calculadas de la distribución exacta del
    portafolio, y asume que ésta es Normal(E[V], σ) para obtener el VaR.

        VaR(α)  = Φ⁻¹(α) · σ                    (pérdida respecto a E[V])
        CVaR(α) = φ(Φ⁻¹(α)) / (1 − α) · σ       (cola esperada bajo normal)

    Éste es el "VaR Paramétrico" que muestra la hoja de Excel (ver paso 5):
    Para σ = 16.13  →  VaR_95% ≈ 26.54,  VaR_99% ≈ 37.54,  VaR_99.9% ≈ 49.86

    Parámetros
    ----------
    ev          : E[V] del portafolio (calculado de la distribución exacta)
    sigma       : σ del portafolio   (calculado de la distribución exacta)
    conf_levels : niveles de confianza (fracción decimal)

    Devuelve
    --------
    dict: {conf: {'EV', 'sigma', 'VaR', 'CVaR'}}
    """
    out = {}
    for conf in conf_levels:
        z    = float(norm.ppf(conf))
        var  = max(z * sigma, 0.0)
        cvar = max(float(norm.pdf(z)) / (1.0 - conf) * sigma, var)
        out[conf] = {
            "EV":    ev,
            "sigma": sigma,
            "VaR":   var,
            "CVaR":  cvar,
        }
    return out


def thresholds_per_bond(rating_idx: int, trans_mat: np.ndarray) -> np.ndarray:
    """
    Calcula los umbrales N^-1(P_acumulada) para un bono dado.
    Usados en la cópula gaussiana y para mostrar al usuario.

    Devuelve np.ndarray (18,) — último elemento = +inf
    """
    cum_p = np.cumsum(trans_mat[rating_idx])
    with np.errstate(all='ignore'):
        thresh = norm.ppf(np.clip(cum_p, 1e-15, 1.0 - 1e-15))
    thresh[-1] = np.inf
    return thresh


def expected_value_and_sigma(bonds_data: list, trans_mat: np.ndarray):
    """
    Calcula E[V] y σ[V] para cada bono y para el portafolio total
    (solo válido en el caso independiente).

    Devuelve
    --------
    list de dicts con {'nombre', 'EV', 'sigma'} por bono
    dict con {'EV_port', 'sigma_port_min' (cota inferior independiente)}
    """
    per_bond = []
    ev_port   = 0.0
    var_port  = 0.0
    for b in bonds_data:
        probs = trans_mat[b['rating_idx']]
        ev_b  = float(np.dot(b['values'], probs))
        var_b = float(np.dot((b['values'] - ev_b) ** 2, probs))
        per_bond.append({"nombre": b.get("nombre","?"), "EV": ev_b, "sigma": var_b**0.5})
        ev_port  += ev_b
        var_port += var_b   # independiente: varianzas suman

    return per_bond, {"EV_port": ev_port, "sigma_port": var_port ** 0.5}
