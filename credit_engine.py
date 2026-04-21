"""
credit_engine.py
----------------
Motor de riesgo de crédito — CreditMetrics (J.P. Morgan, 1997).
Optimizado: Utiliza VaR Paramétrico para el caso independiente y Monte Carlo para el correlacionado.
"""

import numpy as np
from scipy.stats import norm


# =============================================================================
# RATINGS Y DIMENSIONES
# =============================================================================
RATINGS_DEST = [
    "AAA","AA+","AA","AA-","A+","A","A-",
    "BBB+","BBB","BBB-","BB+","BB","BB-",
    "B+","B","B-","CCC/C","D","NR"
]

RATINGS_EMIT = [
    "AAA","AA+","AA","AA-","A+","A","A-",
    "BBB+","BBB","BBB-","BB+","BB","BB-",
    "B+","B","B-","CCC/C","D"
]

RATINGS_VAL = RATINGS_EMIT  
N_EMIT = len(RATINGS_EMIT)   
N_DEST = len(RATINGS_DEST)   
RATING_IDX = {r: i for i, r in enumerate(RATINGS_EMIT)}

TRADING_DAYS = 252


# =============================================================================
# MATRIZ RAW S&P 1981-2021  (17 filas x 19 cols — NR incluido)
# =============================================================================
_TM_RAW_17x19 = np.array([
    [0.8709,0.0586,0.025,0.0068,0.0016,0.0024,0.0013,0.0,0.0005,0.0,0.0003,0.0005,0.0003,0.0,0.0003,0.0,0.0005,0.0,0.031],       
    [0.0221,0.7968,0.1059,0.0338,0.0068,0.0032,0.0018,0.0005,0.0009,0.0005,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0279],               
    [0.0041,0.0131,0.8099,0.0842,0.0263,0.0112,0.0035,0.0038,0.0013,0.0008,0.0005,0.0003,0.0002,0.0002,0.0,0.0002,0.0005,0.0002,0.0399],  
    [0.0004,0.001,0.0368,0.7899,0.0974,0.0218,0.0057,0.0024,0.0014,0.0006,0.0003,0.0,0.0,0.0003,0.0008,0.0,0.0,0.0003,0.0409],    
    [0.0,0.0005,0.0041,0.0423,0.7916,0.0854,0.0204,0.0058,0.0032,0.0008,0.0005,0.0008,0.0001,0.0006,0.0003,0.0,0.0,0.0005,0.0431], 
    [0.0003,0.0004,0.0021,0.0039,0.0519,0.7936,0.0661,0.0232,0.0081,0.0025,0.0009,0.001,0.0006,0.0008,0.0002,0.0,0.0001,0.0005,0.044],   
    [0.0003,0.0001,0.0005,0.0014,0.0038,0.0614,0.7887,0.0727,0.0185,0.0053,0.0012,0.0012,0.001,0.001,0.0003,0.0001,0.0003,0.0005,0.0416], 
    [0.0,0.0001,0.0005,0.0006,0.0019,0.0069,0.0677,0.7665,0.0798,0.015,0.0034,0.0026,0.0012,0.0014,0.0009,0.0002,0.0006,0.0009,0.0498],  
    [0.0001,0.0001,0.0004,0.0002,0.0009,0.0028,0.0095,0.0726,0.7668,0.062,0.0128,0.0063,0.0026,0.002,0.001,0.0003,0.0005,0.0014,0.0577], 
    [0.0001,0.0001,0.0002,0.0004,0.0006,0.0013,0.0023,0.0108,0.0903,0.7287,0.0557,0.0201,0.0081,0.0036,0.0021,0.0015,0.0019,0.0023,0.0699], 
    [0.0004,0.0,0.0,0.0003,0.0003,0.0008,0.0008,0.0037,0.0149,0.1078,0.661,0.077,0.0257,0.0097,0.0051,0.0022,0.0033,0.0031,0.0841], 
    [0.0,0.0,0.0003,0.0001,0.0,0.0005,0.0004,0.0015,0.0056,0.0187,0.095,0.6534,0.0865,0.0236,0.0103,0.0035,0.0048,0.0046,0.091],  
    [0.0,0.0,0.0,0.0001,0.0001,0.0001,0.0004,0.0009,0.0022,0.0032,0.0161,0.094,0.6389,0.086,0.0301,0.0079,0.0071,0.0092,0.1037],  
    [0.0,0.0001,0.0,0.0003,0.0,0.0003,0.0006,0.0004,0.0005,0.001,0.0029,0.0138,0.0823,0.6253,0.0947,0.0258,0.0178,0.0194,0.1147], 
    [0.0,0.0,0.0001,0.0001,0.0,0.0003,0.0003,0.0001,0.0005,0.0003,0.0009,0.002,0.0105,0.0726,0.616,0.1001,0.0396,0.0299,0.1266],  
    [0.0,0.0,0.0,0.0,0.0001,0.0003,0.0,0.0005,0.0005,0.0008,0.0007,0.0016,0.0038,0.0202,0.0965,0.5528,0.1215,0.0589,0.1417],     
    [0.0,0.0,0.0,0.0,0.0002,0.0,0.0007,0.0004,0.0007,0.0004,0.0002,0.0013,0.0034,0.0085,0.0253,0.1004,0.4391,0.2655,0.1539],     
], dtype=float)

DEFAULT_SPREADS = np.array([
    [0.0434,0.0426,0.0427,0.04325,0.0438],   
    [0.0442,0.0434,0.0435,0.04405,0.0446],   
    [0.0450,0.0442,0.0443,0.04485,0.0454],   
    [0.0461,0.0453,0.0454,0.04595,0.0465],   
    [0.0472,0.0464,0.0465,0.04705,0.0476],   
    [0.0483,0.0475,0.0476,0.04815,0.0487],   
    [0.0498,0.0490,0.0491,0.04965,0.0502],   
    [0.0517,0.0509,0.0510,0.05155,0.0521],   
    [0.0536,0.0528,0.0529,0.05345,0.0540],   
    [0.0566,0.0558,0.0559,0.05645,0.0570],   
    [0.0596,0.0588,0.0589,0.05945,0.0600],   
    [0.0642,0.0634,0.0635,0.06405,0.0646],   
    [0.0706,0.0698,0.0699,0.07045,0.0710],   
    [0.0770,0.0762,0.0763,0.07685,0.0774],   
    [0.0851,0.0843,0.0844,0.08495,0.0855],   
    [0.0959,0.0951,0.0952,0.09575,0.0963],   
    [0.1311,0.1303,0.1304,0.13095,0.1315],   
    [np.nan]*5,                               
], dtype=float)

DEFAULT_TREASURY = np.array([0.0365, 0.0357, 0.0358, 0.03635, 0.0369])


# =============================================================================
# FUNCIONES PÚBLICAS
# =============================================================================
NR_METHODS = {
    "raw_with_d":       "Excel clásico — S&P crudas (AAA..D), filas NO suman 1 (NR excluido)",
    "redistribute":     "Redistribuir NR proporcionalmente (uso profesional)",
    "simple_normalize": "Normalizar sin NR — escala simple (ejercicio clasico)",
    "raw_no_d_nr":      "Sin normalizar — ignorar D y NR, usar crudas AAA..CCC/C",
}

def build_transition_matrix(raw_17x19: np.ndarray = None,
                             nr_treatment: str = "redistribute") -> np.ndarray:
    if raw_17x19 is None:
        raw_17x19 = _TM_RAW_17x19.copy()

    raw = raw_17x19.astype(float).copy()

    rated_probs = raw[:, :17]   
    d_probs     = raw[:, 17]    
    nr_probs    = raw[:, 18]    

    if nr_treatment == "raw_with_d":
        raw18 = np.column_stack([rated_probs, d_probs]) 
        d_row = np.zeros(18); d_row[-1] = 1.0
        return np.vstack([raw18, d_row])  

    elif nr_treatment == "redistribute":
        rated_plus_d = np.column_stack([rated_probs, d_probs])
        sums_rated_d = rated_plus_d.sum(axis=1, keepdims=True)
        sums_rated_d[sums_rated_d == 0] = 1.0
        adjusted = rated_plus_d + nr_probs[:, None] * (rated_plus_d / sums_rated_d)
        rs = adjusted.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
        adjusted = adjusted / rs   
        d_row = np.zeros(18); d_row[-1] = 1.0
        return np.vstack([adjusted, d_row])   

    elif nr_treatment == "simple_normalize":
        rated_plus_d = np.column_stack([rated_probs, d_probs])  
        sums = rated_plus_d.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        adjusted = rated_plus_d / sums  
        d_row = np.zeros(18); d_row[-1] = 1.0
        return np.vstack([adjusted, d_row])  

    else:  
        return rated_probs.copy() 

DEFAULT_TM = build_transition_matrix()
RATINGS_NO_D = RATINGS_EMIT[:17]  


def bond_values_per_rating(VN: float, cupon_pct: float, T: int,
                            pagos_ano: int, recovery_pct: float,
                            spreads: np.ndarray,
                            include_d: bool = True,
                            spread_times: np.ndarray = None) -> np.ndarray:
    n_rated = 17
    n_cols  = spreads.shape[1]

    if spread_times is None:
        times = np.arange(1, n_cols + 1, dtype=float)   
    else:
        times = np.asarray(spread_times, dtype=float)

    vals  = np.zeros(n_rated + (1 if include_d else 0))
    cupon = VN * cupon_pct / pagos_ano
    cf_times = np.array([(t + 1) / pagos_ano for t in range(T * pagos_ano)])

    for r_idx in range(n_rated):
        y_row = spreads[r_idx]
        pv = 0.0
        for t_yr in cf_times:
            idx = int(np.argmin(np.abs(times - t_yr)))
            if t_yr > times[-1]:
                idx = len(times) - 1
            y = y_row[idx]
            pv += cupon / (1.0 + y) ** t_yr
            
        idx_T = int(np.argmin(np.abs(times - T)))
        if T > times[-1]:
            idx_T = len(times) - 1
        pv += VN / (1.0 + y_row[idx_T]) ** T
        vals[r_idx] = pv

    if include_d:
        vals[17] = recovery_pct * VN

    return vals


def scale_var_cvar(results: dict,
                   conf_levels=(0.90, 0.95, 0.99, 0.999)) -> dict:
    _k1d  = 1.0 / np.sqrt(TRADING_DAYS)          
    _k10d = np.sqrt(10.0 / TRADING_DAYS)          
    CAPITAL_MULTIPLIER = 3.0                       

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
        }
    return out


def expected_value_and_sigma(bonds_data: list, trans_mat: np.ndarray):
    per_bond = []
    ev_port   = 0.0
    var_port  = 0.0
    for b in bonds_data:
        probs = trans_mat[b['rating_idx']]
        ev_b  = float(np.dot(b['values'], probs))
        var_b = float(np.dot((b['values'] - ev_b) ** 2, probs))
        per_bond.append({"nombre": b.get("nombre","?"), "EV": ev_b, "sigma": var_b**0.5})
        ev_port  += ev_b
        var_port += var_b   

    return per_bond, {"EV_port": ev_port, "sigma_port": var_port ** 0.5}


def var_cvar_parametric(ev: float, sigma: float,
                        conf_levels=(0.90, 0.95, 0.99, 0.999)) -> dict:
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


def gaussian_copula_simulation(bonds_data: list, trans_mat: np.ndarray,
                                corr_mat: np.ndarray,
                                n_sims: int = 50_000,
                                seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n   = len(bonds_data)

    thresholds = []
    for b in bonds_data:
        cum_p = np.cumsum(trans_mat[b['rating_idx']])
        with np.errstate(all='ignore'):
            thresh = norm.ppf(np.clip(cum_p, 1e-15, 1.0 - 1e-15))
        thresh[-1] = np.inf
        thresholds.append(thresh)

    C = corr_mat.copy().astype(float)
    np.fill_diagonal(C, 1.0)
    jitter = 0.0
    while True:
        try:
            L = np.linalg.cholesky(C + np.eye(n) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter = jitter * 2 + 1e-8

    Z = rng.standard_normal((n_sims, n))
    X = Z @ L.T  

    port_vals = np.zeros(n_sims)
    for b_idx, (b, thresh) in enumerate(zip(bonds_data, thresholds)):
        n_vals = len(b['values'])
        r_sim = np.searchsorted(thresh, X[:, b_idx]).clip(0, n_vals - 1)
        port_vals += np.array(b['values'], dtype=float)[r_sim]

    return port_vals


def var_cvar_from_simulations(sim_vals: np.ndarray,
                               conf_levels=(0.90, 0.95, 0.99, 0.999)) -> dict:
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


def thresholds_per_bond(rating_idx: int, trans_mat: np.ndarray) -> np.ndarray:
    cum_p = np.cumsum(trans_mat[rating_idx])
    with np.errstate(all='ignore'):
        thresh = norm.ppf(np.clip(cum_p, 1e-15, 1.0 - 1e-15))
    thresh[-1] = np.inf
    return thresh
