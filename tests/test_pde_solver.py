"""Tests du solveur EDP forward de Dupire (issue #3).

Organisation :
  1. opérateur spatial L
  2. structure de la sortie, conditions initiale et de bord
  3. comparaisons avec des solutions exactes (Black-Scholes, Bachelier)
  4. ordre de convergence en K et en T
  5. schéma (theta, Rannacher, schéma explicite)
  6. propriétés financières (absence d'arbitrage, moments, monotonies)
  7. invariances (échelle, causalité) et comparaison à une implémentation dense de référence
  8. cas limites numériques (grilles minimales, vol très faible, T très court...)
  9. validation des entrées
"""
import numpy as np
import pytest
from scipy.stats import norm

from dupire.black_scholes import bs_call_price
from dupire.pde_solver import dupire_operator, solve_dupire_pde

S0, R, SIG = 100.0, 0.03, 0.20


def vol_constante(K, T):
    return SIG


def err_vs_bs(K, T, C, r, sigma, strikes=range(80, 121)):
    """Erreur max (à la dernière maturité) entre le prix EDP et Black-Scholes."""
    return max(abs(np.interp(k, K, C[-1]) - bs_call_price(S0, k, T[-1], r, 0.0, sigma))
               for k in strikes)


def bachelier_call(S, K, T, a):
    """Call dans le modèle normal dS = a dW (taux nul)."""
    d = (S - K) / (a * np.sqrt(T))
    return (S - K) * norm.cdf(d) + a * np.sqrt(T) * norm.pdf(d)


def borne_forward(K, T, r):
    """Borne basse sans arbitrage : max(S0 - K e^{-rT}, 0)."""
    return np.maximum(S0 - K[None, :] * np.exp(-r * T[:, None]), 0.0)


# =====================================================================
# 1. Opérateur spatial
# =====================================================================
def test_operateur_exact_sur_polynomes():
    """L est exact sur 1, K et K^2 (différences finies centrées)."""
    K = np.linspace(0, 300, 301)
    sub, diag, sup = dupire_operator(K, 0.5, vol_constante, R)
    Ki = K[1:-1]

    def appliquer(C):
        return sub * C[:-2] + diag * C[1:-1] + sup * C[2:]

    assert np.allclose(appliquer(np.ones_like(K)), 0.0)                       # L 1 = 0
    assert np.allclose(appliquer(K), -R * Ki)                                  # L K = -rK
    assert np.allclose(appliquer(K**2), SIG**2 * Ki**2 - 2 * R * Ki**2)        # L K^2


def test_operateur_accepte_scalaire_ou_tableau():
    K = np.linspace(0, 100, 51)
    s1 = dupire_operator(K, 0.1, lambda K, t: 0.2, R)
    s2 = dupire_operator(K, 0.1, lambda K, t: 0.2 * np.ones_like(K), R)
    for a, b in zip(s1, s2):
        assert np.allclose(a, b)
        assert a.shape == (49,)


# =====================================================================
# 2. Sortie, condition initiale, bords
# =====================================================================
def test_structure_de_la_sortie():
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 300.0, 2.0, 300, 50)
    assert K.shape == (301,) and T.shape == (51,) and C.shape == (51, 301)
    assert K[0] == 0.0 and K[-1] == 300.0 and T[0] == 0.0 and T[-1] == 2.0
    assert np.allclose(np.diff(K), 1.0) and np.allclose(np.diff(T), 2.0 / 50)
    assert np.isfinite(C).all()


def test_condition_initiale_et_bords():
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 300.0, 1.0, 300, 50)
    assert np.allclose(C[0], np.maximum(S0 - K, 0.0))
    assert np.allclose(C[:, 0], S0)
    assert np.allclose(C[:, -1], 0.0)


def test_entiers_flottants_et_numpy_acceptes():
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, 400.0, np.int64(20))
    assert C.shape == (21, 401)


# =====================================================================
# 3. Solutions exactes
# =====================================================================
@pytest.mark.parametrize("sigma, r, T_max, K_max, n_K, n_T", [
    (0.2, 0.00, 1.0, 400, 400, 200),     # taux nul
    (0.2, -0.01, 1.0, 400, 400, 200),    # taux négatif
    (0.2, 0.10, 1.0, 400, 400, 200),     # taux élevé
    (0.3, 0.03, 0.25, 400, 400, 100),    # maturité courte
    (0.2, 0.03, 10.0, 600, 600, 200),    # maturité longue
    (0.5, 0.03, 1.0, 800, 800, 200),     # vol forte
    (1.0, 0.03, 1.0, 1000, 1000, 200),   # vol très forte
    (1.5, 0.03, 1.0, 2000, 2000, 200),   # vol extrême
])
def test_vol_constante_vs_black_scholes(sigma, r, T_max, K_max, n_K, n_T):
    K, T, C = solve_dupire_pde(S0, r, lambda K, t: sigma, K_max, T_max, n_K, n_T)
    assert err_vs_bs(K, T, C, r, sigma) < 6e-3


def test_vol_dependant_du_temps_vs_vol_effective():
    """sigma(T) seul : prix BS avec la vol effective sqrt(int sigma^2 dt / T)."""
    a, b, T_max = 0.15, 0.10, 1.0
    var_tot = a**2 * T_max + a * b * T_max**2 + b**2 * T_max**3 / 3
    sigma_eff = np.sqrt(var_tot / T_max)
    K, T, C = solve_dupire_pde(S0, R, lambda K, t: a + b * t, 400.0, T_max, 400, 200)
    assert err_vs_bs(K, T, C, R, sigma_eff, strikes=range(70, 131)) < 6e-3


def test_vol_dependant_du_strike_vs_bachelier():
    """sigma_loc = a/K (taux nul) <=> modèle normal, qui a une formule exacte."""
    a = 20.0
    K, T, C = solve_dupire_pde(S0, 0.0, lambda K, t: a / K, 400.0, 1.0, 400, 200)
    err = max(abs(C[-1, k] - bachelier_call(S0, k, 1.0, a)) for k in range(50, 151))
    assert err < 6e-3


# =====================================================================
# 4. Convergence
# =====================================================================
def test_convergence_ordre_2_en_K():
    erreurs = []
    for n_K in (100, 200, 400):
        K, T, C = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, n_K, 400)
        erreurs.append(err_vs_bs(K, T, C, R, SIG))
    assert erreurs[0] / erreurs[1] > 3.0
    assert erreurs[1] / erreurs[2] > 3.0


def test_convergence_bachelier_ordre_2_en_K():
    a, erreurs = 20.0, []
    for n_K in (100, 200, 400):
        K, T, C = solve_dupire_pde(S0, 0.0, lambda K, t: a / K, 400.0, 1.0, n_K, 400)
        erreurs.append(max(abs(np.interp(k, K, C[-1]) - bachelier_call(S0, k, 1.0, a))
                           for k in range(50, 151)))
    assert erreurs[0] / erreurs[1] > 3.0
    assert erreurs[1] / erreurs[2] > 3.0


def _err_temporelle(theta, n_rannacher, n_T, ref):
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, 400, n_T,
                               theta=theta, n_rannacher=n_rannacher)
    return np.abs(C[-1] - ref)[80:121].max()


def test_ordre_temporel_crank_nicolson_est_2():
    """Même grille en K, référence n_T = 1600 : l'erreur doit être divisée par ~4."""
    ref = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, 400, 1600)[2][-1]
    e = [_err_temporelle(0.5, 2, n, ref) for n in (25, 50, 100)]
    assert e[0] / e[1] > 3.5 and e[1] / e[2] > 3.5


def test_ordre_temporel_implicite_est_1():
    """Même chose en implicite : l'erreur n'est divisée que par ~2."""
    ref = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, 400, 1600,
                           theta=1.0, n_rannacher=0)[2][-1]
    e = [_err_temporelle(1.0, 0, n, ref) for n in (25, 50, 100)]
    for ratio in (e[0] / e[1], e[1] / e[2]):
        assert 1.7 < ratio < 2.5


def test_crank_nicolson_plus_precis_que_implicite():
    K, T, C1 = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, 400, 200, theta=0.5)
    K, T, C2 = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, 400, 200,
                                theta=1.0, n_rannacher=0)
    assert err_vs_bs(K, T, C1, R, SIG) < err_vs_bs(K, T, C2, R, SIG)


# =====================================================================
# 5. Schéma : Rannacher, theta
# =====================================================================
def _gamma_min(n_rannacher):
    """Plus petite dérivée seconde discrète autour de S0, pas de temps grossiers."""
    K, T, C = solve_dupire_pde(S0, R, lambda K, t: 0.10, 400.0, 0.2, 400, 4,
                               theta=0.5, n_rannacher=n_rannacher)
    return min(np.diff(C[i], 2)[80:120].min() for i in range(1, 5))


def test_rannacher_supprime_les_oscillations():
    assert _gamma_min(0) < -1e-4      # CN seul : densité négative (oscillations)
    assert _gamma_min(2) > -1e-9      # avec Rannacher : densité positive


def test_rannacher_plus_de_pas_que_n_T():
    """n_rannacher > n_T : tous les pas sont lissés, ça doit marcher."""
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 400.0, 1.0, 400, 10, n_rannacher=50)
    assert np.isfinite(C).all() and C.min() >= 0.0
    assert err_vs_bs(K, T, C, R, SIG) < 0.1


def test_theta_zero_stable_avec_petit_pas():
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 200.0, 0.25, 100, 400, theta=0.0)
    assert err_vs_bs(K, T, C, R, SIG) < 0.03


def test_theta_zero_explose_si_pas_trop_grand():
    """Schéma explicite : condition de stabilité (CFL). Avec 20 pas, ça diverge."""
    with np.errstate(all="ignore"):
        K, T, C = solve_dupire_pde(S0, R, vol_constante, 200.0, 0.25, 100, 20,
                                   theta=0.0, n_rannacher=0)
    assert (not np.isfinite(C).all()) or np.abs(C).max() > 1e3


def test_crank_nicolson_stable_meme_avec_tres_gros_pas():
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 200.0, 0.25, 100, 5)
    assert np.isfinite(C).all()
    assert C.min() >= -1e-9 and C.max() <= S0 + 1e-9


# =====================================================================
# 6. Propriétés financières
# =====================================================================
def _verifier_sans_arbitrage(K, T, C, r, tol=1e-5):
    assert np.isfinite(C).all()
    assert (np.diff(C, axis=1) <= tol).all()                     # décroissant en K
    assert (np.diff(C[1:, 20:300], 2, axis=1) >= -tol).all()     # convexe en K (butterfly)
    assert (np.diff(C[:, 20:300], axis=0) >= -tol).all()         # croissant en T (calendar)
    assert (C - borne_forward(K, T, r) >= -tol).all()            # >= S0 - K e^{-rT}
    assert (C <= S0 + tol).all()                                 # <= S0


def test_sans_arbitrage_smile_regulier():
    def smile(K, T):
        return 0.20 - 0.05 * np.tanh((K - 100.0) / 50.0) + 0.02 * np.exp(-T)
    K, T, C = solve_dupire_pde(S0, R, smile, 400.0, 1.0, 400, 200)
    _verifier_sans_arbitrage(K, T, C, R)


@pytest.mark.parametrize("nom, sigma_loc", [
    ("saut en K", lambda K, t: np.where(K < 100, 0.40, 0.15)),
    ("saut en T", lambda K, t: 0.10 if t < 0.5 else 0.50),
    ("très oscillante", lambda K, t: 0.05 + 0.6 * np.abs(np.sin(K / 10))),
    ("vol forte à gauche", lambda K, t: 0.8 * np.exp(-K / 100) + 0.05),
])
def test_sans_arbitrage_vols_discontinues_ou_extremes(nom, sigma_loc):
    K, T, C = solve_dupire_pde(S0, R, sigma_loc, 400.0, 1.0, 400, 200)
    _verifier_sans_arbitrage(K, T, C, R)


def test_moments_de_la_densite():
    """Breeden-Litzenberger : int C_KK = e^{-rT} et int K C_KK = S0 (martingale)."""
    K, T, C = solve_dupire_pde(S0, R, lambda K, t: 0.3, 1000.0, 1.0, 1000, 200)
    dK = K[1] - K[0]
    dens = np.diff(C[-1], 2) / dK**2
    Ki = K[1:-1]
    assert abs(dens.sum() * dK - np.exp(-R * 1.0)) < 1e-4
    assert abs((Ki * dens).sum() * dK - S0) < 1e-2


def test_monotone_en_sigma_et_en_r():
    prix_sigma = [solve_dupire_pde(S0, R, lambda K, t, s=s: s, 400.0, 1.0, 400, 100)[2][-1]
                  for s in (0.1, 0.2, 0.3)]
    assert (np.diff(np.array(prix_sigma), axis=0) >= -1e-10).all()
    prix_r = [solve_dupire_pde(S0, r, vol_constante, 400.0, 1.0, 400, 100)[2][-1]
              for r in (0.0, 0.03, 0.06)]
    assert (np.diff(np.array(prix_r), axis=0) >= -1e-10).all()


# =====================================================================
# 7. Invariances
# =====================================================================
def test_invariance_par_changement_d_echelle():
    """(S0, K) -> (S0/2, K/2) avec la même grille relative : C est divisé par 2."""
    _, _, C1 = solve_dupire_pde(100.0, R, vol_constante, 400.0, 1.0, 400, 100)
    _, _, C2 = solve_dupire_pde(50.0, R, vol_constante, 200.0, 1.0, 400, 100)
    assert np.allclose(C2, 0.5 * C1, atol=1e-10)


def test_causalite_meme_pas_de_temps():
    """Résoudre jusqu'à T=0.5 ou jusqu'à T=1 donne les mêmes prix avant T=0.5."""
    def f(K, t):
        return 0.2 + 0.1 * np.exp(-t) * (1 + np.sin(K / 20))
    _, _, C_long = solve_dupire_pde(S0, R, f, 400.0, 1.0, 400, 200)
    _, _, C_court = solve_dupire_pde(S0, R, f, 400.0, 0.5, 400, 100)
    assert np.allclose(C_long[:101], C_court, atol=1e-12)


def test_appels_a_sigma_loc():
    """sigma_loc reçoit les noeuds intérieurs de K et un temps dans [0, T_max]."""
    appels = []

    def espion(K, t):
        appels.append((K.copy(), t))
        return 0.2

    solve_dupire_pde(S0, R, espion, 300.0, 1.0, 60, 10)
    assert len(appels) > 0
    for K_recu, t in appels:
        assert K_recu.shape == (59,)
        assert K_recu[0] == pytest.approx(5.0) and K_recu[-1] == pytest.approx(295.0)
        assert np.isscalar(t) and 0.0 <= t <= 1.0


def _solveur_dense_reference(S0, r, sigma_loc, K_max, T_max, n_K, n_T, theta, n_rannacher):
    """Implémentation indépendante (matrices denses + np.linalg.solve) du même schéma.

    Sert uniquement à vérifier le solveur bande : mêmes formules, autre algèbre linéaire.
    """
    K = np.linspace(0.0, K_max, n_K + 1)
    dT = T_max / n_T
    C = np.maximum(S0 - K, 0.0)

    def un_pas(C, t_old, dt, th):
        sub, diag, sup = dupire_operator(K, t_old + th * dt, sigma_loc, r)
        L = np.diag(diag) + np.diag(sub[1:], -1) + np.diag(sup[:-1], 1)
        M = len(diag)
        U = C[1:-1]
        b_bord = np.zeros(M)
        b_bord[0] = dt * sub[0] * S0                      # bord K=0 (ancien + nouveau temps)
        rhs = (np.eye(M) + (1 - th) * dt * L) @ U + b_bord
        U_new = np.linalg.solve(np.eye(M) - th * dt * L, rhs)
        return np.concatenate(([S0], U_new, [0.0]))

    for n in range(n_T):
        t = n * dT
        if n < n_rannacher:
            C = un_pas(C, t, dT / 2, 1.0)
            C = un_pas(C, t + dT / 2, dT / 2, 1.0)
        else:
            C = un_pas(C, t, dT, theta)
    return C


@pytest.mark.parametrize("theta, n_rannacher", [(0.5, 0), (0.5, 2), (1.0, 0), (0.0, 0), (0.5, 8)])
def test_identique_a_une_implementation_dense_de_reference(theta, n_rannacher):
    """Vol dépendant de K et de T, gros pas de temps : tous les temps d'évaluation comptent."""
    def f(K, t):
        return 0.15 + 0.25 * t + 0.1 * np.exp(-K / 50.0)

    n_T = 8 if theta != 0.0 else 400                       # theta=0 : petit pas pour rester stable
    _, _, C = solve_dupire_pde(S0, R, f, 300.0, 1.0 if theta != 0.0 else 0.05, 60, n_T,
                               theta=theta, n_rannacher=n_rannacher)
    ref = _solveur_dense_reference(S0, R, f, 300.0, 1.0 if theta != 0.0 else 0.05, 60, n_T,
                                   theta, n_rannacher)
    assert np.allclose(C[-1], ref, atol=1e-10, rtol=0)



# =====================================================================
# 8. Cas limites numériques
# =====================================================================
@pytest.mark.parametrize("n_K, n_T", [(2, 1), (3, 1), (2, 5), (4, 2)])
def test_grilles_minimales(n_K, n_T):
    """Un seul noeud intérieur (n_K=2) : le système bande est de taille 1."""
    K, T, C = solve_dupire_pde(S0, R, vol_constante, 300.0, 1.0, n_K, n_T)
    assert C.shape == (n_T + 1, n_K + 1)
    assert np.isfinite(C).all()
    assert C.min() >= -1e-9 and C.max() <= S0 + 1e-9


def _erreurs_raffinement(sigma, T_max, n_T):
    erreurs = []
    for n_K in (800, 1600, 3200):
        K, T, C = solve_dupire_pde(S0, R, lambda K, t: sigma, 400.0, T_max, n_K, n_T)
        erreurs.append(err_vs_bs(K, T, C, R, sigma))
    return erreurs


def test_vol_tres_faible_converge_par_raffinement():
    """sigma = 1% : le coin du payoff est plus fin que dK=1, il faut raffiner."""
    e = _erreurs_raffinement(0.01, 1.0, 200)
    assert e[0] / e[1] > 3.0 and e[1] / e[2] > 3.0
    assert e[2] < 5e-3


def test_maturite_tres_courte_converge_par_raffinement():
    e = _erreurs_raffinement(0.2, 0.001, 20)
    assert e[0] / e[1] > 3.0 and e[1] / e[2] > 3.0
    assert e[2] < 5e-3


def test_vol_nulle_donne_la_solution_deterministe():
    """sigma = 0 : C = max(S0 - K e^{-rT}, 0). Pas de crash, résultat borné."""
    K, T, C = solve_dupire_pde(S0, R, lambda K, t: 0.0, 400.0, 1.0, 400, 200)
    exact = np.maximum(S0 - K * np.exp(-R * 1.0), 0.0)
    assert np.isfinite(C).all()
    assert C.min() >= -1e-9 and C.max() <= S0 + 1e-9
    assert np.abs(C[-1] - exact).max() < 0.5


def test_K_max_trop_petit_degrade_les_prix():
    """La condition C(K_max)=0 n'est valable que si K_max est grand."""
    erreurs = []
    for K_max in (130, 150, 200, 400):
        K, T, C = solve_dupire_pde(S0, R, lambda K, t: 0.4, float(K_max), 1.0, K_max, 200)
        erreurs.append(err_vs_bs(K, T, C, R, 0.4))
    assert erreurs == sorted(erreurs, reverse=True)       # décroît avec K_max
    assert erreurs[0] > 1.0 and erreurs[-1] < 5e-3


@pytest.mark.parametrize("S", [100.0, 100.25, 100.5, 100.99])
def test_S0_hors_de_la_grille(S):
    """S0 entre deux noeuds : un peu moins précis mais toujours correct."""
    K, T, C = solve_dupire_pde(S, R, vol_constante, 400.0, 1.0, 400, 200)
    err = max(abs(np.interp(k, K, C[-1]) - bs_call_price(S, k, 1.0, R, 0.0, SIG))
              for k in range(80, 121))
    assert err < 6e-3


# =====================================================================
# 9. Validation des entrées
# =====================================================================
@pytest.mark.parametrize("kwargs", [
    dict(S0=0.0), dict(S0=-5.0),
    dict(K_max=100.0), dict(K_max=50.0),
    dict(T_max=0.0), dict(T_max=-1.0),
    dict(n_K=1), dict(n_K=0), dict(n_K=2.5),
    dict(n_T=0), dict(n_T=1.5),
    dict(n_rannacher=-1), dict(n_rannacher=0.5),
    dict(theta=-0.1), dict(theta=1.1),
    dict(r=np.nan), dict(r=np.inf),
])
def test_parametres_invalides(kwargs):
    params = dict(S0=S0, r=R, sigma_loc=vol_constante, K_max=300.0, T_max=1.0,
                  n_K=100, n_T=10, theta=0.5, n_rannacher=2)
    params.update(kwargs)
    with pytest.raises(ValueError):
        solve_dupire_pde(**params)


@pytest.mark.parametrize("sigma_loc", [
    lambda K, t: np.nan,
    lambda K, t: np.inf,
    lambda K, t: np.where(K > 150, np.nan, 0.2),
])
def test_sigma_loc_non_finie(sigma_loc):
    """Une vol locale NaN/inf (ex. issue de local_vol.py) doit lever une erreur claire."""
    with pytest.raises(ValueError, match="non finies"):
        solve_dupire_pde(S0, R, sigma_loc, 300.0, 1.0, 100, 10)


def test_sigma_loc_de_mauvaise_taille():
    with pytest.raises(ValueError):
        solve_dupire_pde(S0, R, lambda K, t: np.ones(3) * 0.2, 300.0, 1.0, 100, 10)
