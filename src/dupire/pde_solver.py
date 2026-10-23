"""Solveur de l'EDP forward de Dupire (issue #3).

EDP (q = 0, S0 fixé) :
    dC/dT = 1/2 * sigma_loc(K, T)^2 * K^2 * d2C/dK2 - r * K * dC/dK

Conditions :
    C(K, 0) = max(S0 - K, 0)   (payoff)
    C(0, T) = S0               (bord gauche)
    C(K_max, T) = 0            (bord droit)

Schéma : différences finies en K, theta-schéma en T
(theta = 0.5 : Crank-Nicolson, theta = 1 : implicite),
avec lissage de Rannacher sur les premiers pas.
"""
import numpy as np
from scipy.linalg import solve_banded


def dupire_operator(K, t, sigma_loc, r):
    """Coefficients de l'opérateur spatial L sur les noeuds intérieurs de K.

    (L C)_j = sub_j * C_{j-1} + diag_j * C_j + sup_j * C_{j+1}

    Retourne (sub, diag, sup), trois vecteurs de taille len(K) - 2.
    """
    dK = K[1] - K[0]
    K_int = K[1:-1]

    # sigma_loc peut renvoyer un scalaire (vol constante) ou un tableau
    sig = np.asarray(sigma_loc(K_int, t), dtype=float) * np.ones_like(K_int)
    if not np.all(np.isfinite(sig)):
        raise ValueError("sigma_loc a renvoyé des valeurs non finies (NaN ou inf)")

    a = 0.5 * sig**2 * K_int**2      # coefficient de d2C/dK2
    b = r * K_int                    # coefficient de dC/dK (avec signe moins dans l'EDP)

    sub = a / dK**2 + b / (2 * dK)
    diag = -2 * a / dK**2
    sup = a / dK**2 - b / (2 * dK)
    return sub, diag, sup


def solve_dupire_pde(S0, r, sigma_loc, K_max, T_max, n_K, n_T,
                     theta=0.5, n_rannacher=2):
    """Résout l'EDP forward de Dupire et renvoie (K, T, C).

    C[i, j] = prix du call de strike K[j] et de maturité T[i].

    Paramètres
    ----------
    sigma_loc : fonction (K, T) -> vol locale ; K est un tableau, T un scalaire.
    n_K, n_T  : nombre d'intervalles en K et en T.
    theta     : 0.5 (Crank-Nicolson) ou 1.0 (implicite).
              theta < 0.5 est conditionnellement stable (pas de temps très petit).
    n_rannacher : nombre de premiers pas remplacés par 2 demi-pas implicites.

    Conseils : choisir K_max / n_K pour que S0 tombe sur un noeud de la grille,
    et K_max assez grand (>= 3 ou 4 fois S0) pour que la condition C(K_max) = 0
    soit raisonnable.
    """
    # ---- Vérification des paramètres ----
    if S0 <= 0:
        raise ValueError("S0 doit être strictement positif")
    if K_max <= S0:
        raise ValueError("K_max doit être supérieur à S0")
    if T_max <= 0:
        raise ValueError("T_max doit être strictement positif")
    if not np.isfinite(r):
        raise ValueError("r doit être fini")
    if int(n_K) != n_K or n_K < 2:
        raise ValueError("n_K doit être un entier >= 2")
    if int(n_T) != n_T or n_T < 1:
        raise ValueError("n_T doit être un entier >= 1")
    if int(n_rannacher) != n_rannacher or n_rannacher < 0:
        raise ValueError("n_rannacher doit être un entier >= 0")
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta doit être dans [0, 1]")
    n_K, n_T = int(n_K), int(n_T)

    # ---- Grilles et condition initiale ----
    dT = T_max / n_T
    K = np.linspace(0.0, K_max, n_K + 1)
    T = np.linspace(0.0, T_max, n_T + 1)

    C = np.zeros((n_T + 1, n_K + 1))
    C[0, :] = np.maximum(S0 - K, 0.0)

    # ---- Un pas de temps : (I - theta*dt*L) C_new = (I + (1-theta)*dt*L) C_old ----
    def step(C_cur, t_old, dt, cur_theta):
        t_eval = t_old + cur_theta * dt          # temps où l'on évalue la vol locale
        sub, diag, sup = dupire_operator(K, t_eval, sigma_loc, r)

        # Membre de droite : U + (1-theta)*dt*L U  (noeuds intérieurs)
        U = C_cur[1:-1]
        LU = diag * U
        LU[1:] += sub[1:] * U[:-1]               # voisin de gauche
        LU[:-1] += sup[:-1] * U[1:]              # voisin de droite
        RHS = U + (1.0 - cur_theta) * dt * LU

        # Bord K = 0 : C(0, t) = S0 à l'ancien (explicite) et au nouveau (implicite) temps
        RHS[0] += (1.0 - cur_theta) * dt * sub[0] * S0 + cur_theta * dt * sub[0] * S0
        # Bord K = K_max : C = 0, rien à ajouter

        # Matrice (I - theta*dt*L) au format bande de solve_banded
        M = len(diag)
        ab = np.zeros((3, M))
        ab[0, 1:] = -cur_theta * dt * sup[:-1]   # sur-diagonale
        ab[1, :] = 1.0 - cur_theta * dt * diag   # diagonale
        ab[2, :-1] = -cur_theta * dt * sub[1:]   # sous-diagonale

        U_new = solve_banded((1, 1), ab, RHS)

        C_new = np.empty_like(C_cur)
        C_new[0] = S0
        C_new[1:-1] = U_new
        C_new[-1] = 0.0
        return C_new

    # ---- Boucle en temps (avec lissage de Rannacher) ----
    for n in range(n_T):
        if n < n_rannacher:
            dt_half = 0.5 * dT
            C_mid = step(C[n], T[n], dt_half, 1.0)
            C[n + 1] = step(C_mid, T[n] + dt_half, dt_half, 1.0)
        else:
            C[n + 1] = step(C[n], T[n], dT, theta)

    return K, T, C
