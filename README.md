# Projet pricing d'option — Volatilité locale de Dupire

Mini-projet de calcul numérique pour la finance : mise en œuvre du **modèle de volatilité locale de Dupire**, résolution numérique de l'EDP associée et analyse des résultats.

**Polytech Nice Sophia — MAM, parcours IMAFA**
**Rendu : 30 octobre 2026, 18h** (rapport PDF de 10 à 15 pages + code)

## Équipe

- Amiel Metier
- Nicolas Muratore
- Tahitowa Senis

## Objectifs

Dans le modèle de Dupire, la volatilité dépend du sous-jacent et du temps :

$$dS_t = r S_t\,dt + \sigma_{loc}(S_t, t)\,S_t\,dW_t$$

Elle est calibrée pour reproduire exactement les prix des options européennes du marché (le *smile*), là où Black-Scholes suppose une volatilité constante. Le projet s'articule autour de quatre étapes :

1. **Pricer de référence** — formules fermées de Black-Scholes et volatilité implicite.
2. **Résolution de l'EDP forward de Dupire** en (K, T) par θ-schéma / Crank-Nicolson, pour une volatilité locale donnée.
3. **Extraction de la volatilité locale** à partir d'une grille de prix de calls européens (formule de Dupire), avec validation aller-retour sur un cas synthétique.
4. **Validation et applications** — Monte Carlo (schéma d'Euler) sous volatilité locale, étude de convergence et de robustesse au bruit, pricing d'une option barrière comparé à Black-Scholes, puis extension sur données réelles (options SPX, paramétrisation SVI).

## Structure du dépôt

```
.
├── src/dupire/
│   ├── black_scholes.py   # formules fermées, vol implicite
│   ├── pde_solver.py      # EDP de Dupire, θ-schéma / Crank-Nicolson
│   ├── local_vol.py       # extraction de σ_loc (formule de Dupire)
│   └── monte_carlo.py     # simulation de l'EDS sous vol locale
├── tests/                 # tests unitaires (pytest)
├── notebooks/             # expériences, figures pour le rapport
├── data/                  # données de marché (non versionnées)
├── report/                # rapport LaTeX
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 ou plus récent.

```bash
git clone <url-du-depot>
cd <nom-du-depot>
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

Lancer les tests :

```bash
pytest
```

Lancer les notebooks :

```bash
jupyter lab
```

Les données de marché sont téléchargées avec `yfinance` et enregistrées dans `data/`, qui n'est pas versionné. Chaque notebook qui en a besoin les régénère.

## Organisation du travail

- Une branche par issue : `feat/<numéro>-<description>` (ex. `feat/3-pde-solver`).
- Pas de push direct sur `main` : chaque modification passe par une pull request relue par un autre membre du groupe.
- Écrire `Closes #<numéro>` dans la description de la PR pour fermer l'issue au merge.
- Toute nouvelle fonction numérique est accompagnée d'au moins un test.

## Références

- B. Dupire, *Pricing with a Smile*, Risk, 1994.
- J. Gatheral, *The Volatility Surface: A Practitioner's Guide*, Wiley, 2006.
- J. Gatheral, A. Jacquier, *Arbitrage-free SVI volatility surfaces*, Quantitative Finance, 2014.
