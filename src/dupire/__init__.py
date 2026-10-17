"""Modèle de volatilité locale de Dupire.

Modules :

- ``black_scholes`` : formules fermées de Black-Scholes et volatilité implicite ;
- ``pde_solver`` : EDP forward de Dupire en (K, T), θ-schéma / Crank-Nicolson ;
- ``local_vol`` : extraction de σ_loc depuis une grille de prix (formule de Dupire) ;
- ``monte_carlo`` : simulation de l'EDS sous volatilité locale.
"""

__version__ = "0.1.0"

__all__ = ["black_scholes", "pde_solver", "local_vol", "monte_carlo"]
