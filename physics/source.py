# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : source.py
# ===========================================================================

import numpy as np

def nozzle_source(U:np.ndarray, pressure:np.ndarray, dareadx:np.ndarray) -> np.ndarray:

    S_geom = np.zeros_like(U)
    S_geom[..., 1] = pressure * dareadx

    return S_geom