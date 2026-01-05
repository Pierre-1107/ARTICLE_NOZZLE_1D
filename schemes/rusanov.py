# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : rusanov.py
# ===========================================================================

import numpy as np
from physics.euler_equation_1d import *

class Rusanov:

    def __init__(self, gas):
        self.gas = gas

    def compute_scheme_flux(self, UL:np.ndarray, UR:np.ndarray) -> np.ndarray:

        FL = compute_flux(U=UL, gas=self.gas)
        FR = compute_flux(U=UR, gas=self.gas)

        s = np.max([
            wave_speed(U=UL, gas=self.gas),
            wave_speed(U=UR, gas=self.gas)
        ])

        return 0.5 * (FL + FR) - 0.5 * s * (UR - UL)