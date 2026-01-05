# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : gas.py
# ===========================================================================

from dataclasses import dataclass

@dataclass(frozen=True)
class Gas:
    gamma:float = 1.4
    r:float = 1 # normalisation pour être conforme au modèle de Anderson
    Cp:float= r / (gamma-1.0)
    Cv:float = (gamma * r) / (gamma - 1)