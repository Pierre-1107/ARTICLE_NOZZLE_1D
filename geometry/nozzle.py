# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : nozzle.py
# ===========================================================================

import numpy as np
from abc import ABC, abstractmethod

class NozzleGeometry(ABC):
    """
    Docstring pour NozzleGeometry

    Classe abstraite pour toutes géométries de tuyère 1D
    """

    @abstractmethod
    def area(self, x_mesh:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def dareadx(self, x_mesh:np.ndarray) -> np.ndarray:
        pass


class AndersonNozzle(NozzleGeometry):
    """
    Docstring pour AndersonNozzle

    Tuyère convergente - divergente utilisée par J.D Anderson (Modern Compressible Flow)
    Annexe B

    A(x) = 1 + 2.2 * (x - 1.5)^2
    """

    def __init__(self, x_throat:float=1.5):
        self.x_throat = x_throat

    def area(self, x_mesh:np.ndarray) -> np.ndarray:
        return 1.0 + 2.2 * (x_mesh - self.x_throat)**2
    
    def dareadx(self, x_mesh:np.ndarray) -> np.ndarray:
        return 4.4 * (x_mesh - self.x_throat)