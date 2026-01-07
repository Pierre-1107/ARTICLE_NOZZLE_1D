# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : euler_equation_1d.py
# ===========================================================================

# ===== DEFINITION ===== #
# -> cons : conservatif (rho, rhoV, rhoE)
# -> prim : primitif (rho, V, p)

import numpy as np
from physics.gas import Gas


def cons_to_prim(U:np.ndarray, gas:Gas) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    density = U[..., 0]
    momentum_density = U[..., 1]
    volumetric_density_energy = U[..., 2]

    velocity = momentum_density / density
    energy = volumetric_density_energy / density 

    pressure = (gas.gamma - 1.0) * density * (energy - 0.5 * velocity**2)

    return density, velocity, pressure


def prim_to_cons(density:np.ndarray, velocity:np.ndarray, 
                 pressure:np.ndarray, gas:Gas) -> np.ndarray:
    
    energy = pressure / ((gas.gamma - 1.0) * density) + 0.5 * velocity**2

    return np.stack([density, density * velocity, density * energy], axis=-1)


def sound_speed(density:np.ndarray, pressure:np.ndarray, gas:Gas) -> np.ndarray:

    return np.sqrt(gas.gamma * pressure/density)


def compute_pressure(U:np.ndarray, gas:Gas) -> np.ndarray:

    _, _, pressure = cons_to_prim(U=U, gas=gas)

    return pressure


def compute_temperature(U:np.ndarray, gas:Gas) -> np.ndarray:

    density = U[..., 0]
    momentum_density = U[..., 1]
    volumetric_density_energy = U[..., 2]

    velocity = momentum_density / density
    energy = volumetric_density_energy / density

    temperature = (gas.gamma - 1.0) * (energy - 0.5 * velocity**2)
    
    return temperature


def compute_flux(U:np.ndarray, gas:Gas) -> np.ndarray:

    density, velocity, pressure = cons_to_prim(U=U, gas=gas)
    volumetric_density_energy = U[..., 2]

    F1 = density * velocity
    F2 = density * velocity**2 + pressure
    F3 = (volumetric_density_energy + pressure) * velocity

    return np.stack([F1, F2, F3], axis=-1)


def wave_speed(U:np.ndarray, gas:Gas) -> np.ndarray:

    density, velocity, pressure = cons_to_prim(U=U, gas=gas)
    sound_velocity = sound_speed(density=density, pressure=pressure, gas=gas)

    return np.abs(velocity) + sound_velocity