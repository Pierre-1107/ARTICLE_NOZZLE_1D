# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : maccormack_scheme.py
# ===========================================================================

import numpy as np

def physics_initialisation(x_mesh:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    density = 1 - 0.3146 * x_mesh
    temperature = 1 - 0.2314 * x_mesh
    velocity = (0.1 + 1.09 * x_mesh) * np.sqrt(temperature)

    return density, velocity, temperature


def nozzle_geometry(x_mesh:np.ndarray, x_throat:float=1.5) -> tuple[np.ndarray, np.ndarray]:

    area = 1 + 2.2 * (x_mesh - x_throat)**2
    dareadx = 4.4 * (x_mesh - x_throat)

    return area, dareadx


def boundary_conditions(density:np.ndarray, velocity:np.ndarray, temperature:np.ndarray):

    density[0] = 1.0
    temperature[0] = 1.0
    velocity[0] = 2 * velocity[1] - velocity[2]

    density[-1] = 2 * density[-2] - density[-3]
    velocity[-1]   = 2 * velocity[-2] - velocity[-3]
    temperature[-1] = 2 * temperature[-2] - temperature[-3]

    return density, velocity, temperature


def time_step(velocity:np.ndarray, temperature:np.ndarray, CFL:float, dx:float) -> float:

    try:
        dt = CFL * dx / (np.abs(velocity) + np.sqrt(temperature)) 
        return np.min(dt)
    
    except Exception as e:
        raise RuntimeError(
            f"Simulation stopped in scheme << MacCormack >>: {e}"
        )
    

def compute_error(ref:np.ndarray, sol:np.ndarray) -> float:

    return 100 * np.max(np.divide(np.abs(sol - ref), ref))


def mac_cormack_scheme(RHO:np.ndarray, V:np.ndarray, T:np.ndarray, A:np.ndarray, 
                         dx:float, dt:float, gamma:float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # +--> expression des vecteurs
    n = len(RHO)

    RHO_PRED = RHO.copy()
    V_PRED = V.copy()
    T_PRED = T.copy()

    dRHO = np.zeros_like(RHO_PRED)
    dV = np.zeros_like(V_PRED)
    dT = np.zeros_like(T_PRED)

    dRHO_PRED = np.zeros_like(RHO_PRED)
    dV_PRED = np.zeros_like(V_PRED)
    dT_PRED = np.zeros_like(T_PRED)

    RHO_NEW = np.zeros_like(RHO)
    V_NEW = np.zeros_like(V)
    T_NEW = np.zeros_like(T)

    # +---+  schéma de prédiction +---+ 

        # +--> boucle spatiale

    for idx in range(n - 1):

            # -> continuité
        dRHO[idx] = (1/dx) * (
            -RHO[idx] * (V[idx+1] - V[idx])
            - RHO[idx] * V[idx] * (np.log(A[idx+1]) - np.log(A[idx]))
            - V[idx] * (RHO[idx+1] - RHO[idx])
        )

            # -> quantité de mouvement
        dV[idx] = - V[idx] * (V[idx+1] - V[idx]) / dx \
                    - (1/gamma) * ((T[idx+1] - T[idx]) / dx
                    + (T[idx]/RHO[idx]) * (RHO[idx+1] - RHO[idx]) / dx)

            # -> énergie
        dT[idx] = - V[idx] * (T[idx+1] - T[idx]) / dx \
                    - (gamma-1) * T[idx] * (
                        (V[idx+1] - V[idx]) / dx
                        + V[idx] * (np.log(A[idx+1]) - np.log(A[idx])) / dx
                    )
        
            # -> mise à jour des grandeurs
        RHO_PRED[idx] = RHO[idx] + dt * dRHO[idx]
        V_PRED[idx] = V[idx] + dt * dV[idx]
        T_PRED[idx] = T[idx] + dt * dT[idx]

        # +--> application des conditions aux limites
    RHO_PRED[0] = RHO[0]
    T_PRED[0]   = T[0]
    V_PRED[0]   = 2*V_PRED[1] - V_PRED[2]

    RHO_PRED[-1] = 2*RHO_PRED[-2] - RHO_PRED[-3]
    V_PRED[-1]   = 2*V_PRED[-2] - V_PRED[-3]
    T_PRED[-1]   = 2*T_PRED[-2] - T_PRED[-3]

    # +---+  schéma de correction +---+ 

    for idx in range(1, n):

        # -> densité 
        dRHO_PRED[idx] = - RHO_PRED[idx] * (V_PRED[idx] - V_PRED[idx-1]) / dx \
                    - RHO_PRED[idx] * V_PRED[idx] * (np.log(A[idx]) - np.log(A[idx-1])) / dx \
                    - V_PRED[idx] * (RHO_PRED[idx] - RHO_PRED[idx-1]) / dx

        # -> quantité de mouvement
        dV_PRED[idx] = - V_PRED[idx] * (V_PRED[idx] - V_PRED[idx-1]) / dx \
                - (1/gamma) * (
                    (T_PRED[idx] - T_PRED[idx-1]) / dx
                    + (T_PRED[idx]/RHO_PRED[idx]) *
                        (RHO_PRED[idx] - RHO_PRED[idx-1]) / dx
                )

        # -> énergie
        dT_PRED[idx] = - V_PRED[idx] * (T_PRED[idx] - T_PRED[idx-1]) / dx \
                - (gamma-1) * T_PRED[idx] * (
                    (V_PRED[idx] - V_PRED[idx-1]) / dx
                    + V_PRED[idx] * (np.log(A[idx]) - np.log(A[idx-1])) / dx
                )

        # -> mise à jour des grandeurs
    RHO_NEW = RHO + 0.5 * (dRHO_PRED + dRHO) * dt
    V_NEW = V + 0.5 * (dV_PRED + dV) * dt
    T_NEW = T + 0.5 * (dT_PRED + dT) * dt

        # -> application des conditions aux limites
    RHO_NEW, V_NEW, T_NEW = boundary_conditions(density=RHO_NEW, velocity=V_NEW, temperature=T_NEW)

    RES_REF = [
        np.mean(np.abs(0.5 * (dRHO_PRED + dRHO))), np.mean(np.abs(0.5 * (dV_PRED + dV))), np.mean(np.abs(0.5 * (dT_PRED + dT)))
    ]

    return RHO_NEW, V_NEW, T_NEW, RES_REF