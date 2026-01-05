# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : roe.py
# ===========================================================================

import numpy as np
from physics.euler_equation_1d import *

class Roe:

    def __init__(self, gas):
        self.gas = gas


    def roe_mean(self, UL:np.ndarray, UR:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # +---+ quantités +---+ #
        density_L, velocity_L, pressure_L = cons_to_prim(U=UL, gas=self.gas)
        density_R, velocity_R, pressure_R = cons_to_prim(U=UR, gas=self.gas)

        # +---+ energie +---+ #
        energy_L = UL[..., 2] / UL[..., 0]
        energy_R = UR[..., 2] / UR[..., 0]
        
        # +---+ densité +---+ #
        sqrt_density_L = np.sqrt(density_L)
        sqrt_density_R = np.sqrt(density_R)
        density_tilde = np.sqrt(density_L * density_R)

        # +---+ enthalpie +---+ #
        enthalpy_L = energy_L + pressure_L / density_L
        enthalpy_R = energy_R + pressure_R / density_R

        # +---+ grandeur moyenne de Roe +---+ 
        u_tilde = (sqrt_density_L * velocity_L + sqrt_density_R * velocity_R) / (sqrt_density_L + sqrt_density_R)
        H_tilde = (sqrt_density_L * enthalpy_L + sqrt_density_R * enthalpy_R) / (sqrt_density_L + sqrt_density_R)
        c_tilde = np.sqrt((self.gas.gamma - 1.0) * (H_tilde - 0.5 * u_tilde**2))

        return u_tilde, H_tilde, c_tilde, density_tilde
    

    def roe_eigenvector(self, u_tilde:np.ndarray, H_tilde:np.ndarray, c_tilde:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        r1 = np.array([
            np.ones_like(u_tilde),
            u_tilde - c_tilde,
            H_tilde - u_tilde * c_tilde
        ])

        r2 = np.array([
            np.ones_like(u_tilde),
            u_tilde,
            0.5 * u_tilde**2
        ])

        r3 = np.array([
            np.ones_like(u_tilde),
            u_tilde + c_tilde,
            H_tilde + u_tilde * c_tilde
        ])

        return r1, r2, r3
    

    def roe_eigenvalue(self, u_tilde:np.ndarray, c_tilde:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        lambda1 = u_tilde - c_tilde

        lambda2 = u_tilde 

        lambda3 = u_tilde + c_tilde

        return lambda1, lambda2, lambda3

    
    def roe_wave_amplitude(self, delta_pressure:np.ndarray, delta_density:np.ndarray, delta_velocity:np.ndarray,
                           c_tilde:np.ndarray, density_tilde:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        alpha1 = (1 / (2*c_tilde)) * (delta_pressure - density_tilde * c_tilde * delta_velocity)
        
        alpha2 = delta_density - delta_pressure / c_tilde**2

        alpha3 = (1 / (2*c_tilde)) * (delta_pressure + density_tilde * c_tilde * delta_velocity)

        return alpha1, alpha2, alpha3
    

    def entropy(self, lambda_i, c_tilde, coeff=0.1):

        eps = coeff * c_tilde

        return np.where(
            np.abs(lambda_i) < eps,
            0.5 * (lambda_i**2 / eps + eps),
            np.abs(lambda_i)
        )


    def compute_scheme_flux(self, UL:np.ndarray, UR:np.ndarray) -> np.ndarray:

        # +---+ Flux +---+ #
        FL = compute_flux(U=UL, gas=self.gas)
        FR = compute_flux(U=UR, gas=self.gas)

        # +---+ Grandeur moyenne de Roe +---+ #
        u_tilde, H_tilde, c_tilde, density_tilde = self.roe_mean(UL=UL, UR=UR)

        # +---+ Pression +---+ #
        pressure_L = compute_pressure(U=UL, gas=self.gas)
        pressure_R = compute_pressure(U=UR, gas=self.gas)
        delta_pressure = pressure_R - pressure_L

        # +---+ Densité +---+ #
        delta_density = UR[..., 0] - UL[..., 0]

        # +---+ Vitesse +---+ #
        delta_velocity = (UR[..., 1]/UR[..., 0]) - (UL[..., 1]/UL[..., 0])

        # +---+ Vecteur propre +---+ #
        r1, r2, r3 = self.roe_eigenvector(u_tilde=u_tilde, H_tilde=H_tilde, c_tilde=c_tilde)

        # +---+ Valeur propre +---+ #
        lambda1, lambda2, lambda3 = self.roe_eigenvalue(u_tilde=u_tilde, c_tilde=c_tilde)
        lambda1 = self.entropy(lambda_i=lambda1, c_tilde=c_tilde)
        # lambda2 = self.entropy(lambda_i=lambda2)
        lambda3 = self.entropy(lambda_i=lambda3, c_tilde=c_tilde)

        # +---+ Amplitude d'onde +---+ #
        alpha1, alpha2, alpha3 = self.roe_wave_amplitude(delta_pressure=delta_pressure, delta_density=delta_density, delta_velocity=delta_velocity, 
                                                         c_tilde=c_tilde, density_tilde=density_tilde)

        # +---+ Dissipation +---+ #
        dissipation = (
            lambda1 * alpha1 * r1 + 
            lambda2 * alpha2 * r2 + 
            lambda3 * alpha3 * r3
        )

        return 0.5 * (FL + FR) - 0.5 * dissipation