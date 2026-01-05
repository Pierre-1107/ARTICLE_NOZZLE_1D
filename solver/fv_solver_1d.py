# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : fv_solver_1d.py
# ===========================================================================

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from termcolor import colored

import numpy as np
import yaml 
import matplotlib.pyplot as plt

from physics.euler_equation_1d import *
from physics.source import *

@dataclass
class FVSolution:
    x: np.ndarray
    Density: np.ndarray
    Velocity: np.ndarray
    Pressure: np.ndarray
    Temperature: np.ndarray
    Mach: np.ndarray
    residuals: np.ndarray
    Nt: int
    converged: bool

class FiniteVolumeSolver1D:

    def __init__(self, cfl, scheme, geometry, gas, config_file:str | Path):
        
        self.scheme = scheme
        self.geometry = geometry
        self.gas = gas
        self.CFL = cfl

        self.config_file = Path(config_file)
        with self.config_file.open("r", encoding="utf-8") as file:
            self.cfg = yaml.safe_load(file)

        # +---+ mesh +---+ #
        mesh = self.cfg["mesh"]
        self.x0, self.xL, self.Nx = mesh["x0"], mesh["xL"], mesh["Nx"]
        self.x_mesh = np.linspace(self.x0, self.xL, self.Nx)
        self.dx = np.mean(np.diff(self.x_mesh))

        # +---+ numerics +---+ #
        numerics = self.cfg["numerics"]
        self.Nt = numerics["Nt"]

        # +---+ live_plot +---+ #
        live_plot = self.cfg["live_plot"]
        self.bool_plot = live_plot["bool"]
        self.frequency = live_plot["frequency"]

        # +---+ convergence +---+ #
        convergence = self.cfg["convergence"]
        self.criterion = float(convergence["criterion"])

        # +---+ nozzle geometry +---+ #
        self.A = self.geometry.area(x_mesh=self.x_mesh)
        self.dAdx = self.geometry.dareadx(x_mesh=self.x_mesh)
        self.A_face = 0.5 * (self.A[1:] + self.A[:-1])

        self.A_total_face = np.empty(self.Nx + 1, dtype=float)
        self.A_total_face[1:-1] = 0.5 * (self.A[1:] + self.A[:-1])
        self.A_total_face[0] = self.A[0]
        self.A_total_face[-1] = self.A[-1]

        # +---+ quantities array +---+ #
        self.U = np.zeros((self.Nx + 2, 3))
        self.residuals = np.full((self.Nt, 3), np.nan, dtype=float)

        # +---+ initial conditions +---+ #
        ic = self.cfg.get("initial_conditions", {})
        self.ic_type = ic.get("type", "uniform")


    def initialize(self) -> None:

            # density = np.ones(self.Nx)
            # velocity = 0.1 * np.ones(self.Nx)
            # pressure = np.ones(self.Nx)

            density = 1 - 0.3146 * self.x_mesh
            temperature = 1 - 0.2314 * self.x_mesh
            velocity = (0.1 + 1.09 * self.x_mesh) * np.sqrt(temperature)

            pressure = density * temperature

            self.U[1:-1, :] = prim_to_cons(density, velocity, pressure, self.gas)
            self.apply_bc()


    def apply_bc(self) -> None:
        """
        Anderson BC applied on primitive variables.
        We only reconstruct primitives where needed to avoid NaNs.
        """

        # ---- interior primitives (cells 1:-1) ----
        density_i, velocity_i, _ = cons_to_prim(self.U[1:-1], self.gas)
        temperature_i = compute_temperature(U=self.U[1:-1], gas=self.gas)

        # Create full arrays for primitives including ghosts
        density = np.empty(self.Nx + 2)
        velocity = np.empty(self.Nx + 2)
        temperature = np.empty(self.Nx + 2)

        density[1:-1], velocity[1:-1], temperature[1:-1] = density_i, velocity_i, temperature_i

        # ================= Inlet (index 0) =================
        density[0] = 1.0
        temperature[0] = 1.0
        velocity[0] = 2.0 * velocity[1] - velocity[2]

        # ================= Outlet (index -1) ===============
        density[-1] = 2.0 * density[-2] - density[-3]
        velocity[-1] = 2.0 * velocity[-2] - velocity[-3]
        temperature[-1] = 2.0 * temperature[-2] - temperature[-3]

        # Safety floors (prevents division by 0)
        density = np.maximum(density, 1e-8)
        temperature = np.maximum(temperature, 1e-8)

        # Pressure consistent with your nondim model
        pressure = self.gas.r * density * temperature

        self.U[:] = prim_to_cons(density=density, velocity=velocity, pressure=pressure, gas=self.gas)

    def compute_dt(self) -> float:

        wave_velocity = wave_speed(U=self.U[1:-1], gas=self.gas)

        return self.CFL * self.dx / float(np.max(wave_velocity))

    def compute_steady_residuals_rho_u_T(self) -> tuple[float, float, float]:
        """
        Steady residuals based on flux-source balance, expressed for (rho, u, T).

        Conservative residual:
            R = (AF_{i+1/2} - AF_{i-1/2})/dx - S

        Then convert to primitive residuals:
            R_rho = R[:,0]
            R_u   = (R_rhou - u*R_rho)/rho
            R_T   = (gamma-1) * ( (R_rhoE - E*R_rho)/rho - u*R_u )
        where E = rhoE/rho
        """

        # ---------- flux ----------
        AF = np.zeros((self.Nx + 1, 3))
        for i in range(self.Nx + 1):
            UL = self.U[i]
            UR = self.U[i + 1]
            Fnum = self.scheme.compute_scheme_flux(UL=UL, UR=UR)
            AF[i] = self.A_total_face[i] * Fnum

        # ---------- source ----------
        _, _, p = cons_to_prim(U=self.U[1:-1], gas=self.gas)
        S = nozzle_source(U=self.U[1:-1], pressure=p, dareadx=self.dAdx)

        # ---------- conservative residual per cell ----------
        R = (AF[1:] - AF[:-1]) / self.dx - S
        R_rho = R[:, 0]
        R_rhou = R[:, 1]
        R_rhoE = R[:, 2]

        # ---------- primitives from current state ----------
        rho = self.U[1:-1, 0]
        u = self.U[1:-1, 1] / rho
        E = self.U[1:-1, 2] / rho

        # ---------- primitive residuals ----------
        R_u = (R_rhou - u * R_rho) / rho
        R_E = (R_rhoE - E * R_rho) / rho
        R_T = (self.gas.gamma - 1.0) * (R_E - u * R_u)

        return (
            float(np.max(np.abs(R_rho))),
            float(np.max(np.abs(R_u))),
            float(np.max(np.abs(R_T))),
        )

    def solver(self, dt:float) -> None:

        # +-----+ compute scheme flux +-----+ #
        AF = np.zeros((self.Nx + 1, 3))
        for idx in range(self.Nx + 1):
            UL = self.U[idx]
            UR = self.U[idx + 1]
            scheme_flux = self.scheme.compute_scheme_flux(UL=UL, UR=UR)
            AF[idx] = self.A_total_face[idx] * scheme_flux

        _, _, p = cons_to_prim(U=self.U[1:-1], gas=self.gas)
        S_geom = nozzle_source(
            U=self.U[1:-1],
            pressure=p,
            dareadx=self.dAdx
        )

        AU = self.A[:, None] * self.U[1:-1]
        AU += 0.5 * dt * S_geom
        AU -= (dt / self.dx) * (AF[1:] - AF[:-1])

        U_star = AU / self.A[:, None]
        _, _, p_star = cons_to_prim(U=U_star, gas=self.gas)
        S_star = nozzle_source(
            U=U_star,
            pressure=p_star,
            dareadx=self.dAdx
        )

        AU += 0.5 * dt * S_star

        self.U[1:-1] = AU / self.A[:, None]


    def run_simulation(self) -> FVSolution:

        self.initialize()

        # ================== Live plot ==================
        fig = axes = None
        if self.bool_plot:
            plt.ion()
            fig, axes = plt.subplots(figsize=(10, 4))

            line_rho, = axes.plot([], [], label="Résidu ρ")
            line_u,   = axes.plot([], [], label="Résidu u")
            line_T,   = axes.plot([], [], label="Résidu T")

            axes.set_yscale("log")
            axes.set_xlabel("Itérations")
            axes.set_ylabel("Résidu stationnaire (max)")
            axes.grid(True, alpha=0.75, linestyle='-.')
            axes.set_title(f"Résidus FV - {self.scheme.__class__.__name__}")
            axes.legend()

        converged = False
        last_it = 0

        # ================== Time loop ==================

        print(f"\n        {colored('+========== DÉBUT DE SIMULATION ==========+', 'yellow')}\n")
        print(f"{colored('Schéma numérique :', 'blue')} {self.scheme.__class__.__name__}")
        print(f"{colored('CFL :', 'blue')} {self.CFL}")
        print(f"{colored('Critère de convergence :', 'blue')} RésiduMax < {self.criterion}\n")

        for it in range(self.Nt):
            last_it = it

            self.apply_bc()
            dt = self.compute_dt()
            self.solver(dt)

            # ---- steady residuals (rho, u, T) ----
            res_density, res_velocity, res_temperature = self.compute_steady_residuals_rho_u_T()

            self.residuals[it, 0] = res_density
            self.residuals[it, 1] = res_velocity
            self.residuals[it, 2] = res_temperature

            # ---- live plot ----
            if self.bool_plot and it % self.frequency == 0:
                it_axis = np.arange(it + 1)

                line_rho.set_data(it_axis, self.residuals[:it + 1, 0])
                line_u.set_data(it_axis, self.residuals[:it + 1, 1])
                line_T.set_data(it_axis, self.residuals[:it + 1, 2])

                axes.set_xlim(0, it + 10)
                axes.relim()
                axes.autoscale_view(scaley=True)

                fig.canvas.draw()
                fig.canvas.flush_events()

            res_max = np.max(np.abs(self.residuals[it, :]))

            if it % 200 == 0:

                print(
                    f"Itération : {it} | "
                    f"RésiduMax = {res_max:.3e} | "
                    f"ρ={self.residuals[it,0]:.3e}, "
                    f"u={self.residuals[it,1]:.3e}, "
                    f"T={self.residuals[it,2]:.3e}"
                )

                # ---- convergence test (stationary!) ----
            if np.max(np.abs(self.residuals[it, :])) < self.criterion:

                print(
                    f"Itération : {it} | "
                    f"RésiduMax = {res_max:.3e} | "
                    f"ρ={self.residuals[it,0]:.3e}, "
                    f"u={self.residuals[it,1]:.3e}, "
                    f"T={self.residuals[it,2]:.3e}"
                )

                converged = True
                print(f"\n        {colored('+========== FIN DE SIMULATION ==========+', 'yellow')}")
                print(f"                       {colored('CONVERGENCE', 'red')}\n")
                break

            if it == self.Nt - 1:
                print(f"\n        {colored('+========== FIN DE SIMULATION ==========+', 'yellow')}\n")

        if self.bool_plot:
            plt.ioff()
            plt.show()

        # ================== Post-processing ==================
        density = self.U[1:-1, 0]
        velocity = self.U[1:-1, 1] / density
        pressure = cons_to_prim(self.U[1:-1], self.gas)[2]

        temperature = compute_temperature(U=self.U[1:-1], gas=self.gas)

        a= np.sqrt(self.gas.gamma * temperature)
        Mach = velocity / a

        return FVSolution(
            x=self.x_mesh,
            Density=density,
            Velocity=velocity,
            Pressure=pressure,
            Temperature=temperature,
            Mach=Mach,
            residuals=self.residuals[:last_it + 1, :],
            Nt=last_it,
            converged=converged,
        )
