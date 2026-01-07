# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : maccormack.py
# ===========================================================================

from maccormack_scheme import *
from termcolor import colored
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# +-----+ solve +-----+ #
def solve_nozzle_maccormack(
    x:np.ndarray,
    gamma:float,
    CFL:float,
    Nt:int,
    live_plot:bool = False,
):

    Nx = len(x)
    dx = np.mean(np.diff(x))

    # +-----+ initial conditions & geometry +-----+ #
    A, _ = nozzle_geometry(x_mesh=x)
    rho0, u0, T0 = physics_initialisation(x_mesh=x)

    RHO = np.zeros((Nt, Nx))
    U = np.zeros((Nt, Nx))
    T = np.zeros((Nt, Nx))
    RES = np.zeros((Nt, 3))

    RHO[0], U[0], T[0] = rho0, u0, T0

    # +-----+ live plot +-----+ #
    if live_plot:
        import matplotlib.pyplot as plt
        plt.ion()

        fig, ax = plt.subplots(figsize=(10, 5))
        line_rho, = ax.plot([], [], label=r"$\rho$")
        line_u,   = ax.plot([], [], label=r"$u$")
        line_T,   = ax.plot([], [], label=r"$T$")

        ax.set_yscale("log")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Residual")
        ax.set_title("MacCormack residuals")
        ax.grid(True)
        ax.legend()

    # --- boucle temporelle ---
    for it in range(Nt - 1):

        dt = time_step(U[it, :], T[it], CFL, dx)

        RHO[it+1], U[it+1], T[it+1], RES[it] = mac_cormack_scheme(
            RHO[it], U[it], T[it], A, dx, dt, gamma
        )

        # --- Live plot mise à jour ---
        if live_plot and it % 10 == 0:
            it_axis = np.arange(it + 1)

            line_rho.set_data(it_axis, RES[:it+1, 0])
            line_u.set_data(it_axis, RES[:it+1, 1])
            line_T.set_data(it_axis, RES[:it+1, 2])

            ax.set_xlim(0, it + 10)
            ax.relim()
            ax.autoscale_view(scaley=True)

            fig.canvas.draw()
            fig.canvas.flush_events()

    if live_plot:
        plt.ioff()
        plt.show()

    return {
        "x": x,
        "Density": RHO[-1],
        "Velocity": U[-1],
        "Pressure": RHO[-1] * T[-1],
        "Temperature": T[-1],
        "Mach": U[-1] / np.sqrt(T[-1]),
        "residuals": RES,
    }


def main():
    # +-----+ configuration +-----+ #
        # yaml
    with open("reference/maccomack_config.yaml") as file:
        cfg = yaml.safe_load(file)

        # mesh
    x_mesh = np.linspace(
        cfg["mesh"]["x0"], cfg["mesh"]["xL"], cfg["mesh"]["Nx"]
    )

    # +-----+ solve maccoramck +-----+ #
    solution = solve_nozzle_maccormack(
        x=x_mesh,
        gamma=cfg["physics"]["gamma"],
        CFL=cfg["numerics"]["CFL"],
        Nt=cfg["numerics"]["Nt"],
        live_plot=cfg["live_plot"]["bool"]
    )

    # +-----+ état de référence +-----+ #
    REF = pd.read_csv("reference/anderson_reference.csv")

    # +-----+ figure +-----+ #
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Solution de référence - MacCormack (tuyère quasi-1D)", fontsize=24)

    axes[0, 0].plot(
        solution['x'], solution['Density'], 
        '-.o', color='black',  markerfacecolor='none', label="Code")
    axes[0, 0].scatter(
        REF['x'], REF['Density'],
        marker='x', color='red', s=40, label="Anderson"
    )

    axes[0, 1].plot(solution['x'], solution['Velocity'],
        '-.o', color='black',  markerfacecolor='none', label="Code")
    axes[0, 1].scatter(
        REF['x'], REF['Velocity'],
        marker='x', color='red', s=40, label="Anderson"
    )

    axes[1, 0].plot(solution['x'], solution['Temperature'],
        '-.o', color='black',  markerfacecolor='none', label="Code")
    axes[1, 0].scatter(
        REF['x'], REF['Temperature'],
        marker='x', color='red', s=40, label="Anderson"
    )

    axes[1, 1].plot(solution['x'], solution['Mach'],
        '-.o', color='black',  markerfacecolor='none', label="Code")
    axes[1, 1].scatter(
        REF['x'], REF['Mach'],
        marker='x', color='red', s=40, label="Anderson"
    )

    ylabel = ["Densité", "Vitesse", "Température", "Mach"]
    for idx, ax in enumerate(axes.flat):
        ax.grid('on', alpha=0.75, linestyle='-.')
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel(ylabel[idx], fontsize=16)
        ax.legend(fontsize=16)

    print(colored("\nValidation du cas de référence (Anderson)", "cyan", attrs=["bold"]))
    print(colored("-" * 41, "cyan"))
    for idx, quantity in enumerate(["Density", "Velocity", "Temperature", "Mach"]):
        err = compute_error(ref=REF[quantity], sol=solution[quantity])
        label = colored(f"{quantity:<12}", "yellow")
        value = colored(f"{err:6.3f} %", "green")
        print(f"{label} : {value}")
    print(colored("-" * 41 + "\n", "cyan"))

    root_dir = Path(__file__).resolve().parents[1]
    mac_dir = root_dir/"results"/"Reference-MacCormack"
    mac_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(mac_dir / "maccormack_reference_figure.png", dpi=300)

    plt.show()

    # +-----+ save solution +-----+ #
    np.savez(mac_dir / "maccormack_reference_solution", **solution)

if __name__ == "__main__":
    main()