# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : main.py
# ===========================================================================

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from physics.gas import Gas
from geometry.nozzle import AndersonNozzle
from schemes.rusanov import Rusanov
from schemes.roe import Roe
from solver.fv_solver_1d import FiniteVolumeSolver1D

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

def main():

    config_file = Path("fv_config.yaml")
    with config_file.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    CFL = cfg["numerics"]["CFL"]

    SCHEME_REGISTRY = {
        "Rusanov": Rusanov,
        "Roe": Roe
    }

    SchemeClass = SCHEME_REGISTRY[cfg["scheme"]["name"]]
    gas = Gas(gamma=1.4)
    geometry = AndersonNozzle()
    scheme = SchemeClass(gas=Gas)

    for cfl in CFL:

        solver = FiniteVolumeSolver1D(
            cfl=cfl,
            scheme=scheme, 
            geometry=geometry, 
            gas=gas, 
            config_file="fv_config.yaml"
        )

        solution = solver.run_simulation()

        # +-----+ dossier de sauvegarde +-----+ #
        scheme_name = scheme.__class__.__name__
        root_dir = Path(__file__).resolve().parents[0]
        save_dir = root_dir/"results"/f"Schema-{scheme_name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # +-----+ graphique +-----+ #
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))

        axes[0, 0].plot(solution.x, solution.Density)
        axes[0, 0].set_ylabel("Densité")

        axes[0, 1].plot(solution.x, solution.Velocity)
        axes[0, 1].set_ylabel("Vitesse")

        axes[1, 0].plot(solution.x, solution.Temperature)
        axes[1, 0].set_ylabel("Température")

        axes[1, 1].plot(solution.x, solution.Mach)
        axes[1, 1].set_ylabel("Mach")

        for ax in axes.flat:
            ax.set_xlabel("x")
            ax.grid(True, alpha=0.75, linestyle="-.")

        fig.suptitle(f"Solution FV - {scheme_name} (tuyère quasi-1D)")
        fig.savefig(save_dir / f"CFL_{cfl}_Schema_{scheme_name}_figure.png", dpi=300)

        plt.show()

        # +-----+ save solution +-----+ #
        np.savez(
            save_dir / f"CFL_{cfl}_Schema_{scheme_name}_solution",
            **vars(solution)
        )


if __name__ == "__main__":
    main()