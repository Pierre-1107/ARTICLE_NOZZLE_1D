# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : post_processing.py
# ===========================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class PostProcess:

    def __init__(self, result_dir:str|Path):

        self.REFERENCE_FOLDER = "Reference-MacCormack"

        self.result_dir = Path(result_dir)
        self.check_output_dir()

        self.folders_names = [
            dir.name for dir in self.result_dir.iterdir() 
            if dir.is_dir()
        ]
        self.check_folders()

        self.nickname = {
            "Pressure": "P",
            "Temperature": "T",
            "Density": r"\rho",
            "Velocity": "V",
            "Mach": "M"
        }

        root_dir = Path(__file__).resolve().parents[1]
        self.comparison_dir = root_dir/"results"/"Comparaison"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)


    def check_output_dir(self) -> None:

        if not self.result_dir.exists():
            raise FileNotFoundError(
                f"[PostProcess] Le dossier '{self.result_dir}' est introuvable."
            )
        
        if not self.result_dir.is_dir():
            raise NotADirectoryError(
                f"[PostProcess] '{self.result_dir}' n'est pas un dossier."
            )


    def check_folders(self) -> None:

        if not self.folders_names:
            raise RuntimeError(
                f"[PostProcess] Aucun dossier détecté dans '{self.result_dir.name}'."
            )
        
        if not self.REFERENCE_FOLDER in self.folders_names:
            raise FileNotFoundError(
                f"[PostProcess] Le dossier de référence '{self.REFERENCE_FOLDER}' est introuvable."
            )
        
        if ('Reference-MacCormack' in self.folders_names) and (len(self.folders_names) == 1):
            raise RuntimeError(
                f"[PostProcess] Seul le fichier de référence est disponible, impossible de faire des comparaisons."
            )


    def load_solution(self, npz_file) -> dict[str, np.ndarray]:

        with np.load(npz_file) as data:
            return {key: data[key] for key in data.files}
        

    def parse_solution_filename(self, npz_file:str|Path) -> tuple[float, str]:

        npz_file = Path(npz_file)
        stem = npz_file.stem
        parts = stem.split("_")

        try:
            cfl = float(parts[1])
            scheme = parts[3]
        except(IndexError, ValueError):
            raise ValueError(
                f"[PostProcess] Nom de fichier invalide : '{npz_file}'"
            )
        
        return cfl, scheme


    def compare_CFL(
            self, 
            scheme_name:str, 
            fields:list=['Density', 'Velocity', 'Pressure', 'Temperature', 'Mach']
        ) -> None:

        # +---+ Normalisation des données +---+ #
        if isinstance(fields, str):
            fields = [fields]

        # +---+ Vérification du dossier de Solver +---+ #
        if scheme_name not in self.folders_names:
            raise FileNotFoundError(
                f"[PostProcess] Le Dossier '{scheme_name}' est introuvable dans '{self.folders_names}'"
            )

        # +---+ Gestion du fichier .npz de référence +---+ #
        ref_dir = self.result_dir / self.REFERENCE_FOLDER
        ref_npz = list(ref_dir.glob("*.npz"))
        ref_sol = self.load_solution(npz_file=ref_npz[0])

       # +---+ Gestion des fichiers .npz de Solver +---+
        solver_dir = self.result_dir / scheme_name
        npz_files = list(solver_dir.glob("*.npz"))

        if not npz_files:
            raise RuntimeError(
                f"[PostProcess] Aucun fichier .npz trouvé dans '{solver_dir}"
            )

        # +---+ Extraction des solutions des fichiers .npz +---+ #
        all_solutions = {}
        all_cfl = np.zeros(shape=len(npz_files))

        for idx, file in enumerate(npz_files):
            all_cfl[idx], _ = self.parse_solution_filename(npz_file=file)
            solution = self.load_solution(npz_file=file)
            all_solutions[f'CFL_{all_cfl[idx]}'] = solution

        # +---+ Gestion de la figure +---+ #

            # +--> nom du fichier
        fields_str = "-".join(fields)
        scheme_str = scheme_name.split("-", 1)[1]
        cfl_str = "-".join(str(cfl_) for cfl_ in sorted(all_cfl))
        file_name = (
            f"comparaison_"
            f"schema_{scheme_str}_"
            f"champs-{fields_str}_"
            f"CFLs-{cfl_str}.png"
        )

        n_fields = len(fields)
        fig, axes = plt.subplots(
            nrows=n_fields,
            ncols=2,
            figsize=(14, 4*n_fields),
            squeeze=False,
            sharex=True
        )
        fig.suptitle(
            f"Comparaison MacCormack - {scheme_name}", 
            fontsize=18
        )

        # +---+ Boucle sur les champs +---+ #
        for idx, field in enumerate(fields):

            # +--> Champs de référence
            axes[idx, 0].plot(
                ref_sol['x'],
                ref_sol[field],
                'k-o', 
                label=f"Référence Anderson"
            )

            for cfl, sol in zip(all_cfl, all_solutions.values()):

                # +--> Champs issus des solutions
                axes[idx, 0].plot(
                    sol['x'], 
                    sol[field],
                    label=f"CFL={cfl}"
                )

                # +--> Alignement des grilles
                if sol[field].shape != ref_sol[field].shape:
                    ref_field = np.interp(
                        sol['x'], 
                        ref_sol['x'], 
                        ref_sol[field]
                    )
                else:
                    ref_field = ref_sol[field]

                # +--> Erreur absolue
                axes[idx, 1].plot(
                    sol['x'], 
                    np.abs(sol[field] - ref_field), 
                    label=rf"CFL={cfl}, $|{self.nickname[field]}_{{ref}} - {self.nickname[field]}_{{sol}}|$"
                )
            
            # +---+ Mise en forme des graphiques +---+ #
            axes[idx, 0].set_ylabel(field, fontsize=14)
            axes[idx, 1].set_ylabel(f"Erreur absolue : {field}", fontsize=14)
            axes[idx, 1].set_yscale("log")

            for col in range(2):
                axes[idx, col].grid('on', alpha=0.75, linestyle='-.')
                axes[idx, col].legend(loc='best', ncol=2, fontsize=10)

            if idx == n_fields - 1:
                axes[idx, 0].set_xlabel("x", fontsize=14)
                axes[idx, 1].set_xlabel("x", fontsize=14)

            axes[0, 0].set_title("Champ")
            axes[0, 1].set_title("Erreur absolue")

        # plt.tight_layout()
        fig.savefig(self.comparison_dir / file_name, dpi=300)
        plt.show()

    
    def compare_scheme(
            self, 
            schemes_names:list[str]|str,
            fields:list[str]|str = ['Density', 'Velocity', 'Pressure', 'Temperature', 'Mach'],
            comp_multi_cfl:bool = False,
            target_cfl:float|None = None
        ) -> None:

        # +---+ Cohérence dans la gestion des CFL +---+ #
        if comp_multi_cfl and target_cfl is not None:
            raise ValueError(
                f"[PostProcess] 'comp_multi_cfl=True' est incompatible avec 'target_cfl'."
            )

        # +---+ Normalisation des données +---+ #
        if isinstance(schemes_names, str):
            schemes_names = [schemes_names]

        if isinstance(fields, str):
            fields = [fields]

        # +---+ Vérification des schémas +---+ #
        for scheme in schemes_names:
            if scheme not in self.folders_names:
                raise FileNotFoundError(
                    f"[PostProcess] Le dossier schéma '{scheme}' est introuvable."
                )
            
        # +---+ Gestion du fichier .npz de référence +---+ #
        ref_dir = self.result_dir / self.REFERENCE_FOLDER
        ref_npz = list(ref_dir.glob("*.npz"))

        if not ref_npz:
            raise RuntimeError(
                f"[PostProcess] Aucun fichier .npz dans '{self.REFERENCE_FOLDER}'."
            )

        ref_sol = self.load_solution(npz_file=ref_npz[0])

        # +---+ Gestion des fichiers .npz de Solver +---+
        all_solutions = {}

        for scheme in schemes_names:

            scheme_dir = self.result_dir / scheme
            npz_files = list(scheme_dir.glob("*.npz"))

            if not npz_files:
                raise RuntimeError(
                    f"[PostProcess] Aucun fichier .npz dans '{scheme_dir}'."
                )
            
            sol_arr = {}

            for file in npz_files:
                cfl, _ = self.parse_solution_filename(npz_file=file)
                sol_arr[cfl] = self.load_solution(npz_file=file)

            all_solutions[scheme] = sol_arr

        # +---+ Séléection des CFL +---+ #
        cfl_sets = [set(sol.keys()) for sol in all_solutions.values()]

        if target_cfl is not None:

            common_cfl = set.intersection(*cfl_sets)

            if target_cfl not in common_cfl:
                raise RuntimeError(
                    f"[PostProcess] Le CFL={target_cfl}, n'est pas commun à tous les schémas."
                )
            cfl_to_compare = [target_cfl]

        elif not comp_multi_cfl:

            cfl_to_compare = sorted(set.intersection(*cfl_sets))

            if not cfl_to_compare:
                raise RuntimeError(
                    "[PostProcess] Aucun CFL commun entre les schémas."
                )

        else:
            cfl_to_compare = sorted(set().union(*cfl_sets))

        # +---+ Gestion de la figure +---+ #
        n_fields = len(fields)

            # +--> nom du fichier
        fields_str = "-".join(fields)
        schemes = sorted(
            scheme.split("-", 1)[1]
            for scheme in schemes_names
        )
        schemes_str = "-vs-".join(schemes)
        file_base = (
            f"comparaison_"
            f"schemas_{schemes_str}_"
            f"champs-{fields_str}"
        )
        if comp_multi_cfl:
            cfl_str = "-".join(str(cfl) for cfl in cfl_to_compare)
            file_name = f"{file_base}_CFLs-{cfl_str}.png"
        else:
            def file_name(cfl:float) -> str:
                return f"{file_base}_CFL-{cfl}.png"

        if comp_multi_cfl:
            
            # +--> Premier mode : multi-CFL / multi-schéma

            fig, axes = plt.subplots(
                nrows=n_fields,
                ncols=2,
                figsize=(14, 4 * n_fields),
                squeeze=False,
                sharex=True
            )

            fig.suptitle(
                "Comparaison multi-schéma / multi-CFL",
                fontsize=18
            )

            for idx, field in enumerate(fields):

                # +--> Champs de référence
                axes[idx, 0].plot(
                    ref_sol['x'],
                    ref_sol[field],
                    'k-o', 
                    label=f"Référence Anderson"
                )

                for scheme, sol_dict in all_solutions.items():

                    name = scheme.split("-", 1)[1]

                    for cfl in cfl_to_compare:

                        if cfl not in sol_dict:
                            continue

                        sol = sol_dict[cfl]
                        label = f"{name} - CFL={cfl}"

                        # +--> Champs issus des solutions
                        axes[idx, 0].plot(
                            sol['x'], 
                            sol[field],
                            label=label
                        )

                        # +--> Alignement des grilles
                        if sol[field].shape != ref_sol[field].shape:
                            ref_field = np.interp(
                                sol['x'], 
                                ref_sol['x'], 
                                ref_sol[field]
                            )
                        else:
                            ref_field = ref_sol[field]

                        # +--> Erreur absolue
                        axes[idx, 1].plot(
                            sol['x'], 
                            np.abs(sol[field] - ref_field), 
                            label=rf"{name} - CFL={cfl} - $|{self.nickname[field]}_{{ref}} - {self.nickname[field]}_{{sol}}|$"
                        )

                # +---+ Mise en forme des graphiques +---+ #
                axes[idx, 0].set_ylabel(field, fontsize=14)
                axes[idx, 1].set_ylabel(f"Erreur absolue : {field}", fontsize=14)
                axes[idx, 1].set_yscale("log")

                for col in range(2):
                    axes[idx, col].grid('on', alpha=0.75, linestyle='-.')
                    axes[idx, col].legend(loc='best', ncol=2, fontsize=8)

                if idx == n_fields - 1:
                    axes[idx, 0].set_xlabel("x", fontsize=14)
                    axes[idx, 1].set_xlabel("x", fontsize=14)

                axes[0, 0].set_title("Champ")
                axes[0, 1].set_title("Erreur absolue")

            # plt.tight_layout()
            fig.savefig(self.comparison_dir / file_name, dpi=300)
            plt.show()

        else:

            # +--> Second/Troisième mode : CFL / multi-schéma

            for cfl in cfl_to_compare:

                fig, axes = plt.subplots(
                    nrows=n_fields,
                    ncols=2,
                    figsize=(14, 4 * n_fields),
                    squeeze=False,
                    sharex=True
                )

                fig.suptitle(
                    f"Comparaison des schémas - CFL = {cfl}",
                    fontsize=18
                )

                for idx, field in enumerate(fields):

                    # +--> Champs de référence
                    axes[idx, 0].plot(
                        ref_sol['x'],
                        ref_sol[field],
                        'k-o', 
                        label=f"Référence Anderson"
                    )

                    for scheme, sol_dict in all_solutions.items():

                        name = scheme.split("-", 1)[1]

                        if cfl not in sol_dict:
                            continue

                        sol = sol_dict[cfl]

                        sol = sol_dict[cfl]

                        # +--> Champs issus des solutions
                        axes[idx, 0].plot(
                            sol['x'], 
                            sol[field],
                            label=name
                        )

                        # +--> Alignement des grilles
                        if sol[field].shape != ref_sol[field].shape:
                            ref_field = np.interp(
                                sol['x'], 
                                ref_sol['x'], 
                                ref_sol[field]
                            )
                        else:
                            ref_field = ref_sol[field]

                        # +--> Erreur absolue
                        axes[idx, 1].plot(
                            sol['x'], 
                            np.abs(sol[field] - ref_field), 
                            label=rf"{name} - $|{self.nickname[field]}_{{ref}} - {self.nickname[field]}_{{sol}}|$"
                        )
                    # +---+ Mise en forme des graphiques +---+ #
                    axes[idx, 0].set_ylabel(field, fontsize=14)
                    axes[idx, 1].set_ylabel(f"Erreur absolue : {field}", fontsize=14)
                    axes[idx, 1].set_yscale("log")

                    for col in range(2):
                        axes[idx, col].grid('on', alpha=0.75, linestyle='-.')
                        axes[idx, col].legend(loc='best', ncol=2, fontsize=10)

                    if idx == n_fields - 1:
                        axes[idx, 0].set_xlabel("x", fontsize=14)
                        axes[idx, 1].set_xlabel("x", fontsize=14)

                    axes[0, 0].set_title("Champ")
                    axes[0, 1].set_title("Erreur absolue")

                # plt.tight_layout()

                fig.savefig(self.comparison_dir / file_name(cfl), dpi=300)

                plt.show()

                
