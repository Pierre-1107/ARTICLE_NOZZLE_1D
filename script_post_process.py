# ===========================================================================
# Projet        : DF-302 - Simulation numérique / Article 
# Titre         : Solveur Volume Fini pour tuyère 1D
# Auteur        : Pierre Lambin--Gosset
# Affiliation   : ISAE-SUPAERO - Mastère Systèmes de Propulsion Aérospatiale
# Année         : 2025-2026
# Fichier       : script_post_process.py
# ===========================================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import yaml

from postprocess.post_processing import PostProcess
from pathlib import Path


def main():

    config_file = Path("post_process_config.yaml")
    with config_file.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    compare_CFL = cfg["compare_CFL"]
    compare_scheme = cfg["compare_scheme"]

    result_dir = Path("results")
    pp = PostProcess(result_dir=result_dir)

    if compare_CFL["activate"]:

        pp.compare_CFL(scheme_name=compare_CFL["scheme_name"], 
                    fields=compare_CFL["fields"]) 
        
    if compare_scheme["activate"]:

        pp.compare_scheme(schemes_names=compare_scheme["schemes_names"], 
                        fields=compare_scheme["fields"], 
                        comp_multi_cfl=compare_scheme["comp_multi_cfl"], 
                        target_cfl=compare_scheme["target_cfl"])

if __name__ == "__main__":

    main()