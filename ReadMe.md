# Solveur 1D d’écoulement compressible – Application à une tuyère

Ce projet consiste en le développement d’un **solveur numérique 1D pour les écoulements compressibles**, appliqué à l’étude d’un écoulement quasi-unidimensionnel dans une tuyère.  
Il repose sur la résolution des **équations d’Euler 1D** par la **méthode des volumes finis**, et permet de comparer plusieurs **schémas numériques classiques** utilisés en mécanique des fluides numérique.

Ce travail a été réalisé dans un **cadre académique (analyse numérique / CFD)**.

## Contexte
L’écoulement compressible dans une tuyère constitue un cas d’étude fondamental en
mécanique des fluides numérique.

## Objectifs
- Implémenter un solveur 1D basé sur la méthode des volumes finis
- Résoudre les équations d’Euler quasi-1D avec terme source
- Comparer différents schémas numériques (Roe, Rusanov, MacCormack)
- Valider les résultats par comparaison avec une solution de référence

## Exécution 
Pour lancer la simulation, se placer dans **ARTICLE_NOZZLE_1D/** :
- État de référence :
```bash
...\ARTICLE_NOZZLE_1D> python .\reference\maccormack.py
```
- Résolution du problème :
```bash
...\ARTICLE_NOZZLE_1D> python .\main.py
```
- Post-traitement des données :
```bash
...\ARTICLE_NOZZLE_1D> python .\script_post_process.py
```

## Configuration
Le projet est fait de telle sorte (si j'ai bien travaillé) que seules les .yaml doivent être modifiées pour générer les résultats. 
- Résolution du problème :
```yaml
physics:
  gamma: 1.4

mesh:
  x0: 0.0
  xL: 3.0
  Nx: 31

numerics:
  Nt: # int
  CFL: # [float] / [float, ..., float]

scheme:
  name: # Roe / Rusanov (nom de la classe)

live_plot:
  bool: # True / False
  frequency: # int

convergence:
  criterion: # float
```
- Post-traitement des données :
```yaml
compare_CFL:
  activate: # True / False
  scheme_name: # 'Schema-Roe' / 'Schema_Rusanov'
  fields: # ['Density', 'Velocity', 'Pressure', 'Temperature', 'Mach']

compare_scheme:
  activate: # True / False
  schemes_names: # ['Schema-Roe', 'Schema-Rusanov']
  fields: # ['Density', 'Velocity', 'Pressure', 'Temperature', 'Mach']
  comp_multi_cfl: # True / False
  target_cfl: # null / float

# fields : choix possible entre :
#   'Density', 'Velocity', 'Pressure', 'Temperature', 'Mach'
# ['Mach', 'Temperature'] / ['Mach', 'Pressure', 'Density'] / ...
```

## Hypothèses
- Écoulement unidimensionnel
- Fluide non visqueux
- Gaz parfait
- Écoulement stationnaire recherché
- Pas de transfert thermique ni de frottements

## Équations gouvernantes
Les équations d'Euler quasi-1D sont résolues sous la forme conservative :

$$\frac{\partial (AU)}{\partial t} + \frac{\partial (AF(U))}{\partial x} = S(U)$$

avec :
- U = [$\rho$, $\rho u$, $\rho E$]
- F(U) = [$\rho u$, $\rho u^2 + p$, $(\rho E + p)u$]
- S(U) = [0, $p\frac{dA}{dx}$, 0]
- A : géométrie de la tuyère

Fermeture thermodynamique : $p = \rho(\gamma-1)(E - 0.5u^2) $

## Schémas numériques
Les schémas suivants sont implémentés :
- Schéma de MacCormack (prédicteur-correcteur, utilisé comme référence)
- Schéma de Rusanov (Lax-Friedrichs local)
- Schéma de Roe (solveur de Riemann approché)

### Organisation du projet
```text
ARTICLE_NOZZLE_1D/
├── main.py                      # Script principal de lancement
├── fv_config.yaml               # Paramètres du solveur
├── post_process_config.yaml     # Paramètres du post-traitement
├── script_post_process.py       # Script de post-traitement
├── ReadMe.md                    # Documentation du projet
│
├── docs/                        # Documentation théorique
│   ├── academic_paper/          # Références bibliographiques
│   │   ├── ModernCompressibleFlow_AnnexeB.pdf
│   │   ├── Roe_s_Method.pdf
│   │   └── Rusanov.pdf
│   └── Article_LaTeX/           # Article LaTeX associé au projet
│
├── geometry/                    # Géométrie du problème
│   └── nozzle.py                # Définition de la tuyère
│
├── physics/                     # Modélisation physique
│   ├── euler_equation_1d.py     # Équations d’Euler 1D
│   ├── gas.py                   # Lois thermodynamiques
│   └── source.py                # Termes sources (variation de section)
│
├── schemes/                     # Schémas numériques
│   ├── roe.py                   # Schéma de Roe
│   └── rusanov.py               # Schéma de Rusanov
│
├── solver/                      # Solveur numérique
│   └── fv_solver_1d.py          # Solveur volumes finis 1D
│
├── reference/                   # Solution de référence
│   └── maccormack_scheme.py     # Schéma de MacCormack
│   └── maccormack.py            # Script de l'état de référence
│
├── postprocess/                 # Post-traitement
│   └── post_processing.py
│
└── results/                     # Résultats numériques
    ├── Reference-MacCormack/    # Résultats de référence
    ├── Schema-Roe/              # Résultats Roe
    ├── Schema-Rusanov/          # Résultats Rusanov
    └── Comparaison/             # Résultats du post-traitement
```

## Amélioration possibles
- Reconstruction MUSCL
- Schéma d'ordre supérieur en temps
- Ajout des termes visqueux / modèle de turbulence
- Implémentation d'autres schémas numériques
- Analyse de stabilité des schémas
- Comparaison au niveau des tailles de maille

## Auteur
**Pierre Lambin--Gosset**

Étudiant en Mastère Systèmes de Propulsion Aérospatiale - 2025/2026

ISAE-SUPAERO