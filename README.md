# Task-Based Programming for Adaptive Mesh Refinement in Compressible Flow Simulations

This repository contains the implementation for [**Task-Based Programming for Adaptive Mesh Refinement in Compressible Flow Simulations**](https://arxiv.org/abs/2508.05020), a [Regent](https://regent-lang.org/)-based framework (built on the [Legion](https://legion.stanford.edu/) programming model) designed to implement adaptive mesh refinement (AMR) for high-order compressible flow solvers.

## Installation

The code is built using **Regent**, a high-level programming language for the Legion programming model. Please follow the Regent installation guide:

- Official language site: https://regent-lang.org/
- Installation instructions: https://github.com/StanfordLegion/legion/tree/stable/language

## Repository Overview

The AMR project is structured into three main parts:

- **Solver design and implementation**  
  Located at `src/`, this directory contains the core AMR solver code, including task-based mesh refinement/coarsening routines.

- **Compressible flow simulations**  
  Found in `tests/`, this folder includes canonical examples and tests for compressible flow problems (e.g., Euler equations).

- **Post-processing for animation**  
  Available in `post/`, this directory provides scripts to generate animations from simulation output.


## Paper and Citation

This project is documented in this paper [arXiv:2508.05020](https://arxiv.org/abs/2508.05020):

```bibtex
@article{wei2025taskbased,
  title={Task-Based Programming for Adaptive Mesh Refinement in Compressible Flow Simulations},
  author={Wei, Anjiang and Song, Hang and Hidayetoglu, Mert and Slaughter, Elliott and Lele, Sanjiva K. and Aiken, Alex},
  journal={arXiv preprint arXiv:2508.05020},
  year={2025},
  url={https://arxiv.org/abs/2508.05020}
}
