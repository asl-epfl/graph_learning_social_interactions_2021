# graph_learning_social_interactions_2021

### This is the code that corresponds to the simulations performed in two papers:

V. Shumovskaia, K. Ntemos, S. Vlaski and A. H. Sayed, "Online Graph Learning from Social Interactions," 2021 55th Asilomar Conference on Signals, Systems, and Computers, 2021, pp. 1263-1267, doi: 10.1109/IEEECONF53345.2021.9723403.

V. Shumovskaia, K. Ntemos, S. Vlaski and A. H. Sayed (2022). Explainability and Graph Learning from Social Interactions. arXiv preprint arXiv:2203.07494.

### Usage

To run experiments, we refer to ```runs_final.txt```, e.g.:

```python main.py --likelihood_regime 0 --times 20000 --adjacency_regime 3 --agents 30 --er_prob 0.2 --seed 25 --lr 0.1 --step_size 0.1 --likelihood_var 0.5 --multistate --states 10```

To understand arguments for the parser we refer to ```parser.py``` or run ```python main.py --help```.

**Author: Valentina Shumovskaia.**
