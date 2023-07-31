# graph_learning_social_interactions_2021

### This is the code that corresponds to the simulations performed in two papers

V. Shumovskaia, K. Ntemos, S. Vlaski and A. H. Sayed, "[Online Graph Learning from Social Interactions](https://asl.epfl.ch/wp-content/uploads/2021/11/asilomar_2021.pdf)," 2021 55th Asilomar Conference on Signals, Systems, and Computers, 2021, pp. 1263-1267. [10.1109/IEEECONF53345.2021.9723403](https://doi.org/10.1109/IEEECONF53345.2021.9723403)[![DOI](https://zenodo.org/badge/477906155.svg)](https://zenodo.org/badge/latestdoi/477906155)



V. Shumovskaia, K. Ntemos, S. Vlaski and A. H. Sayed, "[Explainability and Graph Learning from Social Interactions](https://arxiv.org/abs/2203.07494)," IEEE Transactions on Signal and Information Processing over Networks, vol. 8, pp. 946–959, 2022. [DOI:10.1109/TSIPN.2022.3223805](https://doi.org/10.1109/TSIPN.2022.3223805)[![DOI](https://zenodo.org/badge/477906155.svg)](https://zenodo.org/badge/latestdoi/477906155)



### Usage

To run experiments, we refer to ```runs_final.txt```, e.g.:

```python main.py --likelihood_regime 0 --times 20000 --adjacency_regime 3 --agents 30 --er_prob 0.2 --seed 25 --lr 0.1 --step_size 0.1 --likelihood_var 0.5 --multistate --states 10```

To understand arguments for the parser we refer to ```parser.py``` or run ```python main.py --help```.

**Author: Valentina Shumovskaia**
