# HetSAGE

This repo contains the code used for [HetSAGE: Heterogenous Graph Neural Network
for Relational Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17898). 

Use the `environment.yml` file to install the required packages.
To reproduce the results from the paper:
1. preprocess the data: `python sql2graph.py --opts <opts>`, e.g. `--opts rules/muta188.yml`,
1. run the training: `python exp.py --graph <graph-file> --max-epochs 50
   --workers 2 --logdir <logdir> --device cpu`
