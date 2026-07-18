# multiobjective

This repository involves code for the multiobjective knapsack problem project. Here are the breakdowns:

1. card_game: contains scripts (data generation, pilot data analysis, and TCP server)and data (item lists, eda results, and pareto fronts) for the civilization building card game 
2. cp_model: contains the original code for the conditional probability model
3. cp_eda: contains scripts for cpEDA and human-guided cpEDA
4. cpfn: contains preliminary code about running the push-forward conditional probability neural network
5. paper_related: contain analysis and plotting scripts (heuristic model, hypervolume and dominance ratio computation, and pareto front figures) for the IEEE conference paper
6. food_data: contains code and data related to food nutrients from USDA food data central
7. human_guided_eda: most recent implementation of human guided EDA (weights based on knapsacks instead of items; weights computed using dot product with or distance to the aspiration)
8. pf_modeling: contains several methods for pareto front modeling
9. old_scripts: previous code for generating data and implementing EDA or human guided EDA, which should not be useful

The remaining files are for the general process of generating data, running EDA, and implementing human-agent interactive loop.
