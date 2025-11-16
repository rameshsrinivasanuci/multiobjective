# %%
import numpy as np
import os
from hdf5storage import loadmat, savemat
from multitools import gamma_GC, make_pos_def
from eda import KnapsackEDA, generate_example_data, get_objectives, get_constraints 
from eda import sample_population, non_dominated,organize_results, converged_pf_from_dist
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import islice, combinations
from numpy import random
from scipy.spatial.distance import jensenshannon
import pickle

# %%


# %%
def knapsack_run(n_items,n_selected,n_obj,n_con,shape,scale,r,type = 'sim',n_run=20,pop_size =1000, generations=100, max_no_improve_gen=10):
    capacity = int(shape[-1]*scale[-1]*n_selected)
    allsim = dict()
    allsim['n_items'] = n_items
    allsim['n_selected'] = n_selected
    allsim['n_obj'] = n_obj
    allsim['n_con'] = n_con
    allsim['shape'] = shape
    allsim['scale'] = scale
    allsim['r'] = r
    allsim['capacity'] = capacity
    allsim['pop_size'] = pop_size
    allsim['max_no_improve_gen'] = max_no_improve_gen
    allsim['items']= list()
    allsim['pareto_indices_final'] = list()
    allsim['pareto_front_final'] = list()
    allsim['rpos'] = list()
    for run in range(n_run): #save to file
        #save to file
        print(f"run {run}")
        items_seed = 1211+run
        eda_seed = 1223+run
        f_seed = 1234+run
        # Generate data
        items, rpos = generate_example_data(r, shape, scale, n_items=n_items, seed=items_seed)
        allsim['rpos'].append(rpos)

        # Run EDA
        eda = KnapsackEDA(
            items=items,
            capacity=capacity,
            n_selected=n_selected,
            n_obj=n_obj,
            pop_size=pop_size,
            generations=generations,
            max_no_improve_gen=max_no_improve_gen,
            seed=eda_seed
        )
        #organize results    

        results = eda.run()
        js_div_list, distribution_table, pareto_indices_table, pareto_front_table = organize_results(results)  
        
        # Save results
        """ sim = dict()
        sim['n_items'] = n_items
        sim['n_selected'] = n_selected
        sim['n_obj'] = n_obj
        sim['n_con'] = n_con
        sim['shape'] = shape
        sim['scale'] = scale
        sim['r'] = r
        sim['rpos'] = r
        sim['capacity'] = capacity
        sim['pop_size'] = pop_size
        sim['max_no_improve_gen'] = max_no_improve_gen
        sim['items'] = items
        allsim['items'].append(items)
        sim['items_seed'] = items_seed
        sim['eda_seed'] = eda_seed
        sim['js_div_list'] = js_div_list
        sim['distribution_table'] = distribution_table
        sim['pareto_indices_table'] = pareto_indices_table
        sim['pareto_front_table'] = pareto_front_table
    """    #final refinement 
        pareto_solutions, pareto_objectives, counter = converged_pf_from_dist(distribution_table[-1], items, pareto_indices_table[-1],capacity, n_selected, n_obj, f_seed = f_seed, sample_size=5000, max_iters=50, max_no_change=5)
        pareto_constraints = get_constraints(items, pareto_solutions, n_obj,n_con)
        pareto_front_final = np.concatenate((pareto_objectives, pareto_constraints.reshape(-1,1)), axis =1 )
        #save to file
        allsim['pareto_indices_final'].append(pareto_solutions)
        allsim['pareto_front_final'].append(pareto_front_final)
        return allsim    

#%%

#parameters
type = 'pospair'
n_run = 200
n_items = 120
n_selected = 6
n_obj = 3
n_con = 1
shape = [3.0, 4.0, 2.0, 8.0]
scale = [2.0, 3, 2, 1.0]

capacity = int(shape[-1]*scale[-1]*n_selected)
pop_size = 1000
generations = 100 # do not matter if check convergence
max_no_improve_gen = 10
ct = 0
for r_obj in np.arange(0.1,0.9,0.1):
    for r_con in np.arange(0.1,0.9,0.1):
        ct += 1
        print(f"file = {ct}")
        r = np.array([      
            [1.0, -r_obj, -r_obj, r_con],
            [-r_obj, 1.0, r_obj, r_con],
            [-r_obj, r_obj, 1.0, r_con],
            [r_con, r_con, r_con, 1.0],
        ])

        allsim = knapsack_run(n_items,n_selected,n_obj,n_con,shape,scale,r,type = type,n_run=n_run,pop_size =pop_size, generations=generations, max_no_improve_gen=max_no_improve_gen)

        if not os.path.exists('./results/'):
            os.makedirs('./results/')
        output_dir = './results/'
        savemat(os.path.join(output_dir, f"knapsack_eda_sim_{type}_{n_items}_{n_selected}_{ct}_all.mat"), allsim, store_python_metadata=True)

        file_path = os.path.join(output_dir, f"knapsack_eda_sim_{type}_{n_items}_{n_selected}_{ct}_all.pkl")

        with open(file_path, "wb") as pickle_file: # "wb" for write binary
            pickle.dump(allsim, pickle_file)

# %%
# output_dir = "./results"
# if len(distribution_table) > 0:
#     dist_df = pd.DataFrame(np.vstack(distribution_table))
#     dist_df.index.name = 'generation'
#     dist_df.columns = [f"item_{i}" for i in range(dist_df.shape[1])]
#     dist_df.to_csv(os.path.join(output_dir, f"distribution_table_{n_items}_{n_selected}_{run}.csv"))

# if len(js_div_list) > 0:
#     js_df = pd.DataFrame({
#         'generation': np.arange(1, len(js_div_list) + 1, dtype=int),
#         'js_divergence': js_div_list
#     })
#     js_df.to_csv(os.path.join(output_dir, f"js_div_list_{n_items}_{n_selected}_{run}.csv"), index=False)

# if len(pareto_indices_table) > 0:
#     np.savez_compressed(
#         os.path.join(output_dir, f"pareto_indices_table_{n_items}_{n_selected}_{run}.npz"),
#         **{f"gen_{i+1}": arr for i, arr in enumerate(pareto_indices_table)}
#     )
# if len(pareto_front_table) > 0:
#     np.savez_compressed(
#         os.path.join(output_dir, f"pareto_front_table_{n_items}_{n_selected}_{run}.npz"),
#         **{f"gen_{i+1}": arr for i, arr in enumerate(pareto_front_table)}
#     )

# %%
# sim = dict()
# sim['n_items'] = n_items
# sim['n_selected'] = n_selected
# sim['n_obj'] = n_obj
# sim['n_con'] = n_con
# sim['shape'] = shape
# sim['scale'] = scale
# sim['r'] = r
# sim['capacity'] = capacity
# sim['pop_size'] = pop_size
# sim['max_no_improve_gen'] = max_no_improve_gen
# sim['items'] = items
# sim['items_seed'] = items_seed
# sim['eda_seed'] = eda_seed
# sim['js_div_list'] = js_div_list
# sim['distribution_table'] = distribution_table
# sim['pareto_indices_table'] = pareto_indices_table
# sim['pareto_front_table'] = pareto_front_table      
# savemat(os.path.join(output_dir, f"knapsack_eda_sim_{n_items}_{n_selected}_{run}.mat"), sim, store_python_metadata=True)

# %%


#savemat("temp.mat", allsim, store_python_metadata=True)


# %%




