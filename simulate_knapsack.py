import os
import numpy as np
from hdf5storage import loadmat, savemat
from eda import KnapsackEDA, generate_example_data, organize_results

def main():
    # set parameters
    n_items = 60
    n_selected = 6
    n_obj = 3
    n_con = 1
    shape = [3.0, 4.0, 2.0, 8.0]
    scale = [2.0, 3, 2, 1.0]
    capacity = int(shape[-1]*scale[-1]*n_selected)
    pop_size = 1000
    generations = 100 # do not matter if check convergence
    max_no_improve_gen = 10
    
    corr_type = 'all_pos'
    corr_m_dict = np.load('correlation_matrices.npz')
    r = corr_m_dict[corr_type]
    
    # run simulations
    for run in range(10):
        items_seed = 111+run
        eda_seed = 1123+run
        
        # generate data
        items = generate_example_data(r, shape, scale, n_items=n_items, seed=items_seed)

        # run EDA
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
        results = eda.run()

        # organize results    
        js_div_list, distribution_table, pareto_indices_table, pareto_front_table = organize_results(results)
        
        # save results
        sim = dict()
        sim['n_items'] = n_items
        sim['n_selected'] = n_selected
        sim['n_obj'] = n_obj
        sim['n_con'] = n_con
        sim['shape'] = shape
        sim['scale'] = scale
        sim['r'] = r
        sim['capacity'] = capacity
        sim['pop_size'] = pop_size
        sim['max_no_improve_gen'] = max_no_improve_gen
        sim['items'] = items
        sim['items_seed'] = items_seed
        sim['eda_seed'] = eda_seed
        sim['js_div_list'] = js_div_list
        sim['distribution_table'] = distribution_table
        sim['pareto_indices_table'] = pareto_indices_table
        sim['pareto_front_table'] = pareto_front_table      

        if not os.path.exists('./results/corr_type'):
            os.makedirs('./results/corr_type')
        output_dir = './results/corr_type'
        savemat(os.path.join(output_dir, f"knapsack_eda_sim_{n_items}_{n_selected}_{run}.mat"), sim, store_python_metadata=True)

if __name__ == "__main__":
    main()

