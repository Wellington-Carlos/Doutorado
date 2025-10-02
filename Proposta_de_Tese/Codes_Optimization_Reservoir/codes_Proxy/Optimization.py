# -*- coding: utf-8 -*-
"""
Created on Sun May 18 23:19:24 2025

@author: reza_
"""

import os
os.chdir(r"your\working\directory")
import pygad
# from ObjectiveFunction import objective # For direct optimization uncomment this
from ObjectiveFunctionProxy import objective
import time

# In[]
def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solutions_fitness[-1])

# In[]
start = time.time()
objFunc = objective

num_generations = 20
num_parents_mating = 1

parent_selection_type = "rank"
keep_parents = 1

crossover_type = "single_point"
crossover_prob = 0.7

mutation_type = "random"
mutation_percent_genes = 10

sol_per_pop = 5
num_genes = 4

init_range_low = 1
init_range_high = 60

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=objFunc,
                       on_generation=on_gen,
                       gene_type=int,
                       save_best_solutions=True,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_prob,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

end = time.time()
elapsed = end-start
print(f"Elapsed: {elapsed/3600:.2f}")

filename = '../genetic_egg_proxy_P2'
ga_instance.save(filename=filename)
# loaded_ga_instance = pygad.load(filename=filename)

# In[]
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,4))
plt.plot(ga_instance.best_solutions_fitness)
plt.xlabel("Generation")
plt.xticks([0, 5, 10, 15, 20])
plt.ylabel("Cumulative oil production ($SM^3$)")
plt.legend(["Proxy optimization (GA)"])
plt.title("Second Period")
fig.tight_layout()
plt.show()
fig.savefig("Second Period GA proxy.jpg", dpi=300, format="jpg")
