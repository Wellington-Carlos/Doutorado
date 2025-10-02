import os
import json
import time
import pygad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ObjectiveFunction_2 import objective

with open("params.json", "r") as f:
    params = json.load(f)

# --- Callbacks para mostrar barra de progresso ---
pbar = None
solutions_history = []   # guarda evolução do GA

def on_start(ga_instance):
    global pbar, solutions_history
    pbar = tqdm(total=ga_instance.num_generations,
                desc="Evolução GA", unit="gen")

    # Captura o melhor da população inicial (geração 0)
    solution, best_fitness, _ = ga_instance.best_solution()
    prod_x, prod_y, inj_x, inj_y = [int(v)+1 for v in solution]
    solutions_history.append({
        "Generation": 0,
        "Prod2_X": prod_x,
        "Prod2_Y": prod_y,
        "Inj2_X": inj_x,
        "Inj2_Y": inj_y,
        "COP": best_fitness
    })
    tqdm.write(f"Geração 0 | Best solution = {solution} | Fitness = {best_fitness:.4f}")

def on_gen(ga_instance):
    global pbar, solutions_history
    if pbar is not None:
        pbar.update(1)

    # Melhor solução da geração atual
    solution, best_fitness, _ = ga_instance.best_solution()
    prod_x, prod_y, inj_x, inj_y = [int(v)+1 for v in solution]
    solutions_history.append({
        "Generation": ga_instance.generations_completed,
        "Prod2_X": prod_x,
        "Prod2_Y": prod_y,
        "Inj2_X": inj_x,
        "Inj2_Y": inj_y,
        "COP": best_fitness
    })

    if pbar is not None:
        pbar.set_postfix({"Best fitness": f"{best_fitness:.4f}"})
    tqdm.write(f"Geração {ga_instance.generations_completed} | Best solution = {solution} | Fitness = {best_fitness:.4f}")

def on_stop(ga_instance, last_population_fitness):
    global pbar
    if pbar is not None:
        pbar.close()

# --- Configuração ---
os.chdir("/mnt/c/Users/calva/OneDrive/Documentos/Doutorado/Proposta_de_Tese/Codes_Optimization_Reservoir")

start = time.time()
objFunc = objective

num_generations = params["num_generations"]
num_parents_mating = params["num_parents_mating"]
parent_selection_type = params["parent_selection_type"]
keep_parents = params["keep_parents"]
crossover_type = params["crossover_type"]
crossover_prob = params["crossover_prob"]
mutation_type = params["mutation_type"]
mutation_num_genes = params["mutation_num_genes"]
mutation_percent_genes = params["mutation_percent_genes"]
sol_per_pop = params["sol_per_pop"]
num_genes = params["num_genes"]
init_range_low = params["init_range_low"]
init_range_high = params["init_range_high"]

# --- Instância do GA ---
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=objFunc,
    on_start=on_start,
    on_generation=on_gen,
    on_stop=on_stop,
    gene_type=int,
    save_best_solutions=False,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    crossover_probability=crossover_prob,
    mutation_type=mutation_type,
    mutation_num_genes=mutation_num_genes,
    mutation_percent_genes=mutation_percent_genes
)

# --- Executa GA ---
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"\nParameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# --- Salvar modelo GA ---
filename = '../genetic_egg_direct_P2'
ga_instance.save(filename=filename)

# --- Adiciona linha extra com a melhor solução global ---
prod_x, prod_y, inj_x, inj_y = [int(v)+1 for v in solution]
solutions_history.append({
    "Generation": "Best",
    "Prod2_X": prod_x,
    "Prod2_Y": prod_y,
    "Inj2_X": inj_x,
    "Inj2_Y": inj_y,
    "COP": solution_fitness
})

# --- Salvar resultados em CSV ---
results_df = pd.DataFrame(solutions_history)
results_df.to_csv("GA_results_2.csv", index=False)
print("Resultados salvos em GA_results_2.csv")

#####################################################################################################

# --- Plotar gráfico da evolução ---
df_plot_numeric = results_df[results_df["Generation"] != "Best"]

fig = plt.figure(figsize=(5,4))
plt.plot(df_plot_numeric["Generation"], df_plot_numeric["COP"], marker="o")
plt.xlabel("Generation")
plt.xticks(np.linspace(0, num_generations, 6, dtype=int))
plt.ylabel("Cumulative oil production ($SM^3$)")
plt.legend(["Direct optimization (GA)"])
plt.title("Second Period")
fig.tight_layout()
plt.ticklabel_format(style='plain', axis='y')
fig.savefig("Second Period GA.jpg", dpi=300, format="jpg")

end = time.time()
elapsed = end - start
print(f"Elapsed: {elapsed/60:.2f} minutes.")
