import os
import json
import time
import pygad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ObjectiveFunction_1 import objective

with open("params.json", "r") as f:
    params = json.load(f)

# --- Callbacks para mostrar barra de progresso ---
pbar = None
solutions_history = []   # <--- precisa estar aqui, escopo global

def on_start(ga_instance):
    global pbar, solutions_history
    pbar = tqdm(total=ga_instance.num_generations,
                desc="Evolução GA", unit="gen")

    # Pega o melhor da população inicial (geração 0)
    solution, best_fitness, _ = ga_instance.best_solution()
    prod_x, prod_y, inj_x, inj_y = [int(v)+1 for v in solution]
    solutions_history.append({
        "Generation": 0,
        "Prod_X": prod_x,
        "Prod_Y": prod_y,
        "Inj_X": inj_x,
        "Inj_Y": inj_y,
        "COP": best_fitness
    })

    tqdm.write(f"Geração 0 | Best solution = {solution} | Fitness = {best_fitness:.4f}")


def on_gen(ga_instance):
    global pbar, solutions_history
    if pbar is not None:
        pbar.update(1)

    # Pega melhor solução da geração atual
    solution, best_fitness, _ = ga_instance.best_solution()

    # Salva evolução (coordenadas + fitness)
    prod_x, prod_y, inj_x, inj_y = [int(v)+1 for v in solution]
    solutions_history.append({
        "Generation": ga_instance.generations_completed,
        "Prod1_X": prod_x,
        "Prod1_Y": prod_y,
        "Inj1_X": inj_x,
        "Inj1_Y": inj_y,
        "COP": best_fitness
    })

    # Atualiza postfix da barra
    if pbar is not None:
        pbar.set_postfix({"Best fitness": f"{best_fitness:.4f}"})

    # Usa tqdm.write() para não quebrar a barra
    tqdm.write(f"Geração {ga_instance.generations_completed} | "
               f"Best solution = {solution} | "
               f"Fitness = {best_fitness:.4f}")

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
    on_start=on_start,          # inicia barra
    on_generation=on_gen,       # atualiza barra
    on_stop=on_stop,            # fecha barra
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
    mutation_num_genes = mutation_num_genes,
    mutation_percent_genes = mutation_percent_genes
)

# --- Executa GA ---
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

#####################################################################################################
# atualização do arquivo EGG.DATA para "Egg - Four"

egg_file = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado/Proposta_de_Tese/Codes_Optimization_Reservoir/Egg - Four/EGG.DATA"

# Just remember, the well coordinates are zero-indexed. 
# When writing optimal well coordinates to .DATA file, increment them by 1
prod_x, prod_y, inj_x, inj_y = [int(v)+1 for v in solution]  # dois primeiros = produtor, dois últimos = injetor

with open(egg_file, "r") as f:
    lines = f.readlines()

new_lines = []
inside_welspecs = False

for line in lines:
    # Detecta entrada na seção WELSPECS
    if line.strip().startswith("WELSPECS"):
        inside_welspecs = True
        new_lines.append(line)
        continue

    # Detecta saída da seção WELSPECS
    if inside_welspecs and line.strip() == "/":
        inside_welspecs = False
        new_lines.append(line)
        continue

    # Faz alteração só dentro de WELSPECS
    if inside_welspecs:
        if line.strip().startswith("PROD1"):
            parts = line.split()
            parts[2] = str(prod_x)  # substitui coordenada X
            parts[3] = str(prod_y)  # substitui coordenada Y
            line = " ".join(parts) + "\n"

        elif line.strip().startswith("INJECT1"):
            parts = line.split()
            parts[2] = str(inj_x)
            parts[3] = str(inj_y)
            line = " ".join(parts) + "\n"

    new_lines.append(line)

# Sobrescreve o arquivo
with open(egg_file, "w") as f:
    f.writelines(new_lines)

print("Arquivo EGG.DATA atualizado com sucesso.")

#####################################################################################################
# Adiciona uma linha extra com a melhor solução global
solutions_history.append({
    "Generation": "Best",   # marca como melhor solução final
    "Prod1_X": prod_x,
    "Prod1_Y": prod_y,
    "Inj1_X": inj_x,
    "Inj1_Y": inj_y,
    "COP": solution_fitness
})

# --- Salvar resultados finais ---
results_df = pd.DataFrame(solutions_history)
results_df.to_csv("GA_results_1.csv", index=False)

print("Resultados salvos em GA_results_1.csv")

#####################################################################################################

# --- Salvar modelo GA ---
filename = '../genetic_egg_direct_P1'
ga_instance.save(filename=filename)

# Filtra apenas as linhas numéricas (ignora a linha "Best")
df_plot_numeric = results_df[results_df["Generation"] != "Best"]

fig = plt.figure(figsize=(5,4))
plt.plot(df_plot_numeric["Generation"], df_plot_numeric["COP"], marker="o")
plt.xlabel("Generation")
plt.xticks(np.linspace(0, num_generations, 6, dtype=int))
plt.ylabel("Cumulative oil production ($SM^3$)")
plt.legend(["Direct optimization (GA)"])
plt.title("First Period")   # ou "Second Period", dependendo do script
fig.tight_layout()
plt.ticklabel_format(style='plain', axis='y')
fig.savefig("First Period GA.jpg", dpi=300, format="jpg")

end = time.time()
elapsed = end-start
print(f"Elapsed: {elapsed/60:.2f} minutes.")
