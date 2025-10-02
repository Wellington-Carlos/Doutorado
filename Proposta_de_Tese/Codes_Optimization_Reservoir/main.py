import subprocess, json

params = {
    "num_generations": 3,
    "num_parents_mating": 1,
    "parent_selection_type": "rank",
    "keep_parents": 1,
    "crossover_type": "single_point",
    "crossover_prob": 0.7,
    "mutation_type": "random",
    "mutation_num_genes": 1,
    "mutation_percent_genes": None,
    "sol_per_pop": 2,
    "num_genes": 4,
    "init_range_low": 1,
    "init_range_high": 60,
}

# salva em JSON
with open("params.json", "w") as f:
    json.dump(params, f)


print(">>> Rodando Optimization_1.py ...")
subprocess.run(["python3", "Optimization_1.py"], check=True)

print(">>> Gerando curvas de produção após Optimization_1.py ...")
subprocess.run(["python3", "generate_production_curves.py", "1"], check=True)

print("\n>>> Rodando Optimization_2.py ...")
subprocess.run(["python3", "Optimization_2.py"], check=True)

print(">>> Gerando curvas de produção após Optimization_2.py ...")
subprocess.run(["python3", "generate_production_curves.py", "2"], check=True)

print("\n>>> Finalizado com sucesso!")

