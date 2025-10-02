import subprocess, json, random, csv

# Definir quantas combinações testar
N_EXPERIMENTS = 5  # pode aumentar depois

# Espaços de busca para os hiperparâmetros
search_space = {
    "num_generations": [10, 20, 30, 50],
    "num_parents_mating": [1, 2, 4],
    "parent_selection_type": ["rank", "tournament"],
    "keep_parents": [0, 1],
    "crossover_type": ["single_point", "two_points", "uniform"],
    "crossover_prob": [0.5, 0.7, 0.9],
    "mutation_type": ["random"],
    "mutation_num_genes": [1, 2, 3],
    "mutation_percent_genes": [None, 10, 20],
    "sol_per_pop": [5, 10, 20],
    "num_genes": [4],  # fixo no seu problema
    "init_range_low": [1],
    "init_range_high": [60],
}

# CSV para armazenar resultados
with open("hyperparam_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(list(search_space.keys()) + ["script", "best_fitness"])

    # Loop de experimentos
    for exp in range(N_EXPERIMENTS):
        print(f"\n>>> Rodando experimento {exp+1}/{N_EXPERIMENTS}")

        # Escolhe valores aleatórios dentro do espaço
        params = {k: random.choice(v) for k, v in search_space.items()}

        # Salva em JSON (Optimization_X.py vai ler daqui)
        with open("params.json", "w") as f:
            json.dump(params, f)

        # Rodar Optimization_1.py
        print(">>> Rodando Optimization_1.py ...")
        result1 = subprocess.run(
            ["python3", "Optimization_1.py"],
            text=True, capture_output=True
        )

        # Captura fitness do print (regex mais seguro, mas aqui simplificado)
        fitness1 = None
        for line in result1.stdout.splitlines():
            if "Fitness value of the best solution" in line:
                fitness1 = float(line.split("=")[-1].strip())
                break

        writer.writerow(list(params.values()) + ["Optimization_1", fitness1])

        # Rodar Optimization_2.py
        print(">>> Rodando Optimization_2.py ...")
        result2 = subprocess.run(
            ["python3", "Optimization_2.py"],
            text=True, capture_output=True
        )

        fitness2 = None
        for line in result2.stdout.splitlines():
            if "Fitness value of the best solution" in line:
                fitness2 = float(line.split("=")[-1].strip())
                break

        writer.writerow(list(params.values()) + ["Optimization_2", fitness2])

print("\n>>> Busca de hiperparâmetros finalizada! Resultados salvos em hyperparam_results.csv")

