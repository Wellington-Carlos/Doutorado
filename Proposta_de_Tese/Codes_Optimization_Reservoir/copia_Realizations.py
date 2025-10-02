import os
import shutil
import random

# Diretórios de origem e destino
src_dir = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado/benchmark/OPM_Egg/Permeability_Realizations"
dst_dir = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado/Proposta_de_Tese/Codes_Optimization_Reservoir/Realizations"

# Número de arquivos que você quer copiar (entre 1 e 101)
n_files = 5   # altere aqui

# Lista de arquivos possíveis (PERM0_ECL.INC até PERM100_ECL.INC)
files = [f"PERM{i}_ECL.INC" for i in range(101)]

# Se n_files >= 101 → copia todos
if n_files >= len(files):
    selected_files = files
else:
    selected_files = random.sample(files, n_files)

# Limpar o diretório destino antes de copiar
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)   # remove a pasta inteira com tudo dentro
os.makedirs(dst_dir, exist_ok=True)  # recria pasta vazia

# Copiar arquivos
for fname in selected_files:
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copiado: {fname}")
    else:
        print(f"[AVISO] Arquivo não encontrado: {fname}")

print(f"\nTotal de arquivos copiados: {len(selected_files)}")

