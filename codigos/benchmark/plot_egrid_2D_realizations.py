import numpy as np
import matplotlib.pyplot as plt
from ecl.grid import EclGrid
from ecl.eclfile import EclFile
import os

# --- Parâmetros ---
base_dir = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/benchmark/OPM_Egg"
sim_dirs = [f"simulacao_{i}" for i in range(101)]  # Exemplo para 101 simulações
n_to_plot = 25  # Quantidade de realizações (ex: 25 para 5x5)
z_slice = 0     # Camada a plotar (top layer)
nrows, ncols = int(np.sqrt(n_to_plot)), int(np.sqrt(n_to_plot))  # Subplot 5x5

# --- Selecionar simulações aleatórias ---
#np.random.seed(42)
selected_idx = np.random.choice(len(sim_dirs), n_to_plot, replace=False)
selected_dirs = [sim_dirs[i] for i in selected_idx]

# --- Plotagem ---
fig, axes = plt.subplots(nrows, ncols, figsize=(8, 10), sharex=True, sharey=True)
vmin, vmax = 2, 3.6  # Limites da escala log10

for i, (ax, sim_dir) in enumerate(zip(axes.flat, selected_dirs)):
    sim_path = os.path.join(base_dir, sim_dir)
    egrid_file = os.path.join(sim_path, "EGG_MODEL_ECL.EGRID")
    init_file = os.path.join(sim_path, "EGG_MODEL_ECL.INIT")
    
    # Leitura do grid e perm
    grid = EclGrid(egrid_file)
    nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
    actnum = np.array(grid.export_actnum(), dtype=bool)
    init = EclFile(init_file)
    permx = init["PERMX"][0]
    perm3d = np.full((nx * ny * nz), np.nan)
    perm3d[actnum] = permx
    perm3d = perm3d.reshape((nx, ny, nz), order="F")
    perm_slice = perm3d[:, :, z_slice]
    log_perm = np.log10(np.where(perm_slice > 0, perm_slice, np.nan))

    # Troca os NaNs por vmin (azul)
    log_perm = np.where(np.isnan(log_perm), vmin, log_perm)    

    im = ax.imshow(log_perm.T, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
    x_ticks = np.arange(0, log_perm.shape[0]+1, 20)
    y_ticks = np.arange(0, log_perm.shape[1]+1, 20)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Identificar linha e coluna
    row = i // ncols
    col = i % ncols
    if row == nrows - 1:
        ax.set_xlabel("X direction", fontsize=10)
    if col == 0:
        ax.set_ylabel("Y direction", fontsize=10)
    ax.tick_params(axis='both', labelsize=9)

# Ajuste manual dos espaços antes do colorbar
plt.subplots_adjust(
    left=0.05, right=0.98, top=0.96, bottom=0.16,
    wspace=0.0, hspace=0.08
)

# Colorbar horizontal centralizado abaixo das subplots
cbar = fig.colorbar(
    im, ax=axes, orientation='horizontal',
    fraction=0.035, pad=0.08, aspect=20, format="%.1f"
)
cbar.set_label("Log scale permeability (mD)", fontsize=14)

# Salvar arquivos
output_path_eps = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/paper_com_Bratvold/sn-article-template/egg_model_log_2D_grid.eps"
#output_path_png = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/paper_com_Bratvold/sn-article-template/egg_model_log_2D_grid.png"
plt.savefig(output_path_eps, format="eps", dpi=300, bbox_inches="tight")
#plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Figura de subplots salva!")
