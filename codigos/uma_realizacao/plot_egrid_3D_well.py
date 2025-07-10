import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
from ecl.grid import EclGrid
from ecl.eclfile import EclFile
import re

# === Arquivos do modelo Egg ===
egrid_file = "EGG_MODEL_ECL.EGRID"
init_file = "EGG_MODEL_ECL.INIT"
data_file = "EGG_MODEL_ECL.DATA"

# === Leitura da malha e PERMX ===
grid = EclGrid(egrid_file)
nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
actnum = np.array(grid.export_actnum(), dtype=bool)

init = EclFile(init_file)
permx = init["PERMX"][0]
perm3d = np.full((nx * ny * nz), np.nan)
perm3d[actnum] = permx
perm3d = perm3d.reshape((nx, ny, nz), order="F")

# === Aplica log10 e define faixa de visualização ===
log_perm = np.where(perm3d > 0, np.log10(perm3d), np.nan)
vmin, vmax = 2.0, 3.6
norm_perm = (log_perm - vmin) / (vmax - vmin)
filled = ~np.isnan(log_perm)

# === Cores usando colormap "jet" ===
cmap = colormaps["jet"]
colors = cmap(norm_perm)
colors[np.isnan(log_perm)] = [0, 0, 0, 0]  # transparentes
colors[np.isnan(log_perm), -1] = 0  # reforça alpha = 0

# === Figura 3D ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# === Exagero vertical ===
z_exaggeration = 0.5
ax.set_box_aspect([1, 1, z_exaggeration])

# === Desenha voxels com transparência ===
ax.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.05, alpha=0.8)

# === Função para extrair coordenadas e tipo dos poços ===
def extract_well_coords(data_file):
    wells = []
    reading = False
    with open(data_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.upper().startswith("WELSPECS"):
                reading = True
                continue
            if reading:
                if stripped == "/":
                    break
                tokens = stripped.replace("'", "").split()
                if len(tokens) >= 6:
                    name = tokens[0]
                    i = int(tokens[2]) - 1
                    j = int(tokens[3]) - 1
                    fluid = tokens[5].upper()
                    # Inferir tipo: produtor se não for água
                    if "INJECT" in name.upper() or fluid == "WATER":
                        well_type = "injector"
                    else:
                        well_type = "producer"
                    wells.append((name, i, j, well_type))
    return wells

# === Extrai os poços ===
wells = extract_well_coords(data_file)

# === Calcula profundidade máxima por coluna (última camada ativa) ===
z_limits = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        col_index = [i + j * nx + k * nx * ny for k in range(nz)]
        active_layers = np.where(actnum[col_index])[0]
        if len(active_layers) > 0:
            max_k = active_layers[-1]
            z_limits[i, j] = (max_k + 1) * z_exaggeration
        else:
            z_limits[i, j] = 0  # nenhuma camada ativa

# === Adiciona os poços ===
poço_extra = 6.0  # altura visual acima do modelo

for name, i, j, well_type in wells:
    x = i + 0.5
    y = j + 0.5
    z_base = z_limits[i, j]
    if z_base == 0:
        continue

    z_top = z_base + poço_extra

    # Define cor por tipo
    if well_type == "injector":
        cor_esfera = "blue"
        cor_linha = "blue"
    else:
        cor_esfera = "red"
        cor_linha = "red"

    # ======== CONTROLE INDIVIDUAL DO NOME ========
    # Valores padrão (lado esquerdo)
    dx_name = -1.0
    dy_name = 0.5
    ha_name = 'right'

    # Ajuste individual para poços desejados
    if name.upper() in ['INJECT4', 'INJECT7', 'PROD3', 'PROD4']:
        dx_name = 1.5   # lado direito
        ha_name = 'left'

    # Linha do poço
    ax.plot([x, x], [y, y], [z_top, z_base], color=cor_linha, linewidth=5, zorder=10000)
    # Esfera no topo
    ax.scatter(x, y, z_top, color=cor_esfera, s=30, zorder=10000)
    # Nome do poço (com ajuste individual)
    ax.text(x + dx_name, y + dy_name, z_top, name, color=cor_esfera, fontsize=8, fontweight='bold',
            ha=ha_name, va='bottom', zorder=10000)

# ==== CONFIGURAÇÕES DOS EIXOS E GRID ====
ax.set_xlabel("X", labelpad=10)
ax.set_ylabel("Y", labelpad=10)
ax.set_zlabel("")  # Remove o label do eixo Z

# Remove completamente o eixo Z
ax.set_zticks([])
ax.zaxis.line.set_lw(0.)  # Remove a linha do eixo Z
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparente

# Remove todas as linhas verticais e quadros 3D
ax.xaxis.pane.set_edgecolor('w')  # Branco (invisível)
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Configuração do grid apenas no plano XY
ax.grid(False)  # Desativa todos os grids

# Cria um grid personalizado apenas no plano Z=0
xticks = np.arange(0, nx+1, 10)  # Ajuste o passo conforme necessário
yticks = np.arange(0, ny+1, 10)

for x in xticks:
    ax.plot([x, x], [yticks[0], yticks[-1]], [0, 0], 
            color='gray', linestyle='-', linewidth=0.5)

for y in yticks:
    ax.plot([xticks[0], xticks[-1]], [y, y], [0, 0],
            color='gray', linestyle='-', linewidth=0.5)

# Inverte os eixos conforme seu original
ax.invert_xaxis()
ax.invert_yaxis()
ax.view_init(elev=25, azim=45)
ax.set_zlim(0, np.max(z_limits) + poço_extra + 20)

# === Colorbar ===
# Colorbar horizontal, menor e mais próxima
mappable = plt.cm.ScalarMappable(cmap="jet")
mappable.set_array(log_perm)
mappable.set_clim(vmin, vmax)
cbar = plt.colorbar(
    mappable, ax=ax, orientation='horizontal',
    fraction=0.035, pad=-0.05, aspect=20, format="%.1f"
)
cbar.set_label("Log scale permeability (mD)")

plt.tight_layout()
output_path = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/paper_com_Bratvold/sn-article-template/egg_model_log_3D_well.eps"
plt.savefig(output_path, format="eps", dpi=300, bbox_inches="tight")
plt.close()


print("✅ Imagem salva!")