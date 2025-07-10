import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ecl.grid import EclGrid
from ecl.eclfile import EclFile
from matplotlib import colormaps

# === Parâmetros do modelo Egg ===
egrid_file = "EGG_MODEL_ECL.EGRID"
init_file = "EGG_MODEL_ECL.INIT"

grid = EclGrid(egrid_file)
nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
actnum = np.array(grid.export_actnum(), dtype=bool)

init = EclFile(init_file)
permx = init["PERMX"][0]
perm3d = np.full((nx * ny * nz), np.nan)
perm3d[actnum] = permx
perm3d = perm3d.reshape((nx, ny, nz), order="F")

log_perm = np.where(perm3d > 0, np.log10(perm3d), np.nan)
vmin, vmax = 2.0, 3.6
norm_perm = (log_perm - vmin) / (vmax - vmin)
filled = ~np.isnan(log_perm)

# Coloração
cmap = colormaps["jet"]
colors = cmap(norm_perm)
colors[np.isnan(log_perm)] = [0, 0, 0, 0]
colors[np.isnan(log_perm), -1] = 0

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

z_exaggeration = 0.1
ax.set_box_aspect([1, 1, z_exaggeration])
ax.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.05, alpha=0.8)

# ==== CONFIGURAÇÕES DOS EIXOS E GRID ====
ax.set_xlabel("X", labelpad=10)
ax.set_ylabel("Y", labelpad=10)
ax.set_zlabel("")  # Remove o label do eixo Z

# Remove completamente o eixo Z e as molduras/painéis
ax.set_zticks([])
ax.zaxis.line.set_lw(0.)  # Remove a linha do eixo Z
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparente

ax.xaxis.pane.set_edgecolor('w')  # Branco (invisível)
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove grid padrão
ax.grid(False)

# Cria um grid personalizado apenas no plano Z=0
xticks = np.arange(0, nx+1, 10)  # Ajuste o passo conforme necessário
yticks = np.arange(0, ny+1, 10)

for x in xticks:
    ax.plot([x, x], [yticks[0], yticks[-1]], [0, 0], 
            color='gray', linestyle='-', linewidth=0.5)

for y in yticks:
    ax.plot([xticks[0], xticks[-1]], [y, y], [0, 0],
            color='gray', linestyle='-', linewidth=0.5)

# Inverte os eixos conforme padrão Egg
ax.invert_xaxis()
ax.invert_yaxis()
ax.view_init(elev=25, azim=45)
#ax.set_zlim(0, nz * z_exaggeration)  # Ajusta altura conforme exagero vertical

# === Colorbar ===
# Colorbar horizontal, menor e mais próxima
mappable = plt.cm.ScalarMappable(cmap="jet")
mappable.set_array(log_perm)
mappable.set_clim(vmin, vmax)
cbar = plt.colorbar(
    mappable, ax=ax, orientation='horizontal',
    fraction=0.04, pad=-0.2, aspect=20, format="%.1f"
)
cbar.set_label("Log scale permeability (mD)")

plt.tight_layout()
output_path = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/paper_com_Bratvold/sn-article-template/egg_model_log_3D.eps"
plt.savefig(output_path, format="eps", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Figura salva!")
