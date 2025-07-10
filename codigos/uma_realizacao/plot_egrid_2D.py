import numpy as np
import matplotlib.pyplot as plt
from ecl.grid import EclGrid
from ecl.eclfile import EclFile

# Caminhos dos arquivos
egrid_file = "EGG_MODEL_ECL.EGRID"
init_file = "EGG_MODEL_ECL.INIT"

# Carrega o grid e a permeabilidade
grid = EclGrid(egrid_file)
nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
actnum = np.array(grid.export_actnum(), dtype=bool)

init = EclFile(init_file)
permx = init["PERMX"][0]
perm3d = np.full((nx * ny * nz), np.nan)
perm3d[actnum] = permx
perm3d = perm3d.reshape((nx, ny, nz), order="F")

# Seleciona uma camada

#z_slice = nz // 2

z_slice = 0        # camada mais rasa
# z_slice = 6        # camada mais profunda
# z_slice = 2        # qualquer outra camada

perm_slice = perm3d[:, :, z_slice]

# Aplica log10 (para mD)
log_perm = np.log10(np.where(perm_slice > 0, perm_slice, np.nan))

# Limites de cor
vmin, vmax = 2, 3.6  # 10^2 = 100 mD até 10^3.6 ≈ 3981 mD

# Plot
plt.figure(figsize=(6, 5))
im = plt.imshow(log_perm.T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
plt.xlabel("X direction")
plt.ylabel("Y direction")
cbar = plt.colorbar(im, label="Log scale permeability (mD)")
plt.title(f"Egg Model – Layer Z = {z_slice}")
plt.tight_layout()

output_path = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/paper_com_Bratvold/sn-article-template/egg_model_log_2D.eps"
plt.savefig(output_path, format="eps", dpi=300, bbox_inches="tight")

output_path2 = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/paper_com_Bratvold/sn-article-template/egg_model_log_2D.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
plt.close()

print("✅ Imagem salva!")




