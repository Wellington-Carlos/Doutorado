
import numpy as np
from ecl.grid import EclGrid
from ecl.eclfile import EclFile
import os, sys

model_path = sys.argv[1]
output_npy = sys.argv[2]

# Arquivos corretos do modelo base Egg
egrid = os.path.join(model_path, "EGG_MODEL_ECL.EGRID")
initf = os.path.join(model_path, "EGG_MODEL_ECL.INIT")

grid = EclGrid(egrid)
nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
actnum = np.array(grid.export_actnum(), dtype=bool)

init = EclFile(initf)
permx = init["PERMX"][0]

perm3d = np.full((nx * ny * nz,), np.nan)
perm3d[actnum] = permx
perm3d = perm3d.reshape((nx, ny, nz), order="F")

layer0 = perm3d[:,:,0]
np.save(output_npy, layer0)
