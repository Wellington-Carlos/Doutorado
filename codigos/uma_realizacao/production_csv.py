import pandas as pd
from ecl.summary import EclSum
import glob
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_summary_data(smspec_file):
    """Carrega dados de produção usando a biblioteca ecl"""
    try:
        case_path = str(Path(smspec_file).with_suffix(''))
        eclsum = EclSum(case_path)
        prod_vars = [key for key in eclsum.keys() 
                     if any(x in key for x in ['WOPR', 'WWPR', 'WLPR'])]
        prod_vars = [var for var in prod_vars 
                     if not any(x in var.upper() for x in ['INJ', 'INJECT'])]
        inj_well = "INJECT1"
        inj_var = f"WWIR:{inj_well}"
        if inj_var in eclsum.keys():
            prod_vars.append(inj_var)
        dates = eclsum.dates
        data = {'DATE': [d for d in dates]}
        for var in prod_vars:
            data[var] = eclsum.numpy_vector(var)
        return pd.DataFrame(data).set_index('DATE')
    except Exception as e:
        print(f"Erro ao carregar {case_path}: {str(e)}")
        return None

# Busca arquivo SMSPEC na pasta atual
smspec_files = glob.glob("*.SMSPEC")

if not smspec_files:
    print("Nenhum arquivo .SMSPEC encontrado na pasta atual.")
else:
    smspec = smspec_files[0]
    print(f"Processando: {smspec}")
    case_name = Path(smspec).stem
    df = load_summary_data(smspec)
    if df is not None and not df.empty:
        csv_file = f"{case_name}_production.csv"
        df.to_csv(csv_file)
        print(f"Dados salvos em {csv_file}")
    else:
        print("Nenhum dado de produção válido encontrado.")
