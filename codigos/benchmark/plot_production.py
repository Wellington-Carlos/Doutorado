#!/usr/bin/env python3
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ecl.summary import EclSum
import glob
import os
from pathlib import Path
import warnings

# Configurações atualizadas
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')

def load_summary_data(smspec_file):
    """Carrega dados usando a biblioteca ecl diretamente"""
    try:
        # Remove a extensão .SMSPEC se presente
        case_path = str(Path(smspec_file).with_suffix(''))

        # Carrega o arquivo de sumário
        eclsum = EclSum(case_path)

        # Obtém todas as variáveis de produção desejadas (sem WGPR)
        prod_vars = [key for key in eclsum.keys() 
                     if any(x in key for x in ['WOPR', 'WWPR', 'WLPR'])]

        # Filtra poços injetores
        prod_vars = [var for var in prod_vars 
                     if not any(x in var.upper() for x in ['INJ', 'INJECT'])]

        # Inclui a curva de injeção de água (por exemplo, INJECT1)
        inj_well = "INJECT1"
        inj_var = f"WWIR:{inj_well}"
        if inj_var in eclsum.keys():
            prod_vars.append(inj_var)

        # Extrai os dados
        dates = eclsum.dates
        data = {'DATE': [d for d in dates]}

        for var in prod_vars:
            data[var] = eclsum.numpy_vector(var)

        return pd.DataFrame(data).set_index('DATE')

    except Exception as e:
        print(f"Erro ao carregar {case_path}: {str(e)}")
        return None

def plot_production_curves(df, output_dir="plots"):
    """Gera e salva gráficos de produção"""
    if df is None or df.empty:
        print("Nenhum dado válido para plotagem")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Processa cada poço individualmente
    wells = set(var.split(':')[1] for var in df.columns if ':' in var and not var.startswith('WWIR'))

    for well in wells:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Líquido
        liquid_vars = [var for var in df.columns if f'WLPR:{well}' in var]
        if liquid_vars:
            df[liquid_vars].plot(ax=axes[0], style=['-o'], linewidth=2)
            axes[0].set_title(f'Produção de Líquido - {well}')
            axes[0].set_ylabel('m³/d')
            axes[0].grid(True)

        # Óleo
        oil_vars = [var for var in df.columns if f'WOPR:{well}' in var]
        if oil_vars:
            df[oil_vars].plot(ax=axes[1], style=['-s'], linewidth=2)
            axes[1].set_title(f'Produção de Óleo - {well}')
            axes[1].set_ylabel('m³/d')
            axes[1].grid(True)

        # Água
        water_vars = [var for var in df.columns if f'WWPR:{well}' in var]
        if water_vars:
            df[water_vars].plot(ax=axes[2], style=['-^'], color='blue', linewidth=2)
            axes[2].set_title(f'Produção de Água - {well}')
            axes[2].set_ylabel('m³/d')
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/production_{well}.png")
        plt.close()

    # Gera gráfico para o poço injetor INJECT1 se existir
    if 'WWIR:INJECT1' in df.columns:
        plt.figure(figsize=(12, 6))
        df['WWIR:INJECT1'].plot(style='-x', color='purple', linewidth=2)
        plt.title('Injeção de Água - INJECT1')
        plt.ylabel('m³/d')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/injection_INJECT1.png")
        plt.close()

def main():
    # Caminho para as simulações no Windows acessado via WSL
    base_path = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado_PUC/benchmark/OPM_Egg"

    # Itera sobre todas as pastas simulacao_0 até simulacao_100
    for i in range(101):
        sim_dir = os.path.join(base_path, f"simulacao_{i}")
        smspec_files = glob.glob(os.path.join(sim_dir, "*.SMSPEC"))

        if not smspec_files:
            print(f"Nenhum arquivo .SMSPEC encontrado em {sim_dir}")
            continue

        for smspec in smspec_files:
            print(f"\nProcessando: {smspec}")
            case_name = Path(smspec).stem

            # Carrega dados
            df = load_summary_data(smspec)

            if df is not None and not df.empty:
                # Salva CSV
                csv_file = os.path.join(sim_dir, f"{case_name}_production.csv")
                df.to_csv(csv_file)
                print(f"Dados salvos em {csv_file}")

                # Gera gráficos
                #plot_dir = os.path.join(sim_dir, f"{case_name}_plots")
                plot_dir = os.path.join(sim_dir, f"{case_name}_plots")
                if os.path.exists(plot_dir):
                    shutil.rmtree(plot_dir)
                os.makedirs(plot_dir)
                plot_production_curves(df, plot_dir)
                #plot_production_curves(df, plot_dir)
                print(f"Gráficos salvos em {plot_dir}")
            else:
                print("Nenhum dado de produção válido encontrado")

if __name__ == "__main__":
    main()
