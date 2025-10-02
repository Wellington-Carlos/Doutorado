#!/usr/bin/env python3
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ecl.summary import EclSum
import glob
import os
from pathlib import Path
import warnings
import sys

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')


def load_summary_data(smspec_file, wells_filter=None):
    """Carrega dados de produção e injeção por poço"""
    try:
        case_path = os.path.splitext(smspec_file)[0]
        eclsum = EclSum(case_path)

        all_vars = eclsum.keys()
        data = {"DATE": [d for d in eclsum.dates]}

        selected_vars = []
        for var in all_vars:
            if ":" in var:
                prefix, well = var.split(":")
                if wells_filter is None or well in wells_filter:
                    if prefix in ["WOPR", "WWPR", "WLPR", "WWIR"]:
                        data[var] = eclsum.numpy_vector(var)
                        selected_vars.append(var)

        # garante que também captura curvas de injeção por poço
        if wells_filter:
            for well in wells_filter:
                inj_var = f"WWIR:{well}"
                if inj_var in all_vars and inj_var not in selected_vars:
                    data[inj_var] = eclsum.numpy_vector(inj_var)
                    selected_vars.append(inj_var)

        if not selected_vars:
            print("Nenhum dado de produção/injeção encontrado para os poços filtrados")
            return None

        return pd.DataFrame(data).set_index("DATE")

    except Exception as e:
        print(f"Erro ao carregar {smspec_file}: {str(e)}")
        return None


def save_field_totals(smspec_file, output_dir):
    """Exporta variáveis totais de campo para CSV"""
    try:
        case_path = os.path.splitext(smspec_file)[0]
        eclsum = EclSum(case_path)

        field_vars = [k for k in eclsum.keys() if k.startswith("F")]
        if not field_vars:
            print("Nenhuma variável de campo encontrada")
            return

        data = {"DATE": [d for d in eclsum.dates]}
        for var in field_vars:
            data[var] = eclsum.numpy_vector(var)

        df_field = pd.DataFrame(data).set_index("DATE")

        csv_file = os.path.join(output_dir, "field_totals.csv")
        df_field.to_csv(csv_file)
        print(f"Variáveis de campo salvas em {csv_file}")

    except Exception as e:
        print(f"Erro ao salvar variáveis de campo: {e}")


def plot_production_curves(df, wells_filter, output_dir="plots"):
    """Plota curvas de produção e injeção por poço"""
    if df is None or df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    for well in wells_filter:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        if f"WLPR:{well}" in df.columns:
            df[f"WLPR:{well}"].plot(ax=axes[0], style="-o", linewidth=2)
            axes[0].set_title(f"Produção de Líquido - {well}")
            axes[0].set_ylabel("m³/d")
            axes[0].grid(True)

        if f"WOPR:{well}" in df.columns:
            df[f"WOPR:{well}"].plot(ax=axes[1], style="-s", linewidth=2)
            axes[1].set_title(f"Produção de Óleo - {well}")
            axes[1].set_ylabel("m³/d")
            axes[1].grid(True)

        if f"WWPR:{well}" in df.columns:
            df[f"WWPR:{well}"].plot(ax=axes[2], style="-^", color="blue", linewidth=2)
            axes[2].set_title(f"Produção de Água - {well}")
            axes[2].set_ylabel("m³/d")
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/production_{well}.png")
        plt.close()

        # curva de injeção (se existir)
        if f"WWIR:{well}" in df.columns:
            plt.figure(figsize=(12, 6))
            df[f"WWIR:{well}"].plot(style="-x", color="purple", linewidth=2)
            plt.title(f"Injeção de Água - {well}")
            plt.ylabel("m³/d")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/injection_{well}.png")
            plt.close()


def main():
    base_path = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado/Proposta_de_Tese/Codes_Optimization_Reservoir/Temp"

    # Identifica período (1 ou 2)
    period = sys.argv[1] if len(sys.argv) > 1 else "1"
    if period == "1":
        wells_filter = ["PROD1", "INJECT1"]
        output_dir = os.path.join(base_path, "EGG_plots_1")
        csv_file = os.path.join(output_dir, "EGG_production_1.csv")
    else:
        wells_filter = ["PROD1", "PROD2", "INJECT1", "INJECT2"]
        output_dir = os.path.join(base_path, "EGG_plots_2")
        csv_file = os.path.join(output_dir, "EGG_production_2.csv")

    smspec_files = glob.glob(os.path.join(base_path, "*.SMSPEC"))
    if not smspec_files:
        print(f"Nenhum arquivo .SMSPEC encontrado em {base_path}")
        return

    for smspec in smspec_files:
        print(f"\nProcessando: {smspec}")
        df = load_summary_data(smspec, wells_filter=wells_filter)

        if df is not None and not df.empty:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            # Salva CSV com variáveis por poço (inclui injetores)
            df.to_csv(csv_file)
            print(f"CSV salvo em {csv_file}")

            # Salva figuras por poço
            plot_production_curves(df, wells_filter, output_dir)
            print(f"Gráficos salvos em {output_dir}")

            # Salva variáveis de campo
            save_field_totals(smspec, output_dir)


if __name__ == "__main__":
    main()
