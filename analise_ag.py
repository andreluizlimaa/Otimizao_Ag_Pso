# analise_ag.py

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

# Importe a função principal do seu algoritmo genético.
from Genetico import algoritmo_genetico

# --- Definição de Parâmetros para Análise do AG ---
PARAMETROS_AG = {
    "tamanho_populacao": 35,
    "limites": (-500, 500),
    "num_geracoes": 200,
    "taxa_cruzamento": 0.7,
    "taxa_mutacao": 0.07,
    "geracoes_sem_melhora_limite": 20, # Critério de parada por convergência
    "tolerancia": 1e-6 # Tolerância para o critério de parada
}

NUM_SIMULACOES = 20 
OUTPUT_FOLDER_ANALYSIS = "resultados_analise_ag" # Pasta para salvar os resultados da análise
OUTPUT_FOLDER_SUMMARY = os.path.join(OUTPUT_FOLDER_ANALYSIS, "sumario_estatistico") # Nova pasta para o CSV sumarizado

class AnaliseAG:
    def __init__(self, parametros_ag, num_simulacoes, output_folder):
        self.parametros_ag = parametros_ag
        self.num_simulacoes = num_simulacoes
        self.output_folder = output_folder
        self.output_folder_summary = OUTPUT_FOLDER_SUMMARY
        
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.output_folder_summary, exist_ok=True)

    def executar(self):
        print(f"Iniciando análise do Algoritmo Genético com {self.num_simulacoes} simulações...\n")

        resultados_simulacoes = []
        
        # NOVAS LISTAS para armazenar os históricos de cada simulação
        all_historico_melhor_geracao = []
        all_historico_melhor_global = []

        for i in range(self.num_simulacoes):
            print(f"Executando simulação AG {i+1}/{self.num_simulacoes}...")
            
            results_ag = algoritmo_genetico(
                tamanho_populacao=self.parametros_ag["tamanho_populacao"],
                limites=self.parametros_ag["limites"],
                num_geracoes=self.parametros_ag["num_geracoes"],
                taxa_cruzamento=self.parametros_ag["taxa_cruzamento"],
                taxa_mutacao=self.parametros_ag["taxa_mutacao"],
                geracoes_sem_melhora_limite=self.parametros_ag["geracoes_sem_melhora_limite"],
                tolerancia=self.parametros_ag["tolerancia"]
            )
            
            resultados_simulacoes.append({
                "Simulacao": i + 1,
                "Melhor Valor Encontrado": results_ag["melhor_valor_global"],
                "Gerações para Convergência": results_ag["iteracoes_executadas"],
                "Avaliações Função Objetivo (Total)": results_ag["avaliacoes_funcao_total"],
                "Multiplicações (Total)": results_ag["multiplicacoes_total"],
                "Divisões (Total)": results_ag["divisoes_total"],
                "Avaliações (no Melhor Global)": results_ag["avaliacoes_minimo_global"],
                "Multiplicações (no Melhor Global)": results_ag["multiplicacoes_minimo_global"],
                "Divisões (no Melhor Global)": results_ag["divisoes_minimo_global"],
            })
            
            # Coleta os dois históricos de cada simulação
            all_historico_melhor_geracao.append(results_ag["historico_melhor_geracao"])
            all_historico_melhor_global.append(results_ag["historico_melhor_global"])


        # --- Geração de Estatísticas e Relatórios (sem alterações nesta parte) ---
        df_resultados_completos = pd.DataFrame(resultados_simulacoes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_filename_bruto = os.path.join(self.output_folder, f"resultados_ag_simulacoes_{timestamp}.csv")
        excel_filename_bruto = os.path.join(self.output_folder, f"resultados_ag_simulacoes_{timestamp}.xlsx")
        
        df_resultados_completos.to_csv(csv_filename_bruto, index=False)
        
        try:
            df_resultados_completos.to_excel(excel_filename_bruto, index=False)
            print(f"Resultados detalhados de cada simulação salvos em Excel: {excel_filename_bruto}")
        except ModuleNotFoundError:
            print("AVISO: openpyxl não está instalado. Não foi possível salvar os resultados em formato Excel.")
            print("Para habilitar o salvamento em Excel, execute: pip install openpyxl")
        
        print(f"\nResultados detalhados de cada simulação salvos em CSV: {csv_filename_bruto}")


        colunas_metrica = [col for col in df_resultados_completos.columns if col != 'Simulacao']
        stats = df_resultados_completos[colunas_metrica].describe().loc[['mean', 'std']].transpose()
        stats.rename(columns={'mean': 'Media', 'std': 'Desvio_Padrao'}, inplace=True)
        df_sumario_ag = stats.reset_index()
        df_sumario_ag.rename(columns={'index': 'Métrica'}, inplace=True)

        ordem_metrica = [
            "Melhor Valor Encontrado",
            "Gerações para Convergência",
            "Avaliações Função Objetivo (Total)",
            "Multiplicações (Total)",
            "Divisões (Total)",
            "Avaliações (no Melhor Global)",
            "Multiplicações (no Melhor Global)",
            "Divisões (no Melhor Global)"
        ]
        ordem_final = [m for m in ordem_metrica if m in df_sumario_ag['Métrica'].values]
        for m in df_sumario_ag['Métrica'].values:
            if m not in ordem_final:
                ordem_final.append(m)
        
        df_sumario_ag['Métrica'] = pd.Categorical(df_sumario_ag['Métrica'], categories=ordem_final, ordered=True)
        df_sumario_ag = df_sumario_ag.sort_values('Métrica')
        df_sumario_ag.reset_index(drop=True, inplace=True)

        sumario_csv_filename = os.path.join(self.output_folder_summary, f"Sumario_Desempenho_AG_{timestamp}.csv")
        df_sumario_ag.to_csv(sumario_csv_filename, index=False)
        print(f"\nSumário de desempenho AG (Média e Desvio Padrão) salvo em: {sumario_csv_filename}")
        print("\n--- Sumário de Desempenho AG ---")
        print(df_sumario_ag)

        report_filename = os.path.join(self.output_folder, f"relatorio_analise_ag_{timestamp}.txt")
        with open(report_filename, "w") as f:
            f.write(f"Análise do Algoritmo Genético ({self.num_simulacoes} simulações)\n")
            f.write("------------------------------------------------------------------\n")
            f.write("Parâmetros do AG:\n")
            for param, value in self.parametros_ag.items():
                f.write(f"  {param.replace('_', ' ').capitalize()}: {value}\n")
            f.write("------------------------------------------------------------------\n")
            f.write("Estatísticas Sumárias das Simulações (Formato 'describe'):\n")
            f.write(df_resultados_completos.describe().loc[['mean', 'std', 'min', 'max']].transpose().to_string())
            f.write("\n------------------------------------------------------------------\n")
            f.write("\nSumário de Desempenho (Média e Desvio Padrão):\n")
            f.write(df_sumario_ag.to_string())
            f.write("\n------------------------------------------------------------------\n")
            print(f"Relatório de análise salvo em: {report_filename}")

        # Garantir que todos os históricos tenham o mesmo comprimento para calcular média/desvio
        max_len = max(len(h) for h in all_historico_melhor_global) # Pode usar qualquer um dos históricos para o max_len
        if not all_historico_melhor_global: # Adiciona esta verificação
            print("Nenhum histórico de dados para plotar o gráfico de convergência AG.")
            return

        padded_historico_melhor_geracao = []
        padded_historico_melhor_global = []

        for i in range(self.num_simulacoes):
            hg_geracao = all_historico_melhor_geracao[i]
            hg_global = all_historico_melhor_global[i]

            padded_historico_melhor_geracao.append(
                hg_geracao + [hg_geracao[-1]] * (max_len - len(hg_geracao)) if hg_geracao else [np.nan] * max_len
            )
            padded_historico_melhor_global.append(
                hg_global + [hg_global[-1]] * (max_len - len(hg_global)) if hg_global else [np.nan] * max_len
            )

        # Calcula a média e o desvio padrão ao longo das simulações
        mean_historico_melhor_geracao = np.nanmean(padded_historico_melhor_geracao, axis=0)
        std_historico_melhor_geracao = np.nanstd(padded_historico_melhor_geracao, axis=0)
        
        mean_historico_melhor_global = np.nanmean(padded_historico_melhor_global, axis=0) # Média do melhor global

        geracoes = np.arange(1, max_len + 1)
        
        # Estilo e plotagem idênticos ao PSO
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Linha azul tracejada (Média do Melhor Z da Geração)
        ax.plot(geracoes, mean_historico_melhor_geracao, 'b--', label=f'Média do Melhor Z da Geração (AG)')
        
        # Área sombreada do desvio padrão
        upper_bound_geracao = mean_historico_melhor_geracao + std_historico_melhor_geracao
        lower_bound_geracao = mean_historico_melhor_geracao - std_historico_melhor_geracao
        ax.fill_between(geracoes, lower_bound_geracao, upper_bound_geracao, 
                        color='blue', alpha=0.1, label='Desvio Padrão (±1σ)')
        
        # Linha vermelha sólida (Média do Melhor Valor Global Encontrado)
        ax.plot(geracoes, mean_historico_melhor_global, 'r-', label='Média do Melhor Valor Global Encontrado', linewidth=2)

        ax.set_title(f'Gráfico de Convergência - Algoritmo Genético (Média de {self.num_simulacoes} Execuções)', fontsize=16)
        ax.set_xlabel('Número de Iterações/Gerações', fontsize=12)
        ax.set_ylabel('Valor da Função Objetivo (Z Ótimo)', fontsize=12)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True)
        ax.set_ylim(bottom=None)

        plt.tight_layout()
        
        plot_filename = os.path.join(self.output_folder, f"convergencia_media_ag_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de convergência média AG salvo em: {plot_filename}")

# Bloco principal para executar a análise
if __name__ == "__main__":
    analisador = AnaliseAG(PARAMETROS_AG, NUM_SIMULACOES, OUTPUT_FOLDER_ANALYSIS)
    analisador.executar()