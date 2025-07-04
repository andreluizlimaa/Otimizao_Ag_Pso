# analise_ag.py

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd # Para organizar os resultados em um DataFrame

# Importe a função principal do seu algoritmo genético.
# Certifique-se de que 'genetico' é o nome correto do arquivo (sem .py)
from Genetico import algoritmo_genetico # Ajustado para 'genetico' conforme padrão

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

# Altere para o número desejado de simulações (ex: 30) para obter estatísticas significativas
NUM_SIMULACOES = 20 # <-- IMPORTANTE: Mude para 30 ou mais para ter médias e desvios padrão úteis
OUTPUT_FOLDER_ANALYSIS = "resultados_analise_ag" # Pasta para salvar os resultados da análise
OUTPUT_FOLDER_SUMMARY = os.path.join(OUTPUT_FOLDER_ANALYSIS, "sumario_estatistico") # Nova pasta para o CSV sumarizado

class AnaliseAG:
    def __init__(self, parametros_ag, num_simulacoes, output_folder):
        self.parametros_ag = parametros_ag
        self.num_simulacoes = num_simulacoes
        self.output_folder = output_folder
        self.output_folder_summary = OUTPUT_FOLDER_SUMMARY # Usar a nova pasta para o sumário
        
        # Garante que as pastas de saída existam
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.output_folder_summary, exist_ok=True) # Cria a pasta para o sumário

    def executar(self):
        print(f"Iniciando análise do Algoritmo Genético com {self.num_simulacoes} simulações...\n")

        # Listas para armazenar os resultados de cada simulação individualmente
        resultados_simulacoes = []
        
        # DataFrame para armazenar o histórico de convergência de cada simulação
        historicos_convergencia_df = pd.DataFrame()

        for i in range(self.num_simulacoes):
            print(f"Executando simulação AG {i+1}/{self.num_simulacoes}...")
            
            # Chama a função algoritmo_genetico e recebe o dicionário de resultados
            results_ag = algoritmo_genetico(
                tamanho_populacao=self.parametros_ag["tamanho_populacao"],
                limites=self.parametros_ag["limites"],
                num_geracoes=self.parametros_ag["num_geracoes"],
                taxa_cruzamento=self.parametros_ag["taxa_cruzamento"],
                taxa_mutacao=self.parametros_ag["taxa_mutacao"],
                geracoes_sem_melhora_limite=self.parametros_ag["geracoes_sem_melhora_limite"],
                tolerancia=self.parametros_ag["tolerancia"]
            )
            
            # Adiciona os resultados desta simulação à lista
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
            
            # Adiciona o histórico de convergência desta simulação ao DataFrame
            # Isso garante que mesmo se as simulações tiverem comprimentos diferentes,
            # o Pandas lidará com isso com NaNs, que serão preenchidos depois.
            historicos_convergencia_df[f'Simulação {i+1}'] = pd.Series(results_ag["historico_melhor_global"])


        # --- Geração de Estatísticas e Relatórios ---
        # Converte a lista de resultados em um DataFrame para facilitar a análise estatística
        df_resultados_completos = pd.DataFrame(resultados_simulacoes)

        # Cria um timestamp para os arquivos de saída
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- SALVAR RESULTADOS BRUTOS EM CSV E EXCEL (como antes) ---
        csv_filename_bruto = os.path.join(self.output_folder, f"resultados_ag_simulacoes_{timestamp}.csv")
        excel_filename_bruto = os.path.join(self.output_folder, f"resultados_ag_simulacoes_{timestamp}.xlsx")
        
        df_resultados_completos.to_csv(csv_filename_bruto, index=False)
        
        try: # Bloco try-except para o Excel, caso openpyxl não esteja instalado
            df_resultados_completos.to_excel(excel_filename_bruto, index=False)
            print(f"Resultados detalhados de cada simulação salvos em Excel: {excel_filename_bruto}")
        except ModuleNotFoundError:
            print("AVISO: openpyxl não está instalado. Não foi possível salvar os resultados em formato Excel.")
            print("Para habilitar o salvamento em Excel, execute: pip install openpyxl")
        
        print(f"\nResultados detalhados de cada simulação salvos em CSV: {csv_filename_bruto}")


        # --- NOVO: GERAR E SALVAR CSV COM SUMÁRIO ESTATÍSTICO (formato PSO) ---
        # As colunas 'Simulacao' não devem ser incluídas nas estatísticas de média/desvio padrão
        colunas_metrica = [col for col in df_resultados_completos.columns if col != 'Simulacao']

        # Calcular as estatísticas descritivas (média e desvio padrão)
        stats = df_resultados_completos[colunas_metrica].describe().loc[['mean', 'std']].transpose()

        # Renomear as colunas para 'Media' e 'Desvio_Padrao' para combinar com o formato do PSO
        stats.rename(columns={'mean': 'Media', 'std': 'Desvio_Padrao'}, inplace=True)

        # Resetar o índice para transformar os nomes das métricas em uma coluna 'Métrica'
        df_sumario_ag = stats.reset_index()
        df_sumario_ag.rename(columns={'index': 'Métrica'}, inplace=True)

        # Definir uma ordem específica para as métricas, se desejar (opcional, para consistência visual)
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
        # Garantir que todas as métricas existentes estejam na ordem, adicionando novas ao final se houver
        ordem_final = [m for m in ordem_metrica if m in df_sumario_ag['Métrica'].values]
        # Adicionar métricas que podem não estar na lista pré-definida, mas estão no DataFrame
        for m in df_sumario_ag['Métrica'].values:
            if m not in ordem_final:
                ordem_final.append(m)
        
        df_sumario_ag['Métrica'] = pd.Categorical(df_sumario_ag['Métrica'], categories=ordem_final, ordered=True)
        df_sumario_ag = df_sumario_ag.sort_values('Métrica')
        df_sumario_ag.reset_index(drop=True, inplace=True)


        # Salvar o CSV sumarizado na nova pasta
        sumario_csv_filename = os.path.join(self.output_folder_summary, f"Sumario_Desempenho_AG_{timestamp}.csv")
        df_sumario_ag.to_csv(sumario_csv_filename, index=False)
        print(f"\nSumário de desempenho AG (Média e Desvio Padrão) salvo em: {sumario_csv_filename}")
        print("\n--- Sumário de Desempenho AG ---")
        print(df_sumario_ag) # Imprime o sumário no console

        # --- SALVAR RELATÓRIO DE ESTATÍSTICAS (TXT) ---
        report_filename = os.path.join(self.output_folder, f"relatorio_analise_ag_{timestamp}.txt")
        with open(report_filename, "w") as f:
            f.write(f"Análise do Algoritmo Genético ({self.num_simulacoes} simulações)\n")
            f.write("------------------------------------------------------------------\n")
            f.write("Parâmetros do AG:\n")
            for param, value in self.parametros_ag.items():
                f.write(f"  {param.replace('_', ' ').capitalize()}: {value}\n")
            f.write("------------------------------------------------------------------\n")
            f.write("Estatísticas Sumárias das Simulações (Formato 'describe'):\n")
            # Usa o stats_sumario original do describe() para o TXT
            f.write(df_resultados_completos.describe().loc[['mean', 'std', 'min', 'max']].transpose().to_string())
            f.write("\n------------------------------------------------------------------\n")
            f.write("\nSumário de Desempenho (Média e Desvio Padrão):\n")
            f.write(df_sumario_ag.to_string()) # Adiciona o novo sumário formatado ao TXT
            f.write("\n------------------------------------------------------------------\n")
            print(f"Relatório de análise salvo em: {report_filename}")

        # --- Plotar o gráfico de convergência médio ---
        plt.figure(figsize=(12, 7)) # Aumenta um pouco o tamanho para melhor visualização
        
        # Preenche NaNs com o último valor válido para que a média e desvio padrão sejam calculados corretamente
        historicos_convergencia_filled = historicos_convergencia_df.fillna(method='ffill', axis=0) 
        
        # Calcula a média e o desvio padrão ao longo das simulações
        media_historico = historicos_convergencia_filled.mean(axis=1)
        std_historico = historicos_convergencia_filled.std(axis=1)

        geracoes = np.arange(len(media_historico))
        
        plt.plot(geracoes, media_historico, label=f'Média da Aptidão do Melhor Indivíduo (AG)', color='blue')
        plt.fill_between(geracoes, media_historico - std_historico, media_historico + std_historico, 
                         alpha=0.2, color='lightblue', label='Desvio Padrão (AG)')
        
        plt.title('Convergência Média do Algoritmo Genético ao Longo das Simulações')
        plt.xlabel('Gerações')
        plt.ylabel('Melhor Aptidão (Valor da Função Objetivo)')
        plt.grid(True)
        plt.legend()
        plt.yscale('log') # Escala logarítmica é útil para ver a convergência em funções com grande faixa de valores
        plt.tight_layout() # Ajusta o layout para evitar corte de rótulos
        
        plot_filename = os.path.join(self.output_folder, f"convergencia_media_ag_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Gráfico de convergência média salvo em: {plot_filename}")

# Bloco principal para executar a análise
if __name__ == "__main__":
    analisador = AnaliseAG(PARAMETROS_AG, NUM_SIMULACOES, OUTPUT_FOLDER_ANALYSIS)
    analisador.executar()