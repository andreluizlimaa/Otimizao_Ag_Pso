# analise_pso.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Adiciona o diretório atual ao PATH para que possamos importar PSO
# Isso é necessário se analise_pso.py não estiver na raiz do projeto ou se for executado de um subdiretório
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa a classe PSO e a função objetivo W4
# Certifique-se de que 'funcoes_otimizacao.py' e 'Grafico.py' também estejam acessíveis
from PSO import PSO
from funcoes_otimizacao import funcao_w4

def run_pso_simulations(num_simulations=20):
    """
    Roda múltiplas simulações do algoritmo PSO com os mesmos parâmetros
    e coleta os resultados.
    """
    # --- Parâmetros fixos para todas as simulações do PSO ---
    # Estes parâmetros DEVEM ser os mesmos que você usa no seu PSO.py
    # para que a análise seja consistente.
    limites_funcao = (-500, 500)
    num_particulas = 20
    num_iteracoes = 100
    w_max = 0.7
    w_min = 0.2
    c1 = 2
    c2 = 2
    tolerancia = 1e-6
    iteracoes_sem_melhora_limite = 20

    all_results = [] # Lista para armazenar os dicionários de resultados de cada simulação

    print(f"\nIniciando {num_simulations} simulações do PSO...")
    for i in range(num_simulations):
        print(f"Executando simulação {i+1}/{num_simulations}...")
        # Cria uma nova instância do PSO para cada simulação
        otimizador_pso = PSO(
            funcao_w4, limites_funcao, num_particulas, num_iteracoes,
            w_max, w_min, c1, c2, tolerancia, iteracoes_sem_melhora_limite
        )
        # Executa o algoritmo e armazena o dicionário de resultados
        results = otimizador_pso.executar()
        all_results.append(results)
        print("-" * 30) # Separador visual

    # Retorna todos os resultados e os parâmetros comuns usados
    return all_results, {
        "limites_funcao": limites_funcao,
        "num_particulas": num_particulas,
        "num_iteracoes_max": num_iteracoes,
        "w_max": w_max,
        "w_min": w_min,
        "c1": c1,
        "c2": c2,
        "tolerancia": tolerancia,
        "iteracoes_sem_melhora_limite": iteracoes_sem_melhora_limite
    }

def analyze_and_plot_results(all_results, common_params):
    """
    Analisa os resultados coletados, calcula médias/desvio padrão e gera um gráfico.
    """
    if not all_results:
        print("Nenhum resultado para analisar. Verifique as simulações.")
        return

    # Extrai os dados relevantes de cada simulação
    melhores_valores = [r['melhor_valor_global'] for r in all_results]
    iteracoes_executadas = [r['iteracoes_executadas'] for r in all_results]
    avaliacoes_total = [r['avaliacoes_funcao_total'] for r in all_results]
    multiplicacoes_total = [r['multiplicacoes_total'] for r in all_results]
    divisoes_total = [r['divisoes_total'] for r in all_results]
    avaliacoes_minimo_global = [r['avaliacoes_minimo_global'] for r in all_results]
    multiplicacoes_minimo_global = [r['multiplicacoes_minimo_global'] for r in all_results]
    divisoes_minimo_global = [r['divisoes_minimo_global'] for r in all_results]

    # Calcula médias e desvio padrão
    mean_melhor_valor = np.mean(melhores_valores)
    std_melhor_valor = np.std(melhores_valores)

    mean_iteracoes = np.mean(iteracoes_executadas)
    std_iteracoes = np.std(iteracoes_executadas)

    mean_avaliacoes_total = np.mean(avaliacoes_total)
    std_avaliacoes_total = np.std(avaliacoes_total)

    mean_multiplicacoes_total = np.mean(multiplicacoes_total)
    std_multiplicacoes_total = np.std(multiplicacoes_total)
    
    mean_divisoes_total = np.mean(divisoes_total)
    std_divisoes_total = np.std(divisoes_total)

    mean_avaliacoes_min_global = np.mean(avaliacoes_minimo_global)
    std_avaliacoes_min_global = np.std(avaliacoes_minimo_global)

    mean_multiplicacoes_min_global = np.mean(multiplicacoes_minimo_global)
    std_multiplicacoes_min_global = np.std(multiplicacoes_minimo_global)
    
    mean_divisoes_min_global = np.mean(divisoes_minimo_global)
    std_divisoes_min_global = np.std(divisoes_minimo_global)


    # --- Impressão dos resultados médios e desvio padrão ---
    print("\n" + "="*50)
    print("--- Análise Média dos Resultados do PSO ---")
    print(f"Número de Simulações: {len(all_results)}")
    print("\nParâmetros Utilizados nas Simulações:")
    print(f"  Limites da Função: {common_params['limites_funcao']}")
    print(f"  Número de Partículas: {common_params['num_particulas']}")
    print(f"  Número de Iterações Máximo: {common_params['num_iteracoes_max']}")
    print(f"  Peso de Inércia (W_max): {common_params['w_max']}")
    print(f"  Peso de Inércia (W_min): {common_params['w_min']}")
    print(f"  Coeficiente Cognitivo (c1): {common_params['c1']}")
    print(f"  Coeficiente Social (c2): {common_params['c2']}")
    print(f"  Tolerância para Convergência: {common_params['tolerancia']}")
    print(f"  Iterações sem Melhora Limite: {common_params['iteracoes_sem_melhora_limite']}")
    print("\nResultados Médios (com Desvio Padrão):")
    print(f"  Melhor Valor Global: {mean_melhor_valor:.4f} (DP: {std_melhor_valor:.4f})")
    print(f"  Iterações Executadas: {mean_iteracoes:.2f} (DP: {std_iteracoes:.2f})")
    print(f"  Avaliações da Função Objetivo (Total): {mean_avaliacoes_total:.2f} (DP: {std_avaliacoes_total:.2f})")
    print(f"  Multiplicações (Total): {mean_multiplicacoes_total:.2f} (DP: {std_multiplicacoes_total:.2f})")
    print(f"  Divisões (Total): {mean_divisoes_total:.2f} (DP: {std_divisoes_total:.2f})")
    print(f"  Avaliações para o 'Melhor Global' (momento): {mean_avaliacoes_min_global:.2f} (DP: {std_avaliacoes_min_global:.2f})")
    print(f"  Multiplicações para o 'Melhor Global' (momento): {mean_multiplicacoes_min_global:.2f} (DP: {std_multiplicacoes_min_global:.2f})")
    print(f"  Divisões para o 'Melhor Global' (momento): {mean_divisoes_min_global:.2f} (DP: {std_divisoes_min_global:.2f})")
    print("="*50)

    # --- Plotagem dos resultados médios com desvio padrão ---
    metrics_names = [
        "Melhor Valor Global", 
        "Iterações Executadas", 
        "Avaliações Função (Total)",
        "Multiplicações (Total)",
        "Divisões (Total)",
        "Avaliações Melhor Global",
        "Multiplicações Melhor Global",
        "Divisões Melhor Global"
    ]
    mean_values = [
        mean_melhor_valor, 
        mean_iteracoes, 
        mean_avaliacoes_total,
        mean_multiplicacoes_total,
        mean_divisoes_total,
        mean_avaliacoes_min_global,
        mean_multiplicacoes_min_global,
        mean_divisoes_min_global
    ]
    std_devs = [
        std_melhor_valor, 
        std_iteracoes, 
        std_avaliacoes_total,
        std_multiplicacoes_total,
        std_divisoes_total,
        std_avaliacoes_min_global,
        std_multiplicacoes_min_global,
        std_divisoes_min_global
    ]

    # Cria subplots para organizar melhor os gráficos, pois são muitos dados
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    axes = axes.flatten() # Achata a matriz de eixos para fácil iteração

    # Gráfico 1: Melhor Valor Global e Iterações Executadas
    ax1 = axes[0]
    bars1 = ax1.bar(metrics_names[0:2], # Pegando 'Melhor Valor Global' e 'Iterações Executadas'
                    mean_values[0:2], 
                    yerr=std_devs[0:2], 
                    capsize=5, color=['skyblue', 'lightcoral'])
    ax1.set_title('Média e Desvio Padrão (Melhor Valor e Iterações)')
    ax1.set_ylabel('Valor')
    ax1.ticklabel_format(style='plain', axis='y') # Evita notação científica no eixo Y se os valores forem grandes
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=9)

    # Gráfico 2: Avaliações e Operações (Total)
    ax2 = axes[1]
    bars2 = ax2.bar(metrics_names[2:5], # Pegando 'Avaliações Função (Total)', 'Multiplicações (Total)', 'Divisões (Total)'
                    mean_values[2:5], 
                    yerr=std_devs[2:5], 
                    capsize=5, color=['lightgreen', 'salmon', 'plum'])
    ax2.set_title('Média e Desvio Padrão (Avaliações e Operações - Total)')
    ax2.set_ylabel('Quantidade')
    ax2.ticklabel_format(style='plain', axis='y')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}', va='bottom', ha='center', fontsize=9, rotation=45) # Formata como inteiro, com vírgulas

    # Gráfico 3: Avaliações e Operações (Melhor Global)
    ax3 = axes[2]
    bars3 = ax3.bar(metrics_names[5:8], # Pegando 'Avaliações Melhor Global', 'Multiplicações Melhor Global', 'Divisões Melhor Global'
                    mean_values[5:8], 
                    yerr=std_devs[5:8], 
                    capsize=5, color=['gold', 'darkorange', 'mediumpurple'])
    ax3.set_title('Média e Desvio Padrão (Avaliações e Operações - Melhor Global)')
    ax3.set_ylabel('Quantidade')
    ax3.ticklabel_format(style='plain', axis='y')
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}', va='bottom', ha='center', fontsize=9, rotation=45) # Formata como inteiro, com vírgulas

    # Adiciona a legenda dos parâmetros em um subplot separado
    ax4 = axes[3]
    ax4.axis('off') # Desliga os eixos para este subplot
    
    # Prepara o texto da legenda com os parâmetros
    legend_text = (
        f"Parâmetros da Simulação:\n"
        f"Função Otimizada: W4 (Minimização)\n"
        f"Limites da Função: {common_params['limites_funcao']}\n"
        f"Número de Partículas: {common_params['num_particulas']}\n"
        f"Número de Iterações Máximo: {common_params['num_iteracoes_max']}\n"
        f"Peso de Inércia (W_max): {common_params['w_max']}\n"
        f"Peso de Inércia (W_min): {common_params['w_min']}\n"
        f"Coeficiente Cognitivo (c1): {common_params['c1']}\n"
        f"Coeficiente Social (c2): {common_params['c2']}\n"
        f"Tolerância para Convergência: {common_params['tolerancia']}\n"
        f"Iterações sem Melhora Limite: {common_params['iteracoes_sem_melhora_limite']}\n"
        f"Número de Simulações: {len(all_results)}\n"
    )
    # Adiciona o texto ao subplot
    ax4.text(0.05, 0.95, legend_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout() # Ajusta o layout para evitar sobreposição
    plt.suptitle(f"Análise Média de {len(all_results)} Simulações PSO", y=1.02, fontsize=16) # Título geral para todos os subplots
    plt.show()
    plt.savefig("analise_pso_resultados.png", dpi=300, bbox_inches='tight') # Salva a figura com alta resolução
    plt.close(fig) # Fecha a janela do gráfico para liberar recursos

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    num_simulations_to_run = 20 # Defina quantas vezes você quer rodar o PSO
    
    # Roda as simulações e captura os resultados e parâmetros comuns
    results_data, simulation_params = run_pso_simulations(num_simulations=num_simulations_to_run)
    
    # Analisa e plota os resultados
    analyze_and_plot_results(results_data, simulation_params)