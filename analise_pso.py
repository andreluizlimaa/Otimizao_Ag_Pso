# analise_pso.py

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd # Certifique-se de que pandas está importado

# Importa a classe PSO e a função objetivo W4
# Certifique-se de que PSO.py e funcoes_otimizacao.py estão no mesmo diretório
from PSO import PSO
from funcoes_otimizacao import funcao_w4

def plot_convergence_graph(all_historico_melhor_global, all_historico_media_melhores_locais,
                           all_historico_desvio_padrao_melhores_locais, algorithm_name="PSO", num_runs=1):
    """
    Gera o gráfico de convergência, calculando a média e desvio padrão
    dos históricos de múltiplas execuções.
    """
    if not all_historico_melhor_global:
        print("Nenhum histórico de dados para plotar o gráfico de convergência.")
        return

    max_len = max(len(h) for h in all_historico_media_melhores_locais)

    padded_historico_melhor_global = []
    padded_historico_media_melhores_locais = []
    padded_historico_desvio_padrao_melhores_locais = []

    for i in range(num_runs):
        hg = all_historico_melhor_global[i]
        hm = all_historico_media_melhores_locais[i]
        hdp = all_historico_desvio_padrao_melhores_locais[i]

        padded_historico_melhor_global.append(
            hg + [hg[-1]] * (max_len - len(hg)) if hg else [np.nan] * max_len
        )
        padded_historico_media_melhores_locais.append(
            hm + [hm[-1]] * (max_len - len(hm)) if hm else [np.nan] * max_len
        )
        padded_historico_desvio_padrao_melhores_locais.append(
            hdp + [hdp[-1]] * (max_len - len(hdp)) if hdp else [np.nan] * max_len
        )

    mean_historico_melhor_global = np.nanmean(padded_historico_melhor_global, axis=0)
    mean_historico_media_melhores_locais = np.nanmean(padded_historico_media_melhores_locais, axis=0)
    std_historico_media_melhores_locais = np.nanstd(padded_historico_media_melhores_locais, axis=0)

    iteracoes = np.arange(1, max_len + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(iteracoes, mean_historico_media_melhores_locais, 'b--', label=f'Média do Melhor Z da Geração ({algorithm_name})')
    upper_bound = mean_historico_media_melhores_locais + std_historico_media_melhores_locais
    lower_bound = mean_historico_media_melhores_locais - std_historico_media_melhores_locais
    ax.fill_between(iteracoes, lower_bound, upper_bound, color='blue', alpha=0.1, label='Desvio Padrão (±1σ)')
    ax.plot(iteracoes, mean_historico_melhor_global, 'r-', label='Média do Melhor Valor Global Encontrado', linewidth=2)

    ax.set_title(f'Gráfico de Convergência - {algorithm_name} (Média de {num_runs} Execuções)', fontsize=16)
    ax.set_xlabel('Número de Iterações/Gerações', fontsize=12)
    ax.set_ylabel('Valor da Função Objetivo (Z Ótimo)', fontsize=12)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)

    output_folder_analysis = "resultados_analise"
    os.makedirs(output_folder_analysis, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"Grafico_Convergencia_{algorithm_name}_media_{num_runs}_runs_{timestamp}.png"
    image_path = os.path.join(output_folder_analysis, image_name)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de convergência salvo em: {image_path}")

    plt.show()


def plot_bar_graphs(results_df, algorithm_name="PSO", num_runs=1, pso_params_display=None):
    """
    Gera gráficos de barra para as estatísticas de avaliações e operações computacionais.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Calcula médias e desvios padrão
    mean_best_global = results_df['melhor_valor_global'].mean()
    std_best_global = results_df['melhor_valor_global'].std()
    mean_iter_executed = results_df['iteracoes_executadas'].mean()
    std_iter_executed = results_df['iteracoes_executadas'].std()

    mean_fo_evals_total = results_df['avaliacoes_funcao_total'].mean()
    std_fo_evals_total = results_df['avaliacoes_funcao_total'].std()
    mean_mult_total = results_df['multiplicacoes_total'].mean()
    std_mult_total = results_df['multiplicacoes_total'].std()
    mean_div_total = results_df['divisoes_total'].mean()
    std_div_total = results_df['divisoes_total'].std()

    mean_fo_evals_min_global = results_df['avaliacoes_minimo_global'].mean()
    std_fo_evals_min_global = results_df['avaliacoes_minimo_global'].std()
    mean_mult_min_global = results_df['multiplicacoes_minimo_global'].mean()
    std_mult_min_global = results_df['multiplicacoes_minimo_global'].std()
    mean_div_min_global = results_df['divisoes_minimo_global'].mean()
    std_div_min_global = results_df['divisoes_minimo_global'].std()

    # --- Criação dos Gráficos de Barra ---
    fig2, axes = plt.subplots(2, 2, figsize=(16, 9)) # 2x2 subplots
    fig2.suptitle(f'Análise de Desempenho - {algorithm_name} (Média de {num_runs} Execuções)', fontsize=18)

    # Gráfico 1: Melhor Valor Global e Iterações Executadas
    ax1 = axes[0, 0]
    labels1 = ['Melhor Valor Global', 'Iterações Executadas']
    values1 = [mean_best_global, mean_iter_executed]
    errors1 = [std_best_global, std_iter_executed]
    colors1 = ['skyblue', 'lightcoral']
    bars1 = ax1.bar(labels1, values1, yerr=errors1, capsize=5, color=colors1)
    ax1.set_title('Média e Desvio Padrão (Melhor Valor e Iterações)', fontsize=14)
    ax1.set_ylabel('Valor / Iterações', fontsize=12)
    # Adicionar os valores acima das barras
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval),
                 f'{yval:.2f}', ha='center', va='bottom' if yval > 0 else 'top')

    # Gráfico 2: Avaliações e Operações - Total (para Convergência)
    ax2 = axes[0, 1]
    labels2 = ['Avaliações Função', 'Multiplicações', 'Divisões']
    values2 = [mean_fo_evals_total, mean_mult_total, mean_div_total]
    errors2 = [std_fo_evals_total, std_mult_total, std_div_total]
    colors2 = ['lightgreen', 'salmon', 'plum']
    bars2 = ax2.bar(labels2, values2, yerr=errors2, capsize=5, color=colors2)
    ax2.set_title('Média e Desvio Padrão (Avaliações e Operações - Total)', fontsize=14)
    ax2.set_ylabel('Quantidade', fontsize=12)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval),
                 f'{yval:.0f}', ha='center', va='bottom' if yval > 0 else 'top')


    # Gráfico 3: Avaliações e Operações - Melhor Global (para encontrar o mínimo global)
    ax3 = axes[1, 0]
    labels3 = ['Avaliações', 'Multiplicações', 'Divisões']
    values3 = [mean_fo_evals_min_global, mean_mult_min_global, mean_div_min_global]
    errors3 = [std_fo_evals_min_global, std_mult_min_global, std_div_min_global]
    colors3 = ['orange', 'darkorange', 'mediumpurple']
    bars3 = ax3.bar(labels3, values3, yerr=errors3, capsize=5, color=colors3)
    ax3.set_title('Média e Desvio Padrão (Avaliações e Operações - Melhor Global)', fontsize=14, pad=20)
    ax3.set_ylabel('Quantidade', fontsize=12)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * yval if yval > 0 else -0.05 * yval),
                 f'{yval:.0f}', ha='center', va='bottom' if yval > 0 else 'top')


    # Texto com Parâmetros da Simulação (no quarto subplot vazio)
    ax4 = axes[1, 1]
    ax4.set_axis_off() # Desliga os eixos para este subplot

    if pso_params_display:
        params_text = (
            f'Parâmetros da Simulação:\n'
            f'Função Otimizada: W4 (Minimização)\n'
            f'Limites da Função: {pso_params_display["limites_funcao"]}\n'
            f'Número de Partículas: {pso_params_display["num_particulas"]}\n'
            f'Número de Iterações Máximo: {pso_params_display["num_iteracoes"]}\n'
            f'Peso de Inércia (W_max): {pso_params_display["w_max"]:.2f}\n'
            f'Peso de Inércia (W_min): {pso_params_display["w_min"]:.2f}\n'
            f'Coeficiente Cognitivo (c1): {pso_params_display["c1"]:.0f}\n'
            f'Coeficiente Social (c2): {pso_params_display["c2"]:.0f}\n'
            f'Tolerância para Convergência: {pso_params_display["tolerancia"]}\n'
            f'Iterações sem Melhora Limite: {pso_params_display["iteracoes_sem_melhora_limite"]}\n'
        )
        ax4.text(0.5, 0.9, params_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', horizontalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7, ec='black', lw=1.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_folder_analysis = "resultados_analise"
    os.makedirs(output_folder_analysis, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"Graficos_Barras_{algorithm_name}_media_{num_runs}_runs_{timestamp}.png"
    image_path = os.path.join(output_folder_analysis, image_name)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"Gráficos de barra salvos em: {image_path}")

    plt.show()

# Função MODIFICADA para salvar o sumário em um arquivo CSV com UTF-8
def save_summary_to_csv(results_df, algorithm_name="PSO", num_runs=1, pso_params_display=None):
    """
    Salva as médias e desvios padrão das estatísticas de desempenho em um arquivo CSV.
    """
    output_folder_analysis = "resultados_analise"
    os.makedirs(output_folder_analysis, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"Sumario_Desempenho_{algorithm_name}_media_{num_runs}_runs_{timestamp}.csv"
    file_path = os.path.join(output_folder_analysis, file_name)

    # Calcula médias e desvios padrão (permanece o mesmo)
    mean_best_global = results_df['melhor_valor_global'].mean()
    std_best_global = results_df['melhor_valor_global'].std()
    mean_iter_executed = results_df['iteracoes_executadas'].mean()
    std_iter_executed = results_df['iteracoes_executadas'].std()

    mean_fo_evals_total = results_df['avaliacoes_funcao_total'].mean()
    std_fo_evals_total = results_df['avaliacoes_funcao_total'].std()
    mean_mult_total = results_df['multiplicacoes_total'].mean()
    std_mult_total = results_df['multiplicacoes_total'].std()
    mean_div_total = results_df['divisoes_total'].mean()
    std_div_total = results_df['divisoes_total'].std()

    mean_fo_evals_min_global = results_df['avaliacoes_minimo_global'].mean()
    std_fo_evals_min_global = results_df['avaliacoes_minimo_global'].std()
    mean_mult_min_global = results_df['multiplicacoes_minimo_global'].mean()
    std_mult_min_global = results_df['multiplicacoes_minimo_global'].std()
    mean_div_min_global = results_df['divisoes_minimo_global'].mean()
    std_div_min_global = results_df['divisoes_minimo_global'].std()

    # Cria um DataFrame para salvar no CSV (permanece o mesmo)
    data_to_csv = {
        'Métrica': [
            'Melhor Valor Global Final',
            'Iterações Executadas (até convergência)',
            'Avaliações da Função Objetivo (Total)',
            'Multiplicações (Total)',
            'Divisões (Total)',
            'Avaliações da Função Objetivo (Mínimo Global)',
            'Multiplicações (Mínimo Global)',
            'Divisões (Mínimo Global)'
        ],
        'Media': [
            mean_best_global,
            mean_iter_executed,
            mean_fo_evals_total,
            mean_mult_total,
            mean_div_total,
            mean_fo_evals_min_global,
            mean_mult_min_global,
            mean_div_min_global
        ],
        'Desvio_Padrao': [
            std_best_global,
            std_iter_executed,
            std_fo_evals_total,
            std_mult_total,
            std_div_total,
            std_fo_evals_min_global,
            std_mult_min_global,
            std_div_min_global
        ]
    }
    summary_df = pd.DataFrame(data_to_csv)

    # Adiciona os parâmetros de simulação como comentários no CSV
    header_comments = [
        f"# --- Sumário de Desempenho do Algoritmo {algorithm_name} ---",
        f"# Número de Execuções: {num_runs}",
        f"# Data e Hora da Análise: {timestamp}",
        "#",
        "# Parâmetros da Simulação:"
    ]
    if pso_params_display:
        for key, value in pso_params_display.items():
            # A função .title() pode introduzir caracteres especiais que precisam ser em UTF-8
            header_comments.append(f"#   {key.replace('_', ' ').title()}: {value}")
    header_comments.append("#") 

    # Salva o DataFrame no arquivo CSV, especificando encoding='utf-8'
    # Use 'newline=""' para evitar problemas de linhas em branco extras no CSV
    # MODIFICADO AQUI: encoding='utf-8'
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        for line in header_comments:
            f.write(line + '\n')
        summary_df.to_csv(f, index=False, float_format='%.4f') # encoding é tratado pelo 'f' aqui

    print(f"Sumário de desempenho salvo em: {file_path}")


if __name__ == "__main__":
    # --- VARIÁVEL PARA CONTROLAR QUANTAS VEZES O CÓDIGO RODA ---
    num_repeticoes_simulacao = 20 # Altere este valor para definir quantas vezes o PSO será executado

    # --- Configuração dos Parâmetros do PSO ---
    limites_funcao = (-500, 500)
    num_particulas = 15
    num_iteracoes = 100 # Número MÁXIMO de iterações por execução do PSO
    w_max = 0.9
    w_min = 0.4
    c1 = 2
    c2 = 2
    tolerancia = 1e-6
    iteracoes_sem_melhora_limite = 20

    # Dicionário para passar os parâmetros para o display no gráfico de barras e TXT
    pso_params_for_display = {
        "limites_funcao": limites_funcao,
        "num_particulas": num_particulas,
        "num_iteracoes": num_iteracoes,
        "w_max": w_max,
        "w_min": w_min,
        "c1": c1,
        "c2": c2,
        "tolerancia": tolerancia,
        "iteracoes_sem_melhora_limite": iteracoes_sem_melhora_limite
    }

    # Listas para armazenar os históricos de cada execução (para gráfico de convergência)
    all_historico_melhor_global = []
    all_historico_media_melhores_locais = []
    all_historico_desvio_padrao_melhores_locais = []

    # Lista para armazenar as estatísticas finais de cada execução (para gráficos de barra e CSV)
    all_final_stats = []

    print(f"\n--- Iniciando {num_repeticoes_simulacao} execuções do PSO ---")

    for i in range(num_repeticoes_simulacao):
        print(f"\nExecução {i+1}/{num_repeticoes_simulacao}:")
        otimizador_pso = PSO(funcao_w4, limites_funcao, num_particulas, num_iteracoes,
                             w_max, w_min, c1, c2, tolerancia, iteracoes_sem_melhora_limite)

        pso_results = otimizador_pso.executar()

        # Armazena os históricos desta execução
        all_historico_melhor_global.append(pso_results["historico_melhor_global"])
        all_historico_media_melhores_locais.append(pso_results["historico_media_melhores_locais"])
        all_historico_desvio_padrao_melhores_locais.append(pso_results["historico_desvio_padrao_melhores_locais"])

        # Armazena as estatísticas finais desta execução
        all_final_stats.append({
            'melhor_valor_global': pso_results['melhor_valor_global'],
            'iteracoes_executadas': pso_results['iteracoes_executadas'],
            'avaliacoes_funcao_total': pso_results['avaliacoes_funcao_total'],
            'multiplicacoes_total': pso_results['multiplicacoes_total'],
            'divisoes_total': pso_results['divisoes_total'],
            'avaliacoes_minimo_global': pso_results['avaliacoes_minimo_global'],
            'multiplicacoes_minimo_global': pso_results['multiplicacoes_minimo_global'],
            'divisoes_minimo_global': pso_results['divisoes_minimo_global'],
        })

    print(f"\n--- {num_repeticoes_simulacao} execuções concluídas. Gerando gráficos e sumário ---")

    # Converter a lista de dicionários para um DataFrame do pandas para facilitar cálculos estatísticos
    results_df = pd.DataFrame(all_final_stats)

    # 1. Gerar o Gráfico de Convergência (gráfico de linha)
    plot_convergence_graph(
        all_historico_melhor_global,
        all_historico_media_melhores_locais,
        all_historico_desvio_padrao_melhores_locais,
        algorithm_name="PSO",
        num_runs=num_repeticoes_simulacao
    )

    # 2. Gerar os Gráficos de Barra
    plot_bar_graphs(results_df, algorithm_name="PSO", num_runs=num_repeticoes_simulacao,
                    pso_params_display=pso_params_for_display)

    # 3. NOVO: Salvar o sumário em CSV
    save_summary_to_csv(results_df, algorithm_name="PSO", num_runs=num_repeticoes_simulacao,
                        pso_params_display=pso_params_for_display)

    print("\nAnálise completa. Verifique a pasta 'resultados_analise' para os gráficos e o sumário.")