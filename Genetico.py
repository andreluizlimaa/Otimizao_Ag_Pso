import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os # Importado para salvar arquivos
from datetime import datetime # Importado para gerar timestamp

from funcoes_otimizacao import funcao_w4
from Grafico import GraficoAG
from utils import global_op_counter, FuncaoObjetivoWrapper

# --- Funções Auxiliares do Algoritmo Genético ---

def inicializar_populacao(tamanho_populacao, limites):
    """
    Inicializa a população inicial do algoritmo genético.
    Cada indivíduo é um par (x, y) gerado aleatoriamente dentro dos limites.
    """
    populacao = []
    for _ in range(tamanho_populacao):
        individuo = np.array([
            random.uniform(limites[0], limites[1]),
            random.uniform(limites[0], limites[1]),
        ])
        populacao.append(individuo)
    return populacao

def avaliar_populacao(populacao, funcao_wrapper):
    """
    Avalia a aptidão (valor da função objetivo) de cada indivíduo na população.
    """
    aptidoes = []
    for individuo in populacao:
        x, y = individuo
        aptidao = funcao_wrapper(x, y)
        aptidoes.append(aptidao)
    return aptidoes

def selecao_roleta(populacao, aptidoes):
    """
    Seleciona um indivíduo da população usando o método de seleção por roleta.
    Retorna o indivíduo selecionado.
    Como é minimização, transformamos a aptidão para que valores menores tenham maior "fatia" na roleta.
    """
    if not aptidoes:
        return None

    max_aptidao = max(aptidoes)
    fit_scores = [max_aptidao - apt + 1e-6 for apt in aptidoes]
    
    total_fit = sum(fit_scores)
    
    if total_fit == 0:
        return random.choice(populacao)

    probabilidades = [score / total_fit for score in fit_scores]

    acumulado = 0
    roleta = []
    for i, prob in enumerate(probabilidades):
        acumulado += prob
        roleta.append(acumulado)

    r = random.random()
    for i, lim_superior in enumerate(roleta):
        if r <= lim_superior:
            return populacao[i]
    
    return populacao[-1]


def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)):
    """
    Realiza o cruzamento BLX-alpha (Blend Crossover Alpha) entre os DOIS pais fornecidos.
    Gera DOIS novos filhos com base nos genes dos pais, com uma chance de ocorrer o cruzamento.
    """
    pai1, pai2 = pais[0], pais[1]

    if random.random() < taxa_cruzamento:
        filho1 = np.zeros_like(pai1)
        filho2 = np.zeros_like(pai2)

        for i in range(len(pai1)):
            gene1_pai = pai1[i]
            gene2_pai = pai2[i]

            I = abs(gene1_pai - gene2_pai)
            global_op_counter.add_mult(1)
            d = alpha * I

            lower_bound = min(gene1_pai, gene2_pai) - d
            upper_bound = max(gene1_pai, gene2_pai) + d

            lower_bound = max(lower_bound, limites[0])
            upper_bound = min(upper_bound, limites[1])

            filho1[i] = random.uniform(lower_bound, upper_bound)
            filho2[i] = random.uniform(lower_bound, upper_bound)
            
        return [filho1, filho2]
    else:
        return [pai1.copy(), pai2.copy()]

def mutacao(populacao, taxa_mutacao, limites):
    """
    Aplicação de mutação a indivíduos da população.
    """
    for individuo in populacao:
        if random.random() < taxa_mutacao:
            indice_mutacao = random.randint(0, len(individuo) - 1)
            individuo[indice_mutacao] = random.uniform(limites[0], limites[1])
    return populacao


# --- ALGORITMO GENÉTICO PRINCIPAL ---
def algoritmo_genetico(tamanho_populacao, limites, num_geracoes, taxa_cruzamento, taxa_mutacao, geracoes_sem_melhora_limite=50, tolerancia=1e-6):
    """
    Implementa o algoritmo genético principal para otimização.
    """
    global_op_counter.reset()
    funcao_w4_wrapper_ag = FuncaoObjetivoWrapper(funcao_w4, global_op_counter)

    populacao = inicializar_populacao(tamanho_populacao, limites)
    melhor_solucao = None
    melhor_aptidao = float('inf')
    melhor_geracao = -1

    avaliacoes_ag_melhor_solucao = 0
    operacoes_ag_melhor_solucao_mult = 0
    operacoes_ag_melhor_solucao_div = 0

    geracoes_sem_melhora = 0
    ultima_melhor_aptidao_global = float('inf')

    # --- INICIALIZAÇÃO DO GRÁFICO PARA AG ---
    fig = plt.figure(figsize=(11, 8)) # Aumentado para acomodar a legenda
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(right=0.7) # Ajusta para deixar espaço para a legenda

    # Dicionário para armazenar os parâmetros e estatísticas do AG para a legenda
    ag_params_for_plot = {
        "tamanho_populacao": tamanho_populacao,
        "taxa_mutacao": taxa_mutacao,
        "taxa_crossover": taxa_cruzamento,
        "selecao_tipo": "Roleta", # Definido como Roleta
        "elitismo": False, # Definido explicitamente como False
        "iteracoes_totais": num_geracoes,
        "avaliacoes_funcao": 0, # Será atualizado em cada iteração
        "multiplicacoes_total": 0, # Será atualizado em cada iteração
        "divisoes_total": 0, # Será atualizado em cada iteração
        "avaliacoes_minimo_global": 0, # Será atualizado quando o melhor global for encontrado
        "multiplicacoes_minimo_global": 0, # Será atualizado quando o melhor global for encontrado
        "divisoes_minimo_global": 0 # Será atualizado quando o melhor global for encontrado
    }

    for i in range(num_geracoes):
        aptidoes = avaliar_populacao(populacao, funcao_w4_wrapper_ag)

        if not aptidoes or np.isinf(min(aptidoes)):
            continue

        melhor_aptidao_geracao = min(aptidoes)
        idx_melhor_geracao = aptidoes.index(melhor_aptidao_geracao)
        melhor_solucao_geracao_atual = populacao[idx_melhor_geracao]

        if melhor_aptidao_geracao < melhor_aptidao:
            melhor_aptidao = melhor_aptidao_geracao
            melhor_solucao = melhor_solucao_geracao_atual.copy()
            melhor_geracao = i
            avaliacoes_ag_melhor_solucao = funcao_w4_wrapper_ag.evaluations
            operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications
            operacoes_ag_melhor_solucao_div = global_op_counter.divisions
            
            # Atualiza as estatísticas no dicionário para a legenda
            ag_params_for_plot["avaliacoes_minimo_global"] = avaliacoes_ag_melhor_solucao
            ag_params_for_plot["multiplicacoes_minimo_global"] = operacoes_ag_melhor_solucao_mult
            ag_params_for_plot["divisoes_minimo_global"] = operacoes_ag_melhor_solucao_div
        
        # Lógica de parada por convergência.
        if i > 0 and abs(melhor_aptidao - ultima_melhor_aptidao_global) < tolerancia:
            geracoes_sem_melhora += 1
        else:
            geracoes_sem_melhora = 0
            
        ultima_melhor_aptidao_global = melhor_aptidao

        if geracoes_sem_melhora >= geracoes_sem_melhora_limite:
            print(f"\n[AG] Parada por convergência: Mudança no melhor valor da aptidão menor que {tolerancia} por {geracoes_sem_melhora_limite} gerações.")
            break
            
        # --- Construção da próxima geração sem elitismo ---
        proxima_geracao = []
        
        # O loop deve rodar `tamanho_populacao // 2` vezes para gerar pares de filhos
        # Se for ímpar, um indivíduo extra será selecionado no final
        for _ in range(tamanho_populacao // 2): 
            pai1 = selecao_roleta(populacao, aptidoes)
            pai2 = selecao_roleta(populacao, aptidoes)

            if pai1 is None or pai2 is None:
                # Se a seleção falhar (população muito pequena ou problemas), pode pular ou tentar outra estratégia
                continue 

            filhos_gerados = cruzamento_blx_alpha([pai1, pai2], taxa_cruzamento, alpha=0.5, limites=limites)
            proxima_geracao.extend(filhos_gerados)

        # Se o tamanho da população for ímpar, seleciona um último indivíduo
        if len(proxima_geracao) < tamanho_populacao:
            extra_pai = selecao_roleta(populacao, aptidoes)
            if extra_pai is not None:
                proxima_geracao.append(extra_pai.copy())

        filhos_apos_mutacao = mutacao(proxima_geracao, taxa_mutacao, limites)

        populacao = filhos_apos_mutacao[:tamanho_populacao] 

        # Atualiza as estatísticas globais para o dicionário da legenda a cada iteração
        ag_params_for_plot["avaliacoes_funcao"] = funcao_w4_wrapper_ag.evaluations
        ag_params_for_plot["multiplicacoes_total"] = global_op_counter.multiplications
        ag_params_for_plot["divisoes_total"] = global_op_counter.divisions

        # Passa o dicionário de parâmetros para a função GraficoAG
        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)

    plt.show()

    # --- Prepara as estatísticas para impressão e salvamento ---
    stats_output = []
    stats_output.append("--- Resultados Finais do Algoritmo Genético ---")
    stats_output.append(f"Melhor solução encontrada (AG): {melhor_solucao}")
    stats_output.append(f"Valor da função para a melhor solução (AG): {melhor_aptidao:.4f}")
    stats_output.append(f"Gerações executadas (AG): {i}")
    stats_output.append(f"Avaliações da função objetivo (AG): {funcao_w4_wrapper_ag.evaluations}")
    stats_output.append(f"Operações de Multiplicação (AG): {global_op_counter.multiplications}")
    stats_output.append(f"Operações de Divisão (AG): {global_op_counter.divisions}")
    stats_output.append(f"Avaliações para o 'melhor global' (AG): {avaliacoes_ag_melhor_solucao}")
    stats_output.append(f"Multiplicações para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_mult}")
    stats_output.append(f"Divisões para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_div}")
    stats_output.append("--------------------------------------------")

    for line in stats_output:
        print(line)

    # Salva as estatísticas em um arquivo .txt
    output_folder = "resultados_ag"
    os.makedirs(output_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"estatisticas_ag_{timestamp}.txt"
    output_path = os.path.join(output_folder, file_name)

    with open(output_path, "w") as f:
        for line in stats_output:
            f.write(line + "\n")

    print(f"\nEstatísticas salvas em: {output_path}")

    return melhor_solucao, melhor_aptidao, i

if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    tamanho_populacao = 35
    limites = (-500, 500)
    num_geracoes = 1000
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 50
    tolerancia = 1e-6

    melhor_solucao, melhor_aptidao, geracao = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite,
        tolerancia=tolerancia
    )