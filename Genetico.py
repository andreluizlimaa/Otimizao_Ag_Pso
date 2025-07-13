# Genetico.py

# Importa as bibliotecas necessárias para o Algoritmo Genético
import numpy as np # NumPy para operações numéricas e manipulação de arrays
import random # Para geração de números aleatórios
import matplotlib.pyplot as plt # Matplotlib para criação de gráficos
from mpl_toolkits.mplot3d import Axes3D 
import os # Para interagir com o sistema operacional (criação de pastas, manipulação de arquivos)
from datetime import datetime # Para trabalhar com datas e horas (gerar timestamps para nomes de arquivos)

# Importa as funções personalizadas do projeto
from funcoes_otimizacao import funcao_w4 # A função objetivo W4 a ser minimizada
from Grafico import GraficoAG # Importa a função para plotar o gráfico do AG (visualização 3D)
from utils import global_op_counter, FuncaoObjetivoWrapper # Contadores globais e o wrapper para a função objetivo

# --- Funções Auxiliares do Algoritmo Genético ---

def inicializar_populacao(tamanho_populacao, limites):
    """
    Inicializa a população inicial do algoritmo genético.
    Cada indivíduo é um par (x, y) gerado aleatoriamente dentro dos limites.
    
    Parâmetros:
        tamanho_populacao (int): O número de indivíduos na população.
        limites (tuple): Uma tupla (min, max) definindo os limites do espaço de busca.
        
    Retorna:
        list: Uma lista de arrays NumPy, onde cada array representa um indivíduo (x, y).
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
    
    Parâmetros:
        populacao (list): A lista de indivíduos da população.
        funcao_wrapper (FuncaoObjetivoWrapper): Um wrapper da função objetivo que conta as avaliações.
        
    Retorna:
        list: Uma lista de aptidões, onde cada aptidão corresponde a um indivíduo.
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
    
    Parâmetros:
        populacao (list): A lista de indivíduos da população.
        aptidoes (list): A lista de aptidões correspondentes aos indivíduos.
        
    Retorna:
        np.array or None: O indivíduo selecionado ou None se a lista de aptidões estiver vazia.
    """
    if not aptidoes:
        return None

    max_aptidao = max(aptidoes)
    # Transforma as aptidões para que valores menores resultem em scores maiores (para minimização).
    fit_scores = [max_aptidao - apt + 1e-6 for apt in aptidoes] 
    
    total_fit = sum(fit_scores)
    
    if total_fit == 0:
        return random.choice(populacao)

    probabilidades = [score / total_fit for score in fit_scores]

    r = random.random()
    acumulado = 0
    for i, prob in enumerate(probabilidades):
        acumulado += prob
        if r <= acumulado:
            return populacao[i]
    
    return populacao[-1]


def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)):
    """
    Realiza o cruzamento BLX-alpha (Blend Crossover Alpha) entre os DOIS pais fornecidos.
    Gera DOIS novos filhos com base nos genes dos pais, com uma chance de ocorrer o cruzamento.
    
    Parâmetros:
        pais (list): Uma lista contendo dois indivíduos (pais) para cruzamento.
        taxa_cruzamento (float): A probabilidade de ocorrer o cruzamento.
        alpha (float): Parâmetro alfa do BLX-alpha, define a faixa de mistura.
        limites (tuple): Uma tupla (min, max) definindo os limites do espaço de busca.
        
    Retorna:
        list: Uma lista contendo os dois filhos gerados pelo cruzamento, ou cópias dos pais se não houver cruzamento.
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
    
    Parâmetros:
        populacao (list): A lista de indivíduos da população a ser mutada.
        taxa_mutacao (float): A probabilidade de um indivíduo sofrer mutação em qualquer um de seus genes.
        limites (tuple): Uma tupla (min, max) definindo os limites do espaço de busca para a mutação.
        
    Retorna:
        list: A população após a aplicação das mutações.
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
    
    Parâmetros:
        tamanho_populacao (int): O número de indivíduos na população.
        limites (tuple): Uma tupla (min, max) definindo os limites do espaço de busca.
        num_geracoes (int): O número máximo de gerações.
        taxa_cruzamento (float): A probabilidade de ocorrência de cruzamento.
        taxa_mutacao (float): A probabilidade de ocorrência de mutação.
        geracoes_sem_melhora_limite (int): Limite de gerações consecutivas sem melhora para o critério de parada.
        tolerancia (float): Tolerância para o critério de parada por convergência.
        
    Retorna:
        dict: Um dicionário contendo os resultados e estatísticas do algoritmo,
              incluindo os históricos de convergência.
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

    # --- LISTAS PARA ARMAZENAR O HISTÓRICO DE CONVERGÊNCIA ---
    historico_melhor_geracao = []
    historico_melhor_global = []

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(right=0.7)

    ag_params_for_plot = {
        "tamanho_populacao": tamanho_populacao,
        "taxa_mutacao": taxa_mutacao,
        "taxa_crossover": taxa_cruzamento,
        "selecao_tipo": "Roleta",
        "iteracoes_totais": num_geracoes,
        "avaliacoes_funcao": 0,
        "multiplicacoes_total": 0,
        "divisoes_total": 0,
        "avaliacoes_minimo_global": 0,
        "multiplicacoes_minimo_global": 0,
        "divisoes_minimo_global": 0,
        "geracoes_sem_melhora": geracoes_sem_melhora,
        "limite_geracoes_sem_melhora": geracoes_sem_melhora_limite
    }

    for i in range(num_geracoes): 
        aptidoes = avaliar_populacao(populacao, funcao_w4_wrapper_ag) 

        if not aptidoes or np.isinf(min(aptidoes)):
            if historico_melhor_global:
                historico_melhor_geracao.append(historico_melhor_global[-1])
                historico_melhor_global.append(historico_melhor_global[-1])
            else:
                historico_melhor_geracao.append(float('inf'))
                historico_melhor_global.append(float('inf'))
            
            GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)
            continue

        melhor_aptidao_geracao = min(aptidoes)
        historico_melhor_geracao.append(melhor_aptidao_geracao)

        if melhor_aptidao_geracao < melhor_aptidao:
            if abs(melhor_aptidao - melhor_aptidao_geracao) > tolerancia:
                melhor_aptidao = melhor_aptidao_geracao
                melhor_solucao = populacao[aptidoes.index(melhor_aptidao_geracao)].copy()
                melhor_geracao = i
                
                avaliacoes_ag_melhor_solucao = funcao_w4_wrapper_ag.evaluations
                operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications
                operacoes_ag_melhor_solucao_div = global_op_counter.divisions
                
                geracoes_sem_melhora = 0
            else:
                geracoes_sem_melhora += 1
        else:
            geracoes_sem_melhora += 1
            
        historico_melhor_global.append(melhor_aptidao)

        if geracoes_sem_melhora >= geracoes_sem_melhora_limite: 
            print(f"\n[AG] Parada por convergência: Mudança no melhor valor da aptidão menor que {tolerancia} por {geracoes_sem_melhora_limite} gerações. (Geração: {i + 1})")
            # Garante que o último frame do gráfico seja mostrado antes de fechar
            GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)
            plt.show(block=True) # Mostra o gráfico final e espera o usuário fechar
            break
            
        proxima_geracao = []
        
        for _ in range(tamanho_populacao // 2): 
            pai1 = selecao_roleta(populacao, aptidoes)
            pai2 = selecao_roleta(populacao, aptidoes)

            if pai1 is None or pai2 is None:
                continue

            filhos_gerados = cruzamento_blx_alpha([pai1, pai2], taxa_cruzamento, alpha=0.5, limites=limites)
            proxima_geracao.extend(filhos_gerados)

        if len(proxima_geracao) < tamanho_populacao:
            extra_pai = selecao_roleta(populacao, aptidoes)
            if extra_pai is not None:
                proxima_geracao.append(extra_pai.copy())

        filhos_apos_mutacao = mutacao(proxima_geracao, taxa_mutacao, limites) 

        populacao = filhos_apos_mutacao[:tamanho_populacao] 

        ag_params_for_plot["avaliacoes_funcao"] = funcao_w4_wrapper_ag.evaluations
        ag_params_for_plot["multiplicacoes_total"] = global_op_counter.multiplications
        ag_params_for_plot["divisoes_total"] = global_op_counter.divisions
        ag_params_for_plot["geracoes_sem_melhora"] = geracoes_sem_melhora
        
        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)

    # Força a exibição do gráfico final se o loop terminar por num_geracoes
    if i + 1 == num_geracoes:
        print(f"\n[AG] Loop de gerações concluído. Total de gerações: {i + 1}")
        plt.show(block=True) # Mostra o gráfico final e espera o usuário fechar
    
    # --- Prepara os resultados para retorno ---
    return {
        "melhor_solucao": melhor_solucao,
        "melhor_valor_global": melhor_aptidao,
        "iteracoes_executadas": i + 1,
        "avaliacoes_funcao_total": funcao_w4_wrapper_ag.evaluations,
        "multiplicacoes_total": global_op_counter.multiplications,
        "divisoes_total": global_op_counter.divisions,
        "avaliacoes_minimo_global": avaliacoes_ag_melhor_solucao,
        "multiplicacoes_minimo_global": operacoes_ag_melhor_solucao_mult,
        "divisoes_minimo_global": operacoes_ag_melhor_solucao_div,
        "historico_melhor_geracao": historico_melhor_geracao,
        "historico_melhor_global": historico_melhor_global,
    }

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    # Define os parâmetros de teste para uma execução direta.
    limites = (-500, 500)
    tamanho_populacao = 35
    num_geracoes = 200
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 20
    tolerancia = 1e-6

    # Chama a função do algoritmo genético com os parâmetros de teste.
    results = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite,
        tolerancia=tolerancia
    )
    # Impressão dos resultados do teste direto (usando o dicionário retornado)
    print("\n--- Resultados Finais do Teste Direto do AG ---")
    print(f"Melhor solução encontrada: {results['melhor_solucao']}")
    print(f"Melhor valor global: {results['melhor_valor_global']:.4f}")
    print(f"Gerações executadas: {results['iteracoes_executadas']}")
    print(f"Avaliações da função (total): {results['avaliacoes_funcao_total']}")
    print(f"Multiplicações (total): {results['multiplicacoes_total']}")
    print(f"Divisões (total): {results['divisoes_total']}")
    print(f"Avaliações (no melhor global): {results['avaliacoes_minimo_global']}")
    print(f"Multiplicações (no melhor global): {results['multiplicacoes_minimo_global']}")
    print(f"Divisões (no melhor global): {results['divisoes_minimo_global']}")
    print(f"Tamanho do histórico_melhor_geracao: {len(results['historico_melhor_geracao'])}")
    print(f"Tamanho do historico_melhor_global: {len(results['historico_melhor_global'])}")
