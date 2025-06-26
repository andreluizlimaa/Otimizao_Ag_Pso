import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Importa sua função w4 do novo arquivo
from funcoes_otimizacao import funcao_w4
# Importa a função de gráfico AG do arquivo 'grafico.py'
from Grafico import GraficoAG # Com 'G' maiúsculo para o arquivo

# Importa o contador e o wrapper (do seu arquivo utils.py)
from utils import global_op_counter, FuncaoObjetivoWrapper

# --- Funções Auxiliares do Algoritmo Genético ---

# INICIALIZA A POPULAÇÃO COM INDIVÍDUOS ALEATÓRIOS
def inicializar_populacao(tamanho_populacao, limites):
    populacao = []
    for _ in range(tamanho_populacao):
        individuo = np.array([
            random.uniform(limites[0], limites[1]), # GENE X
            random.uniform(limites[0], limites[1]), # GENE Y
        ])
        populacao.append(individuo)
    return populacao

# AVALIA A APTIDÃO DE CADA INDIVÍDUO NA POPULAÇÃO
def avaliar_populacao(populacao, funcao_wrapper): # Recebe o wrapper
    aptidoes = []
    for individuo in populacao:
        x, y = individuo
        aptidao = funcao_wrapper(x, y) # Usa o wrapper
        aptidoes.append(aptidao)
    return aptidoes

# SELECIONA OS PAIS PARA A REPRODUÇÃO COM BASE NA APTIDÃO (MENORES VALORES)
def selecionar_pais(populacao, aptidoes, num_pais):
    individuos_com_aptidoes = list(zip(populacao, aptidoes))
    individuos_ordenados = sorted(individuos_com_aptidoes, key=lambda x: x[1])
    pais_selecionados = [individuo[0] for individuo in individuos_ordenados[:num_pais]]
    return pais_selecionados

# REALIZA CRUZAMENTO (CROSSOVER) ENTRE OS PAIS PARA CRIAR NOVA GERAÇÃO COM BLENDING (BLX-Alfa)
def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)): # Adicionei 'limites'
    nova_geracao = []
    if len(pais) < 2:
        return []

    global_op_counter.add_div(1) # Para len(pais) // 2
    num_pares = len(pais) // 2

    for _ in range(num_pares):
        pai1, pai2 = random.sample(pais, 2)
        if random.random() < taxa_cruzamento:
            filho1 = np.zeros_like(pai1) # Cria arrays vazios com o mesmo formato
            filho2 = np.zeros_like(pai2)

            # Para cada gene, aplica a lógica do BLX-Alfa
            for i in range(len(pai1)):
                gene1_pai = pai1[i]
                gene2_pai = pai2[i]

                # Calcula o intervalo I
                I = abs(gene1_pai - gene2_pai)
                # Calcula a extensão d
                d = alpha * I

                # Define o novo intervalo para a geração dos filhos
                lower_bound = min(gene1_pai, gene2_pai) - d
                upper_bound = max(gene1_pai, gene2_pai) + d

                # Garante que os limites não ultrapassem os limites gerais da função
                lower_bound = max(lower_bound, limites[0])
                upper_bound = min(upper_bound, limites[1])

                # Gera os genes dos filhos aleatoriamente dentro do novo intervalo
                filho1[i] = random.uniform(lower_bound, upper_bound)
                filho2[i] = random.uniform(lower_bound, upper_bound)

            nova_geracao.extend([filho1, filho2])
        else:
            nova_geracao.extend([pai1.copy(), pai2.copy()])
    return nova_geracao

# REALIZA A MUTAÇÃO EM ALGUNS INDIVÍDUOS DA NOVA GERAÇÃO
def mutacao(populacao, taxa_mutacao, limites):
    for individuo in populacao:
        if random.random() < taxa_mutacao:
            indice_mutacao = random.randint(0, len(individuo) - 1)
            individuo[indice_mutacao] = random.uniform(limites[0], limites[1])
    return populacao


# ALGORITMO GENÉTICO PRINCIPAL
def algoritmo_genetico(tamanho_populacao, limites, num_geracoes, taxa_cruzamento, taxa_mutacao, geracoes_sem_melhora_limite=50):
    # Resetar contadores antes de iniciar a otimização
    global_op_counter.reset()
    funcao_w4_wrapper_ag = FuncaoObjetivoWrapper(funcao_w4, global_op_counter) # Instancia o wrapper para AG

    populacao = inicializar_populacao(tamanho_populacao, limites)
    melhor_solucao = None
    melhor_aptidao = float('inf')
    melhor_geracao = -1

    # Variáveis para o item b) e c) - AG
    avaliacoes_ag_melhor_solucao = 0
    operacoes_ag_melhor_solucao_mult = 0
    operacoes_ag_melhor_solucao_div = 0

    # Variáveis para a decisão de parada por convergência
    geracoes_sem_melhora = 0
    ultima_melhor_aptidao = float('inf')

    # INICIALIZAÇÃO DO GRÁFICO PARA AG
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_geracoes):
        aptidoes = avaliar_populacao(populacao, funcao_w4_wrapper_ag) # Usa o wrapper

        if not aptidoes or np.isinf(min(aptidoes)):
            continue

        # Encontra a melhor solução da geração atual
        melhor_aptidao_geracao = min(aptidoes)
        idx_melhor_geracao = aptidoes.index(melhor_aptidao_geracao)
        melhor_solucao_geracao_atual = populacao[idx_melhor_geracao]

        # Lógica da decisão de parada por convergência
        if melhor_aptidao_geracao < ultima_melhor_aptidao - 1e-6: # -1e-6 para considerar pequenas melhorias
            ultima_melhor_aptidao = melhor_aptidao_geracao
            geracoes_sem_melhora = 0 # Reseta o contador
        else:
            geracoes_sem_melhora += 1

        # Verifica se o limite de gerações sem melhora foi atingido
        if geracoes_sem_melhora >= geracoes_sem_melhora_limite:
            print(f"\n[AG] Parada por convergência: Nenhuma melhora em {geracoes_sem_melhora_limite} gerações.")
            break # Sai do loop principal do AG

        # Atualiza a melhor solução global
        if melhor_aptidao_geracao < melhor_aptidao:
            melhor_aptidao = melhor_aptidao_geracao
            melhor_solucao = melhor_solucao_geracao_atual.copy() # Copia para evitar referência
            melhor_geracao = i
            # Registra as contagens no momento em que o "melhor global" é atualizado para o AG
            avaliacoes_ag_melhor_solucao = funcao_w4_wrapper_ag.evaluations
            operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications
            operacoes_ag_melhor_solucao_div = global_op_counter.divisions


        # *** AQUI ESTÁ A MUDANÇA: USANDO A NOVA FUNÇÃO DE CRUZAMENTO ***
        pais = selecionar_pais(populacao, aptidoes, tamanho_populacao // 2)
        filhos = cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=limites) # Passe os limites aqui também
        # ***************************************************************

        filhos = mutacao(filhos, taxa_mutacao, limites)

        nova_populacao = pais + filhos
        # Garante que a nova população tenha o tamanho correto
        while len(nova_populacao) < tamanho_populacao:
            nova_populacao.append(np.array([
                random.uniform(limites[0], limites[1]),
                random.uniform(limites[0], limites[1])
            ]))
        populacao = nova_populacao[:tamanho_populacao]

        # CHAMADA PARA A FUNÇÃO DE GRÁFICO DO AG
        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao) # Passando o melhor_aptidao

    plt.show() # Mantém o gráfico final aberto para o AG

    # Imprimir resultados detalhados do AG
    print(f"Melhor solução encontrada (AG): {melhor_solucao}")
    print(f"Valor da função para a melhor solução (AG): {melhor_aptidao}")
    print(f"Geração de convergência (AG): {i}") # A iteração 'i' onde parou
    print(f"Avaliações da função objetivo (AG): {funcao_w4_wrapper_ag.evaluations}")
    print(f"Operações de Multiplicação (AG): {global_op_counter.multiplications}")
    print(f"Operações de Divisão (AG): {global_op_counter.divisions}")
    print(f"Avaliações para o 'melhor global' (AG): {avaliacoes_ag_melhor_solucao}")
    print(f"Multiplicações para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_mult}")
    print(f"Divisões para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_div}")

    return melhor_solucao, melhor_aptidao, i # Retorna a geração de parada

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    tamanho_populacao = 35
    limites = (-500, 500)
    num_geracoes = 1000
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 50

    melhor_solucao, melhor_aptidao, geracao = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite
    )
    # Os prints já estão dentro da função algoritmo_genetico agora.