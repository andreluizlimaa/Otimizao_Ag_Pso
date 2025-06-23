import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

# Importa sua função w4 do novo arquivo
from funcoes_otimizacao import funcao_w4
# Importa a função de gráfico AG do arquivo 'grafico.py' (com 'g' minúsculo)
from Grafico import GraficoAG

# Variável global para controlar o estado de pausa do AG
pausado_ag = False

# Função chamada pelo botão Pause para o AG
def alternar_pausa_ag(event):
    global pausado_ag
    pausado_ag = not pausado_ag

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
def avaliar_populacao(populacao):
    aptidoes = []
    for individuo in populacao:
        x, y = individuo
        aptidao = funcao_w4(x, y) # CHAMA A FUNÇÃO W4 IMPORTADA
        aptidoes.append(aptidao)
    return aptidoes

# SELECIONA OS PAIS PARA A REPRODUÇÃO COM BASE NA APTIDÃO (MENORES VALORES)
def selecionar_pais(populacao, aptidoes, num_pais):
   individuos_com_aptidoes = list(zip(populacao, aptidoes))
   individuos_ordenados = sorted(individuos_com_aptidoes, key=lambda x: x[1])
   pais_selecionados = [individuo[0] for individuo in individuos_ordenados[:num_pais]]
   return pais_selecionados

# REALIZA CRUZAMENTO (CROSSOVER) ENTRE OS PAIS PARA CRIAR NOVA GERAÇÃO
def cruzamento(pais, taxa_cruzamento):
    nova_geracao = []
    if len(pais) < 2:
        return []

    num_pares = len(pais) // 2
    for _ in range(num_pares):
        pai1, pai2 = random.sample(pais, 2)
        if random.random() < taxa_cruzamento:
            ponto_corte = random.randint(0, len(pai1) - 1)
            filho1 = np.concatenate((pai1[:ponto_corte], pai2[ponto_corte:]))
            filho2 = np.concatenate((pai2[:ponto_corte], pai1[ponto_corte:]))
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
    populacao = inicializar_populacao(tamanho_populacao, limites)
    melhor_solucao = None
    melhor_aptidao = float('inf')
    melhor_geracao = -1

    # Variáveis para a decisão de parada por convergência
    geracoes_sem_melhora = 0
    ultima_melhor_aptidao = float('inf')


    # INICIALIZAÇÃO DO GRÁFICO PARA AG
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # ADICIONA O BOTÃO DE PAUSE PARA AG
    pause_posicao_ag = plt.axes([0.85, 0.03, 0.1, 0.05])
    pause_botao_ag = Button(pause_posicao_ag, 'Pause AG')
    pause_botao_ag.hovercolor = 'lightgreen'
    pause_botao_ag.on_clicked(alternar_pausa_ag)


    for i in range(num_geracoes):
        aptidoes = avaliar_populacao(populacao)

        if not aptidoes or np.isinf(min(aptidoes)):
            continue

        # Encontra a melhor solução da geração atual
        melhor_aptidao_geracao = min(aptidoes)
        idx_melhor_geracao = aptidoes.index(melhor_aptidao_geracao)
        melhor_solucao_geracao_atual = populacao[idx_melhor_geracao]

        # Lógica da decisão de parada por convergência
        # Usamos uma pequena tolerância para flutuações de ponto flutuante
        if melhor_aptidao_geracao < ultima_melhor_aptidao - 1e-6: # -1e-6 para considerar pequenas melhorias
            # Houve melhora significativa
            ultima_melhor_aptidao = melhor_aptidao_geracao
            geracoes_sem_melhora = 0 # Reseta o contador
        else:
            # Não houve melhora significativa
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


        pais = selecionar_pais(populacao, aptidoes, tamanho_populacao // 2)
        filhos = cruzamento(pais, taxa_cruzamento)
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
        GraficoAG(populacao, melhor_solucao, i + 1, ax)

        # Espera o botão de pausa ser desativado
        while pausado_ag:
            plt.pause(0.1)

    plt.show() # Mantém o gráfico final aberto para o AG

    return melhor_solucao, melhor_aptidao, melhor_geracao

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
# Ele NÃO será executado quando genetico.py for importado como um módulo por outro arquivo (ex: main.py).
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    # PARÂMETROS DO ALGORITMO (para teste direto)
    tamanho_populacao = 35
    limites = (-500, 500) # Formato para o AG
    num_geracoes = 1000
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 50

    # EXECUTA O ALGORITMO GENÉTICO
    melhor_solucao, melhor_aptidao, geracao = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite
    )

    print("Melhor solução encontrada:", melhor_solucao)
    print("Valor da função para a melhor solução:", melhor_aptidao)
    print("Geração:", geracao)