import numpy as np # Importa a biblioteca NumPy, muito usada para operações numéricas e arrays multidimensionais, com o apelido 'np'.
import random # Importa o módulo 'random' para gerar números aleatórios, essenciais em AG para inicialização, seleção, cruzamento e mutação.
import matplotlib.pyplot as plt # Importa a biblioteca Matplotlib para criar gráficos e visualizações, com o apelido 'plt'.
from mpl_toolkits.mplot3d import Axes3D # Importa Axes3D do Matplotlib para permitir gráficos 3D.

# Importa sua função w4 do novo arquivo
from funcoes_otimizacao import funcao_w4 # Importa a função objetivo 'funcao_w4', que o Algoritmo Genético tentará minimizar.
# Importa a função de gráfico AG do arquivo 'grafico.py'
from Grafico import GraficoAG # Importa a função 'GraficoAG', usada para visualizar o progresso do AG.

# Importa o contador e o wrapper (do seu arquivo utils.py)
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa 'global_op_counter' para rastrear operações e 'FuncaoObjetivoWrapper' para encapsular a função objetivo e contar suas avaliações.

# --- Funções Auxiliares do Algoritmo Genético ---

# INICIALIZA A POPULAÇÃO COM INDIVÍDUOS ALEATÓRIOS
def inicializar_populacao(tamanho_populacao, limites):
    populacao = [] # Cria uma lista vazia para armazenar os indivíduos da população.
    # Loop para criar o número especificado de indivíduos.
    for _ in range(tamanho_populacao):
        # Cria um indivíduo como um array NumPy, com dois "genes" (coordenadas x e y).
        # Cada gene é um número aleatório dentro dos limites fornecidos.
        individuo = np.array([
            random.uniform(limites[0], limites[1]), # GENE X (coordenada x do indivíduo)
            random.uniform(limites[0], limites[1]), # GENE Y (coordenada y do indivíduo)
        ])
        populacao.append(individuo) # Adiciona o indivíduo recém-criado à população.
    return populacao # Retorna a população inicial.

# AVALIA A APTIDÃO DE CADA INDIVÍDUO NA POPULAÇÃO
def avaliar_populacao(populacao, funcao_wrapper): # Esta função recebe a população e o wrapper da função.
    aptidoes = [] # Cria uma lista vazia para armazenar as aptidões (valores da função objetivo) de cada indivíduo.
    # Itera sobre cada indivíduo na população.
    for individuo in populacao:
        x, y = individuo # Desempacota as coordenadas x e y do indivíduo.
        aptidao = funcao_wrapper(x, y) # Avalia a função objetivo (funcao_w4) para o indivíduo usando o wrapper. O wrapper contabiliza as avaliações e operações.
        aptidoes.append(aptidao) # Adiciona a aptidão calculada à lista.
    return aptidoes # Retorna a lista de aptidões.

# SELECIONA OS PAIS PARA A REPRODUÇÃO COM BASE NA APTIDÃO (MENORES VALORES)
def selecionar_pais(populacao, aptidoes, num_pais):
    # Combina cada indivíduo com sua aptidão em uma lista de tuplas.
    individuos_com_aptidoes = list(zip(populacao, aptidoes))
    # Ordena a lista de indivíduos com base em suas aptidões (do menor para o maior, pois estamos minimizando).
    individuos_ordenados = sorted(individuos_com_aptidoes, key=lambda x: x[1])
    # Seleciona os 'num_pais' primeiros indivíduos (os de melhor aptidão) como pais.
    pais_selecionados = [individuo[0] for individuo in individuos_ordenados[:num_pais]]
    return pais_selecionados # Retorna a lista de pais selecionados.

# REALIZA CRUZAMENTO (CROSSOVER) ENTRE OS PAIS PARA CRIAR NOVA GERAÇÃO COM BLENDING (BLX-Alfa)
def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)): # Adicionei 'limites'
    nova_geracao = [] # Lista para armazenar os filhos gerados.
    if len(pais) < 2: # Se não houver pais suficientes para formar pares, retorna uma lista vazia.
        return []

    global_op_counter.add_div(1) # Contabiliza uma operação de divisão (para len(pais) // 2).
    num_pares = len(pais) // 2 # Calcula o número de pares de pais que serão formados.

    for _ in range(num_pares):
        pai1, pai2 = random.sample(pais, 2) # Seleciona dois pais aleatórios (sem repetição) para o cruzamento.
        if random.random() < taxa_cruzamento: # Verifica se o cruzamento deve ocorrer com base na taxa de cruzamento.
            filho1 = np.zeros_like(pai1) # Cria um array NumPy vazio para o primeiro filho, com o mesmo formato do pai.
            filho2 = np.zeros_like(pai2) # Cria um array NumPy vazio para o segundo filho.

            # Para cada gene (dimensão), aplica a lógica do cruzamento BLX-Alfa.
            for i in range(len(pai1)):
                gene1_pai = pai1[i] # Pega o gene 'i' do pai 1.
                gene2_pai = pai2[i] # Pega o gene 'i' do pai 2.

                # Calcula o intervalo I entre os genes dos pais.
                I = abs(gene1_pai - gene2_pai)
                # Calcula a extensão 'd' do intervalo expandido (alpha * I).
                d = alpha * I

                # Define o novo intervalo (limites inferior e superior) para a geração dos filhos.
                # Este intervalo é expandido por 'd' para permitir maior exploração.
                lower_bound = min(gene1_pai, gene2_pai) - d
                upper_bound = max(gene1_pai, gene2_pai) + d

                # Garante que os limites dos filhos não ultrapassem os limites gerais da função objetivo.
                lower_bound = max(lower_bound, limites[0]) # Ajusta o limite inferior para não ser menor que o limite geral.
                upper_bound = min(upper_bound, limites[1]) # Ajusta o limite superior para não ser maior que o limite geral.

                # Gera os genes dos filhos aleatoriamente dentro do novo intervalo calculado.
                filho1[i] = random.uniform(lower_bound, upper_bound)
                filho2[i] = random.uniform(lower_bound, upper_bound)

            nova_geracao.extend([filho1, filho2]) # Adiciona os dois filhos gerados à lista da nova geração.
        else:
            # Se não houver cruzamento, os pais são copiados diretamente para a nova geração.
            nova_geracao.extend([pai1.copy(), pai2.copy()])
    return nova_geracao # Retorna a nova geração de indivíduos resultantes do cruzamento.

# REALIZA A MUTAÇÃO EM ALGUNS INDIVÍDUOS DA NOVA GERAÇÃO
def mutacao(populacao, taxa_mutacao, limites):
    # Itera sobre cada indivíduo na população.
    for individuo in populacao:
        # Decide se o indivíduo sofrerá mutação com base na taxa de mutação.
        if random.random() < taxa_mutacao:
            # Escolhe um índice de gene aleatório para mutar.
            indice_mutacao = random.randint(0, len(individuo) - 1)
            # Altera o gene selecionado para um novo valor aleatório dentro dos limites.
            individuo[indice_mutacao] = random.uniform(limites[0], limites[1])
    return populacao # Retorna a população com os indivíduos possivelmente mutados.


# ALGORITMO GENÉTICO PRINCIPAL
def algoritmo_genetico(tamanho_populacao, limites, num_geracoes, taxa_cruzamento, taxa_mutacao, geracoes_sem_melhora_limite=50):
    # Resetar contadores antes de iniciar a otimização
    global_op_counter.reset() # Zera os contadores de operações globais.
    # Instancia o wrapper para a função objetivo, específico para o AG.
    funcao_w4_wrapper_ag = FuncaoObjetivoWrapper(funcao_w4, global_op_counter)

    # Inicializa a população usando a função auxiliar.
    populacao = inicializar_populacao(tamanho_populacao, limites)
    melhor_solucao = None # Variável para armazenar o melhor indivíduo encontrado em todas as gerações.
    melhor_aptidao = float('inf') # Variável para armazenar a melhor aptidão (menor valor) encontrada globalmente.
    melhor_geracao = -1 # Guarda a geração em que a melhor solução foi encontrada.

    # Variáveis para rastrear contagens no momento em que a melhor solução global do AG é atualizada.
    avaliacoes_ag_melhor_solucao = 0
    operacoes_ag_melhor_solucao_mult = 0
    operacoes_ag_melhor_solucao_div = 0

    # Variáveis para a decisão de parada por convergência (se o algoritmo não melhorar por X gerações).
    geracoes_sem_melhora = 0 # Contador de gerações consecutivas sem melhora significativa.
    ultima_melhor_aptidao = float('inf') # Armazena a melhor aptidão da geração anterior para comparação.

    # INICIALIZAÇÃO DO GRÁFICO PARA AG
    fig = plt.figure(figsize=(8, 8)) # Cria uma nova figura para o gráfico.
    ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura.

    # Loop principal das gerações do Algoritmo Genético.
    for i in range(num_geracoes):
        # Avalia a aptidão de todos os indivíduos na população atual usando o wrapper.
        aptidoes = avaliar_populacao(populacao, funcao_w4_wrapper_ag)

        # Verifica se a lista de aptidões não está vazia ou contém infinitos.
        if not aptidoes or np.isinf(min(aptidoes)):
            continue # Pula para a próxima iteração se houver problemas na avaliação.

        # Encontra a melhor aptidão (menor valor) da geração atual.
        melhor_aptidao_geracao = min(aptidoes)
        # Encontra o índice do indivíduo com a melhor aptidão na geração atual.
        idx_melhor_geracao = aptidoes.index(melhor_aptidao_geracao)
        # Pega a melhor solução (indivíduo) da geração atual.
        melhor_solucao_geracao_atual = populacao[idx_melhor_geracao]

        # Lógica da decisão de parada por convergência
        # Verifica se a melhor aptidão da geração atual é significativamente melhor que a última melhor aptidão.
        # Usa -1e-6 para considerar pequenas melhorias e evitar arredondamento.
        if melhor_aptidao_geracao < ultima_melhor_aptidao - 1e-6:
            ultima_melhor_aptidao = melhor_aptidao_geracao # Atualiza a última melhor aptidão.
            geracoes_sem_melhora = 0 # Reseta o contador de gerações sem melhora.
        else:
            geracoes_sem_melhora += 1 # Incrementa o contador se não houve melhora significativa.

        # Verifica se o limite de gerações sem melhora foi atingido.
        if geracoes_sem_melhora >= geracoes_sem_melhora_limite:
            print(f"\n[AG] Parada por convergência: Nenhuma melhora em {geracoes_sem_melhora_limite} gerações.")
            break # Sai do loop principal do AG se a convergência for detectada.

        # Atualiza a melhor solução global (a melhor encontrada em todas as gerações).
        if melhor_aptidao_geracao < melhor_aptidao:
            melhor_aptidao = melhor_aptidao_geracao # Atualiza o melhor valor de aptidão global.
            melhor_solucao = melhor_solucao_geracao_atual.copy() # Copia a melhor solução para evitar problemas de referência.
            melhor_geracao = i # Registra a geração em que essa melhor solução foi encontrada.
            # Registra as contagens de avaliações e operações no momento em que o "melhor global" para o AG é atualizado.
            avaliacoes_ag_melhor_solucao = funcao_w4_wrapper_ag.evaluations
            operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications
            operacoes_ag_melhor_solucao_div = global_op_counter.divisions

        # Seleciona os pais para a próxima geração.
        pais = selecionar_pais(populacao, aptidoes, tamanho_populacao // 2)
        # Realiza o cruzamento BLX-Alfa para gerar filhos a partir dos pais selecionados.
        filhos = cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=limites)
        # Aplica mutação nos filhos gerados.
        filhos = mutacao(filhos, taxa_mutacao, limites)

        # Constrói a nova população combinando os pais e os filhos.
        nova_populacao = pais + filhos
        # Garante que a nova população tenha o tamanho correto, adicionando novos indivíduos aleatórios se necessário.
        while len(nova_populacao) < tamanho_populacao:
            nova_populacao.append(np.array([
                random.uniform(limites[0], limites[1]),
                random.uniform(limites[0], limites[1])
            ]))
        populacao = nova_populacao[:tamanho_populacao] # Trunca a população para o tamanho desejado.

        # CHAMADA PARA A FUNÇÃO DE GRÁFICO DO AG
        # Atualiza o gráfico com a população atual, a melhor solução encontrada e a aptidão.
        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao)

    plt.show() # Mantém o gráfico final aberto para o AG após todas as gerações ou a parada por convergência.

    # Imprime os resultados detalhados do Algoritmo Genético.
    print(f"Melhor solução encontrada (AG): {melhor_solucao}") # Exibe a posição da melhor solução.
    print(f"Valor da função para a melhor solução (AG): {melhor_aptidao}") # Exibe o valor da função nessa posição.
    print(f"Geração de convergência (AG): {i}") # Mostra a última geração processada (onde parou).
    print(f"Avaliações da função objetivo (AG): {funcao_w4_wrapper_ag.evaluations}") # Total de avaliações da função objetivo.
    print(f"Operações de Multiplicação (AG): {global_op_counter.multiplications}") # Total de operações de multiplicação.
    print(f"Operações de Divisão (AG): {global_op_counter.divisions}") # Total de operações de divisão.
    # Exibe as contagens no momento em que a melhor solução global foi atingida.
    print(f"Avaliações para o 'melhor global' (AG): {avaliacoes_ag_melhor_solucao}")
    print(f"Multiplicações para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_mult}")
    print(f"Divisões para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_div}")

    # Retorna a melhor solução, sua aptidão e a geração de parada.
    return melhor_solucao, melhor_aptidao, i

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    # Define os parâmetros para a execução do Algoritmo Genético.
    tamanho_populacao = 35
    limites = (-500, 500)
    num_geracoes = 1000
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 50 # Limite de gerações sem melhora para acionar a parada por convergência.

    # Chama a função principal do algoritmo genético com os parâmetros definidos.
    melhor_solucao, melhor_aptidao, geracao = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite
    )
    # Os prints já estão dentro da função algoritmo_genetico agora, então não são necessários aqui.