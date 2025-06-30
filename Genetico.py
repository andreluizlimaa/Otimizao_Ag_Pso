import numpy as np # Importa a biblioteca NumPy, essencial para operações numéricas e arrays eficientes.
import random # Importa a biblioteca random para geração de números aleatórios, usada em inicialização, seleção e mutação.
import matplotlib.pyplot as plt # Importa o módulo pyplot da Matplotlib para criar gráficos (geralmente para visualização de dados).
from mpl_toolkits.mplot3d import Axes3D # Importa Axes3D da Matplotlib para permitir gráficos 3D.

from funcoes_otimizacao import funcao_w4 # Importa a função objetivo 'funcao_w4' do arquivo 'funcoes_otimizacao.py'.
from Grafico import GraficoAG # Importa a função 'GraficoAG' do arquivo 'Grafico.py', para plotar o progresso do AG.
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa o contador global de operações e o wrapper da função objetivo do arquivo 'utils.py'.

# --- Funções Auxiliares do Algoritmo Genético ---

def inicializar_populacao(tamanho_populacao, limites):
    """
    Inicializa a população inicial do algoritmo genético.
    Cada indivíduo é um par (x, y) gerado aleatoriamente dentro dos limites.
    """
    populacao = [] # Cria uma lista vazia para armazenar os indivíduos da população.
    for _ in range(tamanho_populacao): # Loop para criar o número especificado de indivíduos.
        individuo = np.array([ # Cria um novo indivíduo como um array NumPy.
            random.uniform(limites[0], limites[1]), # Gera um valor aleatório para a coordenada x dentro dos limites.
            random.uniform(limites[0], limites[1]), # Gera um valor aleatório para a coordenada y dentro dos limites.
        ])
        populacao.append(individuo) # Adiciona o indivíduo criado à população.
    return populacao # Retorna a população inicializada.

def avaliar_populacao(populacao, funcao_wrapper):
    """
    Avalia a aptidão (valor da função objetivo) de cada indivíduo na população.
    """
    aptidoes = [] # Lista para armazenar os valores de aptidão de cada indivíduo.
    for individuo in populacao: # Itera sobre cada indivíduo na população.
        x, y = individuo # Desempacota as coordenadas x e y do indivíduo.
        aptidao = funcao_wrapper(x, y) # Avalia a função objetivo para as coordenadas (x, y) usando o wrapper.
        aptidoes.append(aptidao) # Adiciona a aptidão calculada à lista.
    return aptidoes # Retorna a lista de aptidões.

def selecionar_pais(populacao, aptidoes, num_pais):
    """
    Seleciona os pais para a próxima geração usando o método de seleção por torneio ou elite (neste caso, elite simples).
    Os indivíduos com as menores aptidões (melhores) são selecionados.
    """
    individuos_com_aptidoes = list(zip(populacao, aptidoes)) # Combina indivíduos com suas aptidões em tuplas.
    individuos_ordenados = sorted(individuos_com_aptidoes, key=lambda x: x[1]) # Ordena os indivíduos com base na aptidão (o segundo elemento da tupla), do menor para o maior (minimização).
    pais_selecionados = [individuo[0] for individuo in individuos_ordenados[:num_pais]] # Seleciona os 'num_pais' melhores indivíduos (os primeiros da lista ordenada).
    return pais_selecionados # Retorna a lista de pais selecionados.

def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)):
    """
    Realiza o cruzamento BLX-alpha (Blend Crossover Alpha) entre os pais selecionados.
    Gera novos filhos com base nos genes dos pais, com uma chance de ocorrer o cruzamento.
    """
    nova_geracao = [] # Lista para armazenar os filhos gerados.
    if len(pais) < 2: # Se não houver pelo menos dois pais, o cruzamento não pode ocorrer.
        return [] # Retorna uma lista vazia.

    # global_op_counter.add_div(1) # Este comentário foi mantido da versão anterior, mas a linha foi removida na última revisão para maior precisão, pois a divisão aqui não é uma operação aritmética do problema.
    num_pares = len(pais) // 2 # Calcula o número de pares de pais que serão formados.

    for _ in range(num_pares): # Itera para formar pares de pais.
        pai1, pai2 = random.sample(pais, 2) # Seleciona dois pais aleatoriamente (sem substituição) da lista de pais.
        if random.random() < taxa_cruzamento: # Verifica se o cruzamento deve ocorrer com base na taxa de cruzamento.
            filho1 = np.zeros_like(pai1) # Cria um array NumPy para o primeiro filho, com o mesmo formato do pai.
            filho2 = np.zeros_like(pai2) # Cria um array NumPy para o segundo filho.

            for i in range(len(pai1)): # Itera sobre cada gene (dimensão) do indivíduo.
                gene1_pai = pai1[i] # Pega o gene 'i' do primeiro pai.
                gene2_pai = pai2[i] # Pega o gene 'i' do segundo pai.

                I = abs(gene1_pai - gene2_pai) # Calcula a diferença absoluta entre os genes dos pais.
                global_op_counter.add_mult(1) # Conta a multiplicação para `alpha * I`.
                d = alpha * I # Calcula 'd', que define a extensão da área de mistura.

                # Calcula os limites inferior e superior para a geração dos genes dos filhos.
                # A faixa de valores para os novos genes é [min(gene1, gene2) - d, max(gene1, gene2) + d].
                lower_bound = min(gene1_pai, gene2_pai) - d
                upper_bound = max(gene1_pai, gene2_pai) + d

                # Garante que os limites calculados não excedam os limites globais do problema.
                lower_bound = max(lower_bound, limites[0])
                upper_bound = min(upper_bound, limites[1])

                # Gera o gene do filho dentro da faixa calculada.
                filho1[i] = random.uniform(lower_bound, upper_bound)
                filho2[i] = random.uniform(lower_bound, upper_bound)
                
            nova_geracao.extend([filho1, filho2]) # Adiciona os dois filhos gerados à lista da nova geração.
        else: # Se o cruzamento não ocorrer.
            nova_geracao.extend([pai1.copy(), pai2.copy()]) # Os pais são passados diretamente (ou cópias deles) para a nova geração.
    return nova_geracao # Retorna a lista de indivíduos da nova geração (filhos e pais que não cruzaram).

def mutacao(populacao, taxa_mutacao, limites):
    """
    Aplica mutação a indivíduos da população com base na taxa de mutação.
    A mutação altera um gene de um indivíduo para um novo valor aleatório dentro dos limites.
    """
    for individuo in populacao: # Itera sobre cada indivíduo na população.
        if random.random() < taxa_mutacao: # Verifica se a mutação deve ocorrer para este indivíduo.
            indice_mutacao = random.randint(0, len(individuo) - 1) # Escolhe um índice de gene aleatoriamente para mutar.
            individuo[indice_mutacao] = random.uniform(limites[0], limites[1]) # Altera o gene selecionado para um novo valor aleatório dentro dos limites.
    return populacao # Retorna a população após a aplicação das mutações.


# ALGORITMO GENÉTICO PRINCIPAL
def algoritmo_genetico(tamanho_populacao, limites, num_geracoes, taxa_cruzamento, taxa_mutacao, geracoes_sem_melhora_limite=50, tolerancia=1e-6):
    """
    Implementa o algoritmo genético principal para otimização.
    """
    # Resetar contadores antes de iniciar a otimização
    global_op_counter.reset() # Reseta o contador global de operações para garantir uma contagem limpa para esta execução do AG.
    funcao_w4_wrapper_ag = FuncaoObjetivoWrapper(funcao_w4, global_op_counter) # Cria uma instância do wrapper para a função objetivo, que contará as avaliações e operações.

    populacao = inicializar_populacao(tamanho_populacao, limites) # Inicializa a população com indivíduos aleatórios.
    melhor_solucao = None # Variável para armazenar a melhor solução (indivíduo) encontrada até agora.
    melhor_aptidao = float('inf') # Variável para armazenar a melhor aptidão (menor valor da função) encontrada até agora, inicializada com infinito.
    melhor_geracao = -1 # Guarda a geração em que a melhor solução foi encontrada.

    avaliacoes_ag_melhor_solucao = 0 # Contador de avaliações no momento em que a melhor solução global foi encontrada.
    operacoes_ag_melhor_solucao_mult = 0 # Contador de multiplicações no momento em que a melhor solução global foi encontrada.
    operacoes_ag_melhor_solucao_div = 0 # Contador de divisões no momento em que a melhor solução global foi encontrada.

    # Variáveis para a decisão de parada por convergência (se o algoritmo não melhorar por X gerações).
    geracoes_sem_melhora = 0 # Contador de gerações consecutivas sem melhoria significativa.
    ultima_melhor_aptidao_global = float('inf') # Guarda a melhor aptidão da geração anterior para comparar a convergência.

    # INICIALIZAÇÃO DO GRÁFICO PARA AG
    fig = plt.figure(figsize=(8, 8)) # Cria uma nova figura para o gráfico.
    ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura para visualização.

    for i in range(num_geracoes): # Loop principal que executa o algoritmo por um número definido de gerações.
        aptidoes = avaliar_populacao(populacao, funcao_w4_wrapper_ag) # Avalia a aptidão de cada indivíduo na população atual.

        if not aptidoes or np.isinf(min(aptidoes)): # Verifica se a lista de aptidões está vazia ou contém valores infinitos (problemas na avaliação).
            continue # Se houver problemas, pula para a próxima geração.

        melhor_aptidao_geracao = min(aptidoes) # Encontra a melhor aptidão (menor valor) na geração atual.
        idx_melhor_geracao = aptidoes.index(melhor_aptidao_geracao) # Encontra o índice do indivíduo com a melhor aptidão.
        melhor_solucao_geracao_atual = populacao[idx_melhor_geracao] # Pega o indivíduo correspondente à melhor aptidão.

        # Atualiza a melhor solução global (a melhor encontrada em todas as gerações).
        # É importante atualizar o melhor global ANTES da lógica de convergência para ter o valor mais recente.
        if melhor_aptidao_geracao < melhor_aptidao: # Se a melhor aptidão da geração atual for melhor que a melhor global anterior.
            melhor_aptidao = melhor_aptidao_geracao # Atualiza a melhor aptidão global.
            melhor_solucao = melhor_solucao_geracao_atual.copy() # Atualiza a melhor solução global (faz uma cópia para evitar referência).
            melhor_geracao = i # Registra a geração em que essa melhoria ocorreu.
            avaliacoes_ag_melhor_solucao = funcao_w4_wrapper_ag.evaluations # Guarda o número total de avaliações até este ponto.
            operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications # Guarda o total de multiplicações até este ponto.
            operacoes_ag_melhor_solucao_div = global_op_counter.divisions # Guarda o total de divisões até este ponto.
        
        # --- Lógica da decisão de parada por convergência usando tolerância ---
        # Compara o melhor_aptidao atual (que é o melhor global até agora) com o da geração anterior.
        # Evita a comparação na primeira iteração (i == 0), pois não há uma "anterior".
        if i > 0 and abs(melhor_aptidao - ultima_melhor_aptidao_global) < tolerancia:
            # Se a diferença entre o melhor atual e o último melhor for menor que a tolerância.
            geracoes_sem_melhora += 1 # Incrementa o contador de gerações sem melhoria.
        else:
            # Se houve uma melhoria maior que a tolerância, ou se for a primeira iteração, reseta o contador.
            geracoes_sem_melhora = 0
            
        # Atualiza o "último melhor" para a próxima iteração
        ultima_melhor_aptidao_global = melhor_aptidao # Guarda o melhor global atual para ser comparado na próxima geração.

        # Verifica se o limite de gerações sem melhora foi atingido.
        if geracoes_sem_melhora >= geracoes_sem_melhora_limite: # Se o contador de gerações sem melhora atingiu o limite.
            print(f"\n[AG] Parada por convergência: Mudança no melhor valor da aptidão menor que {tolerancia} por {geracoes_sem_melhora_limite} gerações.") # Informa a parada.
            break # Sai do loop principal do AG se a convergência for detectada.
            
        # Seleciona os pais para a próxima geração.
        pais = selecionar_pais(populacao, aptidoes, tamanho_populacao // 2) # Seleciona um subconjunto da população como pais.
        filhos = cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=limites) # Realiza o cruzamento para gerar novos filhos.
        filhos = mutacao(filhos, taxa_mutacao, limites) # Aplica mutação aos filhos gerados.

        nova_populacao = pais + filhos # Combina os pais (que podem ser mantidos por elitismo) com os filhos.
        while len(nova_populacao) < tamanho_populacao: # Garante que a nova população tenha o tamanho correto.
            nova_populacao.append(np.array([ # Se a nova população for menor, preenche com novos indivíduos aleatórios.
                random.uniform(limites[0], limites[1]),
                random.uniform(limites[0], limites[1])
            ]))
        populacao = nova_populacao[:tamanho_populacao] # Atualiza a população para a próxima geração, mantendo o tamanho original.

        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao) # Atualiza o gráfico com a população atual e a melhor solução.

    plt.show() # Exibe a janela do gráfico após a conclusão do algoritmo.

    # Imprime os resultados finais do algoritmo genético.
    print(f"Melhor solução encontrada (AG): {melhor_solucao}")
    print(f"Valor da função para a melhor solução (AG): {melhor_aptidao}")
    print(f"Gerações executadas (AG): {i}") # Mostra a última geração processada (onde parou).
    print(f"Avaliações da função objetivo (AG): {funcao_w4_wrapper_ag.evaluations}") # Total de avaliações da função objetivo.
    print(f"Operações de Multiplicação (AG): {global_op_counter.multiplications}") # Total de multiplicações.
    print(f"Operações de Divisão (AG): {global_op_counter.divisions}") # Total de divisões.
    print(f"Avaliações para o 'melhor global' (AG): {avaliacoes_ag_melhor_solucao}") # Avaliações até encontrar o melhor global.
    print(f"Multiplicações para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_mult}") # Multiplicações até encontrar o melhor global.
    print(f"Divisões para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_div}") # Divisões até encontrar o melhor global.

    return melhor_solucao, melhor_aptidao, i # Retorna a melhor solução, sua aptidão e o número de gerações executadas.

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    # Define os parâmetros para uma execução de teste direto do algoritmo genético.
    tamanho_populacao = 35
    limites = (-500, 500)
    num_geracoes = 1000
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 50
    tolerancia = 1e-6 # Define a tolerância para o teste direto

    # Chama a função principal do algoritmo genético com os parâmetros definidos.
    melhor_solucao, melhor_aptidao, geracao = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite,
        tolerancia=tolerancia
    )