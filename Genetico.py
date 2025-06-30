import numpy as np # Importa a biblioteca NumPy, essencial para operações numéricas e arrays eficientes (como vetores de genes dos indivíduos).
import random # Importa a biblioteca 'random' para geração de números aleatórios, usada na inicialização da população, seleção por torneio, cruzamento e mutação.
import matplotlib.pyplot as plt # Importa o módulo 'pyplot' da Matplotlib como 'plt', para criar e exibir gráficos estáticos.
from mpl_toolkits.mplot3d import Axes3D # Importa 'Axes3D' da Matplotlib, necessário para criar gráficos em 3 dimensões, visualizando o espaço de busca.

from funcoes_otimizacao import funcao_w4 # Importa a 'funcao_w4' do arquivo 'funcoes_otimizacao.py', que é a função objetivo que o Algoritmo Genético (AG) tentará minimizar.
from Grafico import GraficoAG # Importa a função 'GraficoAG' do arquivo 'Grafico.py', utilizada para plotar a população e a melhor solução ao longo das gerações.
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa 'global_op_counter' (um objeto global para contar operações aritméticas) e 'FuncaoObjetivoWrapper' (para envolver a função objetivo e contar suas avaliações) do módulo 'utils.py'.

# --- Funções Auxiliares do Algoritmo Genético ---

def inicializar_populacao(tamanho_populacao, limites):
    """
    Inicializa a população inicial do algoritmo genético.
    Cada indivíduo é um par (x, y) gerado aleatoriamente dentro dos limites.
    """
    populacao = [] # Cria uma lista vazia para armazenar os indivíduos que comporão a população.
    for _ in range(tamanho_populacao): # Loop que se repete 'tamanho_populacao' vezes para criar cada indivíduo.
        individuo = np.array([ # Cria um novo indivíduo como um array NumPy de duas dimensões (x, y).
            random.uniform(limites[0], limites[1]), # Gera um valor aleatório para a primeira coordenada (x) dentro dos limites fornecidos.
            random.uniform(limites[0], limites[1]), # Gera um valor aleatório para a segunda coordenada (y) dentro dos mesmos limites.
        ])
        populacao.append(individuo) # Adiciona o indivíduo recém-criado à lista da população.
    return populacao # Retorna a população inicializada, pronta para a primeira avaliação.

def avaliar_populacao(populacao, funcao_wrapper):
    """
    Avalia a aptidão (valor da função objetivo) de cada indivíduo na população.
    """
    aptidoes = [] # Cria uma lista vazia para armazenar os valores de aptidão (fitnes) de cada indivíduo.
    for individuo in populacao: # Itera sobre cada 'individuo' presente na lista 'populacao'.
        x, y = individuo # Desempacota o array 'individuo' em suas coordenadas x e y.
        aptidao = funcao_wrapper(x, y) # Chama a função objetivo (através do wrapper) para calcular a aptidão do indivíduo (quanto menor, melhor, pois é minimização).
        aptidoes.append(aptidao) # Adiciona a aptidão calculada à lista 'aptidoes'.
    return aptidoes # Retorna a lista completa de aptidões, na mesma ordem dos indivíduos da população.

def selecao_torneio(populacao, aptidoes, tamanho_torneio=3):
    """
    Seleciona um único indivíduo da população usando o método de seleção por torneio.
    Retorna o indivíduo "vencedor" do torneio.
    """
    # 1. Escolhe aleatoriamente 'tamanho_torneio' índices de indivíduos da população.
    # 'random.sample' garante que os índices selecionados sejam únicos (sem repetição).
    competidores_indices = random.sample(range(len(populacao)), tamanho_torneio)
    
    # 2. Assume que o primeiro competidor sorteado é o melhor inicialmente.
    melhor_competidor_indice = competidores_indices[0]
    
    # 3. Itera sobre os índices dos competidores para encontrar aquele com a melhor aptidão.
    # Como o objetivo é minimização, a "melhor" aptidão é o menor valor.
    for idx in competidores_indices: # 'idx' representa o índice de um competidor sorteado.
        # Compara a aptidão do competidor atual ('aptidoes[idx]') com a aptidão do atual "melhor competidor".
        if aptidoes[idx] < aptidoes[melhor_competidor_indice]:
            melhor_competidor_indice = idx # Se o competidor atual for melhor, atualiza 'melhor_competidor_indice' para ele.
            
    # Retorna o indivíduo (seu genótipo) que corresponde ao índice do melhor competidor encontrado no torneio.
    return populacao[melhor_competidor_indice]

def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)):
    """
    Realiza o cruzamento BLX-alpha (Blend Crossover Alpha) entre os DOIS pais fornecidos.
    Gera DOIS novos filhos com base nos genes dos pais, com uma chance de ocorrer o cruzamento.
    """
    # A função agora espera receber `pais` como uma lista ou tupla contendo exatamente dois indivíduos (pai1, pai2).
    pai1, pai2 = pais[0], pais[1] # Desempacota os dois pais fornecidos.

    if random.random() < taxa_cruzamento: # Verifica se o cruzamento deve ocorrer com base na 'taxa_cruzamento'.
        filho1 = np.zeros_like(pai1) # Cria um array NumPy para o primeiro filho, com o mesmo formato (dimensões) do pai1, preenchido com zeros.
        filho2 = np.zeros_like(pai2) # Cria um array NumPy para o segundo filho, com o mesmo formato do pai2.

        for i in range(len(pai1)): # Itera sobre cada gene (dimensão, e.g., x e y) dos pais.
            gene1_pai = pai1[i] # Pega o gene 'i' do primeiro pai.
            gene2_pai = pai2[i] # Pega o gene 'i' do segundo pai.

            I = abs(gene1_pai - gene2_pai) # Calcula a diferença absoluta entre os genes correspondentes dos dois pais.
            global_op_counter.add_mult(1) # Conta uma multiplicação para a operação 'alpha * I'.
            d = alpha * I # Calcula 'd', que define a amplitude da "mistura" de genes para criar os filhos.

            # Calcula os limites inferior e superior para a geração dos genes dos filhos.
            # A faixa de valores para os novos genes será centrada entre os pais, mas estendida por 'd'.
            lower_bound = min(gene1_pai, gene2_pai) - d
            upper_bound = max(gene1_pai, gene2_pai) + d

            # Garante que os limites calculados para o novo gene não excedam os limites globais do problema.
            lower_bound = max(lower_bound, limites[0]) # Ajusta o limite inferior se for menor que o limite global.
            upper_bound = min(upper_bound, limites[1]) # Ajusta o limite superior se for maior que o limite global.

            # Gera o gene 'i' para o filho1 aleatoriamente dentro da nova faixa de mistura.
            filho1[i] = random.uniform(lower_bound, upper_bound)
            # Gera o gene 'i' para o filho2 aleatoriamente dentro da mesma faixa de mistura.
            filho2[i] = random.uniform(lower_bound, upper_bound)
            
        return [filho1, filho2] # Retorna uma lista contendo os dois filhos gerados pelo cruzamento.
    else: # Se a condição de cruzamento (baseada na taxa_cruzamento) não for satisfeita.
        return [pai1.copy(), pai2.copy()] # Retorna cópias dos pais originais, pois não houve cruzamento para este par.

def mutacao(populacao, taxa_mutacao, limites):
    """
    Aplica mutação a indivíduos da população com base na taxa de mutação.
    A mutação altera um gene de um indivíduo para um novo valor aleatório dentro dos limites.
    """
    for individuo in populacao: # Itera sobre cada 'individuo' na lista 'populacao' (geralmente filhos).
        if random.random() < taxa_mutacao: # Verifica se a mutação deve ocorrer para este indivíduo, com base na 'taxa_mutacao'.
            indice_mutacao = random.randint(0, len(individuo) - 1) # Escolhe aleatoriamente qual gene (dimensão) do indivíduo será mutado.
            individuo[indice_mutacao] = random.uniform(limites[0], limites[1]) # Altera o gene selecionado para um novo valor aleatório dentro dos limites globais.
    return populacao # Retorna a população (agora com mutações aplicadas a alguns indivíduos).


# --- ALGORITMO GENÉTICO PRINCIPAL ---
def algoritmo_genetico(tamanho_populacao, limites, num_geracoes, taxa_cruzamento, taxa_mutacao, geracoes_sem_melhora_limite=50, tolerancia=1e-6, tamanho_torneio=3):
    """
    Implementa o algoritmo genético principal para otimização.
    Recebe parâmetros como tamanho da população, limites do espaço de busca, número de gerações, taxas de cruzamento e mutação,
    critérios de parada por convergência e o tamanho do torneio para seleção.
    """
    global_op_counter.reset() # Reseta o contador global de operações para garantir uma contagem limpa para esta nova execução do AG.
    funcao_w4_wrapper_ag = FuncaoObjetivoWrapper(funcao_w4, global_op_counter) # Cria uma instância do wrapper para 'funcao_w4', que será usada para avaliar indivíduos e contar as chamadas da função objetivo.

    populacao = inicializar_populacao(tamanho_populacao, limites) # Inicializa a primeira geração da população com indivíduos gerados aleatoriamente dentro dos limites.
    melhor_solucao = None # Variável para armazenar o indivíduo (genótipo) que representa a melhor solução encontrada em todas as gerações.
    melhor_aptidao = float('inf') # Variável para armazenar o valor da aptidão da melhor solução encontrada, inicializada com infinito (para minimização).
    melhor_geracao = -1 # Guarda o número da geração em que a 'melhor_solucao' foi encontrada pela última vez.

    avaliacoes_ag_melhor_solucao = 0 # Contador de avaliações da função objetivo no momento em que a 'melhor_solucao' foi atualizada.
    operacoes_ag_melhor_solucao_mult = 0 # Contador de multiplicações no momento em que a 'melhor_solucao' foi atualizada.
    operacoes_ag_melhor_solucao_div = 0 # Contador de divisões no momento em que a 'melhor_solucao' foi atualizada.

    # Variáveis para a decisão de parada por convergência (se o algoritmo não melhorar significativamente por X gerações consecutivas).
    geracoes_sem_melhora = 0 # Contador de gerações consecutivas sem uma melhoria significativa no 'melhor_aptidao'.
    ultima_melhor_aptidao_global = float('inf') # Armazena o 'melhor_aptidao' da geração anterior para comparação de convergência.

    # --- INICIALIZAÇÃO DO GRÁFICO PARA AG ---
    fig = plt.figure(figsize=(8, 8)) # Cria uma nova figura Matplotlib para o gráfico 3D.
    ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura, que será usado para plotar as populações.

    for i in range(num_geracoes): # Loop principal do Algoritmo Genético, que executa por um número definido de 'num_geracoes'. 'i' representa a geração atual.
        aptidoes = avaliar_populacao(populacao, funcao_w4_wrapper_ag) # Avalia a aptidão de cada indivíduo na 'populacao' atual e armazena os resultados em 'aptidoes'.

        if not aptidoes or np.isinf(min(aptidoes)): # Verifica se a lista de aptidões está vazia ou se o menor valor é infinito (indicando problema na avaliação).
            continue # Se houver problemas, pula para a próxima iteração do loop.

        melhor_aptidao_geracao = min(aptidoes) # Encontra o menor valor (melhor aptidão) na geração atual.
        idx_melhor_geracao = aptidoes.index(melhor_aptidao_geracao) # Encontra o índice do indivíduo que possui a 'melhor_aptidao_geracao'.
        melhor_solucao_geracao_atual = populacao[idx_melhor_geracao] # Obtém o indivíduo (genótipo) correspondente à melhor aptidão da geração atual.

        # Atualiza a melhor solução global encontrada até o momento.
        # Esta atualização é feita ANTES da lógica de convergência para sempre ter o valor mais recente para comparação.
        if melhor_aptidao_geracao < melhor_aptidao: # Se a melhor aptidão da geração atual for menor (melhor) que a 'melhor_aptidao' global.
            melhor_aptidao = melhor_aptidao_geracao # Atualiza a 'melhor_aptidao' global com o novo melhor valor.
            melhor_solucao = melhor_solucao_geracao_atual.copy() # Atualiza a 'melhor_solucao' global (faz uma cópia para evitar que a referência mude).
            melhor_geracao = i # Registra o número da geração em que essa melhoria global ocorreu.
            avaliacoes_ag_melhor_solucao = funcao_w4_wrapper_ag.evaluations # Guarda o total de avaliações da função objetivo até este ponto.
            operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications # Guarda o total de multiplicações até este ponto.
            operacoes_ag_melhor_solucao_div = global_op_counter.divisions # Guarda o total de divisões até este ponto.
        
        # --- Lógica da decisão de parada por convergência usando tolerância ---
        # Compara o 'melhor_aptidao' atual (que é o melhor global encontrado até agora) com o da geração anterior.
        # 'i > 0' evita a comparação na primeira iteração, pois não há uma "anterior".
        if i > 0 and abs(melhor_aptidao - ultima_melhor_aptidao_global) < tolerancia:
            # Se a diferença absoluta entre o melhor atual e o último melhor for menor que a 'tolerancia',
            # significa que a melhoria foi insignificante.
            geracoes_sem_melhora += 1 # Incrementa o contador de gerações sem melhoria significativa.
        else:
            # Se houve uma melhoria maior que a tolerância, ou se for a primeira iteração, zera o contador.
            geracoes_sem_melhora = 0
            
        # Atualiza o "último melhor" para ser usado na próxima iteração para comparação de convergência.
        ultima_melhor_aptidao_global = melhor_aptidao

        # Verifica se o limite de gerações consecutivas sem melhoria foi atingido.
        if geracoes_sem_melhora >= geracoes_sem_melhora_limite: # Se o contador atingiu o limite.
            print(f"\n[AG] Parada por convergência: Mudança no melhor valor da aptidão menor que {tolerancia} por {geracoes_sem_melhora_limite} gerações.") # Imprime uma mensagem informando a parada por convergência.
            break # Sai do loop principal do AG, encerrando o processo de otimização.
            
        # --- Modificação AQUI para usar a seleção por torneio e construir a próxima geração ---
        # A 'proxima_geracao' é a lista que será preenchida com os novos indivíduos.
        # Implementa um elitismo simples: se houver uma 'melhor_solucao' global, ela é copiada para a próxima geração.
        proxima_geracao = [melhor_solucao.copy()] if melhor_solucao is not None else []

        # Calcula quantos filhos ainda são necessários para preencher a população até 'tamanho_populacao'.
        # Isso leva em conta se o elitismo já adicionou um indivíduo.
        num_filhos_necessarios = tamanho_populacao - len(proxima_geracao) 

        # Loop para gerar pares de filhos (cruzamento) até que o número necessário seja atingido.
        for _ in range(num_filhos_necessarios // 2): # // 2 porque cada cruzamento gera 2 filhos.
            # Seleção do primeiro pai usando o método de torneio.
            pai1 = selecao_torneio(populacao, aptidoes, tamanho_torneio)
            # Seleção do segundo pai usando o método de torneio.
            pai2 = selecao_torneio(populacao, aptidoes, tamanho_torneio)

            # Realiza o cruzamento BLX-alpha entre os dois pais selecionados pelo torneio.
            # `cruzamento_blx_alpha` agora espera uma lista de 2 pais e retorna 2 filhos.
            filhos_gerados = cruzamento_blx_alpha([pai1, pai2], taxa_cruzamento, alpha=0.5, limites=limites)
            proxima_geracao.extend(filhos_gerados) # Adiciona os filhos gerados à lista da 'proxima_geracao'.

        # Tratar caso de número ímpar de filhos necessários.
        # Se 'num_filhos_necessarios' for ímpar, um filho ainda falta após o loop acima.
        if num_filhos_necessarios % 2 != 0 and len(proxima_geracao) < tamanho_populacao:
            # Seleciona um indivíduo extra via torneio (pode ser considerado um "filho" sem par de cruzamento ou um pai direto).
            extra_pai = selecao_torneio(populacao, aptidoes, tamanho_torneio)
            proxima_geracao.append(extra_pai.copy()) # Adiciona uma cópia deste indivíduo à 'proxima_geracao'.


        filhos_apos_mutacao = mutacao(proxima_geracao, taxa_mutacao, limites) # Aplica a mutação a todos os indivíduos da 'proxima_geracao' (filhos e elite, se houver).

        # Garante que a 'populacao' da próxima geração tenha exatamente o 'tamanho_populacao' definido, truncando se houver excesso.
        populacao = filhos_apos_mutacao[:tamanho_populacao] 

        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao) # Atualiza e redesenha o gráfico 3D para visualizar o estado da população na geração atual.

    plt.show() # Exibe a janela do gráfico Matplotlib ao final da execução completa do AG.

    # --- Imprime os resultados finais do algoritmo genético ---
    print(f"Melhor solução encontrada (AG): {melhor_solucao}") # Exibe o genótipo da melhor solução encontrada.
    print(f"Valor da função para a melhor solução (AG): {melhor_aptidao}") # Exibe a aptidão (valor da função objetivo) da melhor solução.
    print(f"Gerações executadas (AG): {i}") # Exibe o número total de gerações que foram executadas até a parada.
    print(f"Avaliações da função objetivo (AG): {funcao_w4_wrapper_ag.evaluations}") # Exibe o total de vezes que a função objetivo foi avaliada.
    print(f"Operações de Multiplicação (AG): {global_op_counter.multiplications}") # Exibe o total de operações de multiplicação contadas.
    print(f"Operações de Divisão (AG): {global_op_counter.divisions}") # Exibe o total de operações de divisão contadas.
    print(f"Avaliações para o 'melhor global' (AG): {avaliacoes_ag_melhor_solucao}") # Exibe o número de avaliações no ponto em que o 'melhor_global' foi atingido.
    print(f"Multiplicações para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_mult}") # Exibe as multiplicações no ponto em que o 'melhor_global' foi atingido.
    print(f"Divisões para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_div}") # Exibe as divisões no ponto em que o 'melhor_global' foi atingido.

    return melhor_solucao, melhor_aptidao, i # Retorna os resultados mais importantes do AG.

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
if __name__ == "__main__": # Bloco de código que só é executado quando o script 'genetico.py' é invocado diretamente (não importado como módulo).
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---") # Mensagem indicando que o teste direto está começando.
    # Define os parâmetros para uma execução de teste padrão do algoritmo genético.
    tamanho_populacao = 35 # Define o número de indivíduos na população.
    limites = (-500, 500) # Define os limites superior e inferior para as coordenadas x e y dos indivíduos.
    num_geracoes = 1000 # Define o número máximo de gerações para a execução.
    taxa_cruzamento = 0.7 # Define a probabilidade de um par de pais sofrer cruzamento.
    taxa_mutacao = 0.01 # Define a probabilidade de um gene individual sofrer mutação.
    geracoes_sem_melhora_limite = 50 # Define o número de gerações sem melhoria para a parada por convergência.
    tolerancia = 1e-6 # Define a tolerância para considerar uma melhoria "significativa".
    tamanho_torneio = 3 # **ADICIONADO**: Define o tamanho do torneio para a seleção de pais (quantos indivíduos competirão).

    # Chama a função principal do algoritmo genético com os parâmetros definidos acima.
    melhor_solucao, melhor_aptidao, geracao = algoritmo_genetico( # As variáveis irão armazenar os valores de retorno da função.
        tamanho_populacao=tamanho_populacao, # Passa o tamanho da população.
        limites=limites, # Passa os limites do espaço de busca.
        num_geracoes=num_geracoes, # Passa o número de gerações.
        taxa_cruzamento=taxa_cruzamento, # Passa a taxa de cruzamento.
        taxa_mutacao=taxa_mutacao, # Passa a taxa de mutação.
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite, # Passa o limite de gerações sem melhoria.
        tolerancia=tolerancia, # Passa a tolerância.
        tamanho_torneio=tamanho_torneio # **ADICIONADO**: Passa o tamanho do torneio.
    )