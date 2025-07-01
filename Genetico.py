import numpy as np # Importa a biblioteca NumPy, fundamental para operações numéricas e manipulação de arrays.
import random # Importa o módulo random para geração de números aleatórios.
import matplotlib.pyplot as plt # Importa Matplotlib para criação de gráficos.
from mpl_toolkits.mplot3d import Axes3D # Importa Axes3D para gráficos 3D.
import os # Importa o módulo os para interagir com o sistema operacional (manipulação de arquivos/pastas).
from datetime import datetime # Importa datetime para trabalhar com datas e horas (para timestamps).

from funcoes_otimizacao import funcao_w4 # Importa a função objetivo W4 a ser otimizada.
from Grafico import GraficoAG # Importa a função para plotar o gráfico do AG.
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa contadores de operações e o wrapper da função objetivo.

# --- Funções Auxiliares do Algoritmo Genético ---

def inicializar_populacao(tamanho_populacao, limites):
    """
    Inicializa a população inicial do algoritmo genético.
    Cada indivíduo é um par (x, y) gerado aleatoriamente dentro dos limites.
    """
    populacao = [] # Lista vazia para armazenar os indivíduos da população.
    for _ in range(tamanho_populacao): # Loop para criar 'tamanho_populacao' indivíduos.
        individuo = np.array([ # Cria um indivíduo como um array NumPy de 2 dimensões.
            random.uniform(limites[0], limites[1]), # Coordenada x aleatória dentro dos limites.
            random.uniform(limites[0], limites[1]), # Coordenada y aleatória dentro dos limites.
        ])
        populacao.append(individuo) # Adiciona o indivíduo à população.
    return populacao # Retorna a população inicializada.

def avaliar_populacao(populacao, funcao_wrapper):
    """
    Avalia a aptidão (valor da função objetivo) de cada indivíduo na população.
    """
    aptidoes = [] # Lista vazia para armazenar as aptidões dos indivíduos.
    for individuo in populacao: # Itera sobre cada indivíduo na população.
        x, y = individuo # Desempacota as coordenadas x e y do indivíduo.
        aptidao = funcao_wrapper(x, y) # Avalia a função objetivo para as coordenadas (e conta as avaliações).
        aptidoes.append(aptidao) # Adiciona a aptidão à lista.
    return aptidoes # Retorna a lista de aptidões.

def selecao_roleta(populacao, aptidoes):
    """
    Seleciona um indivíduo da população usando o método de seleção por roleta.
    Retorna o indivíduo selecionado.
    Como é minimização, transformamos a aptidão para que valores menores tenham maior "fatia" na roleta.
    """
    if not aptidoes: # Verifica se a lista de aptidões está vazia.
        return None # Retorna None se não houver aptidões para selecionar.

    max_aptidao = max(aptidoes) # Encontra a maior aptidão na população.
    # Transforma as aptidões para que valores menores resultem em scores maiores (para minimização).
    fit_scores = [max_aptidao - apt + 1e-6 for apt in aptidoes] 
    
    total_fit = sum(fit_scores) # Calcula a soma total dos scores de aptidão.
    
    if total_fit == 0: # Caso todos os scores sejam zero (situação rara, mas possível).
        return random.choice(populacao) # Retorna um indivíduo aleatório.

    probabilidades = [score / total_fit for score in fit_scores] # Calcula a probabilidade de seleção de cada indivíduo.

    acumulado = 0 # Variável para somar as probabilidades acumuladas.
    roleta = [] # Lista para armazenar os limites superiores da "roleta".
    for i, prob in enumerate(probabilidades): # Itera sobre as probabilidades.
        acumulado += prob # Acumula a probabilidade.
        roleta.append(acumulado) # Adiciona o valor acumulado à roleta.

    r = random.random() # Gera um número aleatório entre 0 e 1 (o "giro" da roleta).
    for i, lim_superior in enumerate(roleta): # Itera sobre os limites da roleta.
        if r <= lim_superior: # Se o giro cair dentro do limite, seleciona o indivíduo.
            return populacao[i] # Retorna o indivíduo selecionado.
    
    return populacao[-1] # Retorna o último indivíduo como fallback (caso raro).


def cruzamento_blx_alpha(pais, taxa_cruzamento, alpha=0.5, limites=(-500, 500)):
    """
    Realiza o cruzamento BLX-alpha (Blend Crossover Alpha) entre os DOIS pais fornecidos.
    Gera DOIS novos filhos com base nos genes dos pais, com uma chance de ocorrer o cruzamento.
    """
    pai1, pai2 = pais[0], pais[1] # Desempacota os dois pais.

    if random.random() < taxa_cruzamento: # Verifica se o cruzamento deve ocorrer com base na taxa de cruzamento.
        filho1 = np.zeros_like(pai1) # Cria um array de zeros com a mesma forma do pai1 para o filho1.
        filho2 = np.zeros_like(pai2) # Cria um array de zeros com a mesma forma do pai2 para o filho2.

        for i in range(len(pai1)): # Itera sobre os genes (dimensões) dos pais.
            gene1_pai = pai1[i] # Pega o gene do pai1.
            gene2_pai = pai2[i] # Pega o gene do pai2.

            I = abs(gene1_pai - gene2_pai) # Calcula a diferença absoluta entre os genes.
            global_op_counter.add_mult(1) # Contabiliza uma multiplicação para o cálculo de 'd'.
            d = alpha * I # Calcula o 'd' para definir a faixa de mistura.

            lower_bound = min(gene1_pai, gene2_pai) - d # Define o limite inferior para o novo gene.
            upper_bound = max(gene1_pai, gene2_pai) + d # Define o limite superior para o novo gene.

            lower_bound = max(lower_bound, limites[0]) # Garante que o limite inferior não saia do espaço de busca.
            upper_bound = min(upper_bound, limites[1]) # Garante que o limite superior não saia do espaço de busca.

            filho1[i] = random.uniform(lower_bound, upper_bound) # Gera um gene para o filho1 dentro dos novos limites.
            filho2[i] = random.uniform(lower_bound, upper_bound) # Gera um gene para o filho2 dentro dos novos limites.
            
        return [filho1, filho2] # Retorna os dois filhos gerados pelo cruzamento.
    else:
        return [pai1.copy(), pai2.copy()] # Se não houver cruzamento, retorna cópias dos pais originais.

def mutacao(populacao, taxa_mutacao, limites):
    """
    Aplicação de mutação a indivíduos da população.
    """
    for individuo in populacao: # Itera sobre cada indivíduo na população.
        if random.random() < taxa_mutacao: # Verifica se a mutação deve ocorrer para este indivíduo.
            indice_mutacao = random.randint(0, len(individuo) - 1) # Escolhe um gene aleatório para mutar.
            # Altera o gene selecionado para um novo valor aleatório dentro dos limites.
            individuo[indice_mutacao] = random.uniform(limites[0], limites[1])
    return populacao # Retorna a população após a aplicação das mutações.


# --- ALGORITMO GENÉTICO PRINCIPAL ---
def algoritmo_genetico(tamanho_populacao, limites, num_geracoes, taxa_cruzamento, taxa_mutacao, geracoes_sem_melhora_limite=50, tolerancia=1e-6):
    """
    Implementa o algoritmo genético principal para otimização.
    """
    global_op_counter.reset() # Reseta os contadores globais de operações.
    # Cria um "wrapper" para a função objetivo que também conta suas avaliações.
    funcao_w4_wrapper_ag = FuncaoObjetivoWrapper(funcao_w4, global_op_counter)

    populacao = inicializar_populacao(tamanho_populacao, limites) # Inicializa a primeira população.
    melhor_solucao = None # Variável para armazenar a melhor solução global encontrada.
    melhor_aptidao = float('inf') # Variável para armazenar a melhor aptidão global (inicializada com infinito para minimização).
    melhor_geracao = -1 # Geração em que a melhor solução global foi encontrada.

    avaliacoes_ag_melhor_solucao = 0 # Contagem de avaliações quando o melhor global foi *encontrado*.
    operacoes_ag_melhor_solucao_mult = 0 # Contagem de multiplicações quando o melhor global foi *encontrado*.
    operacoes_ag_melhor_solucao_div = 0 # Contagem de divisões quando o melhor global foi *encontrado*.

    geracoes_sem_melhora = 0 # Contador para gerações consecutivas sem melhora significativa.
    ultima_melhor_aptidao_global = float('inf') # Armazena a última melhor aptidão global para o critério de convergência.

    # --- INICIALIZAÇÃO DO GRÁFICO PARA AG ---
    fig = plt.figure(figsize=(11, 8)) # Cria uma nova figura para o gráfico.
    ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura.
    plt.subplots_adjust(right=0.7) # Ajusta o layout para deixar espaço para a legenda à direita.

    # Dicionário para armazenar os parâmetros e estatísticas do AG para a legenda do gráfico.
    ag_params_for_plot = {
        "tamanho_populacao": tamanho_populacao,
        "taxa_mutacao": taxa_mutacao,
        "taxa_crossover": taxa_cruzamento,
        "selecao_tipo": "Roleta", # Define o tipo de seleção utilizado.
        "iteracoes_totais": num_geracoes,
        "avaliacoes_funcao": 0, # Será atualizado em cada iteração com o total de avaliações.
        "multiplicacoes_total": 0, # Será atualizado em cada iteração com o total de multiplicações.
        "divisoes_total": 0, # Será atualizado em cada iteração com o total de divisões.
        "avaliacoes_minimo_global": 0, # Será atualizado quando o melhor global for encontrado.
        "multiplicacoes_minimo_global": 0, # Será atualizado quando o melhor global for encontrado.
        "divisoes_minimo_global": 0 # Será atualizado quando o melhor global for encontrado.
    }

    for i in range(num_geracoes): # Loop principal do algoritmo genético, uma iteração por geração.
        aptidoes = avaliar_populacao(populacao, funcao_w4_wrapper_ag) # Avalia a aptidão de todos os indivíduos da população.

        if not aptidoes or np.isinf(min(aptidoes)): # Lida com casos onde não há aptidões válidas.
            continue # Pula para a próxima iteração.

        melhor_aptidao_geracao = min(aptidoes) # Encontra a melhor aptidão da geração atual.
        idx_melhor_geracao = aptidoes.index(melhor_aptidao_geracao) # Encontra o índice do melhor indivíduo da geração.
        melhor_solucao_geracao_atual = populacao[idx_melhor_geracao] # Pega o melhor indivíduo da geração.

        # Verifica se o melhor indivíduo da geração atual é melhor que o melhor global encontrado até agora.
        if melhor_aptidao_geracao < melhor_aptidao:
            melhor_aptidao = melhor_aptidao_geracao # Atualiza a melhor aptidão global.
            melhor_solucao = melhor_solucao_geracao_atual.copy() # Atualiza a melhor solução global.
            melhor_geracao = i # Registra a geração em que foi encontrada.
            # Registra as avaliações e operações *no momento* em que o melhor global foi encontrado.
            avaliacoes_ag_melhor_solucao = funcao_w4_wrapper_ag.evaluations
            operacoes_ag_melhor_solucao_mult = global_op_counter.multiplications
            operacoes_ag_melhor_solucao_div = global_op_counter.divisions
            
            # Atualiza as estatísticas no dicionário para a legenda do gráfico.
            ag_params_for_plot["avaliacoes_minimo_global"] = avaliacoes_ag_melhor_solucao
            ag_params_for_plot["multiplicacoes_minimo_global"] = operacoes_ag_melhor_solucao_mult
            ag_params_for_plot["divisoes_minimo_global"] = operacoes_ag_melhor_solucao_div
        
        # Lógica de parada por convergência (se a melhora global for menor que a tolerância).
        if i > 0 and abs(melhor_aptidao - ultima_melhor_aptidao_global) < tolerancia:
            geracoes_sem_melhora += 1 # Incrementa o contador de gerações sem melhora.
        else:
            geracoes_sem_melhora = 0 # Reseta o contador se houve melhora.
            
        ultima_melhor_aptidao_global = melhor_aptidao # Atualiza a última melhor aptidão global para a próxima comparação.

        if geracoes_sem_melhora >= geracoes_sem_melhora_limite: # Verifica se o limite de gerações sem melhora foi atingido.
            print(f"\n[AG] Parada por convergência: Mudança no melhor valor da aptidão menor que {tolerancia} por {geracoes_sem_melhora_limite} gerações.")
            break # Sai do loop principal se houver convergência.
            
        # --- Construção da próxima geração sem elitismo ---
        proxima_geracao = [] # A lista da próxima geração começa vazia (sem elitismo).
        
        # O loop deve rodar `tamanho_populacao // 2` vezes para gerar pares de filhos.
        # Se o tamanho da população for ímpar, um indivíduo extra será selecionado no final.
        for _ in range(tamanho_populacao // 2): 
            pai1 = selecao_roleta(populacao, aptidoes) # Seleciona o primeiro pai por roleta.
            pai2 = selecao_roleta(populacao, aptidoes) # Seleciona o segundo pai por roleta.

            if pai1 is None or pai2 is None: # Lida com casos onde a seleção pode falhar.
                continue # Se a seleção falhar, pula para a próxima iteração do loop de geração de filhos.

            # Realiza o cruzamento BLX-alpha entre os pais.
            filhos_gerados = cruzamento_blx_alpha([pai1, pai2], taxa_cruzamento, alpha=0.5, limites=limites)
            proxima_geracao.extend(filhos_gerados) # Adiciona os filhos gerados à lista da próxima geração.

        # Se o tamanho da população for ímpar, seleciona um último indivíduo para completar a geração.
        if len(proxima_geracao) < tamanho_populacao:
            extra_pai = selecao_roleta(populacao, aptidoes) # Seleciona um pai extra.
            if extra_pai is not None: # Garante que a seleção foi bem-sucedida.
                proxima_geracao.append(extra_pai.copy()) # Adiciona o pai extra à próxima geração.

        filhos_apos_mutacao = mutacao(proxima_geracao, taxa_mutacao, limites) # Aplica mutação aos filhos.

        # A nova população é formada pelos filhos (já mutados), garantindo o tamanho correto.
        populacao = filhos_apos_mutacao[:tamanho_populacao] 

        # Atualiza as estatísticas globais (totais) para o dicionário da legenda a cada iteração.
        ag_params_for_plot["avaliacoes_funcao"] = funcao_w4_wrapper_ag.evaluations
        ag_params_for_plot["multiplicacoes_total"] = global_op_counter.multiplications
        ag_params_for_plot["divisoes_total"] = global_op_counter.divisions

        # Passa o dicionário de parâmetros para a função GraficoAG para plotar a geração atual.
        GraficoAG(populacao, melhor_solucao, i + 1, ax, melhor_aptidao, ag_params=ag_params_for_plot)

    plt.show() # Exibe o gráfico final.

    # --- Prepara as estatísticas para impressão e salvamento ---
    stats_output = [] # Lista para armazenar as linhas de saída.
    stats_output.append("--- Resultados Finais do Algoritmo Genético ---")
    stats_output.append(f"Melhor solução encontrada (AG): {melhor_solucao}") # Posição da melhor solução.
    stats_output.append(f"Valor da função para a melhor solução (AG): {melhor_aptidao:.4f}") # Valor da função para a melhor solução.
    stats_output.append(f"Gerações executadas (AG): {i}") # Número total de gerações executadas.
    stats_output.append(f"Avaliações da função objetivo (AG): {funcao_w4_wrapper_ag.evaluations}") # Avaliações totais (convergência).
    stats_output.append(f"Operações de Multiplicação (AG): {global_op_counter.multiplications}") # Multiplicações totais (convergência).
    stats_output.append(f"Operações de Divisão (AG): {global_op_counter.divisions}") # Divisões totais (convergência).
    stats_output.append(f"Avaliações para o 'melhor global' (AG): {avaliacoes_ag_melhor_solucao}") # Avaliações no momento do melhor global.
    stats_output.append(f"Multiplicações para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_mult}") # Multiplicações no momento do melhor global.
    stats_output.append(f"Divisões para o 'melhor global' (AG): {operacoes_ag_melhor_solucao_div}") # Divisões no momento do melhor global.
    stats_output.append("--------------------------------------------")

    for line in stats_output: # Imprime as linhas de resultado no console.
        print(line)

    # Salva as estatísticas em um arquivo .txt.
    output_folder = "resultados_ag" # Define o nome da pasta.
    os.makedirs(output_folder, exist_ok=True) # Cria a pasta se não existir.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Gera um timestamp.
    file_name = f"estatisticas_ag_{timestamp}.txt" # Define o nome do arquivo.
    output_path = os.path.join(output_folder, file_name) # Cria o caminho completo do arquivo.

    with open(output_path, "w") as f: # Abre o arquivo para escrita.
        for line in stats_output:
            f.write(line + "\n") # Escreve cada linha no arquivo.

    print(f"\nEstatísticas salvas em: {output_path}") # Informa onde o arquivo foi salvo.

    return melhor_solucao, melhor_aptidao, i # Retorna os principais resultados do algoritmo.

# Este bloco só será executado se genetico.py for o script principal rodado (ex: python genetico.py).
if __name__ == "__main__":
    print("\n--- Teste direto do Algoritmo Genético (se executado como script principal) ---")
    # Define os parâmetros de teste para uma execução direta.
    tamanho_populacao = 35
    limites = (-500, 500)
    num_geracoes = 200
    taxa_cruzamento = 0.7
    taxa_mutacao = 0.01
    geracoes_sem_melhora_limite = 50
    tolerancia = 1e-6

    # Chama a função do algoritmo genético com os parâmetros de teste.
    melhor_solucao, melhor_aptidao, geracao = algoritmo_genetico(
        tamanho_populacao=tamanho_populacao,
        limites=limites,
        num_geracoes=num_geracoes,
        taxa_cruzamento=taxa_cruzamento,
        taxa_mutacao=taxa_mutacao,
        geracoes_sem_melhora_limite=geracoes_sem_melhora_limite,
        tolerancia=tolerancia
    )