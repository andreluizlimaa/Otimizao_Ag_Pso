from Enxame import Enxame # Importa a classe 'Enxame' do arquivo 'Enxame.py', que representa uma única partícula no enxame.
from Grafico import GraficoPSO # Importa a classe 'GraficoPSO' do arquivo 'Grafico.py', responsável por gerar a visualização 3D do processo de otimização.
import matplotlib.pyplot as plt # Importa a biblioteca 'matplotlib.pyplot' como 'plt', usada para criar e exibir gráficos.
from funcoes_otimizacao import funcao_w4 # Importa a 'funcao_w4' do módulo 'funcoes_otimizacao', que é a função objetivo que o PSO tentará minimizar.
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa 'global_op_counter' (o contador global de operações) e 'FuncaoObjetivoWrapper' (para envolver a função objetivo e contar suas avaliações) do módulo 'utils.py'.

class PSO: # Define a classe 'PSO', que encapsula a lógica principal do algoritmo de Otimização por Enxame de Partículas.
    def __init__(self, limites, num_particulas, num_iteracoes, iteracoes_sem_melhora_limite, tolerancia): # Define o método construtor da classe 'PSO'. Ele recebe vários parâmetros para configurar o algoritmo.
        # 'limites': Define o espaço de busca da função objetivo (ex: [(-500, 500), (-500, 500)] para x e y).
        # 'num_particulas': O número total de partículas no enxame.
        # 'num_iteracoes': O número máximo de iterações (gerações) que o algoritmo executará.
        # 'iteracoes_sem_melhora_limite': Limite de iterações consecutivas sem melhoria significativa antes de parar o PSO.
        # 'tolerancia': A pequena diferença aceitável para considerar que não houve melhoria significativa.

        # Resetar contadores antes de iniciar a otimização
        global_op_counter.reset() # Zera todos os contadores de operações (multiplicações, divisões, adições) para uma nova execução do PSO.
        # Cria uma instância de 'FuncaoObjetivoWrapper' para 'funcao_w4', permitindo contar as avaliações dessa função.
        self.funcao_w4_wrapper = FuncaoObjetivoWrapper(funcao_w4, global_op_counter)

        melhor_valor_g = float('inf') # Inicializa 'melhor_valor_g' (melhor valor da função objetivo encontrado por *qualquer* partícula no enxame) com infinito.
        melhor_posicao_g = [] # Inicializa 'melhor_posicao_g' (a posição correspondente ao 'melhor_valor_g') como uma lista vazia.
        # Variáveis para armazenar o estado dos contadores quando o "melhor global" é encontrado/atualizado.
        self.avaliacoes_pso_minimo_global = 0 # Contador de avaliações da função objetivo até o melhor global ser atingido.
        self.operacoes_pso_minimo_global_mult = 0 # Contador de multiplicações até o melhor global ser atingido.
        self.operacoes_pso_minimo_global_div = 0 # Contador de divisões até o melhor global ser atingido.
        # (Nota: Faltou 'operacoes_pso_minimo_global_add' aqui se você quiser registrar adições também para o mínimo global)

        # Variáveis para a decisão de parada por convergência no PSO
        iteracoes_sem_melhora = 0 # Contador de iterações consecutivas em que não há melhoria significativa no 'melhor_valor_g'.
        ultima_melhor_valor_g = float('inf') # Armazena o melhor valor global da iteração anterior para comparação de convergência.

        enxame = [] # Cria uma lista vazia que irá armazenar todas as partículas do enxame.
        for i in range(num_particulas): # Loop para criar o número especificado de partículas.
            # Passa os limites para o construtor da Enxame
            enxame.append(Enxame(limites)) # Cria uma nova instância de 'Enxame' (partícula) e a adiciona à lista 'enxame', passando os limites do espaço de busca.

        fig = plt.figure(figsize=(8, 8)) # Cria uma nova figura Matplotlib com um tamanho de 8x8 polegadas para o gráfico 3D.
        ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura. '111' significa 1 linha, 1 coluna, primeiro subplot.

        i = 0 # Inicializa o contador de iterações do PSO.
        while i < num_iteracoes: # Inicia o loop principal do PSO, que continua enquanto o número de iterações for menor que 'num_iteracoes'.
            # Captura o melhor valor global *antes* da atualização nesta iteração
            valor_global_antes_atualizacao = melhor_valor_g # Armazena o 'melhor_valor_g' da iteração anterior para a verificação de convergência.

            for j in range(num_particulas): # Loop sobre cada partícula no enxame.
                enxame[j].avaliar(self.funcao_w4_wrapper) # Chama o método 'avaliar' de cada partícula para calcular seu 'valor_atual_i' e atualizar seu 'melhor_valor_i' e 'melhor_posicao_i'.

                if enxame[j].valor_atual_i < melhor_valor_g: # Verifica se o valor atual da partícula é melhor (menor) que o 'melhor_valor_g' atual do enxame.
                    melhor_posicao_g = list(enxame[j].posicao_i) # Se for, atualiza 'melhor_posicao_g' com a posição da partícula atual. Converte para lista, pois 'posicao_i' é um array NumPy.
                    melhor_valor_g = float(enxame[j].valor_atual_i) # Se for, atualiza 'melhor_valor_g' com o novo melhor valor encontrado. Converte para float para consistência.
                    # Registra o estado dos contadores no momento em que o novo melhor global foi encontrado.
                    self.avaliacoes_pso_minimo_global = self.funcao_w4_wrapper.evaluations # Salva as avaliações da função objetivo.
                    self.operacoes_pso_minimo_global_mult = global_op_counter.multiplications # Salva as multiplicações.
                    self.operacoes_pso_minimo_global_div = global_op_counter.divisions # Salva as divisões.

            # Verifica a diferença absoluta entre o melhor valor global atual e o anterior.
            # Se a iteração atual for a primeira, não há 'anterior' para comparar.
            # A condição 'i > 0' garante que esta verificação não ocorra na primeira iteração.
            if i > 0 and abs(melhor_valor_g - valor_global_antes_atualizacao) < tolerancia:
                iteracoes_sem_melhora += 1 # Se a melhoria for menor que a tolerância, incrementa o contador de iterações sem melhoria.
            else:
                # Se houver uma melhora maior que a tolerância, ou se for a primeira iteração, reseta o contador
                iteracoes_sem_melhora = 0 # Se houve uma melhoria significativa, ou se é a primeira iteração, zera o contador.

            # Verifica se o limite de iterações sem melhoria foi atingido.
            if iteracoes_sem_melhora >= iteracoes_sem_melhora_limite: # Compara o contador com o limite definido.
                print(f"\n[PSO] Parada por convergência: Mudança no melhor valor global menor que {tolerancia} por {iteracoes_sem_melhora_limite} iterações.") # Imprime uma mensagem indicando a parada por convergência.
                break # Sai do loop principal do PSO se a convergência for detectada

            for j in range(num_particulas): # Loop sobre cada partícula para atualizar sua velocidade e posição.
                enxame[j].atualizar_velocidade(melhor_posicao_g, i, num_iteracoes) # Atualiza a velocidade da partícula, usando a melhor posição global e o progresso da iteração.
                enxame[j].atualizar_posicao(limites) # Atualiza a posição da partícula com base na nova velocidade, respeitando os limites do espaço de busca.

            GraficoPSO(enxame, i+1, ax, melhor_valor_g) # Chama a função para atualizar o gráfico 3D, mostrando as partículas e a melhor posição global.
            i += 1 # Incrementa o contador de iterações.

        # --- Impressão dos Resultados Finais do PSO ---
        print(f'POSICAO FINAL (PSO): {melhor_posicao_g}') # Imprime a melhor posição (x, y) encontrada pelo enxame.
        print(f'RESULTADO FINAL (PSO): {melhor_valor_g}') # Imprime o melhor valor da função objetivo (aptidão) encontrado.
        print(f'Iterações executadas (PSO): {i}') # Imprime o número total de iterações executadas.
        print(f'Avaliações da função objetivo (PSO): {self.funcao_w4_wrapper.evaluations}') # Imprime o total de vezes que a função objetivo foi avaliada.
        print(f'Operações de Multiplicação (PSO): {global_op_counter.multiplications}') # Imprime o total de operações de multiplicação contadas.
        print(f'Operações de Divisão (PSO): {global_op_counter.divisions}') # Imprime o total de operações de divisão contadas.
        print(f'Avaliações para o "melhor global" (PSO): {self.avaliacoes_pso_minimo_global}') # Imprime o número de avaliações quando o melhor global foi atingido.
        print(f'Multiplicações para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_mult}') # Imprime o número de multiplicações quando o melhor global foi atingido.
        print(f'Divisões para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_div}') # Imprime o número de divisões quando o melhor global foi atingido.

        plt.show() # Exibe a janela do gráfico 3D ao final da execução do PSO.

# Este bloco só será executado se PSO.py for o script principal rodado (ex: python PSO.py).
if __name__ == "__main__": # Este é um padrão comum em Python. O código dentro deste bloco só é executado se o arquivo 'PSO.py' for o script principal executado diretamente.
    print("\n--- Teste direto da Otimização por Enxame de Partículas (se executado como script principal) ---")
    
    # Define os parâmetros para uma execução de teste direto do PSO.
    limites_xy = [(-500, 500), (-500, 500)] # Define os limites X e Y para o espaço de busca da função objetivo.
    num_particulas_pso = 15 # Define o número de partículas para esta execução de teste.
    num_iteracoes_pso = 100 # Define o número máximo de iterações para esta execução de teste.
    iteracoes_sem_melhora_limite_pso = 50 # Define o limite de iterações sem melhoria para esta execução de teste.
    tolerancia_pso = 1e-6 # Define a tolerância para a verificação de convergência para esta execução de teste.

    # Cria uma instância da classe PSO e inicia a otimização
    pso_instance = PSO( # Cria uma nova instância da classe 'PSO'.
        limites=limites_xy, # Passa os limites do espaço de busca.
        num_particulas=num_particulas_pso, # Passa o número de partículas.
        num_iteracoes=num_iteracoes_pso, # Passa o número de iterações.
        iteracoes_sem_melhora_limite=iteracoes_sem_melhora_limite_pso, # Passa o limite de iterações sem melhoria.
        tolerancia=tolerancia_pso # Passa a tolerância de convergência.
    ) # Ao criar a instância, o método __init__ é chamado e o processo de otimização PSO é iniciado.