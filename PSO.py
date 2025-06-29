from Enxame import Enxame # Importa a classe Enxame, que representa uma partícula no algoritmo PSO.
from Grafico import GraficoPSO # Importa a função GraficoPSO, utilizada para plotar o gráfico do enxame.
import matplotlib.pyplot as plt # Importa a biblioteca matplotlib para criação de gráficos.
from funcoes_otimizacao import funcao_w4 # Importa a função de otimização específica, funcao_w4, que será avaliada pelo PSO.
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa o contador global de operações e um wrapper para funções objetivo.

class PSO: # Define a classe PSO (Particle Swarm Optimization).
    def __init__(self, limites, num_particulas, num_iteracoes): # Método construtor da classe PSO, recebe limites, número de partículas e número de iterações.
        # Resetar contadores antes de iniciar a otimização
        global_op_counter.reset() # Reseta o contador global de operações para garantir que as contagens sejam iniciadas do zero.
        self.funcao_w4_wrapper = FuncaoObjetivoWrapper(funcao_w4, global_op_counter) # Instancia um wrapper para a funcao_w4, que permitirá contar as avaliações e operações.

        melhor_valor_g = float('inf') # Inicializa o melhor valor global encontrado como infinito, para que qualquer valor inicial seja menor.
        melhor_posicao_g = [] # Inicializa a melhor posição global encontrada como uma lista vazia.
        self.avaliacoes_pso_minimo_global = 0 # Inicializa o contador de avaliações da função objetivo no momento do melhor global para o item b).
        self.operacoes_pso_minimo_global_mult = 0 # Inicializa o contador de operações de multiplicação no momento do melhor global para o item c).
        self.operacoes_pso_minimo_global_div = 0 # Inicializa o contador de operações de divisão no momento do melhor global para o item c).


        enxame = [] # Inicializa uma lista vazia para armazenar as partículas do enxame.
        for i in range(num_particulas): # Loop para criar o número especificado de partículas.
            enxame.append(Enxame()) # Adiciona uma nova instância da classe Enxame (partícula) à lista.

        fig = plt.figure(figsize=(8, 8)) # Cria uma nova figura matplotlib com um tamanho específico.
        ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura para visualização.

        i = 0 # Inicializa o contador de iterações.
        while i < num_iteracoes: # Loop principal do algoritmo PSO, continua enquanto o número de iterações não for atingido.
            for j in range(num_particulas): # Loop para iterar sobre cada partícula no enxame.
                enxame[j].avaliar(self.funcao_w4_wrapper) # Avalia a função objetivo para a partícula atual, usando o wrapper para contagem.

                if enxame[j].valor_atual_i < melhor_valor_g: # Verifica se o valor atual da partícula é melhor que o melhor valor global encontrado até agora.
                    melhor_posicao_g = list(enxame[j].posicao_i) # Atualiza a melhor posição global com a posição atual da partícula.
                    melhor_valor_g = float(enxame[j].valor_atual_i) # Atualiza o melhor valor global com o valor atual da partícula.
                    # Registra as contagens no momento em que o "melhor global" é atualizado
                    self.avaliacoes_pso_minimo_global = self.funcao_w4_wrapper.evaluations # Armazena o número de avaliações quando o melhor global foi encontrado.
                    self.operacoes_pso_minimo_global_mult = global_op_counter.multiplications # Armazena o número de multiplicações quando o melhor global foi encontrado.
                    self.operacoes_pso_minimo_global_div = global_op_counter.divisions # Armazena o número de divisões quando o melhor global foi encontrado.

            for j in range(num_particulas): # Loop para iterar sobre cada partícula novamente.
                enxame[j].atualizar_velocidade(melhor_posicao_g, i, num_iteracoes) # Atualiza a velocidade da partícula baseada na melhor posição global e na iteração.
                enxame[j].atualizar_posicao(limites) # Atualiza a posição da partícula, respeitando os limites.

            GraficoPSO(enxame, i+1, ax, melhor_valor_g) # Chama a função para atualizar o gráfico do enxame, passando o melhor valor global.
            i += 1 # Incrementa o contador de iterações.

        print(f'POSICAO FINAL: {melhor_posicao_g}') # Imprime a melhor posição final encontrada pelo PSO.
        print(f'RESULTADO FINAL: {melhor_valor_g}') # Imprime o melhor valor final da função objetivo encontrado pelo PSO.
        print(f'Avaliações da função objetivo (PSO): {self.funcao_w4_wrapper.evaluations}') # Imprime o total de avaliações da função objetivo durante a execução do PSO.
        print(f'Operações de Multiplicação (PSO): {global_op_counter.multiplications}') # Imprime o total de operações de multiplicação durante a execução do PSO.
        print(f'Operações de Divisão (PSO): {global_op_counter.divisions}') # Imprime o total de operações de divisão durante a execução do PSO.
        print(f'Avaliações para o "melhor global" (PSO): {self.avaliacoes_pso_minimo_global}') # Imprime o número de avaliações da função objetivo no momento em que o "melhor global" foi encontrado.
        print(f'Multiplicações para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_mult}') # Imprime o número de multiplicações no momento em que o "melhor global" foi encontrado.
        print(f'Divisões para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_div}') # Imprime o número de divisões no momento em que o "melhor global" foi encontrado.


        plt.show() # Exibe a janela com o gráfico animado do enxame.