# Enxame.py

import random # Importa o módulo random para gerar números aleatórios.
import numpy as np # Importa a biblioteca NumPy, geralmente usada para operações numéricas eficientes, embora não diretamente utilizada em todas as linhas deste código.
from utils import global_op_counter # Importa o objeto global_op_counter do módulo 'utils', usado para rastrear operações.

class Enxame: # Define a classe Enxame, que representa uma partícula no algoritmo de Otimização por Enxame de Partículas (PSO).
    def __init__(self): # Método construtor da classe Enxame.
        self.posicao_i = [] # Inicializa uma lista vazia para armazenar a posição atual da partícula.
        self.velocidade_i = [] # Inicializa uma lista vazia para armazenar a velocidade atual da partícula.
        self.melhor_posicao_i = [] # Inicializa uma lista vazia para armazenar a melhor posição individual encontrada pela partícula até o momento.
        self.melhor_valor_i = float('inf') # Inicializa o melhor valor (custo) encontrado pela partícula como infinito, para garantir que qualquer valor inicial seja menor (focado em minimização).
        self.valor_atual_i = float('inf') # Inicializa o valor (custo) atual da posição da partícula como infinito.

        for i in range(2): # Loop que se repete 2 vezes, pois a função 'w4' é 2D (x, y).
            self.velocidade_i.append(random.uniform(-1, 1)) # Atribui um valor de velocidade aleatório entre -1 e 1 para cada dimensão.
            self.posicao_i.append(random.uniform(-500, 500)) # Atribui uma posição inicial aleatória entre -500 e 500 para cada dimensão.

    def avaliar(self, funcao_wrapper): # Método para avaliar a aptidão da posição atual da partícula, recebendo um 'funcao_wrapper'.
        x = self.posicao_i[0] # Atribui a coordenada x da posição atual da partícula.
        y = self.posicao_i[1] # Atribui a coordenada y da posição atual da partícula.

        self.valor_atual_i = funcao_wrapper(x, y) # Chama a função de aptidão (através do wrapper) com as coordenadas x e y, e armazena o resultado em valor_atual_i. O wrapper também cuida da contagem de operações e avaliações.

        if self.valor_atual_i < self.melhor_valor_i: # Verifica se o valor de aptidão atual é melhor (menor) do que o melhor valor individual encontrado até agora.
            self.melhor_posicao_i = self.posicao_i.copy() # Se for melhor, atualiza a melhor_posicao_i com a cópia da posição atual.
            self.melhor_valor_i = self.valor_atual_i # Se for melhor, atualiza o melhor_valor_i com o valor de aptidão atual.

    def atualizar_velocidade(self, pos_best_g, iteracao_atual, num_iteracoes): # Método para atualizar a velocidade da partícula.
        # AQUI COMEÇA A CONTAGEM DE MULTIPLICAÇÕES E DIVISÕES DO PSO

        # w = 0.9 - iteracao_atual*((0.9 - 0.4)/num_iteracoes) # Comentário da linha original que calcula o peso de inércia 'w'.
        global_op_counter.add_mult(1) # Adiciona 1 à contagem de multiplicações para a operação (0.9 - 0.4) * iteracao_atual.
        global_op_counter.add_div(1)  # Adiciona 1 à contagem de divisões para a operação resultado / num_iteracoes.
        w = 0.9 - iteracao_atual*((0.9 - 0.4)/num_iteracoes) # Calcula o peso de inércia 'w', que diminui linearmente ao longo das iterações.

        c1 = 1 # Define a constante de aceleração cognitiva (influência da melhor posição individual).
        c2 = 1 # Define a constante de aceleração social (influência da melhor posição global).

        for i in range(2): # Loop para atualizar a velocidade para cada uma das 2 dimensões.
            r1 = random.random() # Gera um número aleatório entre 0 e 1 para a componente cognitiva.
            r2 = random.random() # Gera um número aleatório entre 0 e 1 para a componente social.

            # vel_cognitiva = c1 * r1 * (self.melhor_posicao_i[i] - self.posicao_i[i]) # Comentário da linha original que calcula a velocidade cognitiva.
            global_op_counter.add_mult(2) # Adiciona 2 à contagem de multiplicações: c1 * r1 e o resultado * (melhor_posicao_i[i] - posicao_i[i]).
            vel_cognitiva = c1 * r1 * (self.melhor_posicao_i[i] - self.posicao_i[i]) # Calcula a componente cognitiva da velocidade, que puxa a partícula em direção à sua melhor posição individual.

            # vel_social = c2 * r2 * (pos_best_g[i] - self.posicao_i[i]) # Comentário da linha original que calcula a velocidade social.
            global_op_counter.add_mult(2) # Adiciona 2 à contagem de multiplicações: c2 * r2 e o resultado * (pos_best_g[i] - posicao_i[i]).
            vel_social = c2 * r2 * (pos_best_g[i] - self.posicao_i[i]) # Calcula a componente social da velocidade, que puxa a partícula em direção à melhor posição global encontrada pelo enxame.

            # self.velocidade_i[i] = w * self.velocidade_i[i] + vel_cognitiva + vel_social # Comentário da linha original que atualiza a velocidade.
            global_op_counter.add_mult(1) # Adiciona 1 à contagem de multiplicações para w * velocidade_i[i].
            self.velocidade_i[i] = w * self.velocidade_i[i] + vel_cognitiva + vel_social # Atualiza a velocidade da partícula combinando a velocidade anterior (influenciada pelo peso de inércia), a componente cognitiva e a componente social.

    def atualizar_posicao(self, limites): # Método para atualizar a posição da partícula, considerando os limites do espaço de busca.
        # Nenhuma multiplicação/divisão direta significativa aqui, apenas soma e comparações # Comentário indicando que esta seção não tem muitas operações complexas para contar.
        for i in range(2): # Loop para atualizar a posição para cada uma das 2 dimensões.
            self.posicao_i[i] = self.posicao_i[i] + self.velocidade_i[i] # Atualiza a posição da partícula somando a velocidade atual à sua posição.

            if self.posicao_i[i] > limites[i][1]: # Verifica se a posição atual excede o limite superior para a dimensão 'i'.
                self.posicao_i[i] = limites[i][1] # Se exceder, "prende" a partícula no limite superior.
            if self.posicao_i[i] < limites[i][0]: # Verifica se a posição atual está abaixo do limite inferior para a dimensão 'i'.
                self.posicao_i[i] = limites[i][0] # Se estiver abaixo, "prende" a partícula no limite inferior.
                self.velocidade_i[i] = 0 # Anula a velocidade da partícula na dimensão 'i' se ela atingir o limite, para evitar que saia do espaço de busca.