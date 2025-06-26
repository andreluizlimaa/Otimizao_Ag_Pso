# Enxame.py

import random
import numpy as np
from utils import global_op_counter # Importa o contador

class Enxame:
    def __init__(self):
        self.posicao_i = []
        self.velocidade_i = []
        self.melhor_posicao_i = []
        self.melhor_valor_i = float('inf') # Melhorado para minimização
        self.valor_atual_i = float('inf') # Melhorado

        for i in range(2): # Sua função w4 é 2D (x, y)
            self.velocidade_i.append(random.uniform(-1, 1))
            self.posicao_i.append(random.uniform(-500, 500))

    def avaliar(self, funcao_wrapper): # Recebe o wrapper da função
        x = self.posicao_i[0]
        y = self.posicao_i[1]

        self.valor_atual_i = funcao_wrapper(x, y) # A chamada ao wrapper conta ops e avaliações

        if self.valor_atual_i < self.melhor_valor_i:
            self.melhor_posicao_i = self.posicao_i.copy()
            self.melhor_valor_i = self.valor_atual_i

    def atualizar_velocidade(self, pos_best_g, iteracao_atual, num_iteracoes):
        # AQUI COMEÇA A CONTAGEM DE MULTIPLICAÇÕES E DIVISÕES DO PSO

        # w = 0.9 - iteracao_atual*((0.9 - 0.4)/num_iteracoes)
        global_op_counter.add_mult(1) # (0.9 - 0.4) * iteracao_atual
        global_op_counter.add_div(1)  # resultado / num_iteracoes
        w = 0.9 - iteracao_atual*((0.9 - 0.4)/num_iteracoes)

        c1 = 1
        c2 = 1

        for i in range(2): # Para as 2 dimensões (x, y)
            r1 = random.random()
            r2 = random.random()

            # vel_cognitiva = c1 * r1 * (self.melhor_posicao_i[i] - self.posicao_i[i])
            global_op_counter.add_mult(2) # c1 * r1, e o resultado * (p - x)
            vel_cognitiva = c1 * r1 * (self.melhor_posicao_i[i] - self.posicao_i[i])

            # vel_social = c2 * r2 * (pos_best_g[i] - self.posicao_i[i])
            global_op_counter.add_mult(2) # c2 * r2, e o resultado * (g - x)
            vel_social = c2 * r2 * (pos_best_g[i] - self.posicao_i[i])

            # self.velocidade_i[i] = w * self.velocidade_i[i] + vel_cognitiva + vel_social
            global_op_counter.add_mult(1) # w * velocidade_i[i]
            self.velocidade_i[i] = w * self.velocidade_i[i] + vel_cognitiva + vel_social

    def atualizar_posicao(self, limites):
        # Nenhuma multiplicação/divisão direta significativa aqui, apenas soma e comparações
        for i in range(2): # Para as 2 dimensões (x, y)
            self.posicao_i[i] = self.posicao_i[i] + self.velocidade_i[i]

            if self.posicao_i[i] > limites[i][1]:
                self.posicao_i[i] = limites[i][1]
            if self.posicao_i[i] < limites[i][0]:
                self.posicao_i[i] = limites[i][0]
                self.velocidade_i[i] = 0 # Anular velocidade ao bater no limite