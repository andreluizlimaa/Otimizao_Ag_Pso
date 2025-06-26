# pso.py

from Enxame import Enxame
from Grafico import GraficoPSO # Importa a função de gráfico
import matplotlib.pyplot as plt
from funcoes_otimizacao import funcao_w4
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa o contador e o wrapper

class PSO:
    def __init__(self, limites, num_particulas, num_iteracoes):
        # Resetar contadores antes de iniciar a otimização
        global_op_counter.reset()
        self.funcao_w4_wrapper = FuncaoObjetivoWrapper(funcao_w4, global_op_counter) # Instancia o wrapper

        melhor_valor_g = float('inf')
        melhor_posicao_g = []
        self.avaliacoes_pso_minimo_global = 0 # Para o item b)
        self.operacoes_pso_minimo_global_mult = 0 # Para o item c)
        self.operacoes_pso_minimo_global_div = 0 # Para o item c)


        enxame = []
        for i in range(num_particulas):
            enxame.append(Enxame())

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        i = 0
        while i < num_iteracoes:
            for j in range(num_particulas):
                enxame[j].avaliar(self.funcao_w4_wrapper) # Usa o wrapper

                if enxame[j].valor_atual_i < melhor_valor_g:
                    melhor_posicao_g = list(enxame[j].posicao_i)
                    melhor_valor_g = float(enxame[j].valor_atual_i)
                    # Registra as contagens no momento em que o "melhor global" é atualizado
                    self.avaliacoes_pso_minimo_global = self.funcao_w4_wrapper.evaluations
                    self.operacoes_pso_minimo_global_mult = global_op_counter.multiplications
                    self.operacoes_pso_minimo_global_div = global_op_counter.divisions

            for j in range(num_particulas):
                enxame[j].atualizar_velocidade(melhor_posicao_g, i, num_iteracoes)
                enxame[j].atualizar_posicao(limites)

            GraficoPSO(enxame, i+1, ax, melhor_valor_g) # Passando o melhor_valor_g
            i += 1

        print(f'POSICAO FINAL: {melhor_posicao_g}')
        print(f'RESULTADO FINAL: {melhor_valor_g}')
        print(f'Avaliações da função objetivo (PSO): {self.funcao_w4_wrapper.evaluations}')
        print(f'Operações de Multiplicação (PSO): {global_op_counter.multiplications}')
        print(f'Operações de Divisão (PSO): {global_op_counter.divisions}')
        print(f'Avaliações para o "melhor global" (PSO): {self.avaliacoes_pso_minimo_global}')
        print(f'Multiplicações para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_mult}')
        print(f'Divisões para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_div}')


        plt.show()