# pso.py
from Enxame import Enxame
from Grafico import GraficoPSO # Agora importa GraficoPSO
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Importa sua função w4
from funcoes_otimizacao import funcao_w4

# VARIÁVEL GLOBAL PARA CONTROLAR O ESTADO DE PAUSA
pausado = False

# FUNÇÃO CHAMADA PELO BOTÃO PAUSE
def alternar_pausa(event):
    global pausado
    pausado = not pausado

class PSO:
    def __init__(self, limites, num_particulas, num_iteracoes): # Removi 'funcao' do argumento
        melhor_valor_g = float('inf')
        melhor_posicao_g = []

        enxame = []
        for i in range(num_particulas):
            enxame.append(Enxame())

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        pause_posicao = plt.axes([0.85, 0.03, 0.1, 0.05])
        pause_botao = Button(pause_posicao, 'Pause')
        pause_botao.hovercolor = 'lightgreen'
        pause_botao.on_clicked(alternar_pausa)

        i = 0
        while i < num_iteracoes:
            for j in range(num_particulas):
                enxame[j].avaliar(funcao_w4) # Usa funcao_w4 importada

                if enxame[j].valor_atual_i < melhor_valor_g:
                    melhor_posicao_g = list(enxame[j].posicao_i)
                    melhor_valor_g = float(enxame[j].valor_atual_i)

            for j in range(num_particulas):
                enxame[j].atualizar_velocidade(melhor_posicao_g, i, num_iteracoes)
                enxame[j].atualizar_posicao(limites)

            GraficoPSO(enxame, i+1, ax) # Agora chama GraficoPSO sem a função como argumento

            while pausado:
                plt.pause(0.1)

            i += 1

        print(f'POSICAO FINAL: {melhor_posicao_g}')
        print(f'RESULTADO FINAL: {melhor_valor_g}')

        plt.show()