# grafico.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Importa sua função w4 do novo arquivo
from funcoes_otimizacao import funcao_w4

# --- Funções auxiliares de w4 (não duplicar, pois já são importadas) ---
# Se você não criou funcoes_otimizacao.py, você teria que colar elas aqui.
# Como criamos, elas não precisam estar aqui diretamente.


# --- Função de Gráfico para PSO ---
def GraficoPSO(enxame, iteracao, ax): # Removi 'funcao' do argumento, pois agora importamos funcao_w4
    ax.clear()

    # CRIAR A MALHA PARA X, Y
    x_grid = np.linspace(-500, 500, 100)
    y_grid = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # CALCULA A FUNÇÃO NA MALHA X, Y usando a função importada
    Z = funcao_w4(X, Y)

    # CONFIGURAÇÕES DO GRÁFICO
    ax.set_title(f'PSO - Iteração {iteracao}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('F(x, y)')

    # PLOTAGEM DA SUPERFÍCIE
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, rstride=1, cstride=1)


    # LISTA DE CORES PREDEFINIDAS
    colors = ['red', 'yellow', 'green', 'blue', 'pink', 'purple', 'orange', 'black']

    # PLOTAGEM DAS PARTÍCULAS COM CORES PREDEFINIDAS
    for idx, particula in enumerate(enxame):
        particula_x = particula.posicao_i[0]
        particula_y = particula.posicao_i[1]

        # CHAMA A FUNÇÃO COM X, Y DA PARTÍCULA
        particle_z_val = funcao_w4(particula_x, particula_y)

        # ESCOLHE A COR PARA A PARTICULA
        color = colors[idx % len(colors)]

        # PLOTA A PARTICULA
        ax.scatter(particula_x, particula_y, particle_z_val, color=color, s=100)

    # Ajusta os limites Z para garantir que o gráfico seja bem visualizado para a função w4
    ax.set_zlim(-550, 50) # Ajuste manual, teste para sua função

    plt.pause(0.1)

# --- NOVA FUNÇÃO DE GRÁFICO PARA AG ---
def GraficoAG(populacao, melhor_solucao, iteracao, ax): # Removi 'funcao' do argumento
    ax.clear()

    # CRIAR A MALHA PARA X, Y
    x_grid = np.linspace(-500, 500, 100)
    y_grid = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # CALCULA A FUNÇÃO NA MALHA X, Y
    Z = funcao_w4(X, Y) # Usa a função importada

    # CONFIGURAÇÕES DO GRÁFICO
    ax.set_title(f'AG - Geração {iteracao}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('F(x, y)')

    # PLOTAGEM DA SUPERFÍCIE
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.5, rstride=1, cstride=1)

    # PLOTAGEM DOS INDIVÍDUOS DA POPULAÇÃO
    population_color = 'cyan'
    best_solution_color = 'red'

    for individuo in populacao:
        ind_x, ind_y = individuo[0], individuo[1]
        ind_z_val = funcao_w4(ind_x, ind_y) # Usa a função importada
        ax.scatter(ind_x, ind_y, ind_z_val, color=population_color, s=50, alpha=0.7)

    # PLOTAGEM DA MELHOR SOLUÇÃO GLOBAL ENCONTRADA ATÉ AGORA
    if melhor_solucao is not None:
        best_x, best_y = melhor_solucao[0], melhor_solucao[1]
        best_z_val = funcao_w4(best_x, best_y) # Usa a função importada
        ax.scatter(best_x, best_y, best_z_val, color=best_solution_color, s=200, marker='o', edgecolor='black', linewidth=1.5, label='Melhor Solução Global')

    ax.set_zlim(-550, 50)

    plt.pause(0.1)