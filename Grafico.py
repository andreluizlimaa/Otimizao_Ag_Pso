import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from funcoes_otimizacao import funcao_w4 # Certifique-se que esta é a versão corrigida

# --- Função de Gráfico para PSO ---
def GraficoPSO(enxame, iteracao, ax, melhor_valor_global): # Adicionado melhor_valor_global
    ax.clear()

    # Definir o range do meshgrid para plotagem
    x_grid = np.linspace(-500, 500, 100)
    y_grid = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Calcular Z usando a função W4
    Z = funcao_w4(X, Y)

    # Título do gráfico com o melhor valor global
    ax.set_title(f'PSO - Iteração {iteracao} | Melhor Valor: {melhor_valor_global:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('F(x, y)')

    # Garantir que os limites dos eixos X e Y correspondam ao meshgrid
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])

    # --- Opção 1: Superfície com cor sólida (sem cmap) ---
    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.4, rstride=5, cstride=5)
    
    # --- Opção 2 (Alternativa, se quiser apenas o wireframe sem a superfície preenchida): ---
    # ax.plot_wireframe(X, Y, Z, color='gray', rstride=5, cstride=5, linewidth=0.7)


    # Plotar as partículas do enxame (mantido como antes)
    colors = ['red', 'yellow', 'green', 'blue', 'pink', 'purple', 'orange', 'black', 'gray', 'brown']
    for idx, particula in enumerate(enxame):
        particula_x = particula.posicao_i[0]
        particula_y = particula.posicao_i[1]
        particle_z_val = funcao_w4(np.array(particula_x), np.array(particula_y))
        color = colors[idx % len(colors)]
        ax.scatter(particula_x, particula_y, particle_z_val, color=color, s=100, edgecolors='black', linewidth=0.5)

    # Definir limites para o eixo Z
    ax.set_zlim([-600, 4000])

    plt.pause(0.01)

# --- NOVA FUNÇÃO DE GRÁFICO PARA AG ---
def GraficoAG(populacao, melhor_solucao, iteracao, ax, melhor_aptidao_global): # Adicionado melhor_aptidao_global
    ax.clear()

    # Definir o range do meshgrid para plotagem
    x_grid = np.linspace(-500, 500, 100)
    y_grid = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Calcular Z usando a função W4
    Z = funcao_w4(X, Y)

    # Título do gráfico com o melhor valor global
    ax.set_title(f'AG - Geração {iteracao} | Melhor Valor: {melhor_aptidao_global:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('F(x, y)')

    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])

    # --- Opção 1: Superfície com cor sólida (sem cmap) ---
    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.5, rstride=5, cstride=5)
    
    # --- Opção 2 (Alternativa, se quiser apenas o wireframe sem a superfície preenchida): ---
    # ax.plot_wireframe(X, Y, Z, color='gray', rstride=5, cstride=5, linewidth=0.7)

    # Plotar a população do AG (mantido como antes)
    population_color = 'cyan'
    best_solution_color = 'red'
    for individuo in populacao:
        ind_x, ind_y = individuo[0], individuo[1]
        ind_z_val = funcao_w4(np.array(ind_x), np.array(ind_y))
        ax.scatter(ind_x, ind_y, ind_z_val, color=population_color, s=50, alpha=0.7)

    # Plotar a melhor solução encontrada (mantido como antes)
    if melhor_solucao is not None:
        best_x, best_y = melhor_solucao[0], melhor_solucao[1]
        best_z_val = funcao_w4(np.array(best_x), np.array(best_y))
        ax.scatter(best_x, best_y, best_z_val, color=best_solution_color, s=200, marker='o', edgecolor='black', linewidth=1.5, label='Melhor Solução Global')

    # Definir limites para o eixo Z
    ax.set_zlim([-500, 4000])

    plt.pause(0.01)