import numpy as np # Importa a biblioteca NumPy, fundamental para operações numéricas e manipulação de arrays.
import matplotlib.pyplot as plt # Importa a biblioteca Matplotlib, usada para criar gráficos e visualizações.
from mpl_toolkits.mplot3d import Axes3D # Importa Axes3D do Matplotlib para permitir a criação de gráficos 3D.
from funcoes_otimizacao import funcao_w4 # Importa a função de otimização 'funcao_w4', que será a base para a superfície do gráfico.

# --- Função de Gráfico para PSO ---
def GraficoPSO(enxame, iteracao, ax, melhor_valor_global): # Define a função para plotar o gráfico do PSO, recebendo o enxame, a iteração atual, o objeto Axes3D e o melhor valor global.
    ax.clear() # Limpa o conteúdo atual do eixo 3D para desenhar a nova iteração.

    # Definir o range do meshgrid para plotagem
    x_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo x.
    y_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo y.
    X, Y = np.meshgrid(x_grid, y_grid) # Cria uma grade de coordenadas a partir dos arrays x_grid e y_grid.

    # Calcular Z usando a função W4
    Z = funcao_w4(X, Y) # Calcula os valores Z da função 'funcao_w4' para cada ponto da grade (X, Y), formando a superfície.

    # Título do gráfico com o melhor valor global
    ax.set_title(f'PSO - Iteração {iteracao} | Melhor Valor: {melhor_valor_global:.4f}') # Define o título do gráfico, mostrando a iteração atual e o melhor valor global encontrado formatado.
    ax.set_xlabel('x') # Define o rótulo do eixo x.
    ax.set_ylabel('y') # Define o rótulo do eixo y.
    ax.set_zlabel('F(x, y)') # Define o rótulo do eixo z, representando o valor da função.

    # Garantir que os limites dos eixos X e Y correspondam ao meshgrid
    ax.set_xlim([-500, 500]) # Define os limites do eixo x para corresponder ao meshgrid.
    ax.set_ylim([-500, 500]) # Define os limites do eixo y para corresponder ao meshgrid.

    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.4, rstride=5, cstride=5) # Plota a superfície 3D da função, com cor cinza claro, transparência e passos de grade.

    # Plotar as partículas do enxame (mantido como antes)
    colors = ['red', 'yellow', 'green', 'blue', 'pink', 'purple', 'orange', 'black', 'gray', 'brown'] # Define uma lista de cores para as partículas.
    for idx, particula in enumerate(enxame): # Itera sobre cada partícula no enxame.
        particula_x = particula.posicao_i[0] # Pega a coordenada x da posição atual da partícula.
        particula_y = particula.posicao_i[1] # Pega a coordenada y da posição atual da partícula.
        particle_z_val = funcao_w4(np.array(particula_x), np.array(particula_y)) # Calcula o valor Z da função para a posição da partícula.
        color = colors[idx % len(colors)] # Seleciona uma cor da lista, ciclando por elas.
        ax.scatter(particula_x, particula_y, particle_z_val, color=color, s=100, edgecolors='black', linewidth=0.5) # Plota a partícula como um ponto 3D na superfície.

    # Definir limites para o eixo Z
    ax.set_zlim([-600, 4000]) # Define os limites do eixo z para melhor visualização.

    plt.pause(0.01) # Pausa por um curto período para permitir a atualização do gráfico e criar a animação.

# --- Função de Gráfico para AG ---
def GraficoAG(populacao, melhor_solucao, iteracao, ax, melhor_aptidao_global): # Define a função para plotar o gráfico do AG, recebendo a população, a melhor solução, a iteração atual, o objeto Axes3D e a melhor aptidão global.
    ax.clear() # Limpa o conteúdo atual do eixo 3D para desenhar a nova geração.

    # Definir o range do meshgrid para plotagem
    x_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo x.
    y_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo y.
    X, Y = np.meshgrid(x_grid, y_grid) # Cria uma grade de coordenadas a partir dos arrays x_grid e y_grid.

    # Calcular Z usando a função W4
    Z = funcao_w4(X, Y) # Calcula os valores Z da função 'funcao_w4' para cada ponto da grade (X, Y), formando a superfície.

    # Título do gráfico com o melhor valor global
    ax.set_title(f'AG - Geração {iteracao} | Melhor Valor: {melhor_aptidao_global:.4f}') # Define o título do gráfico, mostrando a geração atual e o melhor valor de aptidão global encontrado.
    ax.set_xlabel('x') # Define o rótulo do eixo x.
    ax.set_ylabel('y') # Define o rótulo do eixo y.
    ax.set_zlabel('F(x, y)') # Define o rótulo do eixo z, representando o valor da função.

    ax.set_xlim([-500, 500]) # Define os limites do eixo x para corresponder ao meshgrid.
    ax.set_ylim([-500, 500]) # Define os limites do eixo y para corresponder ao meshgrid.

    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.5, rstride=5, cstride=5) # Plota a superfície 3D da função, com cor cinza claro, transparência e passos de grade.

    # Plotar a população do AG
    population_color = 'cyan' # Define a cor para os indivíduos da população.
    best_solution_color = 'red' # Define a cor para a melhor solução.
    for individuo in populacao: # Itera sobre cada indivíduo na população.
        ind_x, ind_y = individuo[0], individuo[1] # Pega as coordenadas x e y do indivíduo.
        ind_z_val = funcao_w4(np.array(ind_x), np.array(ind_y)) # Calcula o valor Z da função para a posição do indivíduo.
        ax.scatter(ind_x, ind_y, ind_z_val, color=population_color, s=50, alpha=0.7) # Plota o indivíduo como um ponto 3D na superfície.

    # Plotar a melhor solução encontrada
    if melhor_solucao is not None: # Verifica se uma melhor solução global foi encontrada.
        best_x, best_y = melhor_solucao[0], melhor_solucao[1] # Pega as coordenadas x e y da melhor solução.
        best_z_val = funcao_w4(np.array(best_x), np.array(best_y)) # Calcula o valor Z da função para a melhor solução.
        ax.scatter(best_x, best_y, best_z_val, color=best_solution_color, s=200, marker='o', edgecolor='black', linewidth=1.5, label='Melhor Solução Global') # Plota a melhor solução como um ponto maior e distinto.

    # Definir limites para o eixo Z
    ax.set_zlim([-500, 4000]) # Define os limites do eixo z para melhor visualização.

    plt.pause(0.01) # Pausa por um curto período para permitir a atualização do gráfico e criar a animação.