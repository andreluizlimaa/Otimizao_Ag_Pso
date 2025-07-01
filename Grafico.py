import numpy as np # Importa a biblioteca NumPy, fundamental para operações numéricas e manipulação de arrays.
import matplotlib.pyplot as plt # Importa a biblioteca Matplotlib, usada para criar gráficos e visualizações.
from mpl_toolkits.mplot3d import Axes3D # Importa Axes3D do Matplotlib para permitir a criação de gráficos 3D.
from funcoes_otimizacao import funcao_w4 # Importa a função de otimização 'funcao_w4', que será a base para a superfície do gráfico.

# --- Função de Gráfico para PSO ---
# Adicionado 'pso_params' como um novo argumento para receber os parâmetros e estatísticas do PSO.
def GraficoPSO(enxame, iteracao, ax, melhor_valor_global, pso_params=None):
    """
    Função para plotar o gráfico do PSO (Particle Swarm Optimization).
    Recebe o enxame de partículas, a iteração atual, o objeto Axes3D,
    o melhor valor global encontrado e um dicionário opcional com parâmetros e estatísticas do PSO.
    """
    ax.clear() # Limpa o conteúdo atual do eixo 3D para desenhar a nova iteração.

    # Definir o range do meshgrid para plotagem da superfície da função.
    x_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo x.
    y_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo y.
    X, Y = np.meshgrid(x_grid, y_grid) # Cria uma grade de coordenadas a partir dos arrays x_grid e y_grid.

    # Calcular Z usando a função W4
    Z = funcao_w4(X, Y) # Calcula os valores Z da função 'funcao_w4' para cada ponto da grade (X, Y), formando a superfície.

    # Título principal do gráfico com a iteração e o melhor valor global
    ax.set_title(f'PSO - Iteração {iteracao} | Melhor Valor: {melhor_valor_global:.4f}', fontsize=12) # Define o título do gráfico, focando na iteração e no melhor valor.
    
    ax.set_xlabel('x') # Define o rótulo do eixo x.
    ax.set_ylabel('y') # Define o rótulo do eixo y.
    ax.set_zlabel('F(x, y)') # Define o rótulo do eixo z, representando o valor da função.

    # Garantir que os limites dos eixos X e Y correspondam ao meshgrid
    ax.set_xlim([-500, 500]) # Define os limites do eixo x para corresponder ao meshgrid.
    ax.set_ylim([-500, 500]) # Define os limites do eixo y para corresponder ao meshgrid.

    # Plota a superfície 3D da função, com cor cinza claro, transparência e passos de grade.
    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.4, rstride=5, cstride=5)

    # Plotar as partículas do enxame
    colors = ['red', 'yellow', 'green', 'blue', 'pink', 'purple', 'orange', 'black', 'gray', 'brown'] # Define uma lista de cores para as partículas.
    for idx, particula in enumerate(enxame): # Itera sobre cada partícula no enxame, obtendo seu índice e o objeto partícula.
        particula_x = particula.posicao_i[0] # Pega a coordenada x da posição atual da partícula.
        particula_y = particula.posicao_i[1] # Pega a coordenada y da posição atual da partícula.
        # Calcula o valor Z da função para a posição da partícula (garantindo que seja um array NumPy).
        particle_z_val = funcao_w4(np.array(particula_x), np.array(particula_y))
        color = colors[idx % len(colors)] # Seleciona uma cor da lista, ciclando por elas usando o operador de módulo.
        # Plota a partícula como um ponto 3D na superfície.
        ax.scatter(particula_x, particula_y, particle_z_val, color=color, s=100, edgecolors='black', linewidth=0.5)

    # Definir limites para o eixo Z
    ax.set_zlim([-600, 4000]) # Define os limites do eixo z para melhor visualização da função W4.

    # --- Criação da Legenda no Lado Direito ---
    if pso_params: # Verifica se o dicionário 'pso_params' foi fornecido.
        # Constrói o texto da legenda formatado
        legend_text = (
            f'--- Parâmetros PSO ---\n'
            f'C1: {pso_params["c1"]:.2f}\n'
            f'C2: {pso_params["c2"]:.2f}\n'
            f'W_max: {pso_params["w_max"]:.2f}\n'
            f'W_min: {pso_params["w_min"]:.2f}\n'
            f'Partículas: {pso_params["num_particulas"]}\n'
            f'Iterações Totais: {pso_params["iteracoes_totais"]}\n\n'
            f'--- Estatísticas Totais ---\n'
            f'Avaliações FO: {pso_params["avaliacoes_funcao"]}\n'
            f'Mult: {pso_params["multiplicacoes_total"]}\n'
            f'Div: {pso_params["divisoes_total"]}\n\n'
            f'--- Estatísticas no Melhor Global ---\n'
            f'Avaliações: {pso_params["avaliacoes_minimo_global"]}\n'
            f'Mult: {pso_params["multiplicacoes_minimo_global"]}\n'
            f'Div: {pso_params["divisoes_minimo_global"]}'
        )
        
        # Adiciona o bloco de texto ao plot
        # 'bbox_to_anchor=(x, y)' posiciona o canto superior esquerdo do box em relação aos eixos.
        # 'transform=ax.transAxes' significa que (0,0) é o canto inferior esquerdo do plot e (1,1) é o superior direito.
        # 'fontsize', 'verticalalignment', 'horizontalalignment' controlam a aparência do texto.
        # 'bbox' cria a borda ao redor do texto.
        ax.text2D(1.1, 0.98, legend_text, 
                  transform=ax.transAxes, 
                  fontsize=10, 
                  verticalalignment='top', 
                  horizontalalignment='left',
                  bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7, ec='black', lw=1.5)) # Adiciona o texto com borda e fundo.

    plt.pause(0.01) # Pausa por um curto período para permitir a atualização do gráfico e criar a animação.

# --- Função de Gráfico para AG ---
# Adicionado 'ag_params' como um novo argumento para receber os parâmetros e estatísticas do AG.
def GraficoAG(populacao, melhor_solucao, iteracao, ax, melhor_aptidao_global, ag_params=None): # MODIFICADO: Adicionado ag_params
    """
    Função para plotar o gráfico do Algoritmo Genético (AG).
    Recebe a população de indivíduos, a melhor solução encontrada, a iteração atual,
    o objeto Axes3D, a melhor aptidão global e um dicionário opcional com parâmetros e estatísticas do AG.
    """
    ax.clear() # Limpa o conteúdo atual do eixo 3D para desenhar a nova geração.

    # Definir o range do meshgrid para plotagem da superfície da função.
    x_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo x.
    y_grid = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para o eixo y.
    X, Y = np.meshgrid(x_grid, y_grid) # Cria uma grade de coordenadas a partir dos arrays x_grid e y_grid.

    # Calcular Z usando a função W4
    Z = funcao_w4(X, Y) # Calcula os valores Z da função 'funcao_w4' para cada ponto da grade (X, Y), formando a superfície.

    # Título do gráfico com a geração atual e o melhor valor de aptidão global encontrado.
    ax.set_title(f'AG - Geração {iteracao} | Melhor Valor: {melhor_aptidao_global:.4f}')
    ax.set_xlabel('x') # Define o rótulo do eixo x.
    ax.set_ylabel('y') # Define o rótulo do eixo y.
    ax.set_zlabel('F(x, y)') # Define o rótulo do eixo z, representando o valor da função.

    ax.set_xlim([-500, 500]) # Define os limites do eixo x para corresponder ao meshgrid.
    ax.set_ylim([-500, 500]) # Define os limites do eixo y para corresponder ao meshgrid.

    # Plota a superfície 3D da função, com cor cinza claro, transparência e passos de grade.
    surf = ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.5, rstride=5, cstride=5)

    # Plotar a população do AG
    population_color = 'cyan' # Define a cor para os indivíduos da população.
    best_solution_color = 'red' # Define a cor para a melhor solução global.
    for individuo in populacao: # Itera sobre cada indivíduo na população.
        ind_x, ind_y = individuo[0], individuo[1] # Pega as coordenadas x e y do indivíduo.
        # Calcula o valor Z da função para a posição do indivíduo (garantindo que seja um array NumPy).
        ind_z_val = funcao_w4(np.array(ind_x), np.array(ind_y))
        ax.scatter(ind_x, ind_y, ind_z_val, color=population_color, s=50, alpha=0.7) # Plota o indivíduo como um ponto 3D na superfície.

    # Plotar a melhor solução encontrada
    if melhor_solucao is not None: # Verifica se uma melhor solução global foi encontrada.
        best_x, best_y = melhor_solucao[0], melhor_solucao[1] # Pega as coordenadas x e y da melhor solução.
        # Calcula o valor Z da função para a melhor solução (garantindo que seja um array NumPy).
        best_z_val = funcao_w4(np.array(best_x), np.array(best_y))
        # Plota a melhor solução como um ponto maior e distinto, com borda preta.
        ax.scatter(best_x, best_y, best_z_val, color=best_solution_color, s=200, marker='o', edgecolor='black', linewidth=1.5, label='Melhor Solução Global')

    # Definir limites para o eixo Z
    ax.set_zlim([-500, 4000]) # Define os limites do eixo z para melhor visualização.

    # --- Criação da Legenda no Lado Direito (para AG) --- # NOVO BLOCO
    if ag_params: # Verifica se o dicionário 'ag_params' foi fornecido.
        # Constrói o texto da legenda formatado com os parâmetros e estatísticas do AG.
        legend_text = (
            f'--- Parâmetros AG ---\n'
            f'Tamanho Pop: {ag_params["tamanho_populacao"]}\n'
            f'Taxa Mutação: {ag_params["taxa_mutacao"]:.2f}\n'
            f'Taxa Crossover: {ag_params["taxa_crossover"]:.2f}\n'
            f'Iterações Totais: {ag_params["iteracoes_totais"]}\n\n'
            f'--- Estatísticas Totais ---\n'
            f'Avaliações FO: {ag_params["avaliacoes_funcao"]}\n'
            f'Mult: {ag_params["multiplicacoes_total"]}\n'
            f'Div: {ag_params["divisoes_total"]}\n\n'
            f'--- Estatísticas no Melhor Global ---\n'
            f'Avaliações: {ag_params["avaliacoes_minimo_global"]}\n'
            f'Mult: {ag_params["multiplicacoes_minimo_global"]}\n'
            f'Div: {ag_params["divisoes_minimo_global"]}'
        )
        
        # Adiciona o bloco de texto ao plot, posicionado no canto superior direito do eixo.
        ax.text2D(1.1, 0.98, legend_text, 
                  transform=ax.transAxes, 
                  fontsize=10, 
                  verticalalignment='top', 
                  horizontalalignment='left',
                  bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7, ec='black', lw=1.5)) # Estiliza o texto com uma caixa de fundo.

    plt.pause(0.01) # Pausa por um curto período para permitir a atualização do gráfico e criar a animação.