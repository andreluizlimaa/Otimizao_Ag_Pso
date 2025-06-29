# main.py
import numpy as np # Importa a biblioteca NumPy, amplamente utilizada para operações numéricas e matemáticas.
import matplotlib.pyplot as plt # Importa o módulo pyplot da biblioteca Matplotlib, comumente usado para criar gráficos e visualizações.

# Importa a função w4 original (para passar para o wrapper)
from funcoes_otimizacao import funcao_w4 # Importa a função 'funcao_w4' do arquivo 'funcoes_otimizacao.py'. Esta função provavelmente será a função objetivo a ser otimizada.

# Importa as classes dos algoritmos de otimização
from PSO import PSO # Importa a classe 'PSO' do arquivo 'PSO.py', que implementa o algoritmo de Otimização por Enxame de Partículas.
from Genetico import algoritmo_genetico # Importa a função 'algoritmo_genetico' do arquivo 'Genetico.py', que implementa o Algoritmo Genético.
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa 'global_op_counter' e 'FuncaoObjetivoWrapper' do arquivo 'utils.py'. 'global_op_counter' provavelmente é um contador global de operações, e 'FuncaoObjetivoWrapper' um invólucro para a função objetivo.

# --- Parâmetros Comuns ou Específicos ---
limites_xy = [(-500, 500), (-500, 500)] # Define os limites para as variáveis x e y. Neste caso, ambas variam de -500 a 500.

print("==================================================") # Imprime uma linha de separação para melhorar a legibilidade da saída.
print("Iniciando otimização com Enxame de Partículas (PSO)...") # Informa ao usuário que a otimização com PSO está começando.
print("==================================================") # Imprime outra linha de separação.

num_particulas_pso = 15 # Define o número de partículas a serem usadas no algoritmo PSO.
num_iteracoes_pso = 40 # Define o número máximo de iterações para o algoritmo PSO.

# A classe PSO agora lida com a inicialização do wrapper e a impressão dos resultados.
pso_instance = PSO( # Cria uma instância da classe PSO.
    limites=limites_xy, # Passa os limites definidos para as variáveis.
    num_particulas=num_particulas_pso, # Passa o número de partículas.
    num_iteracoes=num_iteracoes_pso # Passa o número de iterações.
)

print("\n==================================================") # Imprime uma quebra de linha e uma linha de separação.
print("Iniciando otimização com Algoritmo Genético (AG)...") # Informa ao usuário que a otimização com AG está começando.
print("==================================================") # Imprime outra linha de separação.

tamanho_populacao_ag = 35 # Define o tamanho da população para o Algoritmo Genético.
num_geracoes_ag = 1000 # Define o número máximo de gerações para o Algoritmo Genético.
taxa_cruzamento_ag = 0.7 # Define a taxa de cruzamento (crossover) para o AG, que é a probabilidade de dois indivíduos trocarem material genético.
taxa_mutacao_ag = 0.01 # Define a taxa de mutação para o AG, que é a probabilidade de um gene individual ser alterado.
geracoes_sem_melhora_limite_ag = 50 # Define o limite de gerações consecutivas sem melhoria antes que o algoritmo possa parar (critério de parada precoce).

melhor_solucao_ag, melhor_aptidao_ag, geracao_ag = algoritmo_genetico( # Chama a função do algoritmo genético e armazena seus resultados.
    tamanho_populacao=tamanho_populacao_ag, # Passa o tamanho da população.
    limites=(-500, 500), # Passa os limites para o AG. Note que é uma tupla simples, diferente do formato para PSO.
    num_geracoes=num_geracoes_ag, # Passa o número de gerações.
    taxa_cruzamento=taxa_cruzamento_ag, # Passa a taxa de cruzamento.
    taxa_mutacao=taxa_mutacao_ag, # Passa a taxa de mutação.
    geracoes_sem_melhora_limite=geracoes_sem_melhora_limite_ag # Passa o limite de gerações sem melhoria.
)

print("\n==================================================") # Imprime uma quebra de linha e uma linha de separação.
print("Otimização Concluída.") # Informa ao usuário que o processo de otimização foi concluído.
print("==================================================") # Imprime outra linha de separação.