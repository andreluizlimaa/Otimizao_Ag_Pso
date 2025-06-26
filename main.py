# main.py
import numpy as np
import matplotlib.pyplot as plt # Importa plt para mostrar o gráfico final

# Importa a função w4 original (para passar para o wrapper)
from funcoes_otimizacao import funcao_w4

# Importa as classes dos algoritmos de otimização
from PSO import PSO # Importa a classe PSO
from Genetico import algoritmo_genetico # Importa a função principal do AG
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa o contador e o wrapper

# --- Parâmetros Comuns ou Específicos ---
limites_xy = [(-500, 500), (-500, 500)] # Para x e y

print("==================================================")
print("Iniciando otimização com Enxame de Partículas (PSO)...")
print("==================================================")

num_particulas_pso = 15
num_iteracoes_pso = 40

# A classe PSO agora lida com a inicialização do wrapper e a impressão dos resultados.
pso_instance = PSO(
    limites=limites_xy,
    num_particulas=num_particulas_pso,
    num_iteracoes=num_iteracoes_pso
)

print("\n==================================================")
print("Iniciando otimização com Algoritmo Genético (AG)...")
print("==================================================")

tamanho_populacao_ag = 35
num_geracoes_ag = 1000
taxa_cruzamento_ag = 0.7
taxa_mutacao_ag = 0.01
geracoes_sem_melhora_limite_ag = 50

melhor_solucao_ag, melhor_aptidao_ag, geracao_ag = algoritmo_genetico(
    tamanho_populacao=tamanho_populacao_ag,
    limites=(-500, 500), # Para o AG, seus limites são passados como uma tupla simples
    num_geracoes=num_geracoes_ag,
    taxa_cruzamento=taxa_cruzamento_ag,
    taxa_mutacao=taxa_mutacao_ag,
    geracoes_sem_melhora_limite=geracoes_sem_melhora_limite_ag
)

print("\n==================================================")
print("Otimização Concluída.")
print("==================================================")