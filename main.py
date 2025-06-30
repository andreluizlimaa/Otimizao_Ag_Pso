import numpy as np
import matplotlib.pyplot as plt

from funcoes_otimizacao import funcao_w4
from PSO import PSO
from Genetico import algoritmo_genetico
from utils import global_op_counter, FuncaoObjetivoWrapper

# --- Parâmetros Comuns ou Específicos ---
limites_xy = [(-500, 500), (-500, 500)]

print("==================================================")
print("Iniciando otimização com Enxame de Partículas (PSO)...")
print("==================================================")

num_particulas_pso = 15
num_iteracoes_pso = 100
iteracoes_sem_melhora_limite_pso = 50
tolerancia_pso = 1e-6

pso_instance = PSO(
    limites=limites_xy,
    num_particulas=num_particulas_pso,
    num_iteracoes=num_iteracoes_pso,
    iteracoes_sem_melhora_limite=iteracoes_sem_melhora_limite_pso,
    tolerancia=tolerancia_pso
)

print("\n==================================================")
print("Iniciando otimização com Algoritmo Genético (AG)...")
print("==================================================")

tamanho_populacao_ag = 35
num_geracoes_ag = 200
taxa_cruzamento_ag = 0.7
taxa_mutacao_ag = 0.01
geracoes_sem_melhora_limite_ag = 50
# Nova variável de tolerância para o AG
tolerancia_ag = 1e-6 # Define a tolerância para considerar uma melhoria no AG.

melhor_solucao_ag, melhor_aptidao_ag, geracao_ag = algoritmo_genetico(
    tamanho_populacao=tamanho_populacao_ag,
    limites=(-500, 500),
    num_geracoes=num_geracoes_ag,
    taxa_cruzamento=taxa_cruzamento_ag,
    taxa_mutacao=taxa_mutacao_ag,
    geracoes_sem_melhora_limite=geracoes_sem_melhora_limite_ag,
    tolerancia=tolerancia_ag # Passa a tolerância para o AG
)

print("\n==================================================")
print("Otimização Concluída.")
print("==================================================")