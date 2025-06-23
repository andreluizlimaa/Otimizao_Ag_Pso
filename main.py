# main.py
import numpy as np

# Importa a função w4 de um módulo centralizado
from funcoes_otimizacao import funcao_w4

# Importa as classes dos algoritmos de otimização
from PSO import PSO # Importa a classe PSO
from Genetico import algoritmo_genetico # Importa a função principal do AG

# --- Parâmetros Comuns ou Específicos ---
# Limites para as variáveis x e y para ambos os algoritmos
limites_xy = [(-500, 500), (-500, 500)] # Para x e y

print("==================================================")
print("Iniciando otimização com Enxame de Partículas (PSO)...")
print("==================================================")

# Parâmetros para o PSO
num_particulas_pso = 15
num_iteracoes_pso = 40

# Executa o PSO (a visualização será aberta automaticamente dentro da classe PSO)
# Note que não passamos mais 'funcao_w4' diretamente aqui, pois ela é importada dentro de PSO.py
pso_instance = PSO(
    limites=limites_xy,
    num_particulas=num_particulas_pso,
    num_iteracoes=num_iteracoes_pso
)
# A classe PSO já imprime os resultados finais.

print("\n==================================================")
print("Iniciando otimização com Algoritmo Genético (AG)...")
print("==================================================")

# Parâmetros para o Algoritmo Genético
tamanho_populacao_ag = 35
num_geracoes_ag = 1000
taxa_cruzamento_ag = 0.7
taxa_mutacao_ag = 0.01

# Executa o Algoritmo Genético (a visualização será aberta automaticamente dentro da função)
# O algoritmo_genetico já imprime os resultados finais.
# Note que a função 'funcao_w4' é importada dentro de 'genetico.py' e usada lá.
melhor_solucao_ag, melhor_aptidao_ag, geracao_ag = algoritmo_genetico(
    tamanho_populacao=tamanho_populacao_ag,
    limites=(-500, 500), # Para o AG, seus limites são passados como uma tupla simples
    num_geracoes=num_geracoes_ag,
    taxa_cruzamento=taxa_cruzamento_ag,
    taxa_mutacao=taxa_mutacao_ag
)

# Se você quiser imprimir novamente os resultados finais do AG (já que a função faz isso)
print("\n--- Resultados Finais do Algoritmo Genético ---")
print(f"Melhor solução encontrada (AG): {melhor_solucao_ag}")
print(f"Valor da função para a melhor solução (AG): {melhor_aptidao_ag}")
print(f"Geração de convergência (AG): {geracao_ag}")

print("\n==================================================")
print("Otimização Concluída.")
print("==================================================")