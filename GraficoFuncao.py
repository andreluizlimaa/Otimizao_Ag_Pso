# graficofuncao.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- SUA FUNÇÃO W4 (MANTIDA IGUAL) ---
# Copie e cole as funções auxiliares e a funcao_w4_plot daqui
# ou, se preferir, importe-as se você as organizou em outro arquivo como funcoes_otimizacao.py
# Exemplo de importação:
# from funcoes_otimizacao import funcao_w4 as funcao_w4_plot # Alias para não confundir com a função dinâmica

# Se você tem as funções auxiliares e a funcao_w4_plot aqui mesmo:
def calculate_z_comp_plot(x, y):
    return -x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

def calculate_r(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def calculate_F10(x1, x2, a=500, b=0.1, c=0.5*np.pi):
    term1_inner = (x1**2 + x2**2) / 2
    term1 = -a * np.exp(-b * np.sqrt(term1_inner))
    term2_inner = (np.cos(c * x1) + np.cos(c * x2)) / 2
    term2 = np.exp(term2_inner)
    return term1 - term2 + np.exp(1)

def calculate_zsh(xs, ys):
    inner_sqrt = np.sqrt(xs**2 + ys**2)
    numerator = (np.sin(inner_sqrt)**2 - 0.5)
    denominator = (1 + 0.1 * (xs**2 + ys**2))**2
    return 0.5 - numerator / denominator

def funcao_w4_plot(x, y): # Função principal a ser plotada
    x_transformed_r_z = x / 250
    y_transformed_r_z = y / 250

    x1 = 25 * x_transformed_r_z
    x2 = 25 * y_transformed_r_z

    xs_normalized = x / 50
    ys_normalized = y / 50

    z_val = calculate_z_comp_plot(x_transformed_r_z, y_transformed_r_z)
    r_val = calculate_r(x_transformed_r_z, y_transformed_r_z)
    F10_val = calculate_F10(x1, x2)
    zsh_val = calculate_zsh(xs_normalized, ys_normalized)

    Fobj_val = F10_val * zsh_val
    w4_val = np.sqrt(r_val**2 + z_val**2) + Fobj_val
    return w4_val


# --- CONFIGURAÇÕES PARA O PLOT DA FUNÇÃO COMPLETA ---

# 1. Defina o domínio de busca completo
min_val = -500
max_val = 1000

# 2. Aumente a densidade de pontos para uma superfície mais suave
# Use um valor maior para 'num_points' para mais detalhes.
# Cuidado: valores muito altos (ex: 500, 1000) podem consumir muita memória e tempo.
num_points = 300 # Um bom equilíbrio entre detalhe e performance. Tente 400 ou 500 se quiser mais.
x_grid = np.linspace(min_val, max_val, num_points)
y_grid = np.linspace(min_val, max_val, num_points)
X, Y = np.meshgrid(x_grid, y_grid)

# 3. Calcule os valores Z da função
Z = funcao_w4_plot(X, Y)

# 4. Crie a figura e o subplot 3D com um tamanho grande
# Ajuste 'figsize' para o tamanho desejado da janela/imagem.
# (largura_polegadas, altura_polegadas)
fig = plt.figure(figsize=(10, 8)) # Tamanho grande para análise
ax = fig.add_subplot(111, projection='3d')

# 5. Defina explicitamente os limites dos eixos para garantir o domínio completo
ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])

# 6. Plotar a superfície
# 'cmap' define o mapa de cores (ex: 'viridis', 'plasma', 'inferno', 'jet').
# 'alpha' controla a transparência.
# 'rstride' e 'cstride' controlam a densidade das linhas na superfície (valores menores = mais densas)
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, rstride=10, cstride=10) # Reduzindo rstride/cstride para melhor desempenho de plotagem em meshes grandes. Se quiser mais linhas visíveis, reduza para 1 ou 5.

# 7. Adicione um Color Bar para entender os valores de Z
fig.colorbar(surf, shrink=0.5, aspect=5, label='F(x, y) Value')

# 8. Títulos e rótulos
ax.set_title('Gráfico 3D da função w4 - Análise Ampla', fontsize=18)
ax.set_xlabel('X Axis', fontsize=14)
ax.set_ylabel('Y Axis', fontsize=14)
ax.set_zlabel('F(x, y) Value', fontsize=14)

# 9. Ajuste os limites do eixo Z para focar na região de interesse
# Sua função tem um mínimo próximo de -500. Valores muito positivos podem distorcer a visualização.
# Ajuste estes valores com base na análise de sua função.
ax.set_zlim([-1000, 1000]) # Foca nos mínimos e em uma faixa superior mais controlada

# 10. Configurar o ângulo de visão (opcional, mas útil para uma vista padrão)
# 'elev' (elevação) é o ângulo vertical. 'azim' (azimute) é o ângulo horizontal.
ax.view_init(elev=20, azim=-120) # Um bom ângulo para começar a análise

# 11. Habilitar a grade (opcional)
ax.grid(True)

# 12. Mostrar o plot
plt.show()

# 13. Opção de salvar a figura em alta resolução (DESCOMENTE PARA USAR)
# filename = 'funcao_w4_analise_ampla.png'
# plt.savefig(filename, dpi=300, bbox_inches='tight') # dpi=300 para alta resolução, bbox_inches='tight' para cortar espaços em branco
# print(f"Gráfico salvo como: {filename}")