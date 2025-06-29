"""
funcoes_otimizacao.py
"""
import numpy as np # Importa a biblioteca NumPy, fundamental para operações numéricas eficientes, especialmente com arrays.

def calculate_z(x_original, y_original):
    """
    Calcula a componente 'z' da função W4 (baseada na função de Schwefel).
    Args:
        x_original (np.ndarray): Coordenada x original (do meshgrid, e.g., -500 a 500).
        y_original (np.ndarray): Coordenada y original (do meshgrid, e.g., -500 a 500).
    Returns:
        np.ndarray: Valor da componente z.
    """
    # A função de Schwefel usa as coordenadas originais, não as reescalonadas para [-2, 2]
    # Retorna o valor da componente 'z' aplicando a fórmula da função de Schwefel.
    return -x_original * np.sin(np.sqrt(np.abs(x_original))) - y_original * np.sin(np.sqrt(np.abs(y_original)))

def calculate_r(x_scaled, y_scaled):
    """
    Calcula a componente 'r' da função W4 (baseada na função de Rosenbrock).
    Args:
        x_scaled (np.ndarray): Coordenada x reescalonada (e.g., -2 a 2).
        y_scaled (np.ndarray): Coordenada y reescalonada (e.g., -2 a 2).
    Returns:
        np.ndarray: Valor da componente r.
    """
    # Retorna o valor da componente 'r' aplicando a fórmula da função de Rosenbrock.
    return 100 * (y_scaled - x_scaled**2)**2 + (1 - x_scaled)**2

def calculate_F10(x1, x2, a=500, b=0.1, c=0.5*np.pi):
    """
    Calcula a componente 'F10' da função W4 (baseada na função de Ackley).
    Args:
        x1 (np.ndarray): Coordenada x reescalonada para F10 (e.g., -50 a 50).
        x2 (np.ndarray): Coordenada y reescalonada para F10 (e.g., -50 a 50).
        a (float): Parâmetro 'a' da função Ackley.
        b (float): Parâmetro 'b' da função Ackley.
        c (float): Parâmetro 'c' da função Ackley.
    Returns:
        np.ndarray: Valor da componente F10.
    """
    term1_inner = (x1**2 + x2**2) / 2 # Calcula o termo interno da primeira parte da função de Ackley.
    term1 = -a * np.exp(-b * np.sqrt(term1_inner)) # Calcula a primeira parte da função de Ackley.
    term2_inner = (np.cos(c * x1) + np.cos(c * x2)) / 2 # Calcula o termo interno da segunda parte da função de Ackley.
    term2 = np.exp(term2_inner) # Calcula a segunda parte da função de Ackley.
    return term1 - term2 + np.exp(1) # Retorna o valor final da componente F10, combinando os termos.

def calculate_zsh(xs_input, ys_input):
    """
    Calcula a componente 'zsh' da função W4 (baseada na função de Shubert/modificada).
    Args:
        xs_input (np.ndarray): Coordenada x reescalonada para zsh (corresponde a x1).
        ys_input (np.ndarray): Coordenada y reescalonada para zsh (corresponde a x2).
    Returns:
        np.ndarray: Valor da componente zsh.
    """
    inner_sqrt = np.sqrt(xs_input**2 + ys_input**2) # Calcula a raiz quadrada da soma dos quadrados das entradas.
    numerator = (np.sin(inner_sqrt)**2 - 0.5) # Calcula o numerador da expressão.
    denominator = (1 + 0.1 * (xs_input**2 + ys_input**2))**2 # Calcula o denominador da expressão.
    # Retorna o valor da componente zsh, adicionando um pequeno epsilon ao denominador para evitar divisão por zero.
    return 0.5 - numerator / (denominator + 1e-9)

def funcao_w4(x, y):
    """
    Calcula a função objetivo W4 conforme as especificações do Scilab.
    Esta é a função principal a ser minimizada.
    Args:
        x (np.ndarray): Coordenada x do espaço de busca (-500 a 500).
        y (np.ndarray): Coordenada y do espaço de busca (-500 a 500).
    Returns:
        np.ndarray: Valor da função W4 para os dados de entrada.
    """
    # 1. Z_func (Schwefel) usa as coordenadas ORIGINAIS x e y
    z_val = calculate_z(x, y) # Calcula a componente Schwefel usando as coordenadas originais.

    # 2. Rosenbrock e as demais usam coordenadas reescalonadas
    x_for_r_z = x / 250.0 # Reescalona a coordenada x para o intervalo [-2, 2], usado para Rosenbrock.
    y_for_r_z = y / 250.0 # Reescalona a coordenada y para o intervalo [-2, 2], usado para Rosenbrock.

    # Cálculo da componente Rosenbrock
    r_val = calculate_r(x_for_r_z, y_for_r_z) # Calcula a componente Rosenbrock com as coordenadas reescalonadas.

    # Variáveis reescalonadas x1 e x2, usadas para F10 (Ackley) e zsh (Shubert)
    # No Scilab, são 25 * as variáveis reescalonadas de r e z,
    # resultando em um range de [-50, 50]
    x1 = 25 * x_for_r_z # Reescalona x_for_r_z para um range de [-50, 50] para Ackley e Shubert.
    x2 = 25 * y_for_r_z # Reescalona y_for_r_z para um range de [-50, 50] para Ackley e Shubert.

    # Cálculo das componentes Ackley e Shubert
    F10_val = calculate_F10(x1, x2) # Calcula a componente Ackley com as coordenadas reescalonadas.
    zsh_val = calculate_zsh(x1, x2) # Calcula a componente Shubert com as coordenadas reescalonadas.
    Fobj_val = F10_val * zsh_val # Combina as componentes Ackley e Shubert multiplicando-as.

    # Combinação final: Z_func (Schwefel em [-500, 500]) é combinada com R_func (Rosenbrock em [-2, 2])
    # e Fobj_val (Ackley*Shubert em [-50, 50])
    w4_val = np.sqrt(r_val**2 + z_val**2) + Fobj_val # Calcula o valor final da função W4 combinando todas as componentes.

    return w4_val # Retorna o valor da função W4.

# Exemplo de uso para verificar os ranges (pode ser comentado para uso em otimização)
if __name__ == '__main__': # Bloco que é executado apenas se o script for rodado diretamente (não importado como módulo).
    # Simula o meshgrid do Scilab
    x_test = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para x.
    y_test = np.linspace(-500, 500, 100) # Cria um array de 100 pontos igualmente espaçados entre -500 e 500 para y.
    X_test, Y_test = np.meshgrid(x_test, y_test) # Cria uma grade de coordenadas 2D a partir dos arrays x_test e y_test.

    # Calcula a função W4
    W4_result = funcao_w4(X_test, Y_test) # Calcula os valores da função W4 para todos os pontos da grade.

    # Imprime os valores min/max para depuração
    print(f"Min/Max W4_result: {np.min(W4_result):.2f} / {np.max(W4_result):.2f}") # Imprime os valores mínimo e máximo da função W4 calculada.

    # Checando os ranges internos para depuração (estes prints já estavam corretos para depuração)
    x_for_r_z_test = X_test / 250.0 # Reescalona X_test para o range de Rosenbrock.
    y_for_r_z_test = Y_test / 250.0 # Reescalona Y_test para o range de Rosenbrock.
    x1_test = 25 * x_for_r_z_test # Reescalona x_for_r_z_test para o range de Ackley/Shubert.
    x2_test = 25 * y_for_r_z_test # Reescalona y_for_r_z_test para o range de Ackley/Shubert.

    r_val_test = calculate_r(x_for_r_z_test, y_for_r_z_test) # Calcula a componente Rosenbrock para teste.
    z_val_test = calculate_z(X_test, Y_test) # <--- Aqui também, usando X_test, Y_test # Calcula a componente Schwefel para teste.
    F10_val_test = calculate_F10(x1_test, x2_test) # Calcula a componente Ackley para teste.
    zsh_val_test = calculate_zsh(x1_test, x2_test) # Calcula a componente Shubert para teste.
    Fobj_val_test = F10_val_test * zsh_val_test # Calcula a combinação Ackley*Shubert para teste.

    print(f"Min/Max r_val: {np.min(r_val_test):.2f} / {np.max(r_val_test):.2f}") # Imprime os valores min/max da componente Rosenbrock.
    print(f"Min/Max z_val: {np.min(z_val_test):.2f} / {np.max(z_val_test):.2f}") # Imprime os valores min/max da componente Schwefel.
    print(f"Min/Max F10_val: {np.min(F10_val_test):.2f} / {np.max(F10_val_test):.2f}") # Imprime os valores min/max da componente Ackley.
    print(f"Min/Max zsh_val: {np.min(zsh_val_test):.2f} / {np.max(zsh_val_test):.2f}") # Imprime os valores min/max da componente Shubert.
    print(f"Min/Max Fobj_val: {np.min(Fobj_val_test):.2f} / {np.max(Fobj_val_test):.2f}") # Imprime os valores min/max da componente combinada Fobj.