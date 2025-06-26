"""
funcoes_otimizacao.py
"""
import numpy as np

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
    term1_inner = (x1**2 + x2**2) / 2
    term1 = -a * np.exp(-b * np.sqrt(term1_inner))
    term2_inner = (np.cos(c * x1) + np.cos(c * x2)) / 2
    term2 = np.exp(term2_inner)
    return term1 - term2 + np.exp(1)

def calculate_zsh(xs_input, ys_input):
    """
    Calcula a componente 'zsh' da função W4 (baseada na função de Shubert/modificada).
    Args:
        xs_input (np.ndarray): Coordenada x reescalonada para zsh (corresponde a x1).
        ys_input (np.ndarray): Coordenada y reescalonada para zsh (corresponde a x2).
    Returns:
        np.ndarray: Valor da componente zsh.
    """
    inner_sqrt = np.sqrt(xs_input**2 + ys_input**2)
    numerator = (np.sin(inner_sqrt)**2 - 0.5)
    denominator = (1 + 0.1 * (xs_input**2 + ys_input**2))**2
    return 0.5 - numerator / (denominator + 1e-9)  # Adiciona um pequeno epsilon para evitar divisão por zero

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
    z_val = calculate_z(x, y)

    # 2. Rosenbrock e as demais usam coordenadas reescalonadas
    x_for_r_z = x / 250.0 # Escala [-2, 2] para Rosenbrock
    y_for_r_z = y / 250.0

    # Cálculo da componente Rosenbrock
    r_val = calculate_r(x_for_r_z, y_for_r_z)

    # Variáveis reescalonadas x1 e x2, usadas para F10 (Ackley) e zsh (Shubert)
    # No Scilab, são 25 * as variáveis reescalonadas de r e z,
    # resultando em um range de [-50, 50]
    x1 = 25 * x_for_r_z
    x2 = 25 * y_for_r_z

    # Cálculo das componentes Ackley e Shubert
    F10_val = calculate_F10(x1, x2)
    zsh_val = calculate_zsh(x1, x2)
    Fobj_val = F10_val * zsh_val

    # Combinação final: Z_func (Schwefel em [-500, 500]) é combinada com R_func (Rosenbrock em [-2, 2])
    # e Fobj_val (Ackley*Shubert em [-50, 50])
    w4_val = np.sqrt(r_val**2 + z_val**2) + Fobj_val

    return w4_val

# Exemplo de uso para verificar os ranges (pode ser comentado para uso em otimização)
if __name__ == '__main__':
    # Simula o meshgrid do Scilab
    x_test = np.linspace(-500, 500, 100)
    y_test = np.linspace(-500, 500, 100)
    X_test, Y_test = np.meshgrid(x_test, y_test)

    # Calcula a função W4
    W4_result = funcao_w4(X_test, Y_test)

    # Imprime os valores min/max para depuração
    print(f"Min/Max W4_result: {np.min(W4_result):.2f} / {np.max(W4_result):.2f}")

    # Checando os ranges internos para depuração (estes prints já estavam corretos para depuração)
    x_for_r_z_test = X_test / 250.0
    y_for_r_z_test = Y_test / 250.0
    x1_test = 25 * x_for_r_z_test
    x2_test = 25 * y_for_r_z_test

    r_val_test = calculate_r(x_for_r_z_test, y_for_r_z_test)
    z_val_test = calculate_z(X_test, Y_test) # <--- Aqui também, usando X_test, Y_test
    F10_val_test = calculate_F10(x1_test, x2_test)
    zsh_val_test = calculate_zsh(x1_test, x2_test)
    Fobj_val_test = F10_val_test * zsh_val_test

    print(f"Min/Max r_val: {np.min(r_val_test):.2f} / {np.max(r_val_test):.2f}")
    print(f"Min/Max z_val: {np.min(z_val_test):.2f} / {np.max(z_val_test):.2f}")
    print(f"Min/Max F10_val: {np.min(F10_val_test):.2f} / {np.max(F10_val_test):.2f}")
    print(f"Min/Max zsh_val: {np.min(zsh_val_test):.2f} / {np.max(zsh_val_test):.2f}")
    print(f"Min/Max Fobj_val: {np.min(Fobj_val_test):.2f} / {np.max(Fobj_val_test):.2f}")