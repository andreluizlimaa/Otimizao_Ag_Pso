import numpy as np
from utils import global_op_counter # Importa o contador global de operações

# O global_op_counter será passado implicitamente pela FuncaoObjetivoWrapper

def calculate_z(x_original, y_original):
    """
    Calcula a componente 'z' da função W4 (baseada na função de Schwefel).
    Args:
        x_original (float): Coordenada x.
        y_original (float): Coordenada y.
    Returns:
        float: Valor da componente z.
    """
    # -x_original * np.sin(np.sqrt(np.abs(x_original)))
    global_op_counter.add_mult(1) # x_original * np.sin(...)
    # np.sqrt(np.abs(x_original)): Contamos np.sqrt como 1 operação "complexa" (multiplicação)
    global_op_counter.add_mult(1)
    # np.sin(...) : Contamos np.sin como 1 operação "complexa" (multiplicação)
    global_op_counter.add_mult(1)

    term1 = -x_original * np.sin(np.sqrt(np.abs(x_original)))

    # -y_original * np.sin(np.sqrt(np.abs(y_original)))
    global_op_counter.add_mult(1) # y_original * np.sin(...)
    # np.sqrt(np.abs(y_original)):
    global_op_counter.add_mult(1)
    # np.sin(...):
    global_op_counter.add_mult(1)

    term2 = -y_original * np.sin(np.sqrt(np.abs(y_original)))

    # term1 + term2 (adição, não contado como mult/div)
    return term1 + term2

def calculate_r(x_scaled, y_scaled):
    """
    Calcula a componente 'r' da função W4 (baseada na função de Rosenbrock).
    Args:
        x_scaled (float): Coordenada x reescalonada.
        y_scaled (float): Coordenada y reescalonada.
    Returns:
        float: Valor da componente r.
    """
    # 100 * (y_scaled - x_scaled**2)**2 + (1 - x_scaled)**2

    # x_scaled**2: 1 multiplicação
    global_op_counter.add_mult(1)
    term_x_squared = x_scaled**2

    # (y_scaled - term_x_squared): Subtração, não contada

    # (y_scaled - term_x_squared)**2: 1 multiplicação
    global_op_counter.add_mult(1)
    inner_term_squared = (y_scaled - term_x_squared)**2

    # 100 * inner_term_squared: 1 multiplicação
    global_op_counter.add_mult(1)
    part1 = 100 * inner_term_squared

    # (1 - x_scaled): Subtração, não contada

    # (1 - x_scaled)**2: 1 multiplicação
    global_op_counter.add_mult(1)
    part2 = (1 - x_scaled)**2

    # part1 + part2: Adição, não contada
    return part1 + part2

def calculate_F10(x1, x2, a=500, b=0.1, c=0.5*np.pi):
    """
    Calcula a componente 'F10' da função W4 (baseada na função de Ackley).
    Args:
        x1 (float): Coordenada x reescalonada para F10.
        x2 (float): Coordenada y reescalonada para F10.
        a (float): Parâmetro 'a' da função Ackley.
        b (float): Parâmetro 'b' da função Ackley.
        c (float): Parâmetro 'c' da função Ackley.
    Returns:
        float: Valor da componente F10.
    """
    # c*np.pi: 1 multiplicação
    global_op_counter.add_mult(1)
    c_pi = c * np.pi

    # term1_inner = (x1**2 + x2**2) / 2
    global_op_counter.add_mult(1) # x1**2
    global_op_counter.add_mult(1) # x2**2
    global_op_counter.add_div(1)  # / 2
    term1_inner = (x1**2 + x2**2) / 2

    # term1 = -a * np.exp(-b * np.sqrt(term1_inner))
    global_op_counter.add_mult(1) # -b * np.sqrt(...)
    global_op_counter.add_mult(1) # np.sqrt(...)
    global_op_counter.add_mult(1) # np.exp(...)
    global_op_counter.add_mult(1) # -a * np.exp(...)
    term1 = -a * np.exp(-b * np.sqrt(term1_inner))

    # term2_inner = (np.cos(c * x1) + np.cos(c * x2)) / 2
    global_op_counter.add_mult(1) # c * x1
    global_op_counter.add_mult(1) # np.cos(...)
    global_op_counter.add_mult(1) # c * x2
    global_op_counter.add_mult(1) # np.cos(...)
    global_op_counter.add_div(1)  # / 2
    term2_inner = (np.cos(c * x1) + np.cos(c * x2)) / 2

    # term2 = np.exp(term2_inner)
    global_op_counter.add_mult(1) # np.exp(...)
    term2 = np.exp(term2_inner)

    # term1 - term2 + np.exp(1)
    global_op_counter.add_mult(1) # np.exp(1)
    return term1 - term2 + np.exp(1)

def calculate_zsh(xs_input, ys_input):
    """
    Calcula a componente 'zsh' da função W4 (baseada na função de Shubert/modificada).
    Args:
        xs_input (float): Coordenada x reescalonada para zsh.
        ys_input (float): Coordenada y reescalonada para zsh.
    Returns:
        float: Valor da componente zsh.
    """
    # inner_sqrt = np.sqrt(xs_input**2 + ys_input**2)
    global_op_counter.add_mult(1) # xs_input**2
    global_op_counter.add_mult(1) # ys_input**2
    global_op_counter.add_mult(1) # np.sqrt(...)
    inner_sqrt = np.sqrt(xs_input**2 + ys_input**2)

    # numerator = (np.sin(inner_sqrt)**2 - 0.5)
    global_op_counter.add_mult(1) # np.sin(...)
    global_op_counter.add_mult(1) # (...**2)
    numerator = (np.sin(inner_sqrt)**2 - 0.5)

    # denominator = (1 + 0.1 * (xs_input**2 + ys_input**2))**2
    global_op_counter.add_mult(1) # 0.1 * (...)
    global_op_counter.add_mult(1) # xs_input**2
    global_op_counter.add_mult(1) # ys_input**2
    global_op_counter.add_mult(1) # (...**2)
    denominator = (1 + 0.1 * (xs_input**2 + ys_input**2))**2

    # return 0.5 - numerator / (denominator + 1e-9)
    global_op_counter.add_div(1) # numerator / (denominator + 1e-9)
    return 0.5 - numerator / (denominator + 1e-9)


def funcao_w4(x, y):
    """
    Calcula a função objetivo W4 conforme as especificações do Scilab.
    Esta é a função principal a ser minimizada.
    Args:
        x (float): Coordenada x do espaço de busca (-500 a 500).
        y (float): Coordenada y do espaço de busca (-500 a 500).
    Returns:
        float: Valor da função W4 para os dados de entrada.
    """
    # 1. Z_func (Schwefel) usa as coordenadas ORIGINAIS x e y
    z_val = calculate_z(x, y) # As operações são contadas dentro de calculate_z

    # 2. Rosenbrock e as demais usam coordenadas reescalonadas
    # x / 250.0: 1 divisão
    global_op_counter.add_div(1)
    x_for_r_z = x / 250.0
    # y / 250.0: 1 divisão
    global_op_counter.add_div(1)
    y_for_r_z = y / 250.0

    # Cálculo da componente Rosenbrock
    r_val = calculate_r(x_for_r_z, y_for_r_z) # As operações são contadas dentro de calculate_r

    # Variáveis reescalonadas x1 e x2, usadas para F10 (Ackley) e zsh (Shubert)
    # 25 * x_for_r_z: 1 multiplicação
    global_op_counter.add_mult(1)
    x1 = 25 * x_for_r_z
    # 25 * y_for_r_z: 1 multiplicação
    global_op_counter.add_mult(1)
    x2 = 25 * y_for_r_z

    # Cálculo das componentes Ackley e Shubert
    F10_val = calculate_F10(x1, x2) # As operações são contadas dentro de calculate_F10
    zsh_val = calculate_zsh(x1, x2) # As operações são contadas dentro de calculate_zsh
    
    # Fobj_val = F10_val * zsh_val: 1 multiplicação
    global_op_counter.add_mult(1)
    Fobj_val = F10_val * zsh_val

    # Combinação final: Z_func (Schwefel em [-500, 500]) é combinada com R_func (Rosenbrock em [-2, 2])
    # e Fobj_val (Ackley*Shubert em [-50, 50])
    # np.sqrt(r_val**2 + z_val**2) + Fobj_val
    # r_val**2: 1 multiplicação
    global_op_counter.add_mult(1)
    # z_val**2: 1 multiplicação
    global_op_counter.add_mult(1)
    # np.sqrt(...) : 1 multiplicação
    global_op_counter.add_mult(1)
    w4_val = np.sqrt(r_val**2 + z_val**2) + Fobj_val

    return w4_val

# Exemplo de uso para verificar os ranges (pode ser comentado para uso em otimização)
if __name__ == '__main__':
    # Para teste, precisamos do global_op_counter
    from utils import global_op_counter
    global_op_counter.reset() # Resetar para o teste
    
    x_test = np.linspace(-500, 500, 100)
    y_test = np.linspace(-500, 500, 100)
    X_test, Y_test = np.meshgrid(x_test, y_test)

    # A funcao_w4_wrapper não é usada aqui diretamente no __main__
    # para simular a chamada direta e contar operações, então chamamos funcao_w4 diretamente
    # CUIDADO: Se você usar isto para obter os contadores exatos,
    # lembre-se que 'funcao_w4' deve ser chamada APENAS UMA VEZ
    # para ter o custo de uma única avaliação.
    # Se você passar meshgrid inteiro, as operações serão contadas para CADA ELEMENTO.

    # Para um cálculo de operações de UMA ÚNICA avaliação:
    # Escolha um ponto de exemplo
    x_single = 0.0
    y_single = 0.0
    
    global_op_counter.reset() # Resetar antes da única avaliação
    single_w4_result = funcao_w4(x_single, y_single)
    
    print(f"\n--- Contagem de Operações para uma ÚNICA avaliação de funcao_w4({x_single}, {y_single}) ---")
    print(f"Multiplicações: {global_op_counter.multiplications}")
    print(f"Divisões: {global_op_counter.divisions}")
    print(f"Resultado: {single_w4_result}")

    # Resetar para a execução da parte de meshgrid (que não será usada para contagem final nos algoritmos)
    global_op_counter.reset() 

    # Calcula a função W4 para o meshgrid (APENAS PARA VISUALIZAÇÃO/TESTE DE RANGE, NÃO PARA CONTAR OPERAÇÕES DE UMA CHAMADA)
    # O `funcao_w4` quando recebe np.ndarray, as operações de numpy são "vetorizadas",
    # ou seja, uma única chamada np.sin(array) faz N operações.
    # Para uma contagem PRECISA por elemento em um meshgrid, precisaria de um loop manual.
    # Mas para o AG/PSO, a função será chamada para escalares individuais, então a instrumentação acima está correta.
    W4_result = funcao_w4(X_test, Y_test) 

    print(f"\nMin/Max W4_result (meshgrid): {np.min(W4_result):.2f} / {np.max(W4_result):.2f}")

    x_for_r_z_test = X_test / 250.0
    y_for_r_z_test = Y_test / 250.0
    x1_test = 25 * x_for_r_z_test
    x2_test = 25 * y_for_r_z_test

    # As chamadas a seguir também incrementariam o contador se não fossem para arrays.
    # Para arrays, o comportamento do contador seria diferente (muitas operações por chamada).
    # Mantenha em mente que para AG/PSO, funcao_w4 recebe escalares.
    r_val_test = calculate_r(x_for_r_z_test, y_for_r_z_test)
    z_val_test = calculate_z(X_test, Y_test)
    F10_val_test = calculate_F10(x1_test, x2_test)
    zsh_val_test = calculate_zsh(x1_test, x2_test)
    Fobj_val_test = F10_val_test * zsh_val_test

    print(f"Min/Max r_val (meshgrid): {np.min(r_val_test):.2f} / {np.max(r_val_test):.2f}")
    print(f"Min/Max z_val (meshgrid): {np.min(z_val_test):.2f} / {np.max(z_val_test):.2f}")
    print(f"Min/Max F10_val (meshgrid): {np.min(F10_val_test):.2f} / {np.max(F10_val_test):.2f}")
    print(f"Min/Max zsh_val (meshgrid): {np.min(zsh_val_test):.2f} / {np.max(zsh_val_test):.2f}")
    print(f"Min/Max Fobj_val (meshgrid): {np.min(Fobj_val_test):.2f} / {np.max(Fobj_val_test):.2f}")