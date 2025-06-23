# funcoes_otimizacao.py
import numpy as np

def calculate_z(x, y):
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

# Sua função w4 principal
def funcao_w4(x, y):
    x_transformed_r_z = x / 250
    y_transformed_r_z = y / 250

    x1 = 25 * x_transformed_r_z
    x2 = 25 * y_transformed_r_z

    xs_normalized = x / 50
    ys_normalized = y / 50

    z_val = calculate_z(x_transformed_r_z, y_transformed_r_z)
    r_val = calculate_r(x_transformed_r_z, y_transformed_r_z)
    F10_val = calculate_F10(x1, x2)
    zsh_val = calculate_zsh(xs_normalized, ys_normalized)

    Fobj_val = F10_val * zsh_val
    w4_val = np.sqrt(r_val**2 + z_val**2) + Fobj_val
    return w4_val