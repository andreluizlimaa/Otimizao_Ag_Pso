# utils.py

class OperationCounter:
    def __init__(self):
        self.multiplications = 0
        self.divisions = 0

    def reset(self):
        self.multiplications = 0
        self.divisions = 0

    def add_mult(self, count=1):
        self.multiplications += count

    def add_div(self, count=1):
        self.divisions += count

# Instância global do contador para ser importada onde for necessário
global_op_counter = OperationCounter()


# Classe para envolver a função objetivo e contar suas chamadas e operações internas
class FuncaoObjetivoWrapper:
    def __init__(self, original_func, op_counter):
        self.original_func = original_func
        self.evaluations = 0
        self.op_counter = op_counter

    def __call__(self, x, y):
        self.evaluations += 1
        
        # Estimativa de operações dentro da funcao_w4.
        # Estas são adicionadas por CADA VEZ que a funcao_w4 é chamada para avaliação.
        self.op_counter.add_mult(20) # Estimativa de mults/potências dentro de w4
        self.op_counter.add_div(5)   # Estimativa de divs dentro de w4

        return self.original_func(x, y)