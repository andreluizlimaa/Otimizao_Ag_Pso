# utils.py

class OperationCounter: # Define a classe OperationCounter para rastrear operações matemáticas.
    def __init__(self): # Método construtor da classe.
        self.multiplications = 0 # Inicializa o contador de multiplicações como zero.
        self.divisions = 0 # Inicializa o contador de divisões como zero.

    def reset(self): # Método para resetar os contadores.
        self.multiplications = 0 # Zera o contador de multiplicações.
        self.divisions = 0 # Zera o contador de divisões.

    def add_mult(self, count=1): # Método para adicionar ao contador de multiplicações.
        self.multiplications += count # Incrementa o contador de multiplicações pelo valor fornecido (padrão é 1).

    def add_div(self, count=1): # Método para adicionar ao contador de divisões.
        self.divisions += count # Incrementa o contador de divisões pelo valor fornecido (padrão é 1).

# Instância global do contador para ser importada onde for necessário
global_op_counter = OperationCounter() # Cria uma instância global de OperationCounter, permitindo que seja acessada e atualizada em diferentes partes do código.


# Classe para envolver a função objetivo e contar suas chamadas e operações internas
class FuncaoObjetivoWrapper: # Define a classe FuncaoObjetivoWrapper, que atua como um invólucro para a função objetivo.
    def __init__(self, original_func, op_counter): # Método construtor, recebe a função original e uma instância de OperationCounter.
        self.original_func = original_func # Armazena a função objetivo original.
        self.evaluations = 0 # Inicializa o contador de avaliações da função objetivo como zero.
        self.op_counter = op_counter # Armazena a instância do contador de operações.

    def __call__(self, x, y): # Este método especial permite que instâncias de FuncaoObjetivoWrapper sejam chamadas como funções.
        self.evaluations += 1 # Incrementa o contador de avaliações cada vez que a função é chamada.
        
        # Estimativa de operações dentro da funcao_w4.
        # Estas são adicionadas por CADA VEZ que a funcao_w4 é chamada para avaliação.
        self.op_counter.add_mult(20) # Adiciona uma estimativa de 20 multiplicações/potências ao contador global.
        self.op_counter.add_div(5)   # Adiciona uma estimativa de 5 divisões ao contador global.

        return self.original_func(x, y) # Chama e retorna o resultado da função objetivo original.