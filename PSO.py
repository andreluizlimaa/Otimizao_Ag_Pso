# PSO.py
from Enxame import Enxame
from Grafico import GraficoPSO
import matplotlib.pyplot as plt
from funcoes_otimizacao import funcao_w4
from utils import global_op_counter, FuncaoObjetivoWrapper
import os # Importado para salvar arquivos
from datetime import datetime # Importado para gerar timestamp

class PSO:
    # MODIFICADO: Adicionados c1_param, c2_param, w_max_param, w_min_param ao construtor
    def __init__(self, limites, num_particulas, num_iteracoes, iteracoes_sem_melhora_limite, tolerancia, c1_param, c2_param, w_max_param, w_min_param):
        self.limites = limites
        self.num_particulas = num_particulas
        self.num_iteracoes = num_iteracoes
        self.iteracoes_sem_melhora_limite = iteracoes_sem_melhora_limite
        self.tolerancia = tolerancia

        # NOVO: Armazena os parâmetros passados no construtor como atributos da instância
        self.c1 = c1_param
        self.c2 = c2_param
        self.w_max = w_max_param
        self.w_min = w_min_param

        global_op_counter.reset()
        self.funcao_w4_wrapper = FuncaoObjetivoWrapper(funcao_w4, global_op_counter)

        melhor_valor_g = float('inf')
        melhor_posicao_g = []
        
        self.avaliacoes_pso_minimo_global = 0
        self.operacoes_pso_minimo_global_mult = 0
        self.operacoes_pso_minimo_global_div = 0

        iteracoes_sem_melhora = 0
        
        enxame = []
        for i in range(self.num_particulas): # Usar self.num_particulas
            # MODIFICADO: Passa os novos parâmetros (c1, c2, w_max, w_min) para o construtor de Enxame
            enxame.append(Enxame(self.limites, self.c1, self.c2, self.w_max, self.w_min))

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(right=0.7)

        i = 0
        while i < self.num_iteracoes: # Usar self.num_iteracoes
            valor_global_antes_atualizacao = melhor_valor_g

            for j in range(self.num_particulas): # Usar self.num_particulas
                enxame[j].avaliar(self.funcao_w4_wrapper)

                if enxame[j].valor_atual_i < melhor_valor_g:
                    melhor_posicao_g = list(enxame[j].posicao_i)
                    melhor_valor_g = float(enxame[j].valor_atual_i)
                    self.avaliacoes_pso_minimo_global = self.funcao_w4_wrapper.evaluations
                    self.operacoes_pso_minimo_global_mult = global_op_counter.multiplications
                    self.operacoes_pso_minimo_global_div = global_op_counter.divisions

            if i > 0 and abs(melhor_valor_g - valor_global_antes_atualizacao) < self.tolerancia: # Usar self.tolerancia
                iteracoes_sem_melhora += 1
            else:
                iteracoes_sem_melhora = 0

            if iteracoes_sem_melhora >= self.iteracoes_sem_melhora_limite: # Usar self.iteracoes_sem_melhora_limite
                print(f"\n[PSO] Parada por convergência: Mudança no melhor valor global menor que {self.tolerancia} por {self.iteracoes_sem_melhora_limite} iterações.")
                break

            for j in range(self.num_particulas): # Usar self.num_particulas
                enxame[j].atualizar_velocidade(melhor_posicao_g, i, self.num_iteracoes) # Usar self.num_iteracoes
                enxame[j].atualizar_posicao(self.limites)

            # MODIFICADO: Agora pego os valores de c1, c2, w_max, w_min dos atributos da instância self
            pso_params_for_plot = {
                "c1": self.c1,
                "c2": self.c2,
                "w_max": self.w_max,
                "w_min": self.w_min,
                "iteracoes_totais": self.num_iteracoes,
                "num_particulas": self.num_particulas,
                "avaliacoes_funcao": self.funcao_w4_wrapper.evaluations,
                "multiplicacoes_total": global_op_counter.multiplications,
                "divisoes_total": global_op_counter.divisions,
                "avaliacoes_minimo_global": self.avaliacoes_pso_minimo_global,
                "multiplicacoes_minimo_global": self.operacoes_pso_minimo_global_mult,
                "divisoes_minimo_global": self.operacoes_pso_minimo_global_div
            }

            GraficoPSO(enxame, i+1, ax, melhor_valor_g, pso_params=pso_params_for_plot)
            i += 1

        # --- Impressão dos Resultados Finais do PSO ---
        output_lines = []
        output_lines.append("\n--- Resultados Finais do PSO ---")
        output_lines.append(f'POSICAO FINAL (PSO): {melhor_posicao_g}')
        output_lines.append(f'RESULTADO FINAL (PSO): {melhor_valor_g:.4f}') # Formata para 4 casas decimais
        output_lines.append(f'Iterações executadas (PSO): {i}')
        output_lines.append(f'Avaliações da função objetivo (PSO): {self.funcao_w4_wrapper.evaluations}')
        output_lines.append(f'Operações de Multiplicação (PSO): {global_op_counter.multiplications}')
        output_lines.append(f'Operações de Divisão (PSO): {global_op_counter.divisions}')
        output_lines.append(f'Avaliações para o "melhor global" (PSO): {self.avaliacoes_pso_minimo_global}')
        output_lines.append(f'Multiplicações para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_mult}')
        output_lines.append(f'Divisões para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_div}')
        output_lines.append("--------------------------------")

        for line in output_lines:
            print(line)

        plt.show()

        # Salvar resultados em um arquivo de texto
        output_folder = "resultados_pso"
        os.makedirs(output_folder, exist_ok=True) # Cria a pasta se não existir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"resultados_pso_{timestamp}.txt"
        file_path = os.path.join(output_folder, file_name)

        with open(file_path, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        print(f"Resultados detalhados salvos em: {file_path}")


# Este bloco só será executado se PSO.py for o script principal rodado (ex: python PSO.py).
if __name__ == "__main__":
    print("\n--- Teste direto da Otimização por Enxame de Partículas (se executado como script principal) ---")
    
    # Define os parâmetros para uma execução de teste direto do PSO.
    limites_xy = [(-500, 500), (-500, 500)]
    num_particulas_pso = 15
    num_iteracoes_pso = 100
    iteracoes_sem_melhora_limite_pso = 50
    tolerancia_pso = 1e-6

    # NOVO: Parâmetros c1, c2, w_max, w_min definidos aqui no bloco principal
    c1_pso = 2.0
    c2_pso = 2.0
    w_max_pso = 0.9
    w_min_pso = 0.4

    # Cria uma instância da classe PSO e inicia a otimização
    pso_instance = PSO(
        limites=limites_xy,
        num_particulas=num_particulas_pso,
        num_iteracoes=num_iteracoes_pso,
        iteracoes_sem_melhora_limite=iteracoes_sem_melhora_limite_pso,
        tolerancia=tolerancia_pso,
        # NOVO: Passando os parâmetros de c1, c2, w_max, w_min para a classe PSO
        c1_param=c1_pso,
        c2_param=c2_pso,
        w_max_param=w_max_pso,
        w_min_param=w_min_pso
    )