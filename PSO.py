# PSO.py
from Enxame import Enxame # Importa a classe Enxame, que representa o grupo de partículas.
from Grafico import GraficoPSO # Importa a função GraficoPSO para visualizar o progresso do algoritmo.
import matplotlib.pyplot as plt # Importa a biblioteca Matplotlib para criação e exibição de gráficos.
from funcoes_otimizacao import funcao_w4 # Importa a função objetivo W4, que será minimizada.
from utils import global_op_counter, FuncaoObjetivoWrapper # Importa contadores de operações e o wrapper da função objetivo.
import os # Importado para salvar arquivos (manipulação de diretórios).
from datetime import datetime # Importado para gerar timestamp (data e hora para nomes de arquivos).

class PSO:
    # Construtor da classe PSO, que inicializa os parâmetros do algoritmo.
    def __init__(self, limites, num_particulas, num_iteracoes, iteracoes_sem_melhora_limite, tolerancia, c1_param, c2_param, w_max_param, w_min_param):
        self.limites = limites # Limites do espaço de busca (ex: [(-500, 500), (-500, 500)] para x e y).
        self.num_particulas = num_particulas # Número de partículas no enxame.
        self.num_iteracoes = num_iteracoes # Número máximo de iterações do algoritmo.
        self.iteracoes_sem_melhora_limite = iteracoes_sem_melhora_limite # Limite de iterações sem melhora para critério de parada.
        self.tolerancia = tolerancia # Valor mínimo de mudança para considerar convergência.

        # NOVO: Armazena os parâmetros passados no construtor como atributos da instância
        self.c1 = c1_param # Coeficiente de aceleração cognitiva (influência da melhor posição da partícula).
        self.c2 = c2_param # Coeficiente de aceleração social (influência da melhor posição global do enxame).
        self.w_max = w_max_param # Peso de inércia máximo (influência da velocidade anterior).
        self.w_min = w_min_param # Peso de inércia mínimo (influência da velocidade anterior).

        global_op_counter.reset() # Reseta os contadores globais de operações (multiplicações e divisões).
        # Cria um "wrapper" para a função objetivo que também conta suas avaliações.
        self.funcao_w4_wrapper = FuncaoObjetivoWrapper(funcao_w4, global_op_counter)

        melhor_valor_g = float('inf') # Inicializa o melhor valor global encontrado como infinito (para minimização).
        melhor_posicao_g = [] # Inicializa a melhor posição global como uma lista vazia.
        
        self.avaliacoes_pso_minimo_global = 0 # Armazena o número de avaliações quando o melhor global foi *encontrado*.
        self.operacoes_pso_minimo_global_mult = 0 # Armazena multiplicações quando o melhor global foi *encontrado*.
        self.operacoes_pso_minimo_global_div = 0 # Armazena divisões quando o melhor global foi *encontrado*.

        iteracoes_sem_melhora = 0 # Contador para iterações consecutivas sem melhora significativa.
        
        enxame = [] # Lista para armazenar as partículas do enxame.
        # Loop para criar e adicionar as partículas ao enxame.
        for i in range(self.num_particulas):
            # Cria uma nova instância de Enxame (partícula) com os limites e parâmetros.
            enxame.append(Enxame(self.limites, self.c1, self.c2, self.w_max, self.w_min))

        # Configuração inicial do gráfico 3D.
        fig = plt.figure(figsize=(11, 8)) # Cria uma nova figura para o gráfico.
        ax = fig.add_subplot(111, projection='3d') # Adiciona um subplot 3D à figura.
        plt.subplots_adjust(right=0.7) # Ajusta o layout para deixar espaço para a legenda à direita.

        i = 0 # Inicializa o contador de iterações.
        # Loop principal do algoritmo PSO, que continua até o número máximo de iterações ou convergência.
        while i < self.num_iteracoes:
            valor_global_antes_atualizacao = melhor_valor_g # Guarda o melhor valor global da iteração anterior para comparação.

            # Loop para avaliar cada partícula no enxame.
            for j in range(self.num_particulas):
                # Avalia a posição atual da partícula usando a função objetivo (e conta as avaliações).
                enxame[j].avaliar(self.funcao_w4_wrapper)

                # Verifica se a posição atual da partícula é melhor que a melhor global encontrada até agora.
                if enxame[j].valor_atual_i < melhor_valor_g:
                    melhor_posicao_g = list(enxame[j].posicao_i) # Atualiza a melhor posição global.
                    melhor_valor_g = float(enxame[j].valor_atual_i) # Atualiza o melhor valor global.
                    # Registra o número de avaliações e operações *no momento* em que o melhor global foi encontrado.
                    self.avaliacoes_pso_minimo_global = self.funcao_w4_wrapper.evaluations
                    self.operacoes_pso_minimo_global_mult = global_op_counter.multiplications
                    self.operacoes_pso_minimo_global_div = global_op_counter.divisions

            # Lógica de parada por convergência (se a melhora global for menor que a tolerância).
            if i > 0 and abs(melhor_valor_g - valor_global_antes_atualizacao) < self.tolerancia:
                iteracoes_sem_melhora += 1 # Incrementa o contador de iterações sem melhora.
            else:
                iteracoes_sem_melhora = 0 # Reseta o contador se houve melhora.

            # Verifica se o limite de iterações sem melhora foi atingido.
            if iteracoes_sem_melhora >= self.iteracoes_sem_melhora_limite:
                print(f"\n[PSO] Parada por convergência: Mudança no melhor valor global menor que {self.tolerancia} por {self.iteracoes_sem_melhora_limite} iterações.")
                break # Sai do loop principal se houver convergência.

            # Loop para atualizar a velocidade e posição de cada partícula.
            for j in range(self.num_particulas):
                # Atualiza a velocidade da partícula com base na melhor posição global e na sua própria melhor posição.
                enxame[j].atualizar_velocidade(melhor_posicao_g, i, self.num_iteracoes)
                # Atualiza a posição da partícula com base em sua nova velocidade e limites.
                enxame[j].atualizar_posicao(self.limites)

            # MODIFICADO: Agora pego os valores de c1, c2, w_max, w_min dos atributos da instância self.
            # Dicionário com parâmetros e estatísticas para passar ao gráfico.
            pso_params_for_plot = {
                "c1": self.c1,
                "c2": self.c2,
                "w_max": self.w_max,
                "w_min": self.w_min,
                "iteracoes_totais": self.num_iteracoes,
                "num_particulas": self.num_particulas,
                "avaliacoes_funcao": self.funcao_w4_wrapper.evaluations, # Avaliações totais até o momento.
                "multiplicacoes_total": global_op_counter.multiplications, # Multiplicações totais até o momento.
                "divisoes_total": global_op_counter.divisions, # Divisões totais até o momento.
                "avaliacoes_minimo_global": self.avaliacoes_pso_minimo_global, # Avaliações no momento do melhor global.
                "multiplicacoes_minimo_global": self.operacoes_pso_minimo_global_mult, # Multiplicações no momento do melhor global.
                "divisoes_minimo_global": self.operacoes_pso_minimo_global_div # Divisões no momento do melhor global.
            }

            # Chama a função para desenhar o gráfico da iteração atual.
            GraficoPSO(enxame, i+1, ax, melhor_valor_g, pso_params=pso_params_for_plot)
            i += 1 # Incrementa o contador de iterações.

        # --- Impressão dos Resultados Finais do PSO ---
        output_lines = [] # Lista para armazenar as linhas de saída.
        output_lines.append("\n--- Resultados Finais do PSO ---")
        output_lines.append(f'POSICAO FINAL (PSO): {melhor_posicao_g}') # Posição final da melhor solução global.
        output_lines.append(f'RESULTADO FINAL (PSO): {melhor_valor_g:.4f}') # Valor final da função para a melhor solução.
        output_lines.append(f'Iterações executadas (PSO): {i}') # Número total de iterações executadas.
        output_lines.append(f'Avaliações da função objetivo (PSO): {self.funcao_w4_wrapper.evaluations}') # Avaliações totais (convergência).
        output_lines.append(f'Operações de Multiplicação (PSO): {global_op_counter.multiplications}') # Multiplicações totais (convergência).
        output_lines.append(f'Operações de Divisão (PSO): {global_op_counter.divisions}') # Divisões totais (convergência).
        output_lines.append(f'Avaliações para o "melhor global" (PSO): {self.avaliacoes_pso_minimo_global}') # Avaliações no momento do melhor global.
        output_lines.append(f'Multiplicações para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_mult}') # Multiplicações no momento do melhor global.
        output_lines.append(f'Divisões para o "melhor global" (PSO): {self.operacoes_pso_minimo_global_div}') # Divisões no momento do melhor global.
        output_lines.append("--------------------------------")

        # Imprime as linhas de resultado no console.
        for line in output_lines:
            print(line)

        plt.show() # Exibe o gráfico final.

        # Salvar resultados em um arquivo de texto.
        output_folder = "resultados_pso" # Define o nome da pasta para salvar os resultados.
        os.makedirs(output_folder, exist_ok=True) # Cria a pasta se ela não existir.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Gera um timestamp para o nome do arquivo.
        file_name = f"resultados_pso_{timestamp}.txt" # Define o nome do arquivo.
        file_path = os.path.join(output_folder, file_name) # Cria o caminho completo do arquivo.

        # Escreve os resultados no arquivo de texto.
        with open(file_path, "w") as f:
            for line in output_lines:
                f.write(line + "\n")
        print(f"Resultados detalhados salvos em: {file_path}") # Informa onde os resultados foram salvos.


# Este bloco só será executado se PSO.py for o script principal rodado (ex: python PSO.py).
if __name__ == "__main__":
    print("\n--- Teste direto da Otimização por Enxame de Partículas (se executado como script principal) ---")
    
    # Define os parâmetros de teste para uma execução direta.
    limites_xy = [(-500, 500), (-500, 500)] # Limites para as variáveis x e y da função.
    num_particulas_pso = 15 # Número de partículas para o teste.
    num_iteracoes_pso = 100 # Número de iterações para o teste.
    iteracoes_sem_melhora_limite_pso = 50 # Limite de iterações sem melhora para o teste.
    tolerancia_pso = 1e-6 # Tolerância para convergência para o teste.

    c1_pso = 2.0 # Parâmetro c1 para o teste.
    c2_pso = 2.0 # Parâmetro c2 para o teste.
    w_max_pso = 0.9 # Parâmetro w_max para o teste.
    w_min_pso = 0.4 # Parâmetro w_min para o teste.

    # Cria uma instância da classe PSO e inicia a otimização com os parâmetros de teste.
    pso_instance = PSO(
        limites=limites_xy,
        num_particulas=num_particulas_pso,
        num_iteracoes=num_iteracoes_pso,
        iteracoes_sem_melhora_limite=iteracoes_sem_melhora_limite_pso,
        tolerancia=tolerancia_pso,
        c1_param=c1_pso,
        c2_param=c2_pso,
        w_max_param=w_max_pso,
        w_min_param=w_min_pso
    )