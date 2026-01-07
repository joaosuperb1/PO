import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from itertools import islice
import pandas as pd
import seaborn as sns

# --- CONFIGURAÇÕES DO EXPERIMENTO ---
NUM_INSTANCES = 10        # Quantos cenários diferentes
NUM_RUNS_GA = 20          # Quantas vezes rodar o GA por cenário
OUTPUT_DIR = "PO/benchmark_data"

# --- PARÂMETROS DO PROBLEMA ---
POPULATION_SIZE = 60
GENERATIONS = 150
MUTATION_RATE = 0.15
K_PATHS = 8
PENALTY_FACTOR = 50000

# ==============================================================================
# 1. GERAÇÃO DE INSTÂNCIAS E PERSISTÊNCIA
# ==============================================================================

def create_complex_instance(num_nodes=50, num_edges=150, num_commodities=15):
    """Cria uma rede densa e congestionada."""
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    for u, v in G.edges():
        # Capacidades baixas para gerar gargalos (8 a 15 unidades)
        G[u][v]['capacity'] = random.randint(8, 15) 
        G[u][v]['cost'] = random.randint(1, 20)
    
    commodities = []
    while len(commodities) < num_commodities:
        s, t = random.sample(range(num_nodes), 2)
        if nx.has_path(G, s, t):
            demand = random.randint(3, 5) # Demanda alta
            commodities.append({'s': s, 't': t, 'd': demand})
    return G, commodities

def generate_and_save_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Gerando {NUM_INSTANCES} novas instâncias em '{OUTPUT_DIR}'...")
        for i in range(NUM_INSTANCES):
            G, comms = create_complex_instance()
            filename = os.path.join(OUTPUT_DIR, f"instancia_{i}.pkl")
            with open(filename, "wb") as f:
                pickle.dump((G, comms), f)
        print("Dataset gerado com sucesso.")
    else:
        print(f"Diretório '{OUTPUT_DIR}' já existe. Usando instâncias salvas.")

def load_instance(index):
    filename = os.path.join(OUTPUT_DIR, f"instancia_{index}.pkl")
    with open(filename, "rb") as f:
        return pickle.load(f)

# ==============================================================================
# 2. VERIFICAÇÃO DE RESTRIÇÕES (FÍSICA DA REDE)
# ==============================================================================

def verify_constraints(G, commodities, paths_indices, all_possible_paths):
    """
    Calcula métricas físicas detalhadas: 
    - Violação de Capacidade
    - Conservação de Fluxo (Garantida pela representação de caminho, mas verificamos totais)
    """
    edge_usage = {e: 0 for e in G.edges()}
    total_flow_routed = 0
    total_demand = sum(c['d'] for c in commodities)
    
    # Mapear uso das arestas
    for i, route_idx in enumerate(paths_indices):
        if route_idx == -1: continue # Rota não encontrada ou inválida
        
        path = all_possible_paths[i][route_idx]
        demand = commodities[i]['d']
        total_flow_routed += demand
        
        for j in range(len(path) - 1):
            u, v = path[j], path[j+1]
            edge_usage[(u,v)] += demand

    # Verificar Capacidades
    violated_edges = 0
    total_excess = 0
    for u, v in G.edges():
        cap = G[u][v]['capacity']
        usage = edge_usage[(u,v)]
        if usage > cap:
            violated_edges += 1
            total_excess += (usage - cap)
            
    metrics = {
        "demand_met_pct": (total_flow_routed / total_demand) * 100 if total_demand > 0 else 0,
        "violated_edges_count": violated_edges,
        "total_capacity_excess": total_excess,
        "is_feasible": violated_edges == 0
    }
    return metrics

# ==============================================================================
# 3. ALGORITMOS (ADAPTADOS PARA O BENCHMARK)
# ==============================================================================

# --- HEURÍSTICA SEQUENCIAL (Determinística) ---
def solve_sequential_mcmf(G, commodities):
    temp_G = G.copy()
    total_cost = 0
    start = time.time()
    
    # Armazena os caminhos escolhidos para validação posterior
    # Como o sequential altera o grafo, vamos simular a escolha de rotas
    # Nota: Para simplificar a comparação exata de métricas, aqui focamos no Custo e Viabilidade
    # retornados pelo próprio processo sequencial.
    
    feasible_count = 0
    edge_usage = {e: 0 for e in G.edges()}
    
    for cmd in commodities:
        s, t, d = cmd['s'], cmd['t'], cmd['d']
        try:
            temp_G.nodes[s]['demand'] = -d
            temp_G.nodes[t]['demand'] = d
            flow_dict = nx.min_cost_flow(temp_G)
            
            # Contabiliza custo e atualiza residual
            path_found = False
            for u, v_dist in flow_dict.items():
                for v, flow in v_dist.items():
                    if flow > 0:
                        path_found = True
                        total_cost += flow * temp_G[u][v]['cost']
                        temp_G[u][v]['capacity'] -= flow
                        edge_usage[(u,v)] += flow # Rastreio para validação
            
            if path_found: feasible_count += 1
            
            temp_G.nodes[s]['demand'] = 0
            temp_G.nodes[t]['demand'] = 0
            
        except nx.NetworkXUnfeasible:
            # Se falhar, penaliza
            total_cost += PENALTY_FACTOR
            
    duration = time.time() - start
    
    # Verifica violações (Neste algoritmo greedy, violação geralmente resulta em NetworkXUnfeasible
    # ou falha em rotear, mas vamos contar "arestas estouradas" como 0 se respeitou o residual,
    # porém a "Inviabilidade" vem de não conseguir rotear a demanda).
    
    is_fully_feasible = (feasible_count == len(commodities))
    
    return {
        "cost": total_cost,
        "time": duration,
        "feasible": is_fully_feasible,
        "method": "Sequential"
    }

# --- ALGORITMO GENÉTICO ---
class GeneticAlgorithmMCF:
    def __init__(self, G, commodities):
        self.G = G
        self.commodities = commodities
        self.possible_routes = [
            list(islice(nx.shortest_simple_paths(G, c['s'], c['t'], weight='cost'), K_PATHS))
            for c in commodities
        ]

    def fitness(self, chromosome):
        cost = 0
        usage = {e: 0 for e in self.G.edges()}
        
        for i, idx in enumerate(chromosome):
            path = self.possible_routes[i][idx]
            d = self.commodities[i]['d']
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                usage[(u,v)] += d
                cost += self.G[u][v]['cost'] * d
        
        penalties = 0
        for e in self.G.edges():
            if usage[e] > self.G[e[0]][e[1]]['capacity']:
                excess = usage[e] - self.G[e[0]][e[1]]['capacity']
                cost += excess * PENALTY_FACTOR
                penalties += 1
        return cost, penalties

    def run(self):
        start = time.time()
        pop = [[random.randint(0, len(r)-1) for r in self.possible_routes] for _ in range(POPULATION_SIZE)]
        best_sol = None
        best_fit = float('inf')
        
        for _ in range(GENERATIONS):
            fits = [self.fitness(ind) for ind in pop]
            # Elitismo
            min_fit = min(fits, key=lambda x: x[0])
            if min_fit[0] < best_fit:
                best_fit = min_fit[0]
                best_sol = pop[fits.index(min_fit)]
            
            # Nova população
            new_pop = [best_sol]
            while len(new_pop) < POPULATION_SIZE:
                # Torneio simples
                p1 = pop[random.randint(0, len(pop)-1)]
                p2 = pop[random.randint(0, len(pop)-1)]
                # Crossover
                cut = random.randint(1, len(self.commodities)-1)
                child = p1[:cut] + p2[cut:]
                # Mutação
                if random.random() < MUTATION_RATE:
                    idx = random.randint(0, len(child)-1)
                    child[idx] = random.randint(0, len(self.possible_routes[idx])-1)
                new_pop.append(child)
            pop = new_pop
            
        duration = time.time() - start
        
        # Coleta métricas detalhadas da melhor solução
        constraints = verify_constraints(self.G, self.commodities, best_sol, self.possible_routes)
        
        return {
            "cost": best_fit,
            "time": duration,
            "feasible": constraints['is_feasible'],
            "violated_edges": constraints['violated_edges_count'],
            "excess_capacity": constraints['total_capacity_excess'],
            "method": "Genetic Algo"
        }

# ==============================================================================
# 4. LOOP PRINCIPAL DE BENCHMARK
# ==============================================================================

def run_benchmark():
    generate_and_save_dataset()
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"INICIANDO BENCHMARK ({NUM_INSTANCES} Instâncias x {NUM_RUNS_GA} Runs GA)")
    print(f"{'='*60}\n")
    
    for i in range(NUM_INSTANCES):
        print(f"--- Processando Instância {i+1}/{NUM_INSTANCES} ---")
        G, comms = load_instance(i)
        
        # 1. Executar Sequencial (Determinístico -> 1 vez basta)
        seq_res = solve_sequential_mcmf(G, comms)
        # Replicamos o resultado para facilitar a plotagem lado a lado, 
        # ou guardamos como baseline.
        results.append({
            "Instance": i,
            "Algorithm": "Sequencial",
            "Run_ID": 0,
            "Cost": seq_res["cost"],
            "Time": seq_res["time"],
            "Feasible": seq_res["feasible"],
            "Violations": "N/A (Failed Route)" if not seq_res["feasible"] else 0
        })
        
        # 2. Executar GA (Estocástico -> Várias vezes)
        ga_costs = []
        ga_feasibility = 0
        
        for run_id in range(NUM_RUNS_GA):
            ga = GeneticAlgorithmMCF(G, comms)
            ga_res = ga.run()
            
            results.append({
                "Instance": i,
                "Algorithm": "Genetic Algorithm",
                "Run_ID": run_id,
                "Cost": ga_res["cost"],
                "Time": ga_res["time"],
                "Feasible": ga_res["feasible"],
                "Violations": ga_res["violated_edges"]
            })
            ga_costs.append(ga_res["cost"])
            if ga_res["feasible"]: ga_feasibility += 1
            
        print(f"   > Sequencial Custo: {seq_res['cost']:.0f} | Viável: {seq_res['feasible']}")
        print(f"   > GA Média Custo: {np.mean(ga_costs):.0f} | Viabilidade: {ga_feasibility}/{NUM_RUNS_GA} runs")

    return pd.DataFrame(results)

# ==============================================================================
# 5. ANÁLISE E VISUALIZAÇÃO (CORRIGIDA E MELHORADA)
# ==============================================================================

def plot_results(df):
    """
    Gera um dashboard analítico comparando GA vs Sequencial.
    - Removeu o plot de trade-off.
    - Garante consistência de cores entre todos os gráficos.
    - Novo layout 5-painéis.
    """
    # Configuração estética do Seaborn
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    # --- DEFINIÇÃO DE CORES CONSISTENTES ---
    # Usamos a paleta padrão 'deep' do seaborn e fixamos as cores para cada algoritmo.
    # Assim, não importa a ordem dos dados, a cor será sempre a mesma.
    default_palette = sns.color_palette("deep")
    algorithm_colors = {
        "Sequencial": default_palette[0],       # Geralmente Azul
        "Genetic Algorithm": default_palette[1] # Geralmente Laranja
    }

    # Cria uma figura grande.
    # Layout novo: 3 linhas. As duas primeiras têm 2 colunas. A última linha ocupa tudo.
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2)
    
    # Identificar instâncias únicas para ordenação dos eixos X
    instances = sorted(df['Instance'].unique())
    
    # --- LÓGICA DE ZOOM INTELIGENTE (Mantida) ---
    cutoff_threshold = df['Cost'].median() * 2.5  
    df_zoom = df[df['Cost'] < cutoff_threshold].copy()

    # =================================================================
    # LINHA 1: ANÁLISE DE CUSTO (MACRO vs MICRO)
    # =================================================================
    
    # [0,0] Visão Geral (Inclui Penalidades) - Escala Log
    ax0 = fig.add_subplot(gs[0, 0])
    sns.boxplot(x='Instance', y='Cost', hue='Algorithm', data=df, ax=ax0, 
                palette=algorithm_colors, showfliers=True, linewidth=1)
    ax0.set_title('1. Visão Macro: Custos Totais (Com Penalidades)', fontweight='bold')
    ax0.set_yscale('log')
    ax0.set_ylabel('Custo Total (R$) - Log Scale')
    ax0.legend(loc='upper right', frameon=True)

    # [0,1] Visão Micro (Zoom nos Vencedores)
    ax1 = fig.add_subplot(gs[0, 1])
    if not df_zoom.empty:
        sns.boxplot(x='Instance', y='Cost', hue='Algorithm', data=df_zoom, ax=ax1, 
                    palette=algorithm_colors, showfliers=False, linewidth=1)
        
        # Adiciona os pontos do Sequencial para destacar (usando a mesma cor definida)
        seq_data = df_zoom[df_zoom['Algorithm'] == 'Sequencial']
        if not seq_data.empty:
            sns.stripplot(x='Instance', y='Cost', data=seq_data, ax=ax1, 
                          color=algorithm_colors["Sequencial"], size=6, marker='D', jitter=False, zorder=5)
        
        ax1.set_title('2. Visão Micro: Detalhe (Sem Penalidades Extremas)', fontweight='bold', color='#333333')
        ax1.set_ylabel('Custo Real (R$)')
        ax1.legend_.remove() # Remove legenda duplicada para limpar
    else:
        ax1.text(0.5, 0.5, "Todas as soluções foram penalizadas (Custo muito alto).", ha='center')
        ax1.set_title('2. Visão Micro (Sem dados)', fontweight='bold')

    # =================================================================
    # LINHA 2: PERFORMANCE E ROBUSTEZ
    # =================================================================

    # [1,0] Tempo de Execução
    ax2 = fig.add_subplot(gs[1, 0])
    sns.barplot(x='Instance', y='Time', hue='Algorithm', data=df, ax=ax2, 
                palette=algorithm_colors, errorbar=('ci', 95), capsize=.1)
    ax2.set_title('3. Custo Computacional (Tempo de Execução)', fontweight='bold')
    ax2.set_ylabel('Tempo (s) - Log Scale')
    ax2.set_yscale('log') # Escala log essencial aqui
    ax2.legend_.remove()

    # [1,1] Taxa de Viabilidade
    ax3 = fig.add_subplot(gs[1, 1])
    # Agrupa para calcular a média de sucesso (0.0 a 1.0)
    feasibility_data = df.groupby(['Instance', 'Algorithm'])['Feasible'].mean().reset_index()
    sns.barplot(x='Instance', y='Feasible', hue='Algorithm', data=feasibility_data, ax=ax3, 
                palette=algorithm_colors, alpha=0.9)
    ax3.set_title('4. Robustez (% de Soluções Válidas)', fontweight='bold')
    ax3.set_ylabel('Taxa de Sucesso (0 a 1)')
    ax3.set_ylim(0, 1.05)
    # Adiciona uma linha de 100% para referência
    ax3.axhline(1.0, color='grey', linestyle='--', linewidth=1)
    ax3.legend_.remove()

    # =================================================================
    # LINHA 3: DIAGNÓSTICO (Ocupa toda a largura)
    # =================================================================

    # [2, :] Média de Violações (Diagnóstico de Falha)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Preenche NaN com 0 para evitar erros no plot (assumindo que NaN significa 0 violações ou método diferente)
    df_vio = df.copy()
    df_vio['Violations'] = pd.to_numeric(df_vio['Violations'], errors='coerce').fillna(0)
    
    sns.barplot(x='Instance', y='Violations', hue='Algorithm', data=df_vio, ax=ax4, 
                palette=algorithm_colors, errorbar=('ci', 68)) # CI 68% é o desvio padrão padrão
    
    ax4.set_title('5. Diagnóstico: Média de Restrições Violadas (Arestas com Capacidade Estourada)', fontweight='bold')
    ax4.set_ylabel('Qtd. Média de Violações')
    # Adiciona grids horizontais menores para facilitar a leitura de valores baixos
    ax4.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
    ax4.minorticks_on()
    ax4.xaxis.grid(False) # Desliga grid vertical para limpar
    ax4.legend_.remove()

    # Ajustes finais de espaçamento
    plt.tight_layout()
    # Adiciona um título global (opcional)
    fig.suptitle("Benchmark de Roteamento: Sequencial vs. Algoritmo Genético", y=1.02, fontsize=16, fontweight='bold')
    plt.show()

    # Imprime resumo textual também
    print("\n" + "="*50)
    print("TABELA RESUMO (MÉDIAS GERAIS)")
    print("="*50)
    summary = df.groupby(['Algorithm']).agg({
        'Cost': 'mean', 
        'Time': 'mean', 
        'Feasible': lambda x: f"{x.mean()*100:.1f}%",
        'Violations': lambda x: pd.to_numeric(x, errors='coerce').fillna(0).mean()
    }).rename(columns={'Feasible': 'Viabilidade', 'Violations': 'Média Violações'})
    print(summary)
    print("="*50)

    
if __name__ == "__main__":
    df_results = run_benchmark()
    plot_results(df_results)