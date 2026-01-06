import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from itertools import islice
import pandas as pd

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
# 5. ANÁLISE E VISUALIZAÇÃO
# ==============================================================================

def plot_results(df):
    # Tabela Resumo (Médias por Instância e Algoritmo)
    summary = df.groupby(['Instance', 'Algorithm']).agg({
        'Cost': 'mean',
        'Time': 'mean',
        'Feasible': 'mean', # % de sucesso
        'Violations': lambda x: np.mean([v for v in x if isinstance(v, (int, float))])
    }).reset_index()
    
    print("\n" + "="*40)
    print("TABELA RESUMO DAS MÉDIAS")
    print("="*40)
    print(summary.to_string())

    # --- PLOT 1: Comparação de Custos (Boxplot) ---
    # Filtramos custos absurdamente altos (penalidades) para o gráfico ficar legível
    # Se quiser ver as falhas, remova o filtro ou use log scale
    
    instances = df['Instance'].unique()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Boxplot de Custos
    # Para plotar, vamos criar arrays para o boxplot do GA e pontos para o Sequencial
    ga_data = [df[(df['Instance']==i) & (df['Algorithm']=='Genetic Algorithm')]['Cost'].values for i in instances]
    seq_data = [df[(df['Instance']==i) & (df['Algorithm']=='Sequencial')]['Cost'].values[0] for i in instances]
    
    axes[0].boxplot(ga_data, positions=instances, widths=0.6, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[0].scatter(instances, seq_data, color='red', zorder=5, label='Sequencial (Único)', s=100, marker='X')
    
    axes[0].set_title('Distribuição de Custos: GA (20 execuções) vs Sequencial')
    axes[0].set_ylabel('Custo Total (Escala Log)')
    axes[0].set_yscale('log') # Essencial devido às penalidades
    axes[0].set_xticks(instances)
    axes[0].set_xticklabels([f"Inst {i}" for i in instances])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- PLOT 2: Taxa de Viabilidade ---
    # Bar chart mostrando % de vezes que encontrou solução viável
    
    width = 0.35
    x = np.arange(len(instances))
    
    # Viabilidade Sequencial (0 ou 1)
    seq_feas = [df[(df['Instance']==i) & (df['Algorithm']=='Sequencial')]['Feasible'].mean() * 100 for i in instances]
    # Viabilidade GA (0 a 100%)
    ga_feas = [df[(df['Instance']==i) & (df['Algorithm']=='Genetic Algorithm')]['Feasible'].mean() * 100 for i in instances]
    
    axes[1].bar(x - width/2, seq_feas, width, label='Sequencial', color='salmon')
    axes[1].bar(x + width/2, ga_feas, width, label='Genetic Algo', color='skyblue')
    
    axes[1].set_title('Taxa de Viabilidade (% de Soluções Válidas)')
    axes[1].set_ylabel('% Viável')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Inst {i}" for i in instances])
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_results = run_benchmark()
    plot_results(df_results)