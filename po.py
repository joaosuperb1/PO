import networkx as nx
import random
import time
import matplotlib.pyplot as plt
from itertools import islice


# CONFIGURAÇÕES E PARÂMETROS (Meta-heurística)
POPULATION_SIZE = 50      # Tamanho da população
GENERATIONS = 100         # Número de iterações
MUTATION_RATE = 0.1       # Chance de mutação de um gene
K_PATHS = 5               # Pré-calcula K caminhos possíveis para cada demanda
PENALTY_FACTOR = 10000    # Penalidade por estourar capacidade da aresta


# 1. GERAÇÃO DE INSTÂNCIA E UTILITÁRIOS
def create_random_instance(num_nodes=10, num_edges=25, num_commodities=3):
    """Cria um grafo direcionado aleatório e uma lista de demandas."""
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    
    # Adiciona capacidades e custos às arestas
    for u, v in G.edges():
        G[u][v]['capacity'] = random.randint(5, 15) # Capacidade da aresta
        G[u][v]['cost'] = random.randint(1, 10)     # Custo por unidade de fluxo
        
    # Gera demandas (commodities): (origem, destino, quantidade_demanda)
    commodities = []
    for _ in range(num_commodities):
        s = random.randint(0, num_nodes-1)
        t = random.randint(0, num_nodes-1)
        while s == t or not nx.has_path(G, s, t): # Garante que existe caminho
            s = random.randint(0, num_nodes-1)
            t = random.randint(0, num_nodes-1)
        demand = random.randint(1, 5)
        commodities.append((s, t, demand))
        
    return G, commodities

def k_shortest_paths(G, source, target, k, weight='cost'):
    """Retorna os K caminhos mais curtos entre s e t."""
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except nx.NetworkXNoPath:
        return []


# 2. META-HEURÍSTICA: ALGORITMO GENÉTICO
class GeneticAlgorithmMCF:
    def __init__(self, G, commodities):
        self.G = G
        self.commodities = commodities
        # Pré-processamento: Descobre rotas possíveis para cada mercadoria
        # O "Gene" será o índice da rota escolhida nesta lista
        self.possible_routes = [] 
        for s, t, d in commodities:
            routes = k_shortest_paths(G, s, t, K_PATHS)
            self.possible_routes.append(routes)

    def fitness(self, chromosome):
        """
        Calcula o custo total. Se estourar capacidade, aplica penalidade.
        chromosome: lista de inteiros, onde cada int é o índice da rota escolhida para a mercadoria i.
        """
        total_cost = 0
        edge_usage = {} # (u,v) -> uso atual
        
        # Inicializa uso das arestas
        for u, v in self.G.edges():
            edge_usage[(u,v)] = 0
            
        penalty_count = 0
        
        for i, route_idx in enumerate(chromosome):
            # Se a rota escolhida for válida (índice existe)
            if route_idx < len(self.possible_routes[i]):
                path = self.possible_routes[i][route_idx]
                demand = self.commodities[i][2]
                
                # Soma custo e computa uso das arestas
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    edge_cost = self.G[u][v]['cost']
                    edge_usage[(u,v)] += demand
                    total_cost += edge_cost * demand

        # Verifica violações de capacidade (Restrição Hard penalizada)
        for u, v in self.G.edges():
            capacity = self.G[u][v]['capacity']
            usage = edge_usage[(u,v)]
            if usage > capacity:
                # Penalidade proporcional ao excesso
                total_cost += (usage - capacity) * PENALTY_FACTOR
                penalty_count += 1
                
        return total_cost, penalty_count

    def run(self):
        # População Inicial: Escolha aleatória de rotas
        population = []
        for _ in range(POPULATION_SIZE):
            ind = [random.randint(0, len(routes)-1) if routes else -1 for routes in self.possible_routes]
            population.append(ind)
            
        best_solution = None
        best_fitness = float('inf')
        history = []

        start_time = time.time()

        for gen in range(GENERATIONS):
            # Avaliação
            population_fitness = []
            for ind in population:
                fit, penalties = self.fitness(ind)
                population_fitness.append((ind, fit))
                
                if fit < best_fitness:
                    best_fitness = fit
                    best_solution = ind
            
            history.append(best_fitness)
            
            # Seleção (Torneio)
            next_generation = []
            # Elitismo: Mantém o melhor
            next_generation.append(best_solution) 
            
            while len(next_generation) < POPULATION_SIZE:
                parent1 = self._tournament(population_fitness)
                parent2 = self._tournament(population_fitness)
                
                # Crossover (Ponto único)
                cut = random.randint(1, len(self.commodities)-1)
                child = parent1[:cut] + parent2[cut:]
                
                # Mutação
                if random.random() < MUTATION_RATE:
                    idx_to_mutate = random.randint(0, len(child)-1)
                    if self.possible_routes[idx_to_mutate]:
                        child[idx_to_mutate] = random.randint(0, len(self.possible_routes[idx_to_mutate])-1)
                
                next_generation.append(child)
            
            population = next_generation
            
        elapsed_time = time.time() - start_time
        return best_solution, best_fitness, elapsed_time, history

    def _tournament(self, pop_fit, k=3):
        candidates = random.sample(pop_fit, k)
        # Retorna o indivíduo com menor custo (fitness)
        return min(candidates, key=lambda x: x[1])[0]


# 3. EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    print("--- Iniciando Otimização de Fluxo Multi-Commodity ---")
    
    # 1. Configurar Instância (Use fixas para benchmark, aqui é aleatória para teste)
    G, commodities = create_random_instance(num_nodes=20, num_edges=60, num_commodities=5)
    
    print(f"Grafo: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    print(f"Demandas: {len(commodities)} mercadorias para rotear.")
    
    # 2. Executar Meta-heurística
    ga = GeneticAlgorithmMCF(G, commodities)
    best_sol, cost, duration, fit_history = ga.run()
    
    # 3. Exibir Resultados
    print("\n--- Resultados Finais ---")
    print(f"Melhor Custo Encontrado: {cost}")
    print(f"Tempo de Execução: {duration:.4f} segundos")
    
    # Verifica se houve penalidade (se a solução é viável)
    final_cost, penalties = ga.fitness(best_sol)
    if penalties > 0:
        print(f"ALERTA: Solução inviável (Violação de {penalties} capacidades)")
    else:
        print("Solução Viável: Todas as capacidades respeitadas.")

    print("\nDetalhes das Rotas Escolhidas:")
    for i, route_idx in enumerate(best_sol):
        s, t, d = commodities[i]
        path = ga.possible_routes[i][route_idx]
        print(f"Commodity {i} (De {s} para {t}, Demanda {d}): Rota {path}")

    # (Opcional) Plotar convergência
    # plt.plot(fit_history)
    # plt.title("Convergência do Algoritmo Genético")
    # plt.xlabel("Geração")
    # plt.ylabel("Custo (Fitness)")
    # plt.show()