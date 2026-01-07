import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os

# --- CONFIGURAÇÕES ---
# Caminho onde estão os arquivos .pkl
INPUT_DIR = "/media/superbi/Vault/Alessandra/PO/benchmark_data"

# Caminho onde vamos salvar as imagens geradas
OUTPUT_IMG_DIR = os.path.join(INPUT_DIR, "visualizacoes_grafos")

def visualize_instances():
    # Cria a pasta de imagens se não existir
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)
        print(f"Criada pasta de saída: {OUTPUT_IMG_DIR}")

    # Lista todos os arquivos na pasta
    try:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pkl')]
        files.sort() # Ordena para processar instancia_0, instancia_1, etc.
    except FileNotFoundError:
        print(f"ERRO: Diretório não encontrado: {INPUT_DIR}")
        return

    if not files:
        print("Nenhum arquivo .pkl encontrado.")
        return

    print(f"Encontrados {len(files)} arquivos. Iniciando geração de imagens...")

    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)
        
        # Carrega o arquivo pickle
        with open(filepath, "rb") as f:
            try:
                # O pickle contém uma tupla (Grafo, Commodities)
                G, comms = pickle.load(f)
            except Exception as e:
                print(f"Erro ao ler {filename}: {e}")
                continue

        # --- PREPARAÇÃO DO PLOT ---
        plt.figure(figsize=(12, 10))
        
        # Define o layout
        pos = nx.spring_layout(G, seed=42, k=0.5) 

        # Identifica nós de origem (S) e destino (T) para colorir
        all_sources = set(c['s'] for c in comms)
        all_targets = set(c['t'] for c in comms)
        
        node_colors = []
        for node in G.nodes():
            if node in all_sources and node in all_targets:
                node_colors.append('#b39eb5') # Roxo Pastel (Misto - Hub)
            elif node in all_sources:
                node_colors.append('#77dd77') # Verde pastel (Origem)
            elif node in all_targets:
                node_colors.append('#ff6961') # Vermelho pastel (Destino)
            else:
                node_colors.append('#aec6cf') # Azul pastel (Nó de passagem)

        # Desenha os nós
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        # Desenha as arestas
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, width=1.0)

        # Rótulos das arestas (Mostra: "Cap: X | R$ Y")
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            cap = data.get('capacity', '?')
            cost = data.get('cost', '?')
            # Formatação compacta para não poluir
            edge_labels[(u, v)] = f"C:{cap}\n${cost}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, alpha=0.8)

        # Título e Legenda
        plt.title(f"Visualização: {filename}\nNodes: {len(G.nodes)} | Edges: {len(G.edges)} | Commodities: {len(comms)}", fontsize=14)
        
        # Legenda manual
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#77dd77', label='Origem (Source)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6961', label='Destino (Target)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#b39eb5', label='Misto (Hub)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#aec6cf', label='Passagem', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Salva o arquivo
        output_filename = filename.replace('.pkl', '.png')
        save_path = os.path.join(OUTPUT_IMG_DIR, output_filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close() 
        
        print(f"Salvo: {output_filename}")

    print(f"\nConcluído! Verifique as imagens em: {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    visualize_instances()