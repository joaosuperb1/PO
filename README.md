📦 Benchmark de Otimização: Multi-Commodity Flow Problem (MCFP)

Este projeto implementa e compara duas abordagens distintas para resolver o problema de Fluxo de Múltiplas Mercadorias em redes congestionadas: um Algoritmo Genético (Meta-heurística) e uma Heurística Sequencial (Guloso).

O objetivo é avaliar o trade-off entre tempo de execução, custo total e viabilidade das soluções em topologias de rede geradas aleatoriamente.
✨ Funcionalidades

    Geração de Instâncias: Criação automática de grafos densos com gargalos de capacidade intencionais.

    Persistência de Dados: Salva e carrega cenários (.pkl) para garantir que ambos os algoritmos sejam testados exatamente nas mesmas condições.

    Benchmark Automatizado: Executa bateria de testes (ex: 10 instâncias x 20 execuções do AG).

    Análise Visual: Gera gráficos comparativos (Boxplot de Custos e Gráfico de Barras de Viabilidade) usando matplotlib.

🚀 Como Executar
Pré-requisitos

Certifique-se de ter o Python 3.x instalado. Este projeto agora requer bibliotecas de análise de dados.
Instalação

Instale as dependências atualizadas via pip:
Bash

pip install networkx matplotlib pandas numpy seaborn

Execução

Execute o script principal para iniciar a geração de dados e o benchmark:
Bash

python main.py

    Nota: Na primeira execução, o script criará uma pasta PO/benchmark_data e gerará as instâncias. Execuções subsequentes reutilizarão esses dados para consistência.

🧠 Configuração dos Algoritmos
1. Parâmetros do Algoritmo Genético

| Parâmetro | Valor | Justificativa Técnica |
| :--- | :--- | :--- |
| **População** | 60 | Aumentado para garantir maior diversidade inicial de rotas. |
| **Gerações** | 150 | Permite tempo suficiente para convergência, dado o aumento na complexidade. |
| **Taxa de Mutação** | 15% | Alta taxa para evitar estagnação, forçando a troca de rotas no conjunto de K-Paths. |
| **K-Paths** | 8 | Limita o espaço de busca aos 8 melhores caminhos topológicos por par (s, t). |
| **Penalidade** | 50.000 | Valor extremo para simular restrições "Hard". Soluções que estouram a capacidade são descartadas. |


2. Heurística Sequencial (Baseline)

Utiliza uma abordagem determinística e gulosa. Roteia uma mercadoria por vez usando o caminho de custo mínimo (min_cost_flow) baseado na capacidade residual atual. Serve como base de comparação para velocidade e qualidade da solução.
📊 Análise de Resultados

Ao final da execução, o sistema exibe no console uma tabela resumo e gera duas visualizações:

    Distribuição de Custos (Log Scale): Um Boxplot comparando a variabilidade das 20 execuções do AG contra o resultado único da Sequencial.

    Taxa de Viabilidade: Percentual de vezes que cada algoritmo conseguiu alocar todas as demandas sem violar capacidades.

🛠️ Tecnologias Utilizadas

    Python: Linguagem core.

    NetworkX: Modelagem de grafos, cálculo de shortest paths e min cost flow.

    Pandas: Agregação estatística dos resultados do benchmark.

    Matplotlib: Visualização gráfica dos dados.

    Pickle: Serialização das instâncias de teste.