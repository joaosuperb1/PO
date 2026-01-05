Otimização de Fluxo de Múltiplas Mercadorias com Algoritmos Genéticos

Este projeto aplica Algoritmos Genéticos (AG) para resolver o problema de Fluxo de Múltiplas Mercadorias (Multi-Commodity Flow Problem). O objetivo é encontrar as melhores rotas para diferentes mercadorias em uma rede, respeitando as capacidades das arestas e minimizando custos ou penalidades.
🚀 Como Executar
Pré-requisitos

Certifique-se de ter o Python instalado em sua máquina. Este projeto utiliza as bibliotecas networkx para manipulação de grafos e matplotlib para visualização.
Instalação

    Instale as dependências necessárias via pip:
    Bash

pip install networkx matplotlib

Execute o script principal:
Bash

    python main.py

🧠 Configuração da Meta-heurística

O algoritmo foi ajustado com parâmetros específicos para balancear a exploração do espaço de busca e a convergência para soluções viáveis:
Parâmetro	Valor	Justificativa Técnica
Tamanho da População	50	Mantém a diversidade genética necessária sem comprometer a performance computacional em instâncias de pequeno a médio porte.
Gerações	100	Critério de parada fixo. Testes demonstraram que a solução tende a estabilizar (congelar) após a 80ª geração.
Taxa de Mutação	10%	Definida como alta para evitar a convergência prematura em ótimos locais, dado que o espaço de busca de caminhos combinatórios é altamente desconexo.
Penalidade	10.000	Valor robusto para converter restrições leves (soft) em rígidas (hard), forçando o descarte de indivíduos que violem a capacidade das arestas.
🛠️ Tecnologias Utilizadas

    Python: Linguagem base.

    NetworkX: Utilizada para modelagem da topologia da rede e cálculos de caminhos.

    Matplotlib: Utilizada para gerar gráficos de evolução da fitness e visualização da rede.

    Nota: Este projeto foi desenvolvido para fins acadêmicos/estudos de otimização combinatória e logística.