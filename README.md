<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-algorithm-impartiality/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  Elaboração de uma estratégia de equidade para sistemas de recomendação
</h3>

<p align="center">Algoritmo de equidade visando a diminuição da injustiça do grupo em sistemas de recomendação. </p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-algorithm-impartiality?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-algorithm-impartiality/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-algorithm-impartiality?style=social">
  </a>
</p>

<p align="center">
  <a href="#-sobre">Sobre o projeto</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-links">Links</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">Licença</a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

Como um aplicativo altamente orientado por dados, os sistemas de recomendação podem ser afetados por distorções de dados, culminando em resultados injustos para diferentes grupos de dados, o que pode ser um motivo a afetar o desempenho do sistema. Portanto, é importante identificar e resolver as questões de injustiça em cenários de recomendação. Desenvolvemos, portanto, um algoritmo de equidade visando a diminuição da injustiça do grupo em sistemas de recomendação. O algoritmo foi testado em três conjuntos de dados existentes (MovieLens, Songs e GoodBooks) com três estratégias de agrupamento de usuários. Conseguimos reduzir a injustiça do grupo nos três conjuntos de dados, considerando as três estratégias de agrupamento. 

### :balance_scale: Medidas de Justiça <a name="-medidas"/></a>

* Polarization (Polarização): Para capturar a polarização, buscamos medir a extensão na qual as avaliações dos usuários discordam. Assim, para medir a polarização dos usuários, consideramos as avaliações estimadas \( \hat{X} \), e definimos a métrica de polarização como a soma normalizada das distâncias euclidianas entre pares de avaliações estimadas de usuários, isto é, entre linhas de \( \hat{X} \).

* Individual fairness (Justiça Individual): Para cada usuário \(i\), definimos \(ℓ_i\), a perda do usuário \(i\), como o erro quadrático médio da estimativa sobre as avaliações conhecidas do usuário \(i\):

* Individual fairness (Justiça do Grupo): Justiça de grupo. Seja \(I\) o conjunto de todos os usuários/itens e \(G = \{G_1, ..., G_{g}\}\) uma partição de usuários/itens em \(g\) grupos, isto é, \(I = \cup_{i \in \{1, ..., g\}} G_i\). Definimos a perda do grupo \(i\) como o erro quadrático médio da estimativa sobre todas as avaliações conhecidas no grupo \(i\):


### :notebook_with_decorative_cover: Algoritmo <a name="-algoritmo"/></a>

<img src="https://github.com/ravarmes/recsys-algorithm-impartiality/blob/main/assets/recsys-algorithm-impartiality-1.png" width="700">


### :chart_with_upwards_trend: Resultados(s) <a name="-resultados"/></a>

<img src="https://github.com/ravarmes/recsys-algorithm-impartiality/blob/main/assets/recsys-algorithm-impartiality-2.png" width="700">

<img src="https://github.com/ravarmes/recsys-algorithm-impartiality/blob/main/assets/recsys-algorithm-impartiality-3.png" width="700">


### Arquivos

| Arquivo                               | Descrição                                                                                                                                                                                                                                   |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmImpartiality                | Classe para promover justiça nas recomendações de algoritmos de sistemas de recomendação.                                                                                                                                                   |
| AlgorithmUserFairness                | Classes para medir a justiça (polarização, justiça individual e justiça do grupo) das recomendações de algoritmos de sistemas de recomendação.                                                                                               |
| RecSys                               | Classe no padrão fábrica para instanciar um sistema de recomendação com base em parâmetros string.                                                                                                                                           |
| RecSysALS                            | Alternating Least Squares (ALS) para Filtragem Colaborativa é um algoritmo que otimiza iterativamente duas matrizes para melhor prever avaliações de usuários em itens, baseando-se na ideia de fatoração de matrizes.                       |
| RecSysKNN                            | K-Nearest Neighbors para Sistemas de Recomendação é um método que recomenda itens ou usuários baseando-se na proximidade ou similaridade entre eles, utilizando a técnica dos K vizinhos mais próximos.                                      |
| RecSysNMF                            | Non-Negative Matrix Factorization para Sistemas de Recomendação utiliza a decomposição de uma matriz de avaliações em duas matrizes de fatores não-negativos, revelando padrões latentes que podem ser usados para prever avaliações faltantes. |
| RecSysSGD                            | Stochastic Gradient Descent para Sistemas de Recomendação é uma técnica de otimização que ajusta iterativamente os parâmetros do modelo para minimizar o erro nas previsões de avaliações, através de atualizações baseadas em gradientes calculados de forma estocástica. |
| RecSysSVD                            | Singular Value Decomposition para Sistemas de Recomendação é um método que fatora a matriz de avaliações em três matrizes menores, capturando informações essenciais sobre usuários e itens, o que facilita a recomendação através da reconstrução da matriz original com dados faltantes preenchidos. |
| RecSysNCF                            | Neural Collaborative Filtering é uma abordagem moderna para filtragem colaborativa que utiliza redes neurais para modelar interações complexas e não-lineares entre usuários e itens, visando aprimorar a qualidade das recomendações.          |
| TestAlgorithmImpartiality_Age        | Script de teste do algoritmo de imparcialidade (AlgorithmImpartiality) considerando o agrupamento dos usuários por idade (Age).                                                                                                              |
| TestAlgorithmImpartiality_Age_SaveTXT| Script de teste do algoritmo de imparcialidade (AlgorithmImpartiality) considerando o agrupamento dos usuários por idade (Age) salvando os resultados em arquivo TXT.                                                                        |
| TestAlgorithmImpartiality_Gender     | Script de teste do algoritmo de imparcialidade (AlgorithmImpartiality) considerando o agrupamento dos usuários por gênero (Gender).                                                                                                          |
| TestAlgorithmImpartiality_Gender_SaveTXT | Script de teste do algoritmo de imparcialidade (AlgorithmImpartiality) considerando o agrupamento dos usuários por gênero (Gender) salvando os resultados em arquivo TXT.                                                                 |
| TestAlgorithmImpartiality_NR         | Script de teste do algoritmo de imparcialidade (AlgorithmImpartiality) considerando o agrupamento dos usuários por número de avaliações (NR).                                                                                                |
| TestAlgorithmImpartiality_NR_SaveTXT | Script de teste do algoritmo de imparcialidade (AlgorithmImpartiality) considerando o agrupamento dos usuários por número de avaliações (NR) salvando os resultados em arquivo TXT.                                                         |



## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink:
