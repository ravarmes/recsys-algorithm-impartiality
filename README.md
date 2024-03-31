<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-algorithm-impartiality/blob/master/assets/logo.jpg" />
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

Como um aplicativo altamente orientado por dados, os sistemas de recomendação podem ser afetados por distorções de dados, culminando em resultados injustos para diferentes grupos de dados, o que pode ser um motivo a afetar o desempenho do sistema. Portanto, é importante identificar e resolver as questões de injustiça em cenários de recomendação. Desenvolvemos, portanto, um algoritmo de equidade visando a diminuição da injustiça do grupo em sistemas de recomendação. O algoritmo foi testado em dois conjunto de dados existentes (MovieLens e Songs) com duas estratégias de agrupamento de usuários. Conseguimos reduzir a injustiça do grupo nos dois conjuntos de dados, considerando as duas estratégias de agrupamento. 

### Funções de Objetivo Social (Social Objective Functions)

* Individual fairness (Justiça Individual): a perda do usuário i é a estimativa do erro quadrático médio sobre as classificações conhecidas do usuário i

### Arquivos

- AlgorithmImpartiality: implementação da estratégia de equidade para sistemas de recomendação
- AlgorithmUserFairness: implementação do cálculo das medidas sociais: justiça individual, justiça do grupo e polarização
- RecSys: implementação da classe genérica para a utilização do sistema de recomendação
- RecSysALS: implementação do sistema de recomendação baseado em filtragem colaborativa utilizando ALS (mínimos quadrados alternados)
- RecSysExampleData20Items: implementação de uma matriz de recomendações estimadas (apenas exemplo com valores aleatórios)
- UserFairness: implementação das funções de objetivo social (polarização, justiça individual e justiça do grupo)

- TestImpartiality_Books_K_G: arquivo para testar a implementação AlgorithmImpartiality com base no conjunto de dados Books com K matrizes estimadas e G grupos (G2: agrupamento 95-5, G3: agrupamento hierárquico)


- TestUserFairness_Books: arquivo para testar a implementação UserFairness com base no conjunto de dados Books
- TestUserFairness_MovieLens_1M: arquivo para testar a implementação UserFairness com base no conjunto de dados MovieLens-1M
- TestUserFairness_MovieLens_Small: arquivo para testar a implementação UserFairness com base no conjunto de dados MovieLens-Small (40 usuários e 20 itens)
- TestCluster_Books_01: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados Books e nas variáveis justiça individual, idade e localização.
- TestCluster_Books_02: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados Books e na variável justiça individual.
- TestCluster_MovieLens_1M_01: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-1M e nas variáveis justiça individual, idade e ocupação.
- TestCluster_MovieLens_1M_02: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-1M e na variável justiça individual.
- TestCluster_MovieLens_Small_01: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-Small e nas variáveis justiça individual, idade, NA, SPI, MA e MR.
- TestCluster_MovieLens_Small_02: notebook com a implementação de análise de agrupamento (dendograma e K-means) com base no conjunto de dados MovieLens-Small e na variável justiça individual.


## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink:
