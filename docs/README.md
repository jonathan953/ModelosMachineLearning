# Projeto de Modelos de Machine Learning

Este repositório reúne uma coleção abrangente de **modelos de Machine Learning** aplicados a diferentes problemas de **classificação, regressão, agrupamento e redes neurais**, com o objetivo de **facilitar o aprendizado prático e teórico** desses algoritmos.

Cada modelo apresentado segue um pipeline completo, incluindo:

- Análise exploratória detalhada 📊  
- Limpeza e tratamento dos dados 🧹  
- Divisão entre treino e teste 🧪  
- Treinamento e validação dos modelos 🧠  
- Avaliação com métricas apropriadas (accuracy, RMSE, R², etc.) 📈  
- Interpretação dos resultados e visualizações ✨  

Todos os notebooks são comentados, com explicações passo a passo para tornar o aprendizado acessível a diferentes níveis de conhecimento.

---

## 🔍 Diferencial do Projeto

Além dos estudos de Machine Learning, este repositório inclui uma seção dedicada à **manipulação e preparação de dados com pandas**, traduzindo operações comuns de **SQL para Python**.

São abordados, na prática, conceitos como:

- SELECT, WHERE, ORDER BY  
- JOINs (INNER, LEFT, FULL)  
- GROUP BY / HAVING  
- Subqueries e CTEs  
- CASE WHEN  
- Funções de janela (ROW_NUMBER, RANK, LAG, LEAD)  

Esse diferencial simula cenários reais de **ETL, Analytics e Data Science**, fortalecendo a base de dados antes da modelagem.

---

## ⚠️ Observação Importante

> Todos os **datasets utilizados são sintéticos**, criados exclusivamente para fins educacionais.  
> Nenhum dado real ou sensível foi utilizado neste projeto.

---

## Tecnologias Utilizadas 🛠️

- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook / VS Code
- Git & GitHub

---

## Estrutura de Pastas 📂

    modelos-machine-learning/
    │
    ├── data/                      # Datasets usados nos estudos (dados sintéticos)
    │   ├── customers.csv
    │   ├── order_items.csv
    │   ├── orders.csv
    │   ├── payments.csv
    │   └── products.csv
    │
    ├── pandas/                    # Estudos de manipulação de dados (SQL → pandas, ETL)
    │   └── etl.ipynb                  → ETL e tradução prática de SQL para pandas
    │
    ├── agrupamento/                # Modelos não supervisionados para agrupamento de dados
    │   ├── dbscan.ipynb                → DBSCAN (Density-Based Spatial Clustering)
    │   ├── gmm.ipynb                   → GMM (Gaussian Mixture Models)
    │   ├── hac.ipynb                   → HAC (Hierarchical Agglomerative Clustering)
    │   └── k-means.ipynb              → K-Means Clustering
    │
    ├── classificacao/             # Modelos supervisionados para classificação
    │   ├── adaboost-classifier.ipynb       → AdaBoost Classifier
    │   ├── decision-tree-classifier.ipynb  → Árvore de Decisão
    │   ├── gradient-boosting-classifier.ipynb → Gradient Boosting Classifier
    │   ├── knn-classifier.ipynb            → K-Nearest Neighbors
    │   ├── logistic-regression.ipynb       → Regressão Logística
    │   ├── naive-bayes-bernoulli.ipynb     → Naive Bayes (Bernoulli)
    │   ├── naive-bayes-gaussiano.ipynb     → Naive Bayes (Gaussiano)
    │   ├── random-forest-classifier.ipynb  → Floresta Aleatória (Random Forest)
    │   └── svm.ipynb                       → Máquinas de Vetores de Suporte (SVM)
    │
    ├── regressao/                 # Modelos supervisionados para regressão
    │   ├── adaboost-regressor.ipynb        → AdaBoost Regressor
    │   ├── decision-tree-regressor.ipynb   → Árvore de Decisão para Regressão
    │   ├── elasticnet-l1el2.ipynb          → ElasticNet (Combina L1 e L2)
    │   ├── glm.ipynb                       → Modelos Lineares Generalizados (GLM)
    │   ├── gradient-boost-regressor.ipynb  → Gradient Boosting Regressor
    │   ├── knn-regressor.ipynb             → KNN Regressor
    │   ├── lasso-l1.ipynb                  → Lasso Regression (L1)
    │   ├── linear-regression.ipynb         → Regressão Linear
    │   ├── random-forest-regressor.ipynb   → Random Forest Regressor
    │   ├── ridge-l2.ipynb                  → Ridge Regression (L2)
    │   └── svr.ipynb                       → Support Vector Regressor
    │
    ├── redes-neurais/            # Modelos de Deep Learning
    │   ├── cnn.ipynb                     → Convolutional Neural Network (CNN)
    │   ├── gan.ipynb                     → Generative Adversarial Network (GAN)
    │   ├── lstm.ipynb                    → Long Short-Term Memory (LSTM)
    │   ├── mlp-classifier.ipynb          → Multi-Layer Perceptron (MLP)
    │   ├── transformer-gpt2.ipynb        → Transformer GPT-2
    │   └── data/
    │       └── MNIST/
    │           └── raw/
    │               └── imagem_gerada.png
    │
    ├── graficos/                # Gráficos para análise e visualização
    │   └── graficos-matplotlib-seaborn.ipynb  → Gráficos com Matplotlib & Seaborn
    │
    ├── docs/                    # Documentação do projeto
    │   ├── LICENSE.txt
    │   └── README.md
    │
    ├── .gitignore
    └── requirements.txt


---

## Como Usar 🚀

1. Clone o repositório

```bash
git clone https://github.com/jonathan953/ModelosMachineLearning.git
```

2. Instale os pacotes do ambiente

```bash
pip install -r requirements.txt
```

3. Execute os notebooks com Jupyter ou VSCode

---

## Objetivo Final 🎯

Este projeto tem como objetivo **consolidar fundamentos de Data Science e Machine Learning**
por meio de **implementações práticas e bem documentadas**, servindo como:

- 📚 Material de estudo estruturado  
- 🧠 Repositório de referência conceitual e prática  
- 💼 Portfólio técnico para projetos acadêmicos e profissionais  

O foco está em unir **teoria, prática e organização**, simulando cenários reais
de análise de dados, preparação de dados (ETL) e modelagem preditiva.


---

Caso tenha sugestões, dúvidas ou queira colaborar, fique à vontade para abrir uma *issue* ou enviar um *pull request*! 🤝
