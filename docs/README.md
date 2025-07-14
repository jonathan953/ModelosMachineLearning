# Projeto de Modelos de Machine Learning

Este repositório reúne uma coleção abrangente de **modelos de Machine Learning** aplicados a diferentes problemas de classificação, regressão, agrupamento e redes neurais, com o objetivo de **facilitar o aprendizado prático e teórico** desses algoritmos.

Cada modelo aqui apresentado segue um pipeline completo com:

* Análise exploratória detalhada 📊
* Limpeza e tratamento dos dados 🧹
* Divisão entre treino e teste 🧪
* Treinamento e validação dos modelos 🧠
* Avaliação com métricas apropriadas (accuracy, RMSE, R2, etc.) 📈
* Interpretação dos resultados e visualizações interativas ✨

Todos os notebooks são comentados, com explicações passo a passo para tornar o aprendizado acessível para todos os níveis.

---

## Estrutura de Pastas 📂

    MODELOS DE MACHINE LEARNING/
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
    ├── docs/                    # Documentação
    │   ├── LICENSE.txt
    │   └── README.md
    │
    ├── .gitignore
    ├── requirements.txt
    └── venv/



## Detalhamento Interno 📝

Cada notebook possui:

* README interno explicando o funcionamento do algoritmo ✅
* Detalhamento sobre o dataset utilizado 📄
* Links de referência e material de apoio 🔗

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

Este projeto serve tanto como:

* Material de estudos 📚
* Base para aplicações práticas
* Inspiração para projetos pessoais e acadêmicos

---

Caso tenha sugestões, dúvidas ou queira colaborar, fique à vontade para abrir uma *issue* ou enviar um *pull request*! 🤝
