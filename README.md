# Modelos de Machine Learning

Este reposit\xC3\xB3rio re\xC3\xBAn\xC3\xA9 diversos exemplos pr\xC3\xA1ticos de Machine Learning organizados em notebooks Jupyter. Os notebooks est\xC3\xA3o agrupados por tarefa (regress\xC3\xA3o, classifica\xC3\xA7\xC3\xA3o, agrupamento e redes neurais) e cont\xC3\xAAm explica\xC3\xA7\xC3\xB5es em portugu\xC3\xAAs sobre cada t\xC3\xA9cnica.

## Estrutura

```
ModelosMachineLearning/
├── Agrupamento/
├── Classifica\xC3\xA7\xC3\xA3o/
│   \xE2\x94\x9C── AdaBoostClassifier.ipynb
│   \xE2\x94\x9C── DecisionTreeClassifier.ipynb
│   \xE2\x94\x9C── GradientBoostingClassifier.ipynb
│   \xE2\x94\x9C── KNN.ipynb
│   \xE2\x94\x9C── LogisticRegression.ipynb
│   \xE2\x94\x9C── NaiveBayesBernoulli.ipynb
│   \xE2\x94\x9C── NaiveBayesGaussiano.ipynb
│   \xE2\x94\x9C── RandomForest.ipynb
│   \xE2\x94\x9C── SVM.ipynb
│   \xE2\x94\x94── Redes Neurais/
│       \xE2\x94\x9C── CNN.ipynb
│       \xE2\x94\x9C── GAN.ipynb
│       \xE2\x94\x9C── MLPClassifier.ipynb
│       \xE2\x94\x9C── RNN.ipynb
│       \xE2\x94\x9C── TransformerGPT2.ipynb
│       \xE2\x94\x94── data/
└── Regress\xC3\xA3o/
    \xE2\x94\x9C── DecisionTreeRegressor.ipynb
    \xE2\x94\x9C── ElasticNet-L1eL2.ipynb
    \xE2\x94\x9C── GLM.ipynb
    \xE2\x94\x9C── Lasso-L1.ipynb
    \xE2\x94\x9C── LinearRegression.ipynb
    \xE2\x94\x94── Ridge-L2.ipynb
```

## Conte\xC3\xBAdo

- **Agrupamento** (`Agrupamento/`): notebooks de clustering como DBSCAN, GMM, HAC e k-means.
- **Classifica\xC3\xA7\xC3\xA3o** (`Classifica\xC3\xA7\xC3\xA3o/`): inclui t\xC3\xA9cnicas diversas (\xE2\x80\x8Bdecision tree, random forest, SVM, k-NN, logistic regression, Naive Bayes, AdaBoost, Gradient Boosting), al\xC3\xA9m de modelos de redes neurais.
- **Redes Neurais** (`Classifica\xC3\xA7\xC3\xA3o/Redes Neurais/`): exemplos de CNN, MLP, RNN, GAN e Transformer, acompanhados do dataset MNIST em `data/`.
- **Regress\xC3\xA3o** (`Regress\xC3\xA3o/`): t\xC3\xA9cnicas como regress\xC3\xA3o linear, Lasso, Ridge, ElasticNet, GLM e arvore de decis\xC3\xA3o.

Cada notebook apresenta explica\xC3\xB5es passo a passo sobre o problema, carregamento de dados, treinamento de modelos e interpreta\xC3\xA7\xC3\xA3o de m\xC3\xA9tricas de desempenho.

## Como utilizar

1. Clone este reposit\xC3\xB3rio:
   ```bash
   git clone <URL-do-repo>
   cd ModelosMachineLearning
   ```
2. Recomenda-se criar um ambiente virtual e instalar as depend\xC3\xAAncias necess\xC3\xA1rias. Cada notebook cont\xC3\xA9m instru\xC3\xA7\xC3\xB5es para instalar suas depend\xC3\xAAncias via `pip`.
3. Abra os arquivos `.ipynb` em um ambiente Jupyter (JupyterLab, VS Code, Colab etc.) e execute c\xC3\xA9lula por c\xC3\xA9lula.

## Observa\xC3\xA7\xC3\xB5es

- Os notebooks est\xC3\xA3o em portugu\xC3\xAAs e cont\xC3\xA9m emojis e coment\xC3\xA1rios para auxiliar no aprendizado.
- Alguns exemplos podem requisitar acesso \xC3\xA0 internet para baixar dados (por exemplo, o notebook de RNN usa `yfinance`).
- O dataset MNIST est\xC3\xA1 incluso em `Classifica\xC3\xA7\xC3\xA3o/Redes Neurais/data` para facilitar a execu\xC3\xA7\xC3\xA3o dos notebooks de CNN e MLP.
- O arquivo `.gitignore` j\xC3\xA1 ignora ambientes virtuais e sa\xC3\xADdas geradas pelos notebooks.

## Contribui\xC3\xA7\xC3\xB5es

Sinta-se \xC3\xA0 vontade para abrir issues ou enviar pull requests com melhorias, corre\xC3\xA7\xC3\xB5es e novos exemplos de modelos.

