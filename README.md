# Classificador ML – Predição de Câncer de Mama

Projeto de Machine Learning que constrói e disponibiliza um classificador baseado em rede neural para predição de câncer de mama utilizando o dataset de câncer de mama do scikit-learn.

## Estrutura do Projeto

```
/mlops_project
├── app/                          # Aplicação web
│   ├── __init__.py               # Marcador de pacote
│   ├── main.py                   # App Flask com API de predição
│   └── templates/                # Templates HTML para a interface web
│       └── index.html            # Interface web para predições
├── artifacts/                    # Artefatos de pré-processamento
├── data/                         # Armazenamento de dados
│   ├── preprocessed/             # Dados limpos
│   ├── processed/                # Dados com engenharia de atributos
│   └── raw/                      # Dataset bruto
├── metrics/                      # Métricas de performance do modelo
├── models/                       # Modelo treinado
├── src/                          # Módulos de código-fonte
│   ├── __init__.py               # Marcador de pacote
│   ├── data_loading/             # Utilitários de carregamento de dados
│   │   ├── __init__.py           # Marcador de pacote
│   │   └── load_data.py          # Carregamento e preparação do dataset
│   ├── data_preprocessing/       # Limpeza e divisão dos dados
│   │   ├── __init__.py           # Marcador de pacote
│   │   └── preprocess_data.py    # Limpeza de dados e imputação
│   ├── feature_engineering/      # Utilitários de transformação de atributos
│   │   ├── __init__.py           # Marcador de pacote
│   │   └── engineer_features.py  # Escalonamento e transformação de atributos
│   ├── model_evaluation/         # Scripts de avaliação do modelo
│   │   ├── __init__.py           # Marcador de pacote
│   │   └── evaluate_model.py     # Avaliação de performance do modelo
│   └── model_training/           # Scripts de treinamento do modelo
│       ├── __init__.py           # Marcador de pacote
│       └── train_model.py        # Treinamento da rede neural
├── .dockerignore                 # Regras de exclusão do Docker
├── Dockerfile                    # Instruções de build do Docker
├── params.yaml                   # Parâmetros de configuração
├── pyproject.toml                # Dependências Python e metadados do projeto
└── README.md                     # Documentação do projeto
```


## Funcionalidades
- **Pipeline de Dados**: Pipeline ETL completo desde os dados brutos até atributos prontos para o modelo
- **Rede Neural**: Modelo de deep learning em TensorFlow/Keras com arquitetura configurável
- **Interface Web**: Aplicação web baseada em Flask para realizar predições
- **Gerenciamento de Artefatos**: Modelos e preprocessadores serializados para deploy
- **Métricas de Avaliação**: Análise completa da performance do modelo

## Dependências

O projeto requer Python 3.12+ e os pacotes informados no arquivo `pyproject.toml`.

## Instalação

1. Clone o repositório:
```bash
git clone <https://github.com/thiagoGna/projeto_mlops.git>
cd mlops_project
```

2. Instale as dependências:
```bash
pip install -e .
```

## Configuração

Os hiperparâmetros do modelo e as configurações de processamento de dados são definidos no arquivo `params.yaml`.

## Arquitetura do Modelo

A rede neural consiste em um perceptron multicamadas (MLP) com 2 camadas ocultas.

## Artefatos

O processo de treinamento gera os seguintes arquivos:

No diretório `models/`:
- `model.keras`: Modelo TensorFlow treinado

No diretório `artifacts/`:
- `[features]_mean_imputer.joblib`: Imputador de média para valores ausentes
- `[features]_scaler.joblib`: Standard scaler para normalização dos atributos
- `[target]_one_hot_encoder.joblib`: Codificador one-hot para as classes alvo

## Métricas

As métricas de performance do modelo são salvas em:
- `metrics/training.json`: Histórico de treinamento e métricas de validação
- `metrics/evaluation.json`: Performance no conjunto de teste e matriz de confusão

## Desenvolvimento

O projeto segue uma estrutura modular com separação clara de responsabilidades:
- **Carregamento de Dados**: Busca e salva o dataset bruto de câncer de mama
- **Pré-processamento**: Trata valores ausentes e realiza a divisão treino/teste
- **Engenharia de Atributos**: Aplica transformações e escalonamento
- **Treinamento do Modelo**: Constrói e treina a rede neural
- **Avaliação do Modelo**: Gera métricas de performance
- **Aplicação Web**: Disponibiliza interface para predições

Cada módulo pode ser executado de forma independente e salva suas saídas para a próxima etapa do pipeline.

## Uso

### Treinamento do Modelo

Execute o pipeline completo de ML (para melhor visualização de logs no terminal, execute como módulos usando `python -m`):

```bash
# 1. Carregar e preparar os dados brutos
python -m src.data_loading.load_data

# 2. Pré-processar os dados (imputação e divisão treino/teste)
python -m src.data_preprocessing.preprocess_data

# 3. Engenharia de atributos (escalonamento)
python -m src.feature_engineering.engineer_features

# 4. Treinar o modelo de rede neural
python -m src.model_training.train_model

# 5. Avaliar a performance do modelo
python -m src.model_evaluation.evaluate_model
```

### Executando a Aplicação Web

#### Flask

Após o treinamento do modelo, inicie o servidor Flask:

```bash
python app/main.py
```

A aplicação estará disponível em `http://localhost:5001`

### Docker

Alternativamente, você pode construir e executar a aplicação usando Docker.

#### Build da imagem Docker

```bash
docker build -t ml-classifier .
```

#### Execução do container Docker

```bash
docker run -p 5001:5001 ml-classifier
```

A aplicação web estará disponível em `http://localhost:5001`.

### Realizando Predições

1. **Interface Web**: Envie um arquivo CSV com os atributos do câncer de mama pela interface web
2. **API**: O `/upload` endpoint aceita arquivos CSV e retorna as predições

#### Formato Obrigatório do CSV

O arquivo CSV deve conter exatamente as 30 features do dataset de câncer de mama, com os nomes de colunas corretos:
- mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.
- Consulte `sklearn.datasets.load_breast_cancer().feature_names` para a lista completa
