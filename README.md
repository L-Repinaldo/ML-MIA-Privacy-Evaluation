# Machine Learning, Ataques de Inferência e Trade-off em Privacidade Diferencial

## Visão Geral

Este repositório contém o núcleo experimental da pesquisa em Privacidade Diferencial. Ele mede, compara e explica o trade-off entre utilidade dos dados e risco de exposição à inferência de associação sob diferentes níveis de privacidade ε (`epsilon`).

```text
  menor ε → maior garantia nominal de privacidade
  maior ε → menor perturbação esperada e, potencialmente, maior utilidade
```

O projeto usa modelos supervisionados de classificação como instrumento de medição de utilidade, e Membership Inference Attack baseado em shadow models como instrumento de medição de risco.

Este projeto **não aplica mecanismos de Privacidade Diferencial internamente**. Ele consome datasets já gerados e privatizados por um pipeline externo.

---

## Arquitetura Atual

A arquitetura atual separa exploração, avaliação de utilidade, avaliação de vazamento e visualização:

```text
Dataset Versionado
  -> Exploração de Dados
  -> Avaliação de Utilidade
  -> Artifact Persistence
  -> Avaliação de Vazamento
  -> Visualization Layer
```

Responsabilidades:

- **Exploração de Dados:** inspeciona o dataset base e auxilia a definição de colunas, target e configurações.
- **Avaliação de Utilidade:** carrega datasets versionados, prepara features, treina classificadores e calcula métricas de desempenho.
- **Artifact Persistence:** grava métricas, metadados e entradas necessárias para a avaliação de vazamento.
- **Avaliação de Vazamento:** consome o artifact de utilidade e executa Membership Inference Attack com shadow models.
- **Visualization Layer:** consome artifacts persistidos e gera gráficos de utilidade, vazamento e trade-off.

A execução experimental é conduzida por notebooks. Os módulos em `src/` concentram contratos, carregamento, preprocessamento, treino, métricas, ataques e plots reutilizáveis.

---

## Papel na Arquitetura do Projeto

O projeto completo é composto por dois sistemas independentes:

1. [**Differential Privacy Data Pipeline Experiment**](https://github.com/L-Repinaldo/Diferential-Privacy-Data-Pipeline-Experiment)
   - Extrai dados provinientes dos dados do Enem.
   - Aplica mecanismos de Privacidade Diferencial.
   - Versiona datasets com diferentes valores de `epsilon`.
   - Gera metadados experimentais.

2. **ML MIA Privacy Evaluation**
   - Carrega datasets versionados.
   - Treina modelos de Machine Learning.
   - Executa Membership Inference Attack.
   - Calcula métricas de utilidade e risco.
   - Persiste artifacts experimentais.
   - Permite análise visual por notebooks.

Este repositório corresponde ao segundo sistema e **não acessa diretamente os dados do Enem**.

---

## Fluxo Experimental

O fluxo principal está dividido em notebooks:

```text
notebooks/
  00_explore_data.ipynb
  01_utility_evaluation.ipynb
  02_attack_evaluation.ipynb
  03_visualization.ipynb
```

Ordem de execução:

1. `00_explore_data.ipynb`
   - Carrega o `baseline.parquet`.
   - Explora colunas, distribuição e amostras do dataset.

2. `01_utility_evaluation.ipynb`
   - Define `DatasetConfig`, `TaskConfig` e `PreprocessingConfig`.
   - Carrega o bundle de datasets.
   - Prepara features de treino, validação e teste.
   - Treina os modelos de classificação.
   - Calcula métricas de utilidade.
   - Persiste `utility_metrics.csv`, `leakage_input.pkl` e `metadata.json`.

3. `02_attack_evaluation.ipynb`
   - Carrega o artifact gerado pela avaliação de utilidade.
   - Reconstrói os modelos e dados necessários para o ataque.
   - Executa shadow-model Membership Inference Attack.
   - Calcula métricas de vazamento.
   - Persiste `attack_metrics.csv`, `attack_results.pkl` e `metadata.json`.

4. `03_visualization.ipynb`
   - Carrega artifacts persistidos.
   - Renderiza gráficos de classificação, ataque e trade-off.
   - Não executa treinamento nem ataques.

---

## Estrutura Geral

```text
.
├── artifacts/
│   ├── persistence.py
│   └── evaluation/
├── notebooks/
│   ├── 00_explore_data.ipynb
│   ├── 01_utility_evaluation.ipynb
│   ├── 02_attack_evaluation.ipynb
│   └── 03_visualization.ipynb
├── src/
│   ├── core/
│   ├── data/
│   ├── experiments/
│   │   ├── lekage_evaluation/
│   │   └── utility_evaluation_services/
│   ├── plots/
│   └── preprocessing/
├── requirements.txt
└── README.md
```

---

## Entidades Centrais

A camada `src/core/` formaliza os contratos principais:

- `DatasetConfig`: define nome, versão, tamanho de amostra e seed dos dados.
- `TaskConfig`: define o tipo de tarefa e o target.
- `PreprocessingConfig`: define colunas categóricas, numéricas e estratégias de imputação.
- `SplitConfig`: define seed e tamanho dos splits.
- `ModelSpec`: define nome, tipo e parâmetros de cada modelo.
- `PreparedFeatures`: encapsula features e targets já separados em treino, validação e teste.
- `PredictionResult`: encapsula predições, probabilidades e targets codificados.
- `UtilityClassificationResult`: encapsula métricas de classificação.
- `ShadowAttackConfig`: define configuração do Membership Inference Attack.
- `ShadowModelMiaResult`: encapsula métricas do ataque.

---

## Fluxo dos Datasets

Os datasets ficam em:

```text
src/data/datasets/<DATASET_NAME>/<DATASET_VERSION>/
```

Exemplo atual:

```text
src/data/datasets/enem/enem_2025 - v-2026-09-08_00-56-16/
```

O registry em `src/data/dataset_registry.py` descobre automaticamente:

- `baseline.parquet`
- `dp_eps_*.parquet`

A ordem preservada é:

```text
baseline
dp_eps_0.1
dp_eps_0.5
dp_eps_1.0
dp_eps_2.0
...
```

Também é possível carregar uma amostra consistente entre todos os datasets usando `data_sample_size` e `data_random_state`.

---

## Preprocessamento

O preprocessamento fica em `src/preprocessing/preprocessor.py`.

O fluxo atual:

1. Seleciona apenas colunas configuradas e existentes no dataset.
2. Imputa colunas categóricas com `most_frequent`.
3. Aplica `OneHotEncoder` nas colunas categóricas.
4. Imputa colunas numéricas com `median`.
5. Remove colunas não configuradas.
6. Codifica o target com `LabelEncoder` em tarefas de classificação.

Os splits são feitos em treino, validação e teste em `src/experiments/utility_evaluation_services/feature_preparation.py`.

---

## Modelos

Modelos suportados em `src/experiments/utility_evaluation_services/model/build_model.py`:

- XGBoost Classifier
- Random Forest Classifier
- Logistic Regression

O experimento atual usa:

- `xgboost_classifier`

Todos seguem o mesmo protocolo:

1. Receber um `ModelSpec`.
2. Instanciar o modelo pelo `model_factory`.
3. Treinar com o conjunto de treino.
4. Gerar predições para treino, validação e teste.
5. Gerar probabilidades com `predict_proba`.
6. Retornar `PredictionResult`.

---

## Métricas

### Utilidade

Calculadas em `src/experiments/utility_evaluation_services/metrics.py`:

- `train_acc`
- `validation_acc`
- `test_acc`
- `train_precision`
- `validation_precision`
- `test_precision`
- `train_recall`
- `validation_recall`
- `test_recall`
- `train_f1`
- `validation_f1`
- `test_f1`
- `generalization_gap`

### Vazamento

Calculadas em `src/experiments/lekage_evaluation/metrics.py`:

- `attack_acc`
- `attack_f1`
- `attack_precision`
- `attack_recall`
- `member_acc`
- `non_member_acc`
- `advantage`

---

## Ataque de Inferência

O ataque avaliado é **Membership Inference Attack (MIA)** com shadow models.

Fluxo:

```text
Artifact de utilidade
  -> leakage_input.pkl
  -> shadow models
  -> features de membership
  -> modelo de ataque
  -> métricas de vazamento
  -> artifact de ataque
```

As features do ataque para classificação são montadas em `src/experiments/lekage_evaluation/feature_preparation.py`:

- probabilidades por classe
- confiança
- entropia
- probabilidade da classe verdadeira
- cross-entropy loss

A configuração atual usa:

- `n_shadow_models = 3`
- `member_fraction = 0.5`
- `attack_test_size = 0.3`
- modelo de ataque `xgboost_classifier`

---

## Artifacts

Os artifacts são gerenciados por `artifacts/persistence.py`.

A avaliação de utilidade gera:

```text
artifacts/evaluation/evaluation_<experiment_id>/
  classification/
    utility_metrics.csv
    leakage_input.pkl
    metadata.json
```

A avaliação de vazamento adiciona:

```text
artifacts/evaluation/evaluation_<experiment_id>/
  membership_attack/
    attack_metrics.csv
    attack_results.pkl
    metadata.json
```

O metadata da utilidade contém:

- `experiment_id`
- `experiment_type`
- `artifact_schema_version`
- `created_at`
- dataset usado
- split usado
- colunas de preprocessamento
- tarefa e target
- modelos avaliados

Modelos treinados não são persistidos como artifacts finais.

---

## Visualization Layer

A camada `src/plots/` contém helpers de visualização reutilizáveis:

```text
src/plots/
  classification_plots.py
  membership_attack_plots.py
  trade_off_polt.py
```

Estado atual:

- Classificação: matplotlib
- Membership Inference Attack: matplotlib
- Trade-off utilidade x vazamento: matplotlib

Essa camada recebe DataFrames vindos dos artifacts. Ela não executa experimentos.

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Execução Experimental

0. É preciso carregar os dados gerados pelo sistema [Differential-Privacy-Data-Pipeline-Experiment](#papel-na-arquitetura-do-projeto) e adicionar à pasta `src/data/datasets`.

1. Configure o dataset, target, colunas e modelos em `notebooks/01_utility_evaluation.ipynb`.

2. Garanta que os arquivos estejam em:

```text
src/data/datasets/<DATASET_NAME>/<DATASET_VERSION>/
```

3. Execute a avaliação de utilidade:

```text
notebooks/01_utility_evaluation.ipynb
```

4. Execute a avaliação de vazamento usando o artifact de utilidade:

```text
notebooks/02_attack_evaluation.ipynb
```

5. Execute a visualização:

```text
notebooks/03_visualization.ipynb
```

---

## Reprodutibilidade

A reprodutibilidade depende de:

- `dataset_name` e `dataset_version` em `DatasetConfig`.
- `data_sample_size` e `data_random_state` em `DatasetConfig`.
- Seed e `test_size` em `SplitConfig`.
- Target e tipo de tarefa em `TaskConfig`.
- Colunas em `PreprocessingConfig`.
- Parâmetros dos modelos em `ModelSpec`.
- Configuração do ataque em `ShadowAttackConfig`.
- Artifacts persistidos em `artifacts/evaluation/`.

O projeto não gera dados primários, não aplica DP internamente e não altera datasets de origem.

---

## Observações

- O projeto é acadêmico e experimental.
- Visualizações têm caráter explicativo, não decisório.
- O foco científico é o fenômeno do trade-off, não a competição entre modelos.

---

## Licença

Uso acadêmico e educacional.
