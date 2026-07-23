# Machine Learning, Ataques de Inferência e Trade-off em Privacidade Diferencial

## Visão Geral

Este repositório contém o núcleo experimental da pesquisa em Privacidade Diferencial. Ele mede, compara e explica o trade-off entre utilidade dos dados e risco de vazamento sob diferentes níveis de privacidade (`epsilon`).

O projeto usa modelos supervisionados de regressão baseados em árvores como instrumento de medição de utilidade, e Membership Inference Attack como instrumento de medição de risco.

Este projeto **não aplica mecanismos de Privacidade Diferencial internamente**. Ele consome datasets já gerados e privatizados por um pipeline externo.

---

## Arquitetura Final

A arquitetura atual separa completamente execução científica e exploração visual:

```text
Pipeline Experimental
  -> Artifact Persistence
  -> Visualization Layer
  -> Streamlit Explorer
```

Responsabilidades:

- **Pipeline Experimental:** carrega datasets, executa modelos, executa ataques, agrega métricas e persiste artifacts.
- **Artifact Persistence:** grava `utility_metrics.csv`, `attack_metrics.csv` e `metadata.json`.
- **Visualization Layer:** contém helpers e figuras reutilizáveis, sem executar experimentos.
- **Streamlit Explorer:** consome apenas artifacts persistidos para análise visual interativa.

O Streamlit não importa treino, preprocessamento, ataques, métricas nem pipeline experimental.

---

## Papel na Arquitetura do Projeto

O projeto completo é composto por três sistemas independentes:

1. **Sistema de RH**
   - Simula um ambiente corporativo.
   - Gera dados limpos, consistentes e sensíveis.
   - Aplica regras de negócio.

2. **DP Data Pipeline**
   - Extrai dados do sistema de RH.
   - Aplica mecanismos de Privacidade Diferencial.
   - Versiona datasets com diferentes valores de `epsilon`.
   - Gera metadados experimentais.

3. **ML e Análise Experimental**
   - Carrega datasets versionados.
   - Treina modelos de Machine Learning.
   - Executa Membership Inference Attack.
   - Calcula métricas de utilidade e risco.
   - Persiste artifacts experimentais.
   - Permite exploração visual independente via Streamlit.

Este repositório corresponde ao terceiro sistema e **não acessa diretamente o banco do sistema de RH**.

---

## Pipeline Experimental

O fluxo experimental é centralizado em `core/experimental_pipeline.py`:

```text
config.py
  -> ExperimentConfig
  -> ExperimentalPipeline
      -> Dataset Registry
      -> Model Runners
      -> Utility Metrics
      -> Attack Feature Extraction
      -> Membership Inference Attack
      -> Attack Metrics
      -> Aggregation
      -> Artifact Persistence
```

O pipeline **não gera visualizações automaticamente**. A execução principal termina após a persistência dos artifacts.

---

## Entidades Centrais

A camada `core/` formaliza os contratos principais:

- `PredictionResult`: encapsula predições e modelo treinado.
- `ExperimentResult`: encapsula métricas de utilidade, métricas de ataque e metadados da execução.
- `ExperimentConfig`: define versão do dataset, seeds, tamanhos de teste, modelos ativos e datasets ativos.
- `ExperimentalPipeline`: coordena a execução científica do experimento.

---

## Fluxo dos Datasets

Os datasets ficam em:

```text
data/datasets/<DATASET_VERSION>/
```

O `DATASET_VERSION` é definido em `config.py`.

O registry em `data/dataset_registry.py` descobre automaticamente:

- `baseline.csv`
- `dp_eps_*.csv`

A ordem preservada é:

```text
baseline
eps_0.1
eps_0.5
eps_1.0
eps_2.0
...
```

---

## Modelos

Modelos implementados:

- XGBoost
- Random Forest
- Extra Trees
- Gradient Boosting

Todos seguem o mesmo protocolo:

1. Validar o target `salario`.
2. Separar `X` e `y`.
3. Executar `train_test_split`.
4. Aplicar o preprocessor.
5. Treinar o regressor.
6. Gerar predições de treino e teste.
7. Retornar `PredictionResult`.

Não há tuning agressivo, otimização competitiva ou alteração dinâmica de hiperparâmetros.

---

## Métricas

### Utilidade

Calculadas em `metrics/utility.py`:

- `mae`
- `rmse`
- `train_abs_error`
- `test_abs_error`

### Vazamento

Calculadas em `metrics/attack.py`:

- `attack_acc`
- `member_acc`
- `non_member_acc`
- `advantage`

Os cálculos das métricas permanecem preservados.

---

## Ataque de Inferência

O ataque avaliado é **Membership Inference Attack (MIA)**.

Fluxo:

```text
PredictionResult
  -> utility metrics
  -> extract_attack_features
  -> run_membership_inference_attack
  -> attack metrics
```

O MIA usa os erros absolutos de treino e teste como sinal de membership. A lógica do ataque não foi alterada.

---

## Agregação

A agregação fica em `experiments/aggregation.py`.

Responsabilidades:

- Agrupar resultados por modelo e dataset.
- Calcular médias.
- Aplicar arredondamentos.
- Produzir `df_utility` e `df_attack`.

`experiments/run_experiment.py` executa resultados brutos por execução e não agrega diretamente.

---

## Artifacts

Cada execução de `python main.py` gera:

```text
artifacts/<experiment_id>/
  utility_metrics.csv
  attack_metrics.csv
  metadata.json
```

O metadata contém:

- `dataset_version`
- `timestamp`
- modelos ativos
- seeds
- test sizes

Modelos treinados ainda não são persistidos.

---

## Visualization Layer

A camada `visualization/` contém helpers e figuras reutilizáveis:

```text
visualization/
  common.py
  utility/
  attacks/
  tradeoff/
  summary/
```

Estado atual:

- Tabelas de utilidade: matplotlib
- Tabelas de ataque: matplotlib
- Trade-off: Plotly
- Tabela de síntese: Plotly

Essa camada recebe DataFrames prontos ou dados vindos dos artifacts. Ela não executa experimentos.

---

## Streamlit Explorer

O app em `streamlit_app/` é um consumidor independente dos artifacts persistidos.

Estrutura:

```text
streamlit_app/
  app.py
  artifact_loader.py
  views/
    overview.py
    utility.py
    leakage.py
    tradeoff.py
    comparison.py
```

Views disponíveis:

- **Overview:** metadata, datasets, modelos, seeds e test sizes.
- **Utility Analysis:** evolução de MAE/RMSE, degradação relativa e heatmaps.
- **Leakage Analysis:** `attack_acc`, `advantage`, `member_acc`, `non_member_acc`.
- **Trade-off Analysis:** perda relativa de utilidade versus `advantage`.
- **Comparison:** comparação cruzada entre modelos e epsilons.

O app permite selecionar manualmente um artifact ou usar automaticamente o mais recente.

---

## Estrutura Geral

```text
.
├── analysis/
├── artifacts/
├── attacks/
├── core/
├── data/
├── experiments/
├── metrics/
├── model/
├── plots/
├── preprocessing/
├── sanity_check/
├── streamlit_app/
├── visualization/
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Execução Experimental

1. Configure a versão do dataset em `config.py`:

```python
DATASET_VERSION = "v-2026-03-02_18-10-54"
```

2. Garanta que os arquivos estejam em:

```text
data/datasets/<DATASET_VERSION>/
```

3. Execute:

```bash
python main.py
```

Essa execução gera apenas artifacts:

- `utility_metrics.csv`
- `attack_metrics.csv`
- `metadata.json`

---

## Exploração Visual

Após gerar artifacts, execute:

```bash
streamlit run streamlit_app/app.py
```

O Streamlit carrega os artifacts persistidos e renderiza as visões analíticas sem executar treinamento, ataques ou agregação experimental.

---

## Sanity Checks

O diretório `sanity_check/` contém validações auxiliares para modelos e MIA.

Executar sanity checks de modelo:

```bash
python sanity_check/sanity_model_validation.py
```

Executar sanity checks de MIA:

```bash
python sanity_check/sanity_mia_validation.py
```

Esses checks não fazem parte dos resultados finais do experimento.

---

## Reprodutibilidade

A reprodutibilidade depende de:

- `DATASET_VERSION` em `config.py`.
- Seeds definidas em `ExperimentConfig`.
- Test sizes definidos em `ExperimentConfig`.
- Modelos ativos definidos em `main.py`.
- Datasets versionados em `data/datasets/`.
- Artifacts persistidos em `artifacts/`.

O projeto não gera dados primários, não aplica DP internamente e não altera datasets de origem.

---

## Observações

- Os dados utilizados são simulados e não representam indivíduos reais.
- O projeto é acadêmico e experimental.
- Visualizações têm caráter explicativo, não decisório.
- O foco científico é o fenômeno do trade-off, não a competição entre modelos.

---

## Imagens

***Utilidade:***   

***Vazamento:***

***Trade-Off:***

***Tabelas de síntese:*** 

---

## Licença

Uso acadêmico e educacional.
