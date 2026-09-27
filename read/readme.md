# README — Что уже реализовано

Этот файл — короткая карта возможностей проекта.  
Без подробной теории: только что уже работает, что можно включать через `config`, и что пока оставлено как placeholder.

---

# 1. Classic ML pipeline

## Полностью реализовано

### Данные и подготовка
- загрузка train/test данных;
- выбор `target`;
- поддержка `id_column`;
- удаление указанных колонок;
- базовые проверки данных и leakage checks;
- numeric imputation;
- categorical imputation;
- scaling числовых признаков;
- encoding категориальных признаков;
- optional reduce-memory/downcast.

### Валидация
Поддерживаются:
- `KFold`;
- `StratifiedKFold`;
- `GroupKFold`;
- `StratifiedGroupKFold`.

Можно:
- задавать количество folds;
- выбирать конкретные folds для обучения;
- использовать group column;
- сохранять OOF predictions.

### Classification models
Реализованы:
- `LogisticRegression`;
- `KNeighborsClassifier`;
- `DecisionTreeClassifier`;
- `RandomForestClassifier`;
- `GradientBoostingClassifier`;
- `XGBClassifier`;
- `LGBMClassifier`;
- `CatBoostClassifier`.

### Regression models
Реализованы:
- `LinearRegression`;
- `KNeighborsRegressor`;
- `DecisionTreeRegressor`;
- `RandomForestRegressor`;
- `GradientBoostingRegressor`;
- `XGBRegressor`;
- `LGBMRegressor`;
- `CatBoostRegressor`.

`LinearRegression` — одно внешнее имя для OLS/Ridge/Lasso/ElasticNet. Вид регуляризации выбирается через `models.LinearRegression.regularization`; отдельного имени `Ridge` в Classic-реестре нет.

Для KNN задайте `model.name="KNeighborsClassifier"` или `"KNeighborsRegressor"`. Параметры обоих находятся в `models.<имя>`; числовые признаки масштабируются внутри каждого fold при `preprocessing.scale_numeric=True` (текущее значение по умолчанию). Классификатор поддерживает `predict_proba()`.

### Обучение
Реализовано:
- CV training;
- обучение по выбранным folds;
- подсчёт metric на каждом fold;
- OOF predictions;
- сохранение fold-моделей;
- final model на всех train-данных;
- test inference после обучения;
- отдельный inference mode;
- сохранение config snapshot;
- сохранение metadata/environment;
- сохранение результатов эксперимента.

### Optuna
Реализован hyperparameter tuning:
- `TPE sampler`;
- `Random sampler`;
- search space отдельно для каждой модели;
- для `LinearRegression`: `none` оценивает baseline один раз без подбора, `ridge`/`lasso` подбирают `alpha`, `elasticnet` — `alpha` и `l1_ratio`;
- для обеих KNN-моделей подбираются `n_neighbors` (нечётные 3–35), `weights` (`uniform`/`distance`) и `p` (1/2); `algorithm`, `metric`, `leaf_size`, `n_jobs` фиксированы в config;
- tuning на выбранных folds;
- сохранение trials;
- сохранение best parameters;
- после Optuna обычный training с лучшими параметрами.

### Classic optimization

Работают:

#### Speed
- `parallel_cpu`;
- `histogram_boosting` для поддерживаемого boosting.

#### Training control
- `boosting_early_stopping`.

#### Prediction
- `threshold_tuning`.

Все дополнительные техники управляются через:

```python
optimization.enabled
```

Если главный флаг выключен, техники внутри не используются.

### Внутренние ансамбли Classic

Через `estimator_strategy` реализованы:
- `bagging`;
- `voting`;
- `average`;
- `weighted_average`;
- `stacking`.

Это ансамбли моделей внутри Classic pipeline.
Оба KNN входят в список `estimator_strategy.models` для своей задачи; списки `weights` содержат по восемь значений.

### Визуализация
Реализованы:
- `validation.png`;
- `cv_scores.png`;
- `feature_importance.png`;
- optional `shap_summary.png`.

Для classification:
- confusion matrix.

Для regression:
- true vs predicted;
- residual plot.

### Logging / tracking
Реализованы:
- console logging;
- txt log;
- csv results;
- Telegram notifications;
- W&B tracking;
- сохранение plots/artifacts.

---

# 2. Deep Learning pipeline

## Полностью реализовано

### Данные
Поддерживаются:
- train features;
- train labels;
- test features;
- train/test IDs;
- groups;
- unlabeled data для self-training.

### Валидация
Поддерживаются:
- `KFold`;
- `StratifiedKFold`;
- `GroupKFold`;
- `StratifiedGroupKFold`;
- выбор folds для training;
- выбор folds для inference;
- debug subset;
- OOF predictions.

### Модели
Есть универсальные примеры:
- `MLP`;
- `TransformerMLP`.

Также архитектура позволяет добавлять custom model в `models.py`.

### MLP
Настраиваются:
- hidden layers;
- activation;
- dropout через блок regularization.

### TransformerMLP
Настраиваются основные параметры:
- `d_model`;
- `nhead`;
- `num_layers`;
- `dim_feedforward`;
- `head_hidden_dims`.

### Training loop
Реализован полноценный PyTorch training loop:
- batches;
- forward;
- loss;
- backward;
- optimizer step;
- validation;
- metric calculation;
- checkpointing;
- best model;
- last checkpoint;
- resume training;
- CV;
- OOF;
- final training;
- inference.

### Reproducibility
Реализован deterministic seed для:
- Python;
- NumPy;
- PyTorch;
- folds;
- Optuna trials;
- DataLoader generator/workers.

Это сделано для того, чтобы одинаковые настройки давали максимально одинаковый результат.

### Regularization
Отдельно и явно реализованы:
- dropout;
- weight decay.

Они по умолчанию выключены.

### DL optimization

Главный переключатель:

```python
optimization.enabled
```

Если он выключен, дополнительные optimization techniques не применяются.

#### Speed — реализовано
- AMP / mixed precision;
- `torch.compile`;
- fused optimizer;
- optimized DataLoader.

#### Memory — реализовано
- gradient accumulation.

#### Training control — реализовано
- early stopping;
- gradient clipping;
- scheduler;
- warmup;
- EMA.

### Scheduler
Можно использовать стандартные PyTorch schedulers по имени.

Реализована поддержка:
- обычного scheduler;
- scheduler step по epoch или step;
- `ReduceLROnPlateau`;
- optional warmup через `LinearLR + SequentialLR`.

### Strategies

Отдельно от optimization реализованы:

#### Fine-tuning / Transfer Learning
- загрузка pretrained checkpoint;
- freeze backbone;
- unfreeze backbone после заданной epoch;
- отдельный LR для backbone и head.

#### Metric Learning
- embedding;
- metric loss;
- supervised + metric loss;
- balanced batches.

#### Hard Negative Mining
- `batch_hard`.

#### Self-training
- prediction на unlabeled data;
- confidence threshold;
- pseudo-label selection;
- retraining;
- несколько rounds.

#### Multi-head
- несколько output heads;
- отдельный task для head;
- отдельный loss;
- отдельный metric;
- loss weights;
- primary head.

### Optuna
Реализовано:
- TPE / Random sampler;
- generic search space через `path`;
- tuning по выбранным folds;
- Optuna pruning;
- best params;
- повторное обычное обучение после tuning.

### Visualization
Реализованы:
- `training_curves.png`;
- `validation.png`;
- `cv_scores.png`.

`training_curves.png` включает:
- train loss;
- validation loss;
- validation metric;
- learning rate, если scheduler включён.

### Tracking
Реализованы:
- local logging;
- TensorBoard;
- W&B;
- Telegram;
- csv/txt результаты.

### Export
Есть model conversion/export infrastructure.
Текущий основной формат:
- `torch_export`.

---

# 3. Universal Ensemble

Отдельный `ensemble.py` объединяет уже обученные эксперименты.

Classic и DL pipeline не зависят друг от друга напрямую.

Они передают ensemble одинаковые artifacts:

```text
oof_predictions.csv
predictions.csv
config.yaml
metadata.json
```

## Реализованные типы ensemble

### Average
Простое среднее predictions.

### Weighted average
Среднее с заданными weights.

### Voting
Для classification:
- hard voting;
- soft voting.

### Stacking
- OOF predictions используются как признаки;
- обучается meta-model;
- test predictions передаются meta-model для final prediction.

По умолчанию:
- classification → `LogisticRegression`;
- regression → `Ridge` (это meta-model корневого ансамбля, независимая от Classic `models.LinearRegression`).

### Optuna weight optimization
Реализован автоматический поиск weights.

Важно:
- базовые модели не переобучаются;
- используются уже сохранённые OOF predictions;
- Optuna ищет только веса ансамбля.

### Presets
Можно хранить несколько готовых ensemble-конфигураций:
- boostings;
- neural networks;
- Classic + DL;
- weighted;
- stacking.

Для нового ансамбля не нужно создавать новый `.py` файл.

### Проверки
Перед ensemble проверяются:
- наличие experiment;
- OOF/test artifacts;
- IDs;
- target;
- shapes;
- task;
- multiclass classes;
- NaN/inf;
- weights.

### Ensemble visualization
Реализованы:
- `validation.png`;
- `cv_scores.png`;
- `ensemble_diagnostics.png`.

В diagnostics:
- base model scores vs ensemble;
- weights/contribution;
- prediction correlation.

---

# 4. Общий формат artifacts

Classic и DL сохраняют совместимые файлы.

Пример:

```text
checkpoints/<experiment>/
├── oof_predictions.csv
├── predictions.csv
├── config.yaml
├── metadata.json
├── environment.json
├── model / checkpoints
└── plots/
```

Это позволяет:
- сравнивать эксперименты;
- строить ансамбли;
- воспроизводить run;
- понимать, с каким config была обучена модель.

---

# 5. Что пока НЕ реализовано полностью

Эти блоки существуют в config, но специально обозначены как placeholders.

## Classic ML
- task-specific feature engineering;
- probability calibration;
- feature selection;
- generic postprocessing.

## Deep Learning
- task-specific preprocessing;
- augmentations;
- gradient checkpointing;
- quantization;
- model pruning;
- knowledge distillation.

Если включить placeholder-технику, pipeline не должен молча делать вид, что она работает — код сообщает, что реализация отсутствует.

---

# 6. Что выключено в baseline

По умолчанию baseline не использует дополнительные техники.

## Classic
Выключены:
- optimization;
- early stopping;
- threshold tuning;
- ensembles;
- Optuna;
- SHAP;
- W&B;
- Telegram.

## Deep Learning
Выключены:
- dropout;
- weight decay;
- AMP;
- compile;
- fused optimizer;
- optimized DataLoader;
- gradient accumulation;
- early stopping;
- gradient clipping;
- scheduler;
- warmup;
- EMA;
- finetuning;
- metric learning;
- hard negative mining;
- self-training;
- multi-head;
- Optuna;
- TensorBoard;
- W&B;
- Telegram.

То есть можно начать с чистого baseline, а затем включать техники по одной и сравнивать результат.

---

# 7. Самая простая схема использования

## Classic

```text
выбрать task
→ указать data/target
→ выбрать model
→ выбрать metric
→ запустить baseline
→ посмотреть CV
→ при необходимости включить Optuna/optimization/ensemble
```

## Deep Learning

```text
указать данные
→ выбрать MLP/TransformerMLP
→ выбрать metric/loss/optimizer
→ запустить baseline
→ посмотреть training curves + CV
→ включать regularization/optimization/strategies по необходимости
```

## Ensemble

```text
обучить несколько experiments
→ получить OOF/test predictions
→ добавить experiments в ensemble preset
→ выбрать average / weighted / voting / stacking
→ запустить ensemble.py
```

---

# 8. README-файлы проекта

- `README_IMPLEMENTED.md` — что уже реализовано.
- `README_ARCHITECTURE.md` — как устроен проект и куда идут данные.
- `README_CONFIG_REFERENCE.md` — что означает каждый параметр config.
- `README_TRAINING_AND_INFERENCE.md` — как физически работает training и inference.
- `README_USAGE_ENSEMBLES.md` — как пользоваться ансамблями.
