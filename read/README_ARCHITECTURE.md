# README — Что делает каждый файл

Этот файл — краткая карта проекта.

Здесь без подробной теории: только что делает каждый файл, где он участвует и какие файлы реально важно понимать в первую очередь.

---

# 1. Корень проекта

| Файл | Что делает |
|---|---|
| `ensemble.py` | Универсально собирает ансамбли из уже готовых OOF/test predictions. Поддерживает average, weighted average, voting, stacking и подбор весов. |
| `ensemble_config.py` | Config для ансамблей: какие эксперименты объединять, какой preset использовать, какие веса и meta-model. |
| `artifact_format.py` | Общий формат сохранения predictions для Classic и DL, чтобы `ensemble.py` мог читать их одинаково. |
| `requirements.txt` | Общие зависимости проекта. |
| `tests/test_ensemble.py` | Тесты системы ансамблей: alignment, stacking, веса и проверки данных. |
| `README_ARCHITECTURE.md` | Простое объяснение, как устроен весь проект. |
| `README_CONFIG_REFERENCE.md` | Полный справочник по config и параметрам. |
| `README_TRAINING_AND_INFERENCE.md` | Как физически происходит обучение и prediction. |
| `README_USAGE_ENSEMBLES.md` | Как пользоваться ансамблями. |
| `README_IMPLEMENTED.md` | Что уже реализовано, что optional и что пока placeholder. |
| `AUDIT_BEFORE_CHANGES.md` | Какие проблемы были найдены до переработки pipeline. |

---

# 2. Classic ML

| Файл | Что делает |
|---|---|
| `config.py` | Все настройки Classic pipeline: данные, модели (в том числе KNN для обеих задач), CV, optimization, Optuna, графики и т.д. |
| `main.py` | Главная точка запуска. Читает config и решает: training, tuning или inference. |
| `data.py` | Загружает CSV, отделяет target, groups и строит preprocessing для числовых и категориальных признаков. |
| `checks.py` | Проверяет данные, в основном возможный leakage и подозрительные признаки. |
| `features.py` | Место для task-specific feature engineering. Сейчас в основном hook/заглушка. |
| `models.py` | Реестр Classic моделей, включая `KNeighborsClassifier`/`KNeighborsRegressor`; при `model.name="LinearRegression"` выбирает OLS/Ridge/Lasso/ElasticNet по `models.LinearRegression.regularization`, передавая только применимые аргументы. |
| `estimator_strategy.py` | Собирает внутренние ансамбли Classic: bagging, voting, average, weighted average, stacking. |
| `train.py` | Главная логика обучения: CV, folds, fit, metric, OOF, final training, early stopping, threshold tuning. |
| `predict.py` | Загружает готовые модели и делает inference на новых/test данных. |
| `tuning.py` | Optuna: создаёт trials, предлагает параметры из `config.tuning.search_spaces`, запускает CV и сохраняет лучшие hyperparameters. Для Classic `LinearRegression` фильтрует `alpha`/`l1_ratio` по выбранной регуляризации; для KNN использует `n_neighbors`, `weights`, `p`. |
| `postprocessing.py` | Финальная обработка predictions после модели. Сейчас базовая расширяемая точка. |
| `visualization.py` | Строит `validation.png`, `cv_scores.png`, feature importance и optional SHAP. |
| `artifact_io.py` | Сохраняет OOF/test predictions в стандартизированном формате. |
| `tracking.py` | W&B и сохранение experiment metrics/artifacts. |
| `notifier.py` | Отправляет уведомления в Telegram. |
| `utils.py` | Общие технические функции: seed, директории, metadata, config snapshot, environment, results. |
| `test_telegram.py` | Быстрая проверка подключения Telegram. |
| `test_wandb.py` | Быстрая проверка подключения W&B. |
| `requirements.txt` | Зависимости именно Classic pipeline. |
| `telegram_credits.example.json` | Пример файла с Telegram credentials. |
| `.gitignore` | Какие файлы Git не должен сохранять. |
| `data/.gitkeep` | Сохраняет пустую папку `data/` в Git. |
| `logs/.gitkeep` | Сохраняет пустую папку `logs/` в Git. |
| `checkpoints/.gitkeep` | Сохраняет пустую папку `checkpoints/` в Git. |

## Главная цепочка Classic

Для линейной регрессии снаружи всегда указывается `model.name="LinearRegression"`. В `config.py` задаётся `models.LinearRegression.regularization`: `none` создаёт обычную `LinearRegression`, `ridge` — Ridge, `lasso` — Lasso, `elasticnet` — ElasticNet. `main.py`, CV, preprocessing, OOF и inference работают с полученным sklearn estimator без отдельных веток. Внутренний Classic stacking использует тот же выбор для регрессионной meta-model; корневой ансамбль готовых экспериментов имеет собственный независимый config.

Для метода ближайших соседей выбирайте `model.name="KNeighborsClassifier"` или `"KNeighborsRegressor"`. Существующий `data.py` стандартизует числовые признаки при `preprocessing.scale_numeric=True` внутри каждого fold; отдельной ветки обучения KNN нет. Обе модели доступны в `estimator_strategy.models` и имеют по одному весу в соответствующем списке из восьми весов.

```text
main.py
→ data.py
→ checks.py
→ features.py
→ models.py / estimator_strategy.py
→ train.py
→ visualization.py
→ artifact_io.py
→ checkpoints
```

---

# 3. Deep Learning

| Файл | Что делает |
|---|---|
| `config.py` | Все настройки DL: model, training, optimizer, regularization, optimization, strategies, tuning и т.д. |
| `main.py` | Главная точка запуска DL pipeline. |
| `data.py` | Загружает данные, создаёт Dataset/DataLoader, folds и seeded workers. |
| `checks.py` | Проверяет корректность training data. |
| `preprocessing.py` | Место для task-specific preprocessing одного sample. |
| `augmentations.py` | Место для augmentation данных. |
| `models.py` | Содержит MLP/Transformer backbone, prediction heads и создаёт итоговую DL-модель. |
| `train.py` | Главный training loop: forward → loss → backward → optimizer → validation → checkpoints → OOF. |
| `predict.py` | Inference готовой DL-модели или нескольких fold-моделей. |
| `losses.py` | Создаёт loss и маршрутизирует loss для обычных или multi-head моделей. |
| `metrics.py` | Считает метрики и преобразует raw output модели в predictions. |
| `optimizers.py` | Создаёт optimizer и группы параметров, например backbone/head. |
| `schedulers.py` | Создаёт scheduler и правильно вызывает его `step()`. |
| `checkpointing.py` | Сохраняет и загружает checkpoints модели/optimizer/scheduler. |
| `tuning.py` | Optuna для DL: trials, search space, pruning, best params. |
| `ema.py` | Реализация Exponential Moving Average весов модели. |
| `finetuning.py` | Pretrained weights, freeze/unfreeze backbone, transfer learning. |
| `metric_learning.py` | Metric-learning loss, например triplet loss. |
| `hard_negative_mining.py` | Выбирает сложные negative examples для metric learning. |
| `samplers.py` | BalancedBatchSampler и другие специальные sampling-механизмы. |
| `self_training.py` | Pseudo-labeling: модель размечает unlabeled data и переобучается. |
| `multi_head.py` | Логика нескольких output heads и primary head. |
| `model_conversion.py` | Экспортирует обученную модель в deployment/export формат. |
| `postprocessing.py` | Финальная обработка prediction. |
| `visualization.py` | Training curves, validation plot, CV scores. |
| `artifact_io.py` | Сохраняет OOF/test predictions в общем формате. |
| `tracking.py` | TensorBoard/W&B/local experiment tracking. |
| `notifier.py` | Telegram notifications. |
| `utils.py` | Seed, device, experiment folders, metadata, environment и другие helper-функции. |
| `generating_dataset.py` | Вспомогательная генерация dataset. |
| `requirements.txt` | Зависимости DL pipeline. |
| `telegram_credits.example.json` | Пример Telegram credentials. |
| `.gitignore` | Что не отправлять в Git. |
| `data/.gitkeep` | Держит пустую `data/` папку в репозитории. |
| `logs/.gitkeep` | Держит пустую `logs/` папку в репозитории. |
| `checkpoints/.gitkeep` | Держит пустую `checkpoints/` папку в репозитории. |

## Главная цепочка DL

```text
main.py
→ data.py
→ preprocessing.py / augmentations.py
→ models.py
→ optimizers.py + losses.py + schedulers.py
→ train.py
→ checkpointing.py
→ visualization.py
→ artifact_io.py
→ predict.py
```

Дополнительные техники подключаются сбоку:

```text
finetuning.py
metric_learning.py
hard_negative_mining.py
self_training.py
multi_head.py
ema.py
```

---

# 4. Какие файлы реально важно понимать сначала

Не нужно сразу разбираться во всех файлах.

Для начала достаточно понимать этот набор.

## Classic

```text
config.py
main.py
data.py
models.py
train.py
predict.py
```

### Что в них происходит

```text
config.py
→ говоришь pipeline, что делать

main.py
→ запускает нужный сценарий

data.py
→ готовит данные

models.py
→ создаёт модель

train.py
→ обучает и валидирует

predict.py
→ делает prediction
```

---

## Deep Learning

```text
config.py
main.py
data.py
models.py
train.py
predict.py
losses.py
optimizers.py
```

### Что в них происходит

```text
config.py
→ настройки эксперимента

main.py
→ запускает pipeline

data.py
→ Dataset / DataLoader / folds

models.py
→ нейросеть

losses.py
→ что модель минимизирует

optimizers.py
→ как обновляются веса

train.py
→ сам цикл обучения

predict.py
→ inference
```

---

## Общее

```text
ensemble.py
ensemble_config.py
```

### Что в них происходит

```text
ensemble_config.py
→ описываешь, какие experiments объединить

ensemble.py
→ берёт их predictions и собирает итоговый ensemble
```

---

# 5. Самая короткая схема проекта

## Classic

```text
данные
→ preprocessing
→ модель
→ CV
→ fit
→ validation
→ OOF
→ final model
→ prediction
```

## Deep Learning

```text
данные
→ DataLoader
→ модель
→ forward
→ loss
→ backward
→ optimizer
→ validation
→ checkpoint
→ OOF
→ prediction
```

## Ensemble

```text
готовые OOF/test predictions
→ average / weighted / voting / stacking
→ final prediction
```

---

# 6. Главное

Не нужно учить все файлы заранее.

Когда начнёшь новый проект, основная работа почти всегда будет идти через:

```text
config
data
models
train
predict
```

Остальные файлы — это отдельные инструменты, которые подключаются по необходимости:

```text
Optuna
scheduler
EMA
finetuning
metric learning
self-training
ensemble
tracking
visualization
```

Поэтому сначала нужно понимать основной путь данных и обучения, а уже потом разбирать дополнительные техники по мере их использования.
