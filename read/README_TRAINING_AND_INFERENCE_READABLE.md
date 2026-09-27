# Как реально работает обучение, prediction и ensemble

Этот README нужен не как справочник по всем файлам проекта, а чтобы понять **что именно выполняет код**.

Главные вопросы здесь:

1. Что происходит во время обучения Classic ML?
2. Что происходит во время обучения Deep Learning?
3. Как после обучения делается prediction?
4. Откуда берётся OOF?
5. Как несколько моделей объединяются в ensemble?

Если нужно узнать, где лежит конкретный файл — см. `README_ARCHITECTURE.md`.  
Если нужно узнать значение параметра config — см. `README_CONFIG_REFERENCE.md`.

# 1. Самая короткая схема

## Classic ML

> train data
> → split на folds
> → preprocessing
> → model.fit()
> → prediction validation
> → metric
> → OOF
> → повторить для всех folds
> → final model на всех train data
> → test prediction

## Deep Learning

> train data
> → DataLoader
> → fold
> → epoch
> → batch
> → forward
> → loss
> → backward
> → optimizer.step()
> → validation
> → best checkpoint
> → OOF
> → test prediction через fold-модели

## Ensemble

> OOF/test predictions нескольких моделей
> → average / weighted average / voting / stacking
> → ensemble OOF
> → честный ensemble score
> → ensemble test prediction

Дальше разберём именно то, **как это реализовано в коде**.

# 2. Classic ML — как начинается обучение

Главная функция обучения:

```python
train(config, X, y, groups=None, ids=None)
```

Она находится в:

> classic_ml_pipeline_v8/train.py

Смысл функции:

> X, y
> ↓
> создать CV
> ↓
> для каждого fold обучить отдельную модель
> ↓
> получить validation prediction
> ↓
> собрать OOF
> ↓
> посчитать CV score
> ↓
> обучить final model на всех данных

# 3. Classic ML — тренировочный цикл по folds

Главная часть `train()` выглядит так:

```python
for fold, (tr, va) in enumerate(
    get_split_iterator(get_cv(config), X, y, groups, config)
):
    if fold not in requested:
        continue

    pipeline = _fit_pipeline(
        build_training_pipeline(X.iloc[tr], config),
        X.iloc[tr],
        y.iloc[tr],
        config,
        X.iloc[va],
        y.iloc[va],
    )

    labels, probabilities = _predict(
        pipeline,
        X.iloc[va],
        config,
        classes,
    )

    score = get_metric(
        config,
        y.iloc[va],
        labels,
        probabilities,
        classes,
    )

    scores.append(score)
    oof[va] = probabilities
    fold_assignment[va] = fold
```

Теперь по шагам.

## 3.1 Получаем train и validation индексы

```python
for fold, (tr, va) in enumerate(...):
```

Например при 5 folds:

> Fold 0:
> tr = 80% объектов
> va = 20%
>
> Fold 1:
> tr = другие 80%
> va = другие 20%

Один объект находится в validation только своего fold.

## 3.2 Создаётся новый pipeline

```python
build_training_pipeline(X.iloc[tr], config)
```

Внутри:

```python
def build_training_pipeline(X, config):
    return Pipeline([
        ("preprocessing", build_preprocessor(X, config)),
        ("model", get_estimator(config)),
    ])
```

То есть одна сущность содержит:

> raw features
> ↓
> preprocessing
> ↓
> model

Например:

> NaN
> ↓
> imputer
> ↓
> scaler / one-hot
> ↓
> XGBoost

Это важно: preprocessing обучается только на `X_train` текущего fold.

Validation data не участвуют в `fit` preprocessing.

Для `KNeighborsClassifier` и `KNeighborsRegressor` используйте `preprocessing.scale_numeric=True` (по умолчанию): `StandardScaler` для числовых колонок обучается отдельно в каждом fold, так что расстояния KNN вычисляются по преобразованным признакам. Выберите имя модели в `model.name`; остальные шаги CV, OOF и final training те же. Классификатор предоставляет `predict_proba()`, регрессор — `predict()`.

# 4. Classic ML — где реально обучается модель

В обычном случае `_fit_pipeline()` приходит к:

```python
pipeline.fit(X_train, y_train)
```

После этого sklearn Pipeline сам делает:

> X_train
> ↓
> preprocessing.fit_transform()
> ↓
> model.fit()

То есть для Classic ML у нас нет ручного цикла:

```python
for epoch:
    for batch:
```

Он скрыт внутри конкретной библиотеки.

Для регрессии внешнее имя остаётся `model.name="LinearRegression"`. `models.py` перед `pipeline.fit()` создаёт OLS, Ridge, Lasso или ElasticNet по `models.LinearRegression.regularization`. Аргументы, не подходящие выбранному estimator, не передаются. Поэтому `train.py`, CV и OOF не меняются.

Для:

- LogisticRegression;
- RandomForest;
- DecisionTree;
- XGBoost;
- LightGBM;
- CatBoost;

наш pipeline вызывает их обычный `.fit()`.

# 5. Classic ML — early stopping у boosting

Если включено:

> optimization.enabled = True
> optimization.training_control.boosting_early_stopping.enabled = True

обычный `pipeline.fit()` немного меняется.

Например для XGBoost:

```python
model.set_params(early_stopping_rounds=rounds)

model.fit(
    xt,
    y_train,
    eval_set=[(xv, y_val)],
    verbose=False,
)
```

Здесь:

> xt = preprocessing(train)
> xv = preprocessing(validation)

Validation используется только для контроля:

> перестала ли модель улучшаться?

Если улучшения долго нет — boosting заканчивается раньше.

Для baseline эта техника выключена.

# 6. Classic ML — prediction validation fold

После `fit`:

```python
labels, probabilities = _predict(
    pipeline,
    X.iloc[va],
    config,
    classes,
)
```

Для classification внутри происходит примерно:

```python
labels = pipeline.predict(X_val)
probabilities = pipeline.predict_proba(X_val)
```

Для regression:

```python
predictions = pipeline.predict(X_val)
```

После этого:

```python
score = get_metric(...)
```

Например:

> Fold 0 F1 = 0.812
> Fold 1 F1 = 0.829
> Fold 2 F1 = 0.805
> ...

# 7. Classic ML — откуда берётся OOF

Перед CV создаётся пустой массив:

```python
oof = np.full(..., np.nan)
```

После prediction каждого fold:

```python
oof[va] = probabilities
```

То есть:

> объект 0
> → prediction модели, которая НЕ обучалась на объекте 0
>
> объект 1
> → prediction модели, которая НЕ обучалась на объекте 1

После всех folds:

> OOF prediction существует почти для каждого train объекта

Именно поэтому OOF можно использовать для:

- честного общего CV score;
- threshold tuning;
- ensemble;
- stacking;
- подбора ensemble weights.

Сохраняется:

> oof_predictions.csv

# 8. Classic ML — зачем потом ещё final model

CV нужен для оценки модели.

Но каждая fold-модель обучалась только на части данных.

Поэтому после CV:

```python
if bool(config.training.train_final_model):
    final_pipeline = _fit_pipeline(
        build_training_pipeline(X, config),
        X,
        y,
        config,
        final_iterations=iterations,
    )

    joblib.dump(
        final_pipeline,
        config.paths.path_to_final_model,
    )
```

Теперь:

> ВСЕ train data
> ↓
> preprocessing
> ↓
> model.fit()
> ↓
> final model

Сохраняется:

> model.joblib

Именно эта модель обычно используется для обычного Classic inference.

# 9. Classic ML — как реализован prediction

Файл:

> classic_ml_pipeline_v8/predict.py

Основная функция:

```python
inference(config)
```

Сначала загружается готовый pipeline:

```python
pipeline = joblib.load(model_path)
```

Потом test data:

```python
df = load_csv(config.paths.path_to_test_dataset)
df = prepare_dataframe(df, config)
df = feature_engineering(df, config)
```

Из DataFrame убираются служебные поля:

```python
X_test = df.drop(
    columns=list(dict.fromkeys(feature_drop)),
    errors="ignore",
)
```

После этого model prediction.

## Classification

```python
classes = list(pipeline.classes_)
probabilities = pipeline.predict_proba(X_test)

predictions = (
    probabilities[:, 1]
    if len(classes) == 2
    else probabilities
)
```

## Regression

```python
predictions = pipeline.predict(X_test)
```

Потом optional postprocessing:

```python
predictions = postprocess_predictions(
    predictions,
    config,
)
```

И сохранение:

```python
save_predictions(...)
```

Итог:

> test.csv
> ↓
> тот же preprocessing
> ↓
> final model
> ↓
> predict / predict_proba
> ↓
> postprocessing
> ↓
> predictions.csv

# 10. Deep Learning — главное отличие

В Classic:

```python
model.fit()
```

и библиотека сама обучает модель.

В Deep Learning цикл написан вручную.

Главная идея:

```python
for epoch in epochs:
    for batch in train_loader:
        outputs = model(features)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
```

Именно это подробно реализовано в:

> dl_pipeline_v1/train.py

# 11. DL — один fold начинается с `run_fold()`

Основная функция:

```python
run_fold(...)
```

В начале каждого fold:

```python
set_seed(
    int(config.general.seed) + int(fold),
    deterministic=bool(config.reproducibility.deterministic),
)
```

Это нужно для воспроизводимости.

Дальше:

```python
train_loader, val_loader, train_idx, val_idx = get_fold_loaders(
    features,
    labels,
    groups,
    fold_ids,
    config,
    fold,
)
```

Получаем:

> train_loader
> val_loader

После этого создаётся новая модель:

```python
model = get_model(config).to(device)
```

Для каждого fold это новая независимая модель.

# 12. DL — создание компонентов обучения

После модели:

```python
optimizer = get_optimizer(config, model)
scheduler = get_scheduler(config, optimizer)
loss_function = get_loss(config)
scaler = _make_grad_scaler(
    device,
    _opt(config, "speed", "amp"),
)
```

Получаем четыре основные части.

## Model

> features → prediction

## Loss

> prediction + correct answer → число ошибки

## Optimizer

Использует gradients и меняет веса модели.

## Scheduler

При необходимости меняет learning rate во время обучения.

# 13. DL — главный тренировочный цикл

Самая важная функция для понимания:

```python
train_one_epoch(...)
```

Вот её основная логика:

```python
model.train()

optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(train_loader):
    features = batch["features"].to(device)
    labels = _move_labels(batch["labels"], device)

    outputs = model(features)

    loss, parts = loss_function(
        outputs,
        labels,
    )

    loss.backward()

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
```

В реальном pipeline код чуть сложнее из-за AMP, accumulation, clipping и EMA.

Но **обычный baseline по смыслу именно такой**.

Теперь разберём каждую строку.

# 14. DL — что происходит с одним batch

Допустим:

> batch_size = 64

DataLoader отдаёт:

```python
features = batch["features"]
labels = batch["labels"]
```

Условно:

> features.shape = (64, num_features)
> labels.shape   = (64,)

Переносим данные на устройство:

```python
features = features.to(
    device,
    non_blocking=True,
)
```

## Шаг 1 — forward

Реальный код:

```python
outputs = model(features)
```

Это:

> 64 объекта
> ↓
> нейросеть
> ↓
> 64 predictions

Например для classification raw output могут выглядеть:

> [
>     [ 2.1, -0.4],
>     [-1.2,  1.8],
>     ...
> ]

Это ещё не обязательно probabilities.

## Шаг 2 — loss

Реальный код:

```python
loss, parts = loss_function(
    outputs,
    labels,
)
```

Loss отвечает:

> насколько плох prediction модели относительно правильного ответа?

Например:

> хороший prediction → маленький loss
> плохой prediction → большой loss

## Шаг 3 — backward

В реальном коде:

```python
scaler.scale(backward_loss).backward()
```

Если AMP выключен, по смыслу это обычный:

```python
loss.backward()
```

PyTorch проходит граф вычислений назад и получает gradients:

> ∂loss / ∂weight

То есть для каждого обучаемого веса:

> как нужно изменить этот вес, чтобы уменьшить loss?

Важно:

```python
backward()
```

**не изменяет веса**.

Он только вычисляет gradients.

## Шаг 4 — optimizer.step()

В реальном pipeline:

```python
scaler.step(optimizer)
scaler.update()
```

Если AMP выключен, по смыслу:

```python
optimizer.step()
```

Вот здесь веса модели реально меняются.

Итак:

> forward
> → prediction
>
> loss
> → насколько prediction плох
>
> backward
> → gradients
>
> optimizer.step
> → изменить weights

Это центральная механика обучения нейросети.

## Шаг 5 — очистить gradients

```python
optimizer.zero_grad(set_to_none=True)
```

PyTorch gradients по умолчанию накапливает.

Поэтому после `optimizer.step()` их нужно очистить перед следующим обычным шагом.

# 15. Реальный `train_one_epoch()` с дополнительными техниками

В pipeline главный участок выглядит так:

```python
for step, batch in enumerate(iterator):
    features = batch["features"].to(
        device,
        non_blocking=True,
    )
    labels = _move_labels(
        batch["labels"],
        device,
    )

    with _autocast_context(
        device,
        _opt(config, "speed", "amp"),
    ):
        outputs = model(features)
        loss, parts = loss_function(
            outputs,
            labels,
        )

        window_size = min(
            accumulation,
            len(train_loader)
            - (step // accumulation) * accumulation,
        )

        backward_loss = loss / window_size

    scaler.scale(
        backward_loss
    ).backward()

    should_step = (
        ((step + 1) % accumulation == 0)
        or ((step + 1) == len(train_loader))
    )

    if should_step:
        if _opt(
            config,
            "training_control",
            "gradient_clipping",
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    config.optimization
                    .training_control
                    .gradient_clipping
                    .max_norm
                ),
            )

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        optimizer.zero_grad(
            set_to_none=True
        )
```

Главное:

> AMP
> gradient accumulation
> gradient clipping
> EMA

не создают другой training loop.

Они просто вставляются в определённые места обычного:

> forward
> → loss
> → backward
> → optimizer.step

Baseline можно понимать без них.

# 16. Что такое epoch

Допустим:

> train fold = 8 000 объектов
> batch_size = 100

Получается примерно:

> 80 batches = 1 epoch

То есть:

> epoch 1
> → модель один раз прошла train fold
>
> epoch 2
> → ещё раз прошла train fold,
>    но уже с обновлёнными weights

В `run_fold()`:

```python
for epoch in range(
    current_epoch,
    int(config.training.num_epochs),
):
    train_loss, train_parts = train_one_epoch(...)

    val_loss, current_metric, outputs, targets, val_parts = (
        validate_one_epoch(...)
    )
```

То есть каждая epoch:

> TRAIN
> ↓
> VALIDATION
> ↓
> TRAIN
> ↓
> VALIDATION
> ...

# 17. DL — как реализована validation

Функция:

```python
validate_one_epoch(...)
```

Начинается:

```python
model.eval()
```

И затем:

```python
with torch.no_grad():
    for batch in val_loader:
        features = batch["features"].to(device)
        labels = _move_labels(
            batch["labels"],
            device,
        )

        outputs = model(features)

        loss, parts = loss_function(
            outputs,
            labels,
        )
```

Ключевое отличие от training:

> НЕТ backward()
> НЕТ optimizer.step()

То есть validation:

> data
> → forward
> → loss
> → metric

Но веса модели не меняются.

После всех validation batches:

```python
metric = get_metric(
    config,
    targets_all,
    outputs_all,
)
```

Например получаем:

> train loss = 0.34
> val loss   = 0.42
> F1         = 0.81

# 18. DL — как выбирается best model

После validation:

```python
improved = is_improvement(
    config,
    current_metric,
    best_metric,
)
```

Если score стал лучше:

```python
if improved:
    best_metric = current_metric

    save_checkpoint(
        best_path,
        model,
        optimizer,
        scheduler,
        scaler,
        ...
    )
```

Получается:

> best.pt
> = состояние модели на лучшей validation metric

Также может сохраняться:

> last.pt
> = состояние после последней epoch

Для inference обычно используется `best.pt`.

# 19. DL — OOF prediction

Каждый fold возвращает prediction своей validation части.

Например:

> Fold 0 model
> → predictions для fold 0 validation rows
>
> Fold 1 model
> → predictions для fold 1 validation rows
>
> Fold 2 model
> → predictions для fold 2 validation rows

Потом они складываются обратно:

```python
for result in fold_results:
    idx = result["val_idx"]

    raw = np.asarray(
        get_primary_raw_outputs(
            config,
            result["outputs"],
        )
    )

    raw_predictions[idx] = values
    folds_array[idx] = int(result["fold"])
```

И сохраняются:

```python
save_predictions(
    config.paths.path_to_oof,
    ids[selected],
    artifact_values,
    task,
    classes,
    target[selected],
    folds_array[selected],
)
```

Получаем:

> oof_predictions.csv

Это DL-аналог OOF из Classic pipeline.

# 20. DL — как делается test prediction

Файл:

> dl_pipeline_v1/predict.py

Сначала для одной fold-модели:

```python
def _predict_one_model(
    config,
    checkpoint_path,
    loader,
    device,
):
    model = get_model(config).to(device)

    load_checkpoint(
        checkpoint_path,
        model,
        map_location=device,
    )

    model.eval()

    with torch.no_grad():
        for features in loader:
            features = features.to(device)

            converted = _convert_model_output(
                config,
                model(features),
            )
```

То есть:

> создать ту же architecture
> ↓
> загрузить weights из best.pt
> ↓
> model.eval()
> ↓
> test forward
> ↓
> prediction

# 21. DL — почему test проходит через несколько моделей

`predict_features()` делает:

```python
predictions = [
    _predict_one_model(
        config,
        path,
        loader,
        device,
    )
    for path in _checkpoint_paths(
        config,
        checkpoint_root,
        folds=folds,
    )
]

return _average_structures(
    predictions
)
```

То есть:

> test data
> ↓
> fold_0/best.pt → prediction 0
> fold_1/best.pt → prediction 1
> fold_2/best.pt → prediction 2
> ...
> ↓
> average
> ↓
> final prediction

Получается, что DL inference уже использует маленький ensemble из fold-моделей.

# 22. DL — финальный `inference()`

Основная часть:

```python
features = load_test_data(config)

checkpoint_root, folds = get_inference_source(
    config
)

probabilities_or_values = predict_features(
    config,
    features,
    checkpoint_root=checkpoint_root,
    folds=folds,
)
```

Потом выбирается основной output:

```python
primary = (
    probabilities_or_values[
        get_primary_head(config)
    ]
    if isinstance(
        probabilities_or_values,
        dict,
    )
    else probabilities_or_values
)
```

И predictions сохраняются:

```python
save_predictions(
    path,
    ids,
    values,
    task,
    classes,
)
```

Итоговая схема:

> test data
> ↓
> DataLoader
> ↓
> несколько fold checkpoints
> ↓
> forward каждой модели
> ↓
> average predictions
> ↓
> postprocessing
> ↓
> predictions.csv

# 23. Теперь главное про ансамбли

До этого были ансамбли внутри отдельных pipeline.

Но общий:

> ensemble.py

работает по другой логике.

Он **НЕ обучает заново XGBoost/MLP/CatBoost/Transformer**.

Он берёт уже готовые:

> oof_predictions.csv
> predictions.csv

от каждого experiment.

Например:

> XGBoost experiment
> ├── oof_predictions.csv
> └── predictions.csv
>
> CatBoost experiment
> ├── oof_predictions.csv
> └── predictions.csv
>
> MLP experiment
> ├── oof_predictions.csv
> └── predictions.csv
>
> Transformer experiment
> ├── oof_predictions.csv
> └── predictions.csv

Дальше их объединяет общий `ensemble.py`.

# 24. Почему ensemble обязательно использует OOF

Допустим есть:

> XGBoost
> CatBoost
> MLP
> Transformer

Для каждого train объекта имеются OOF predictions.

То есть base model не обучалась на объекте, prediction которого мы используем.

Получаем условно:

>              XGB    CAT    MLP    TRANS
> row 0        0.80   0.76   0.71   0.83
> row 1        0.13   0.20   0.09   0.15
> row 2        0.64   0.70   0.55   0.61
> ...

И рядом настоящий:

> target

Поэтому можно честно проверить:

> действительно ли объединение моделей улучшает score?

Если использовать predictions моделей на тех данных, на которых они обучались, получилась бы утечка.

# 25. Как `ensemble.py` загружает модели

На самом деле он модели не загружает.

Он загружает artifacts:

```python
members = [
    _read_artifact(
        cfg,
        member,
    )
    for member in members_cfg
]
```

Потом проверяет alignment:

```python
oof_ids, test_ids, y, fold, task, classes = _aligned(
    cfg,
    members,
)
```

Проверяются:

- одинаковые IDs;
- одинаковые targets;
- одинаковые folds;
- одинаковая task;
- одинаковые classes;
- одинаковый порядок объектов.

# 26. Prediction stack

После загрузки:

```python
oof_stack = np.stack([
    _matrix(
        m["oof"],
        m["columns"],
        task,
        classes,
    )
    for m in members
])

test_stack = np.stack([
    _matrix(
        m["test"],
        m["columns"],
        task,
        classes,
    )
    for m in members
])
```

Если 4 binary models и 10 000 train объектов:

> oof_stack.shape
> ≈ (4 models, 10 000 rows, 2 classes)

То есть теперь все predictions находятся в одном массиве.

# 27. Average ensemble

Самый простой ensemble.

Если:

> XGB prediction = 0.80
> CAT prediction = 0.70
> MLP prediction = 0.60
> TRANS prediction = 0.90

то:

> average
> = (0.80 + 0.70 + 0.60 + 0.90) / 4
> = 0.75

В коде average идёт через:

```python
active_weights = [1.0] * len(members)

oof_pred = _combine(
    oof_stack,
    active_weights,
)

test_pred = _combine(
    test_stack,
    active_weights,
)
```

Функция `_combine()` нормализует веса и объединяет predictions.

# 28. Weighted average

То же самое, но модели имеют разный вес.

Например:

> XGB         0.30
> CatBoost    0.35
> MLP         0.20
> Transformer 0.15

Prediction:

> final
> =
> XGB * 0.30
> +
> CatBoost * 0.35
> +
> MLP * 0.20
> +
> Transformer * 0.15

В коде:

```python
raw_weights = [
    m.get("weight")
    for m in members_cfg
]

weights = (
    np.asarray(raw_weights, dtype=float)
    / sum(raw_weights)
).tolist()

oof_pred = _combine(
    oof_stack,
    weights,
)

test_pred = _combine(
    test_stack,
    weights,
)
```

# 29. Optuna для weights

Веса можно не выбирать вручную.

Optuna получает только OOF predictions.

Она НЕ делает:

> XGBoost.fit()
> MLP training
> CatBoost.fit()

ещё раз.

Она делает только:

> trial 1
> weights = [0.2, 0.3, 0.1, 0.4]
> → объединить OOF
> → score
>
> trial 2
> weights = [0.4, 0.3, 0.2, 0.1]
> → объединить OOF
> → score
>
> ...

Реальный objective:

```python
def objective(trial):
    values = [
        trial.suggest_float(
            f"weight_{i}",
            1e-6,
            1.0,
        )
        for i in range(len(stack))
    ]

    return _scores(
        config,
        y,
        _combine(stack, values),
        task,
        classes,
    )
```

После поиска:

```python
weights = _optimize_weights(...)
test_pred = _combine(
    test_stack,
    weights,
)
```

# 30. Почему weight optimization тоже cross-fitted

Чтобы не переобучиться на OOF при подборе weights, pipeline делает ещё один уровень разделения.

В коде:

```python
for group in folds:
    train_mask = fold != group
    val_mask = fold == group

    cv_weights = _optimize_weights(
        cfg,
        oof_stack[:, train_mask],
        y[train_mask],
        task,
        classes,
        seed,
    )

    oof_pred[val_mask] = _combine(
        oof_stack[:, val_mask],
        cv_weights,
    )
```

То есть веса для fold `0` подбираются **без fold 0**.

Это позволяет получить более честный ensemble OOF score.

После оценки уже подбираются final weights на всех OOF:

```python
weights = _optimize_weights(
    cfg,
    oof_stack,
    y,
    task,
    classes,
    seed,
)
```

И они применяются к test:

```python
test_pred = _combine(
    test_stack,
    weights,
)
```

# 31. Voting

Voting используется только для classification.

## Hard voting

Каждая модель сначала выбирает класс:

> XGB         → class 1
> CatBoost    → class 1
> MLP         → class 0
> Transformer → class 1

Большинство:

> class 1

В коде:

```python
oof_pred = _hard_vote(
    oof_stack,
    classes,
)

test_pred = _hard_vote(
    test_stack,
    classes,
)
```

## Soft voting

Используются probabilities.

По сути это усреднение вероятностей.

Обычно probabilities содержат больше информации, чем просто hard labels.

# 32. Stacking — идея

Stacking не задаёт веса вручную.

Predictions base models становятся **новыми признаками**.

Например:

> Original row
> ↓
> XGB OOF probability        = 0.80
> CatBoost OOF probability   = 0.73
> MLP OOF probability        = 0.62
> Transformer OOF probability= 0.88
> ↓
> [0.80, 0.73, 0.62, 0.88]
> ↓
> meta-model
> ↓
> final prediction

Meta-model сама учится:

> когда какой base model стоит слушать сильнее.

# 33. Stacking — как реализовано в коде

Сначала predictions превращаются в meta-features:

```python
features = np.concatenate([
    oof_stack[i]
    for i in range(len(members))
], axis=1)

test_features = np.concatenate([
    test_stack[i]
    for i in range(len(members))
], axis=1)
```

Потом meta-model обучается cross-fitted.

```python
for group in folds:
    train_mask = fold != group
    val_mask = fold == group

    model, _, _ = _meta_model(
        cfg,
        task,
    )

    _fit_meta(
        model,
        features[train_mask],
        y[train_mask],
    )

    oof_pred[val_mask] = _meta_output(
        model,
        features[val_mask],
        task,
        classes,
    )
```

Это важно.

Для объектов fold 0 meta-model обучается без fold 0.

Поэтому ensemble OOF остаётся честным.

# 34. Stacking — final meta-model

Когда ensemble score уже можно честно оценить:

```python
meta_model, meta_name, meta_params = _meta_model(
    cfg,
    task,
)

_fit_meta(
    meta_model,
    features,
    y,
)

test_pred = _meta_output(
    meta_model,
    test_features,
    task,
    classes,
)
```

То есть:

> ВСЕ OOF base predictions
> ↓
> final meta-model.fit()
> ↓
> test base predictions
> ↓
> final ensemble prediction

По умолчанию:

> classification → LogisticRegression
> regression     → Ridge

# 35. Полный путь ensemble

Весь процесс можно представить так:

> XGBoost training
> → XGB OOF + test
>
> CatBoost training
> → CAT OOF + test
>
> MLP training
> → MLP OOF + test
>
> Transformer training
> → TRANS OOF + test
>
>               ↓
>          ensemble.py
>               ↓
>
> average
> или weighted_average
> или voting
> или stacking
>
>               ↓
>
> ensemble OOF
> → ensemble CV score
>
> ensemble test predictions
> → predictions.csv

# 36. Что ensemble сохраняет

После выполнения:

> checkpoints/ensembles/<experiment>/

создаются:

> oof_predictions.csv
> predictions.csv
> config.yaml
> metadata.json
> metrics.json
> meta_model.joblib       # если stacking
> plots/

`metrics.json` содержит:

- score ensemble;
- fold scores;
- CV mean/std;
- scores base models.

`metadata.json` содержит:

- тип ensemble;
- members;
- weights;
- использовалась ли weight optimization;
- meta-model;
- task;
- classes.

# 37. Чем отличаются три уровня моделей в проекте

Это полезно не путать.

## 1. Обычная модель

Например:

> XGBoost
> CatBoost
> MLP
> Transformer

Она сама делает prediction.

## 2. Fold ensemble в DL

> MLP fold 0
> MLP fold 1
> MLP fold 2
> ...
> ↓
> average

Это несколько версий **одной архитектуры**, обученных на разных folds.

## 3. Общий ensemble.py

> XGBoost experiment
> +
> CatBoost experiment
> +
> MLP experiment
> +
> Transformer experiment

Это объединение уже разных экспериментов/моделей.

# 38. Где здесь Optuna

Есть три разных применения.

## Classic Optuna

Ищет hyperparameters модели:

Для `model.name="LinearRegression"` search space зависит от режима: `none` — один baseline trial без параметров, `ridge`/`lasso` — только `alpha`, `elasticnet` — `alpha` и `l1_ratio`. Режим выбирается в config до запуска, а не внутри trial.

Для обоих KNN-имён Optuna берёт из `tuning.search_spaces` только `n_neighbors` (3–35 с шагом 2), `weights` (`uniform` или `distance`) и `p` (1 или 2). `algorithm`, `metric`, `leaf_size` и `n_jobs` остаются из `models.<имя>`. Если KNN включён во внутренний `estimator_strategy`, конфиг содержит по восемь моделей и восемь весов для каждой задачи.

> max_depth
> learning_rate
> n_estimators
> ...

После поиска обычный Classic pipeline обучается с лучшими params.

## DL Optuna

Ищет:

> learning rate
> hidden sizes
> и другие параметры из search space

Каждый trial запускает реальное DL training.

## Ensemble Optuna

Модели НЕ переобучает.

Ищет только:

> weights base models

на уже готовых OOF predictions.

# 39. Что нужно понимать в первую очередь

Перед следующим проектом достаточно уверенно понимать вот это.

## Classic

> fold
> → preprocessing.fit на train
> → model.fit
> → validation prediction
> → metric
> → OOF
> → final fit
> → test prediction

## DL

> batch
> → model(features)
> → loss
> → backward
> → optimizer.step

и затем:

> epoch
> → train
> → validation
> → best checkpoint

и затем:

> folds
> → OOF
> → test predictions

## Ensemble

> base OOF predictions
> +
> base test predictions
> ↓
> average / weights / meta-model
> ↓
> ensemble OOF score
> +
> ensemble test prediction

Если эти три схемы понятны, вся остальная архитектура уже накладывается сверху.

# 40. Коротко про optional-техники

Их не нужно учить заранее.

Они вставляются в уже понятный основной процесс.

> AMP
> → меняет precision вычислений
>
> gradient accumulation
> → несколько backward перед optimizer.step
>
> gradient clipping
> → ограничивает слишком большие gradients
>
> scheduler
> → меняет learning rate
>
> warmup
> → плавно разгоняет learning rate в начале
>
> EMA
> → хранит сглаженную копию weights
>
> early stopping
> → останавливает обучение, если validation не улучшается
>
> finetuning
> → сначала обучает часть pretrained model, потом может разморозить больше слоёв
>
> metric learning
> → добавляет обучение embedding space
>
> self-training
> → добавляет pseudo-labels
>
> Optuna
> → многократно запускает training с разными параметрами

Все они вторичны относительно главного:

> Classic:
> fit → predict → metric
>
> DL:
> forward → loss → backward → optimizer.step
>
> Ensemble:
> OOF predictions → combine → score
