# Как пользоваться ансамблями

Этот README нужен для практики.

Главная идея очень простая:

```text
сначала отдельно обучаем несколько моделей
↓
каждая сохраняет OOF + test predictions
↓
ensemble.py читает эти файлы
↓
объединяет predictions
↓
получаем новый ensemble experiment
```

Тебе НЕ нужно писать отдельный Python-файл под каждую комбинацию моделей.

---

# 1. Пример

Допустим, ты обучил четыре модели:

```text
XGBoost
CatBoost
MLP
Transformer
```

Каждая уже закончила training.

Получается:

```text
XGBoost ───────┐
CatBoost ──────┤
               ├→ ensemble.py → final prediction
MLP ───────────┤
Transformer ───┘
```

`ensemble.py` не обучает их заново.

Он использует только их сохранённые predictions.

Если один из Classic-экспериментов — линейная регрессия, его `model.name` остаётся `LinearRegression`, а `models.LinearRegression.regularization` задаёт OLS/Ridge/Lasso/ElasticNet при обучении базовой модели. Здесь ансамбль читает только её OOF/test predictions; Ridge как пример meta-model корневого ансамбля настраивается отдельно.

Classic-эксперимент с KNN выбирается через `model.name="KNeighborsClassifier"` или `"KNeighborsRegressor"` и тоже сохраняет OOF/test predictions для этого ансамбля. Внутренний `estimator_strategy` Classic включает KNN в соответствующие списки моделей; для `weighted_average` каждой из восьми моделей соответствует отдельный вес. Держите `preprocessing.scale_numeric=True` для числовых признаков KNN.

---

# 2. Что должна сохранить каждая модель

У каждого experiment должны быть:

```text
checkpoints/<experiment_name>/
├── oof_predictions.csv
├── predictions.csv
└── metadata.json
```

## `oof_predictions.csv`

Это predictions на train, полученные через CV.

Нужны для:

- честной оценки ансамбля;
- подбора weights;
- stacking.

## `predictions.csv`

Prediction на test.

Нужны для получения итогового test prediction ансамбля.

## `metadata.json`

Хранит информацию, необходимую для проверки совместимости экспериментов.

---

# 3. Почему нельзя просто смешать любые CSV

Модели должны предсказывать **одни и те же объекты**.

Например:

```text
XGBoost row id=123
```

должен соответствовать:

```text
MLP row id=123
```

Поэтому ensemble проверяет:

- ID;
- target;
- folds;
- task;
- classes;
- размеры predictions;
- NaN/inf.

Если данные несовместимы, ensemble должен остановиться с понятной ошибкой.

---

# 4. Где настраивается ансамбль

Файл:

```text
ensemble_config.py
```

Главное:

```python
config["ensemble"]["enabled"] = True
```

и выбрать preset:

```python
config["ensemble"]["preset"] = "boosting_dl"
```

Preset — это просто сохранённое описание:

> какие эксперименты взять и каким способом объединить.

---

# 5. Пример preset

Упрощённо:

```python
"boosting_dl": {
    "type": "average",
    "members": [
        {
            "name": "xgb",
            "source": "classic",
            "experiment": "xgb_v1",
        },
        {
            "name": "mlp",
            "source": "dl",
            "experiment": "mlp_v1",
        },
    ],
}
```

Это означает:

```text
взять experiment xgb_v1
+
взять experiment mlp_v1
↓
усреднить их predictions
```

Чтобы создать другой ансамбль, ты просто добавляешь другой preset.

Новый `.py` не нужен.

---

# 6. Как запустить

После того как нужные base experiments уже обучены:

```bash
python ensemble.py
```

Результат появится отдельно:

```text
checkpoints/ensembles/<ensemble_experiment_name>/
```

---

# 7. `average`

Самый простой вариант.

Допустим:

```text
XGBoost probability = 0.80
CatBoost probability = 0.70
MLP probability = 0.50
```

Average:

```text
(0.80 + 0.70 + 0.50) / 3 = 0.667
```

То есть все модели имеют одинаковый вес.

Config:

```text
type = average
```

Когда использовать:

- хочешь простой baseline ensemble;
- модели примерно одинаково хорошие;
- пока не хочешь подбирать weights.

С этого варианта я бы начинал первым.

---

# 8. `weighted_average`

То же усреднение, но у моделей разные веса.

Например:

```text
XGBoost    weight = 0.40
CatBoost   weight = 0.30
MLP        weight = 0.20
Transformer weight = 0.10
```

Итог:

```text
prediction =
XGB * 0.40
+ CatBoost * 0.30
+ MLP * 0.20
+ Transformer * 0.10
```

Вес можно задать руками в preset.

---

# 9. Автоматический подбор weights

Если не хочешь выбирать веса вручную, их может подобрать Optuna.

Для этого включается ensemble weight optimization.

Главная логика:

```text
base models уже обучены
↓
берём только их OOF predictions
↓
Optuna пробует разные weights
↓
считает metric
↓
выбирает лучшие weights
```

ВАЖНО:

```text
XGBoost / CatBoost / MLP / Transformer
НЕ переобучаются на каждом trial
```

Меняются только числа weights.

Поэтому это намного дешевле обычной Optuna для моделей.

---

# 10. `voting`

Используется только для classification.

Есть два типа.

## Hard voting

Каждая модель голосует за класс.

Например:

```text
XGB → class 1
CatBoost → class 1
MLP → class 0
```

Итог:

```text
class 1
```

потому что 2 голоса против 1.

## Soft voting

Усредняются probabilities.

Например:

```text
XGB       P(class1)=0.8
CatBoost  P(class1)=0.7
MLP       P(class1)=0.4
```

Среднее:

```text
0.633
```

Soft voting обычно информативнее, потому что учитывает уверенность модели, а не только готовый label.

---

# 11. `stacking`

Это самый интересный вариант.

Здесь predictions базовых моделей становятся **новыми признаками**.

Например для одного объекта:

```text
XGBoost prediction      = 0.80
CatBoost prediction     = 0.72
MLP prediction          = 0.55
Transformer prediction  = 0.61
```

Получаем новый объект:

```text
[0.80, 0.72, 0.55, 0.61]
```

На таких признаках обучается ещё одна модель:

```text
meta-model
```

Схема:

```text
XGBoost OOF ───────┐
CatBoost OOF ──────┤
MLP OOF ───────────┼→ meta-model → final OOF prediction
Transformer OOF ───┘
```

Для classification по умолчанию используется LogisticRegression.

Для regression — Ridge.

---

# 12. Почему stacking использует именно OOF

Нельзя обучать meta-model на predictions моделей по тем объектам, которые базовые модели уже видели при training.

Иначе будет leakage.

Поэтому используются OOF predictions:

```text
каждый OOF prediction получен моделью,
которая не обучалась на этом объекте
```

В текущей реализации stacking дополнительно оценивается crossfit по OOF folds.

То есть meta-model тоже не оценивается на тех же строках, на которых она была fit.

---

# 13. Что я бы использовал по порядку

На новом проекте не начинай сразу со stacking.

Нормальный порядок:

```text
1. Лучшие отдельные модели
↓
2. average
↓
3. weighted_average
↓
4. Optuna weights
↓
5. stacking
```

Так сразу видно, действительно ли усложнение помогает.

---

# 14. Пример: два бустинга + две нейросети

Допустим уже есть experiments:

```text
xgb_v1
catboost_v1
mlp_v1
transformer_v1
```

Создаёшь preset:

```python
"all_models": {
    "type": "average",
    "members": [
        {"name": "xgb", "source": "classic", "experiment": "xgb_v1"},
        {"name": "cat", "source": "classic", "experiment": "catboost_v1"},
        {"name": "mlp", "source": "dl", "experiment": "mlp_v1"},
        {"name": "transformer", "source": "dl", "experiment": "transformer_v1"},
    ],
}
```

Потом:

```python
config["ensemble"]["preset"] = "all_models"
```

и запускаешь:

```bash
python ensemble.py
```

Всё.

---

# 15. Что сохраняется после ensemble

```text
checkpoints/ensembles/<ensemble_name>/
├── oof_predictions.csv
├── predictions.csv
├── config.yaml
├── metadata.json
├── metrics.json
├── plots/
└── meta_model.joblib    # только stacking
```

## `metrics.json`

Содержит:

- score ensemble;
- fold scores;
- CV mean/std;
- scores базовых моделей.

## `metadata.json`

Содержит:

- members;
- type;
- weights;
- был ли weight optimization;
- meta-model для stacking.

---

# 16. Графики ensemble

Создаются три основных графика.

## `validation.png`

Classification:

```text
confusion matrix
```

Regression:

```text
true vs predicted
+
residual plot
```

## `cv_scores.png`

Показывает score ensemble по folds и mean/std.

## `ensemble_diagnostics.png`

Показывает:

```text
base model scores vs ensemble
```

и при необходимости:

```text
weights / meta-model contribution
```

а также:

```text
prediction correlation
```

Корреляция полезна, потому что ансамбль особенно интересен, когда хорошие модели ошибаются по-разному.

---

# 17. Самые частые ошибки

## Разные ID

```text
XGB predictions относятся к одним объектам,
MLP predictions — к другим
```

Такой ensemble нельзя собирать.

## Разные folds

Для честного stacking/weight optimization желательно, чтобы OOF fold assignment был совместим.

Pipeline это проверяет.

## Нет `predictions.csv`

Значит base experiment не делал inference на test.

Сначала надо получить test predictions.

## Использовать `submission.csv` вместо `predictions.csv`

Не надо.

Для ансамбля используется именно probability/value prediction, а не уже thresholded labels.

## Подбирать weights по test target

Так делать нельзя.

Weights подбираются по OOF train predictions.

---

# 18. Самая короткая инструкция

Если всё остальное забыл:

```text
1. Обучи несколько моделей.
2. Убедись, что у каждой есть oof_predictions.csv и predictions.csv.
3. Добавь их experiment_name в preset ensemble_config.py.
4. Выбери average для первого теста.
5. Поставь ensemble.enabled=True.
6. Запусти python ensemble.py.
7. Сравни score отдельных моделей и ensemble.
8. Только потом пробуй weights/stacking.
```
