# House Prices

Регрессия для [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques): прогноз стоимости дома по табличным признакам. Target — `SalePrice`. Локальная метрика — RMSE в пространстве `log1p(SalePrice)`; меньшее значение лучше. Исходные данные: 1460 обучающих и 1459 тестовых объектов.

Проект объединяет классические регрессоры, MLP и ансамбль Classic stacking + MLP. История экспериментов и результаты ведутся отдельно в заметке Obsidian; текущие метрики также сохраняются в файлы проекта.

## Результаты и эксперименты

История экспериментов, CV и Public LB: [experiments.md](experiments.md).

## Установка

Python 3.12. Все команды выполняются из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

В Windows активация: `.venv\Scripts\activate`.

Исходные CSV должны находиться здесь:

```text
classic_ml_pipeline_v8/data/train.csv
classic_ml_pipeline_v8/data/test.csv
```

В обоих конфигах пайплайнов `PROJECT_ROOT = Path(__file__).resolve().parent`. DL читает исходные CSV из каталога данных Classic.

## Полное обучение

Для обучения с заданными гиперпараметрами установите `tuning.enabled = False` в `classic_ml_pipeline_v8/config.py` и `dl_pipeline_v1/config.py`.

```bash
python main.py --mode train --pipeline final
```

Команда последовательно обучает Classic stacking и MLP с кросс-валидацией по пяти фолдам, сохраняет модели и OOF-предсказания, затем усредняет предсказания и создаёт `outputs/submission.csv`.

Имена новых экспериментов формируются из `new_experiment_prefix` в корневом `config.py`:

```python
"new_experiment_prefix": "clean_cv_v2",
```

В этом примере создаются `clean_cv_v2_classic`, `clean_cv_v2_dl`, `clean_cv_v2_final`. Перед повторным обучением выбирайте свободный префикс: существующие эксперименты защищены от перезаписи.

Отдельное обучение компонентов:

```bash
python main.py --mode train --pipeline classic
python main.py --mode train --pipeline dl
```

Эти команды обучают выбранный компонент; итоговый ансамбль собирается при `--pipeline final`.

## Отчёт и сохранённый сабмит

Для уже выполненного запуска `clean_cv_v1` в корневом `config.py` укажите:

```python
"mode": "report",
"pipeline": "final",
"classic_experiment": "clean_cv_v1_classic",
"dl_experiment": "clean_cv_v1_dl",
"final_experiment": "clean_cv_v1_final",
```

```bash
python main.py
```

Или явно:

```bash
python main.py --mode report --pipeline final
```

Режим `report` собирает таблицу сохранённых метрик и копирует сабмит выбранного `final_experiment` в `outputs/submission.csv`.

| Файл / каталог | Назначение |
|---|---|
| `outputs/submission.csv` | Сабмит Kaggle: `Id`, `SalePrice` |
| `outputs/results.csv` | Таблица CV сохранённых экспериментов |
| `checkpoints/ensembles/<final_experiment>/metrics.json` | Метрики итогового ансамбля |
| `checkpoints/ensembles/<final_experiment>/predictions.csv` | Сохранённый сабмит ансамбля |
| `classic_ml_pipeline_v8/checkpoints/<classic_experiment>/` | Classic-модель, OOF, тестовые предсказания и метрики |
| `dl_pipeline_v1/checkpoints/<dl_experiment>/` | Веса и препроцессоры фолдов, OOF, тестовые предсказания и метрики MLP |

После обучения с новым префиксом обновите `classic_experiment`, `dl_experiment` и `final_experiment` для последующей работы с этим запуском. Сабмит каждой отдельной модели сохраняется в `predictions.csv` внутри её каталога эксперимента.

## Конфиги

| Конфиг | Что настраивает |
|---|---|
| `config.py` в корне | Режим запуска, выбор пайплайна, папка результатов и имена экспериментов |
| `classic_ml_pipeline_v8/config.py` | Признаки, предобработка, регрессоры, гиперпараметры, CV, Classic stacking и Optuna |
| `dl_pipeline_v1/config.py` | Архитектура MLP, optimizer, scheduler, loss, параметры обучения, CV и Optuna |
| `ensemble_config.py` | Способ объединения сохранённых Classic/DL-предсказаний и веса моделей |

Корневой конфиг связывает готовые пайплайны: задаёт, что запускать и какие эксперименты использовать. Подробные параметры обучения остаются в конфигах Classic и DL. Аргументы `--mode` и `--pipeline` переопределяют корневые настройки для конкретного запуска.

В режиме `train` имена создаются из `new_experiment_prefix`. В режиме `report` используется выбранный `final_experiment`. Режим `inference` пересчитывает предсказания из весов выбранных Classic и DL; при `--pipeline final` он создаёт ансамбль `<new_experiment_prefix>_final`, поэтому для этого каталога нужен свободный префикс.

## Подготовка данных и модели

Из обучающей выборки исключены `Id = 524, 1299, 1397`; остаётся 1457 строк. Тестовые строки сохраняются полностью. Удалены признаки `GarageArea`, `GarageYrBlt`, `Exterior2nd`, `GarageCond`.

Пропуски, обозначающие отсутствие объекта, заполняются фиксированными значениями `"None"` или `0`. Остальные числовые пропуски заполняются медианой, категориальные — самым частым значением. Добавлены `MSSubClassCat` и `Spaciousness = (1stFlrSF + 2ndFlrSF) / TotRmsAbvGrd`.

Classic stacking объединяет ElasticNet, GradientBoostingRegressor и LightGBM; мета-модель — ElasticNet. В проекте также доступны LinearRegression baseline, Ridge, Lasso, KNN, RandomForest и XGBoost.

MLP использует скрытые слои `[64, 32, 128]`, SiLU, dropout, Adam, CosineAnnealingLR и MSELoss. Предобработка обучается на тренировочной части каждого фолда; препроцессор MLP сохраняется вместе с весами соответствующего фолда.

Внешняя валидация — KFold, пять фолдов, `shuffle=True`, seed `1027309`. Итоговый ансамбль усредняет предсказания Classic stacking и MLP с весами 50/50 в пространстве `log1p(SalePrice)`, затем применяет `expm1`. Пайплайн сохраняет этот ансамбль и не выбирает модель автоматически по лучшему CV.

## Структура проекта

| Файл / каталог | Назначение |
|---|---|
| `main.py`, `config.py` | Общий запуск и выбор экспериментов |
| `requirements.txt` | Зависимости |
| `EDA.ipynb` | Распределения, пропуски, корреляции, выбросы, Mutual Information и аналитические выводы |
| `classic_ml_pipeline_v8/` | Данные, признаки, регрессоры, обучение, CV и предсказания |
| `dl_pipeline_v1/` | MLP, предобработка фолдов, обучение, CV и предсказания |
| `ensemble.py`, `ensemble_config.py` | Объединение сохранённых предсказаний |
| `checkpoints/ensembles/` | Артефакты итоговых ансамблей |
| `tests/test_ensemble.py` | Проверки ID, фолдов и усреднения |
| `outputs/` | Текущий сабмит и таблица результатов |

Проверки ансамбля:

```bash
python -m unittest discover -s tests
```
