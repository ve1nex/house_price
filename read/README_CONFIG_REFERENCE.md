# Справочник конфигурации

Таблицы сформированы по конечным `config.py` обоих проектов и `ensemble_config.py`. «Сейчас» — фактическое значение в поставке; `${...}` разрешается OmegaConf во время запуска. Корневой master `optimization.enabled=False` отключает все вложенные optimization flags. `strategies`, `regularization`, `tuning` и `postprocessing` независимы и имеют собственные флаги. Для заполнения `models.*` не нужно копировать весь API библиотеки: перечисленные ниже параметры наиболее существенно меняют ML-процесс. Остальные defaults зависят от установленной версии и не передаются явно; версия сохраняется в `environment.json`.

## Classic ML — `classic_ml_pipeline_v8/config.py`

### `general`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `experiment_name` | строка / допустимые варианты блока | `baseline_v1` | Имя изолированного run и каталога артефактов. |
| `seed` | число; ограничения задаёт библиотека/алгоритм | `1027309` | Seed Python/NumPy/sklearn/PyTorch/Optuna. |
| `task` | classification / regression | `classification` | Тип целевой переменной; влияет на модель, loss, метрику и форму вероятностей. |
| `num_classes` | `null` либо значение соответствующего типа | `null` | Число классов; Classic вычисляет по y, DL должен совпадать с метками 0..K−1. |
| `mode` | train / inference | `train` | Выбирает обучение либо запуск только inference. |
| `overwrite_experiment` | `True` / `False` | `False` | При True удаляет существующий каталог этого run. |

### `paths`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `project_root` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8` | Каталог проекта, вычисляется из __file__. |
| `path_to_data` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8/data` | Каталог входных CSV. |
| `path_to_train_dataset` | путь к файлу/каталогу | `${paths.path_to_data}/train.csv` | Classic train CSV. |
| `path_to_test_dataset` | путь к файлу/каталогу | `${paths.path_to_data}/test.csv` | Classic test CSV. |
| `path_to_checkpoints_root` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8/checkpoints` | Корень каталогов экспериментов. |
| `path_to_checkpoints` | путь к файлу/каталогу | `${paths.path_to_checkpoints_root}/${general.experiment_name}` | Каталог данного run. |
| `path_to_fold_models` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/fold_models` | Сохранённые sklearn Pipeline каждого fold. |
| `path_to_final_model` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/model.joblib` | Classic joblib final model. |
| `path_to_oof` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/oof_predictions.csv` | Совместимый OOF CSV с id/target/prediction/fold. |
| `path_to_predictions` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/predictions.csv` | Совместимый test probability CSV. |
| `path_to_plots` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/plots` | Каталог компактных графиков. |
| `path_to_config_snapshot` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/config.yaml` | Путь к разрешённому config.yaml. |
| `path_to_metadata` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/metadata.json` | Run metadata с классами, ID namespace, score. |
| `path_to_environment` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/environment.json` | Версии интерпретатора, библиотек, среды. |
| `path_to_logs` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8/logs` | Общий каталог локальной истории. |
| `path_to_wandb` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8/wandb` | Локальные служебные файлы W&B. |
| `path_to_optuna_root` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8/optuna` | Каталог SQLite Optuna. |
| `path_to_optuna_db` | путь к файлу/каталогу | `${paths.path_to_optuna_root}/${general.experiment_name}.db` | SQLite Optuna study. |
| `path_to_tuning` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/tuning` | Каталог CSV и YAML результатов Optuna. |
| `path_to_tuning_trials` | путь к файлу/каталогу | `${paths.path_to_tuning}/trials.csv` | CSV всех Optuna trials. |
| `path_to_best_params` | путь к файлу/каталогу | `${paths.path_to_tuning}/best_params.yaml` | YAML найденных гиперпараметров. |
| `path_to_results_csv` | путь к файлу/каталогу | `${paths.path_to_logs}/results.csv` | CSV истории runs. |
| `path_to_results_txt` | путь к файлу/каталогу | `${paths.path_to_logs}/results.txt` | Текст истории runs. |
| `telegram_credentials` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8/telegram_credits.json` | Локальный JSON с токеном; не публикуйте. |

### `reproducibility`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `save_config_snapshot` | `True` / `False` | `True` | Сохранять разрешённый config.yaml. |
| `save_environment` | `True` / `False` | `True` | Сохранять Python/OS/package versions. |
| `save_data_info` | `True` / `False` | `True` | Сохранять схему и размеры данных. |
| `calculate_dataset_hash` | `True` / `False` | `False` | Считать SHA-256 исходных данных; на больших файлах медленно. |

### `data`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `target` | строка / допустимые варианты блока | `target` | Имя целевого столбца в Classic CSV. |
| `id_column` | `null` либо значение соответствующего типа | `null` | Имя идентификатора, исключаемого из признаков. |
| `id_namespace` | строка / допустимые варианты блока | `row_position` | Семантика ID; должна совпасть у членов ансамбля. Для надёжности используйте явные ID. |
| `drop_columns` | список совместимых значений | `[]` | Столбцы, удаляемые до feature engineering. |
| `reduce_memory` | `True` / `False` | `False` | Опционально снижает разрядность чисел в pandas; может менять точность. |

### `data_checks`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `True` | Включает этот блок независимо от других категорий, если не указано иное. |
| `known_leakage_columns` | список совместимых значений | `[]` | Имена заведомо протекающих признаков: их присутствие вызывает ошибку. |
| `on_suspicious_feature` | warn / raise | `warn` | Действие при признаке, совпадающем с y. |

### `preprocessing`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `numeric_imputer` | mean / median / most_frequent / constant | `median` | Стратегия SimpleImputer для числовых признаков. |
| `categorical_imputer` | most_frequent / constant | `most_frequent` | Стратегия SimpleImputer для категорий. |
| `scale_numeric` | `True` / `False` | `True` | StandardScaler числовых признаков внутри fold. |
| `encode_categorical` | `True` / `False` | `True` | OneHotEncoder категорий внутри fold. |

### `feature_engineering`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `status` | строка / допустимые варианты блока | `placeholder; implement in features.py before enabling` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |

### `split`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `strategy` | KFold / StratifiedKFold / GroupKFold / StratifiedGroupKFold | `StratifiedKFold` | Выбор способа разбиения/вычисления. Значения указаны в строке. |
| `n_splits` | число; ограничения задаёт библиотека/алгоритм | `5` | Количество CV folds; для ансамбля требуются все folds. |
| `folds_to_train` | список совместимых значений | `[0, 1, 2, 3, 4]` | Номера folds для обучения. Частичный список делает OOF неполным. |
| `shuffle` | `True` / `False` | `True` | Перемешивание перед CV или DataLoader; seed фиксирует порядок. |
| `group_column` | `null` либо значение соответствующего типа | `null` | Столбец группы, исключаемый из X и передаваемый GroupKFold. |

### `training`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `debug` | `True` / `False` | `False` | Включает сокращённый набор данных для быстрой проверки. |
| `debug_n_rows` | число; ограничения задаёт библиотека/алгоритм | `1000` | Количество строк Classic debug sample. |
| `save_fold_models` | `True` / `False` | `False` | Сохранять sklearn Pipeline каждого fold. |
| `save_oof_predictions` | `True` / `False` | `True` | Сохранять совместимый OOF CSV. |
| `train_final_model` | `True` / `False` | `True` | Обучать Classic final model на всех train rows. |
| `predict_test_after_fit` | `True` / `False` | `True` | После final fit сформировать test CSV, если входной файл существует. |

### `model`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | имя из блока models либо MLP / TransformerMLP | `LogisticRegression` | Название выбранной модели/метрики/оптимизатора/плана. |

### `models`

#### `LogisticRegression`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `LogisticRegression.C` | положительное число / ∞ | `∞` | Обратная сила L2/L1 регуляризации: `∞` = нет штрафа, меньше C = сильнее штраф. |
| `LogisticRegression.l1_ratio` | 0 (L2) / 1 (L1) / (0,1) ElasticNet | `0.0` | 0=L2, 1=L1, между 0 и 1=ElasticNet; для L1/ElasticNet нужен saga (для L1 допустим liblinear). |
| `LogisticRegression.solver` | lbfgs / liblinear / newton-cg / newton-cholesky / sag / saga | `lbfgs` | Численный алгоритм для LogisticRegression/Ridge. |
| `LogisticRegression.class_weight` | `null` либо значение соответствующего типа | `null` | Веса классов; balanced повышает вес редких классов. |
| `LogisticRegression.fit_intercept` | `True` / `False` | `True` | Оценивать свободный коэффициент. |
| `LogisticRegression.max_iter` | число; ограничения задаёт библиотека/алгоритм | `1000` | Предел итераций оптимизатора LogisticRegression. |
| `LogisticRegression.tol` | число; ограничения задаёт библиотека/алгоритм | `0.0001` | Порог сходимости solver. |
| `LogisticRegression.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |

#### `KNeighborsClassifier`

Выберите `model.name="KNeighborsClassifier"` для классификации. При `preprocessing.scale_numeric=True` числовые признаки стандартизируются внутри каждого CV fold. Классификатор поддерживает `predict_proba()`.

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `KNeighborsClassifier.n_neighbors` | целое > 0 | `15` | Число ближайших соседей; должно быть не больше числа обучающих строк текущего fold. |
| `KNeighborsClassifier.weights` | `uniform` / `distance` | `uniform` | Равные веса соседей или веса обратно пропорционально расстоянию. |
| `KNeighborsClassifier.algorithm` | `auto` / `ball_tree` / `kd_tree` / `brute` | `auto` | Алгоритм поиска соседей; разреженные признаки могут использовать brute force. |
| `KNeighborsClassifier.leaf_size` | целое > 0 | `30` | Размер листа для tree алгоритмов поиска. |
| `KNeighborsClassifier.p` | 1 / 2 и другие положительные числа | `2` | Степень расстояния Минковского: 1 — Manhattan, 2 — Euclidean. |
| `KNeighborsClassifier.metric` | `minkowski` или метрика sklearn neighbors | `minkowski` | Функция расстояния между объектами. |
| `KNeighborsClassifier.n_jobs` | целое / `None` | `1` | Число CPU workers для поиска соседей; флаг `optimization.speed.parallel_cpu` может переопределить. |

#### `RandomForestClassifier`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `RandomForestClassifier.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `300` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `RandomForestClassifier.max_depth` | `null` либо значение соответствующего типа | `null` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `RandomForestClassifier.max_features` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Доля/число признаков для split/bagging. |
| `RandomForestClassifier.min_samples_leaf` | число; ограничения задаёт библиотека/алгоритм | `1` | Минимум объектов в листе. |
| `RandomForestClassifier.min_samples_split` | число; ограничения задаёт библиотека/алгоритм | `2` | Минимум объектов для разделения узла. |
| `RandomForestClassifier.class_weight` | `null` либо значение соответствующего типа | `null` | Веса классов; balanced повышает вес редких классов. |
| `RandomForestClassifier.bootstrap` | `True` / `False` | `True` | Выборки с возвращением для RandomForest. |
| `RandomForestClassifier.max_samples` | `null` либо значение соответствующего типа | `null` | Доля/число объектов для bagging. |
| `RandomForestClassifier.criterion` | gini / entropy / log_loss | `gini` | Критерий качества разбиения дерева. |
| `RandomForestClassifier.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |
| `RandomForestClassifier.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |

#### `GradientBoostingClassifier`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `GradientBoostingClassifier.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `100` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `GradientBoostingClassifier.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.1` | Шаг обновления параметров/деревьев. |
| `GradientBoostingClassifier.max_depth` | число; ограничения задаёт библиотека/алгоритм | `3` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `GradientBoostingClassifier.subsample` | число в [0, 1] | `1.0` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `GradientBoostingClassifier.max_features` | `null` либо значение соответствующего типа | `null` | Доля/число признаков для split/bagging. |
| `GradientBoostingClassifier.loss` | log_loss / exponential | `log_loss` | Функция потерь sklearn GradientBoosting. |
| `GradientBoostingClassifier.n_iter_no_change` | `null` либо значение соответствующего типа | `null` | Внутренний early stopping sklearn GradientBoosting; None выключает. |
| `GradientBoostingClassifier.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |

#### `LinearRegression`

Внешнее имя модели одно: `model.name="LinearRegression"`. `regularization` выбирает класс sklearn внутри `models.py`; указанные для других режимов поля не передаются конструктору. Базовый режим `none` сохраняет OLS без штрафа.

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `LinearRegression.regularization` | `none` / `ridge` / `lasso` / `elasticnet` | `none` | OLS без штрафа / L2 / L1 / смешанный L1+L2. |
| `LinearRegression.alpha` | число > 0 | `1.0` | Сила штрафа для Ridge, Lasso и ElasticNet; при `none` игнорируется. |
| `LinearRegression.l1_ratio` | число от 0 до 1 | `0.5` | Доля L1 в ElasticNet; в остальных режимах игнорируется. |
| `LinearRegression.fit_intercept` | `True` / `False` | `True` | Оценивать свободный коэффициент. |
| `LinearRegression.positive` | `True` / `False` | `False` | Требовать неотрицательные коэффициенты; у OLS для `True` нужен плотный массив признаков. |
| `LinearRegression.max_iter` | целое > 0 | `5000` | Предел итераций для регуляризованных моделей; при `none` игнорируется, у Ridge влияет лишь на итерационные solver. |
| `LinearRegression.tol` | число > 0 | `0.0001` | Порог сходимости регуляризованных моделей; у прямых Ridge solver может не влиять. |
| `LinearRegression.random_state` | целое число / `None` | `${general.seed}` | Передаётся Ridge/Lasso/ElasticNet; влияет на Ridge с `sag`/`saga` и Lasso/ElasticNet при `selection=random`. |
| `LinearRegression.solver` | auto / svd / cholesky / lsqr / sparse_cg / sag / saga / lbfgs | `auto` | Только Ridge; при `positive=True` используйте `auto` или `lbfgs`. |
| `LinearRegression.selection` | cyclic / random | `cyclic` | Порядок обновления коэффициентов Lasso/ElasticNet; `random_state` влияет при `random`. |
| `LinearRegression.n_jobs` | целое число | `1` | Только OLS; может переопределяться `optimization.speed.parallel_cpu`. |

#### `KNeighborsRegressor`

Выберите `model.name="KNeighborsRegressor"` для регрессии. При `preprocessing.scale_numeric=True` числовые признаки стандартизируются внутри каждого CV fold; предсказание выполняется через `predict()`.

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `KNeighborsRegressor.n_neighbors` | целое > 0 | `15` | Число ближайших соседей; должно быть не больше числа обучающих строк текущего fold. |
| `KNeighborsRegressor.weights` | `uniform` / `distance` | `uniform` | Равные веса соседей или веса обратно пропорционально расстоянию. |
| `KNeighborsRegressor.algorithm` | `auto` / `ball_tree` / `kd_tree` / `brute` | `auto` | Алгоритм поиска соседей; разреженные признаки могут использовать brute force. |
| `KNeighborsRegressor.leaf_size` | целое > 0 | `30` | Размер листа для tree алгоритмов поиска. |
| `KNeighborsRegressor.p` | 1 / 2 и другие положительные числа | `2` | Степень расстояния Минковского: 1 — Manhattan, 2 — Euclidean. |
| `KNeighborsRegressor.metric` | `minkowski` или метрика sklearn neighbors | `minkowski` | Функция расстояния между объектами. |
| `KNeighborsRegressor.n_jobs` | целое / `None` | `1` | Число CPU workers для поиска соседей; флаг `optimization.speed.parallel_cpu` может переопределить. |

#### `RandomForestRegressor`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `RandomForestRegressor.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `300` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `RandomForestRegressor.max_depth` | `null` либо значение соответствующего типа | `null` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `RandomForestRegressor.max_features` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Доля/число признаков для split/bagging. |
| `RandomForestRegressor.min_samples_leaf` | число; ограничения задаёт библиотека/алгоритм | `1` | Минимум объектов в листе. |
| `RandomForestRegressor.min_samples_split` | число; ограничения задаёт библиотека/алгоритм | `2` | Минимум объектов для разделения узла. |
| `RandomForestRegressor.bootstrap` | `True` / `False` | `True` | Выборки с возвращением для RandomForest. |
| `RandomForestRegressor.max_samples` | `null` либо значение соответствующего типа | `null` | Доля/число объектов для bagging. |
| `RandomForestRegressor.criterion` | squared_error / absolute_error / friedman_mse / poisson | `squared_error` | Критерий качества разбиения дерева. |
| `RandomForestRegressor.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |
| `RandomForestRegressor.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |

#### `GradientBoostingRegressor`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `GradientBoostingRegressor.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `100` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `GradientBoostingRegressor.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.1` | Шаг обновления параметров/деревьев. |
| `GradientBoostingRegressor.max_depth` | число; ограничения задаёт библиотека/алгоритм | `3` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `GradientBoostingRegressor.subsample` | число в [0, 1] | `1.0` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `GradientBoostingRegressor.max_features` | `null` либо значение соответствующего типа | `null` | Доля/число признаков для split/bagging. |
| `GradientBoostingRegressor.loss` | squared_error / absolute_error / huber / quantile | `squared_error` | Функция потерь sklearn GradientBoosting. |
| `GradientBoostingRegressor.n_iter_no_change` | `null` либо значение соответствующего типа | `null` | Внутренний early stopping sklearn GradientBoosting; None выключает. |
| `GradientBoostingRegressor.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |

#### `DecisionTreeClassifier`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `DecisionTreeClassifier.max_depth` | `null` либо значение соответствующего типа | `null` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `DecisionTreeClassifier.min_samples_split` | число; ограничения задаёт библиотека/алгоритм | `2` | Минимум объектов для разделения узла. |
| `DecisionTreeClassifier.min_samples_leaf` | число; ограничения задаёт библиотека/алгоритм | `1` | Минимум объектов в листе. |
| `DecisionTreeClassifier.criterion` | gini / entropy / log_loss | `gini` | Критерий качества разбиения дерева. |
| `DecisionTreeClassifier.ccp_alpha` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Сила cost-complexity pruning дерева (0 выключает). |
| `DecisionTreeClassifier.class_weight` | `null` либо значение соответствующего типа | `null` | Веса классов; balanced повышает вес редких классов. |
| `DecisionTreeClassifier.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |

#### `DecisionTreeRegressor`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `DecisionTreeRegressor.max_depth` | `null` либо значение соответствующего типа | `null` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `DecisionTreeRegressor.min_samples_split` | число; ограничения задаёт библиотека/алгоритм | `2` | Минимум объектов для разделения узла. |
| `DecisionTreeRegressor.min_samples_leaf` | число; ограничения задаёт библиотека/алгоритм | `1` | Минимум объектов в листе. |
| `DecisionTreeRegressor.criterion` | squared_error / friedman_mse / absolute_error / poisson | `squared_error` | Критерий качества разбиения дерева. |
| `DecisionTreeRegressor.ccp_alpha` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Сила cost-complexity pruning дерева (0 выключает). |
| `DecisionTreeRegressor.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |

#### `XGBClassifier`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `XGBClassifier.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `300` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `XGBClassifier.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.05` | Шаг обновления параметров/деревьев. |
| `XGBClassifier.max_depth` | число; ограничения задаёт библиотека/алгоритм | `6` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `XGBClassifier.subsample` | число в [0, 1] | `1.0` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `XGBClassifier.colsample_bytree` | число в [0, 1] | `1.0` | Доля признаков на дерево; 1 выключает column sampling. |
| `XGBClassifier.reg_alpha` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `XGBClassifier.reg_lambda` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `XGBClassifier.gamma` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Минимальное улучшение для split XGBoost. |
| `XGBClassifier.min_child_weight` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Минимальная сумма весов в дочернем узле XGBoost. |
| `XGBClassifier.scale_pos_weight` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Вес позитивного класса только в binary XGBoost; >1 меняет баланс. |
| `XGBClassifier.tree_method` | exact / hist | `exact` | XGBoost exact или hist; hist включается optimization flag. |
| `XGBClassifier.objective` | строка / допустимые варианты блока | `binary:logistic` | Loss/objective boosting; binary/multiclass определяется task и числом классов. |
| `XGBClassifier.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |
| `XGBClassifier.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |

#### `LGBMClassifier`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `LGBMClassifier.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `300` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `LGBMClassifier.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.05` | Шаг обновления параметров/деревьев. |
| `LGBMClassifier.num_leaves` | число; ограничения задаёт библиотека/алгоритм | `31` | Ограничение числа листьев LightGBM. |
| `LGBMClassifier.max_depth` | число; ограничения задаёт библиотека/алгоритм | `-1` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `LGBMClassifier.subsample` | число в [0, 1] | `1.0` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `LGBMClassifier.subsample_freq` | число; ограничения задаёт библиотека/алгоритм | `0` | Частота row sampling в LightGBM; 0 выключает. |
| `LGBMClassifier.colsample_bytree` | число в [0, 1] | `1.0` | Доля признаков на дерево; 1 выключает column sampling. |
| `LGBMClassifier.reg_alpha` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMClassifier.reg_lambda` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMClassifier.class_weight` | `null` либо значение соответствующего типа | `null` | Веса классов; balanced повышает вес редких классов. |
| `LGBMClassifier.min_child_samples` | число; ограничения задаёт библиотека/алгоритм | `20` | Минимум наблюдений в листе LightGBM. |
| `LGBMClassifier.min_split_gain` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Минимальный gain для split LightGBM. |
| `LGBMClassifier.boosting_type` | gbdt / dart / rf | `gbdt` | Тип boosting LightGBM. |
| `LGBMClassifier.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |
| `LGBMClassifier.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |
| `LGBMClassifier.verbosity` | число; ограничения задаёт библиотека/алгоритм | `-1` | Параметр выбранного estimator; проверьте допустимый диапазон в официальной документации. |

#### `CatBoostClassifier`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `CatBoostClassifier.iterations` | число; ограничения задаёт библиотека/алгоритм | `300` | Число boosting iterations CatBoost. |
| `CatBoostClassifier.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.05` | Шаг обновления параметров/деревьев. |
| `CatBoostClassifier.depth` | число; ограничения задаёт библиотека/алгоритм | `6` | Глубина деревьев CatBoost. |
| `CatBoostClassifier.l2_leaf_reg` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L2 на листья CatBoost; 0 выключает. |
| `CatBoostClassifier.random_strength` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Шум score при выборе split в CatBoost; 0 выключает. |
| `CatBoostClassifier.bootstrap_type` | No / Bayesian / Bernoulli / MVS / Poisson (совместимость зависит от task/device) | `No` | Метод sampling CatBoost; No отключает bootstrap. |
| `CatBoostClassifier.use_best_model` | `True` / `False` | `False` | Использовать лучшую итерацию CatBoost при validation; baseline выключен. |
| `CatBoostClassifier.allow_writing_files` | `True` / `False` | `False` | Разрешение CatBoost писать служебные файлы в рабочую папку. |
| `CatBoostClassifier.loss_function` | строка / допустимые варианты блока | `Logloss` | Функция потерь CatBoost; MultiClass выбирается при >2 классах. |
| `CatBoostClassifier.thread_count` | число; ограничения задаёт библиотека/алгоритм | `1` | Количество CPU threads CatBoost. |
| `CatBoostClassifier.random_seed` | строка / допустимые варианты блока | `${general.seed}` | Seed CatBoost. |
| `CatBoostClassifier.verbose` | `True` / `False` | `False` | Вывод деталей обучения библиотеки. |

#### `XGBRegressor`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `XGBRegressor.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `300` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `XGBRegressor.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.05` | Шаг обновления параметров/деревьев. |
| `XGBRegressor.max_depth` | число; ограничения задаёт библиотека/алгоритм | `6` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `XGBRegressor.subsample` | число в [0, 1] | `1.0` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `XGBRegressor.colsample_bytree` | число в [0, 1] | `1.0` | Доля признаков на дерево; 1 выключает column sampling. |
| `XGBRegressor.reg_alpha` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `XGBRegressor.reg_lambda` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `XGBRegressor.gamma` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Минимальное улучшение для split XGBoost. |
| `XGBRegressor.min_child_weight` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Минимальная сумма весов в дочернем узле XGBoost. |
| `XGBRegressor.tree_method` | exact / hist | `exact` | XGBoost exact или hist; hist включается optimization flag. |
| `XGBRegressor.objective` | строка / допустимые варианты блока | `reg:squarederror` | Loss/objective boosting; binary/multiclass определяется task и числом классов. |
| `XGBRegressor.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |
| `XGBRegressor.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |

#### `LGBMRegressor`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `LGBMRegressor.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `300` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `LGBMRegressor.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.05` | Шаг обновления параметров/деревьев. |
| `LGBMRegressor.num_leaves` | число; ограничения задаёт библиотека/алгоритм | `31` | Ограничение числа листьев LightGBM. |
| `LGBMRegressor.max_depth` | число; ограничения задаёт библиотека/алгоритм | `-1` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `LGBMRegressor.subsample` | число в [0, 1] | `1.0` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `LGBMRegressor.subsample_freq` | число; ограничения задаёт библиотека/алгоритм | `0` | Частота row sampling в LightGBM; 0 выключает. |
| `LGBMRegressor.colsample_bytree` | число в [0, 1] | `1.0` | Доля признаков на дерево; 1 выключает column sampling. |
| `LGBMRegressor.reg_alpha` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMRegressor.reg_lambda` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMRegressor.min_child_samples` | число; ограничения задаёт библиотека/алгоритм | `20` | Минимум наблюдений в листе LightGBM. |
| `LGBMRegressor.min_split_gain` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Минимальный gain для split LightGBM. |
| `LGBMRegressor.boosting_type` | gbdt / dart / rf | `gbdt` | Тип boosting LightGBM. |
| `LGBMRegressor.random_state` | строка / допустимые варианты блока | `${general.seed}` | Seed данного sklearn/XGBoost/LightGBM estimator. |
| `LGBMRegressor.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |
| `LGBMRegressor.verbosity` | число; ограничения задаёт библиотека/алгоритм | `-1` | Параметр выбранного estimator; проверьте допустимый диапазон в официальной документации. |

#### `CatBoostRegressor`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `CatBoostRegressor.iterations` | число; ограничения задаёт библиотека/алгоритм | `300` | Число boosting iterations CatBoost. |
| `CatBoostRegressor.learning_rate` | число; ограничения задаёт библиотека/алгоритм | `0.05` | Шаг обновления параметров/деревьев. |
| `CatBoostRegressor.depth` | число; ограничения задаёт библиотека/алгоритм | `6` | Глубина деревьев CatBoost. |
| `CatBoostRegressor.l2_leaf_reg` | число; ограничения задаёт библиотека/алгоритм | `0.0` | L2 на листья CatBoost; 0 выключает. |
| `CatBoostRegressor.random_strength` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Шум score при выборе split в CatBoost; 0 выключает. |
| `CatBoostRegressor.bootstrap_type` | No / Bayesian / Bernoulli / MVS / Poisson (совместимость зависит от task/device) | `No` | Метод sampling CatBoost; No отключает bootstrap. |
| `CatBoostRegressor.use_best_model` | `True` / `False` | `False` | Использовать лучшую итерацию CatBoost при validation; baseline выключен. |
| `CatBoostRegressor.allow_writing_files` | `True` / `False` | `False` | Разрешение CatBoost писать служебные файлы в рабочую папку. |
| `CatBoostRegressor.loss_function` | строка / допустимые варианты блока | `RMSE` | Функция потерь CatBoost; MultiClass выбирается при >2 классах. |
| `CatBoostRegressor.thread_count` | число; ограничения задаёт библиотека/алгоритм | `1` | Количество CPU threads CatBoost. |
| `CatBoostRegressor.random_seed` | строка / допустимые варианты блока | `${general.seed}` | Seed CatBoost. |
| `CatBoostRegressor.verbose` | `True` / `False` | `False` | Вывод деталей обучения библиотеки. |

### `optimization`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `speed.parallel_cpu.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `speed.parallel_cpu.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `-1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |
| `speed.histogram_boosting.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `training_control.boosting_early_stopping.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `training_control.boosting_early_stopping.rounds` | число; ограничения задаёт библиотека/алгоритм | `30` | Число boosting iterations без улучшения на fold validation. |
| `prediction.threshold_tuning.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `prediction.threshold_tuning.n_steps` | число; ограничения задаёт библиотека/алгоритм | `101` | Число кандидатов порога между 0 и 1. |
| `prediction.calibration.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `prediction.calibration.status` | строка / допустимые варианты блока | `placeholder` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |
| `features.feature_selection.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `features.feature_selection.status` | строка / допустимые варианты блока | `placeholder` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |

### `estimator_strategy`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `type` | bagging / voting / average / weighted_average / stacking | `bagging` | Тип метода/ансамбля или Optuna distribution. |
| `models.classification` | список совместимых значений | `['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']` | Базовые модели внутреннего Classic-ансамбля для classification. |
| `models.regression` | список совместимых значений | `['LinearRegression', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']` | Базовые модели внутреннего Classic-ансамбля; вариант линейной задаёт общий блок. |
| `weights.classification` | список совместимых значений | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | По одному весу на каждую из восьми classification моделей. |
| `weights.regression` | список совместимых значений | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | По одному весу на каждую из восьми регрессионных моделей. |
| `voting` | hard / soft | `hard` | hard/soft метод классификационного voting. |
| `bagging.n_estimators` | число; ограничения задаёт библиотека/алгоритм | `10` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `bagging.max_samples` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Доля/число объектов для bagging. |
| `bagging.max_features` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Доля/число признаков для split/bagging. |
| `bagging.bootstrap` | `True` / `False` | `True` | Выборки с возвращением для RandomForest. |
| `bagging.bootstrap_features` | `True` / `False` | `False` | Bootstrap по признакам внутри bagging. |
| `bagging.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `-1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |
| `stacking.final_model.classification` | строка / допустимые варианты блока | `LogisticRegression` | Meta estimator/модели для classification. |
| `stacking.final_model.regression` | имя модели из Classic-реестра | `LinearRegression` | Meta-estimator внутреннего stacking; использует текущий `models.LinearRegression.regularization`. |
| `stacking.cv` | число; ограничения задаёт библиотека/алгоритм | `5` | Число внутренних folds sklearn stacking. |
| `stacking.passthrough` | `True` / `False` | `False` | Передавать исходные признаки вместе с OOF meta features. |
| `stacking.n_jobs` | число; ограничения задаёт библиотека/алгоритм | `-1` | Число CPU workers; 1 baseline, −1 доступные CPU при включённом флаге. |

### `metric`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | имя функции sklearn.metrics; accuracy_score, f1_score, roc_auc_score, log_loss, root_mean_squared_error, r2_score и др. | `accuracy_score` | Название выбранной модели/метрики/оптимизатора/плана. |
| `direction` | maximize / minimize | `maximize` | Оптимизировать максимум либо минимум metric. |
| `params` | словарь аргументов (ключи зависят от выбранной функции) | `{}` | Дополнительные keyword arguments выбранной функции; пустой словарь не добавляет аргументов. |

### `tuning`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает Optuna или pruning; от optimization.enabled не зависит. |
| `n_trials` | число; ограничения задаёт библиотека/алгоритм | `30` | Количество Optuna trials. |
| `direction` | auto / maximize / minimize | `auto` | Оптимизировать максимум либо минимум metric. |
| `sampler` | tpe / random | `tpe` | Алгоритм Optuna TPE или Random. |
| `study_name` | строка / допустимые варианты блока | `${general.experiment_name}_tuning` | Имя Optuna study в SQLite. |
| `resume_study` | `True` / `False` | `False` | Продолжать старое Optuna study; используйте только с теми же данными и config. |
| `folds_to_use` | строка / допустимые варианты блока | `${split.folds_to_train}` | Номера folds, по которым Optuna усредняет score. |
| `log_trials_to_wandb` | `True` / `False` | `True` | Отправлять summary каждого Classic trial в W&B. |

#### `tuning.search_spaces`

Для каждой модели и каждого параметра указан Optuna distribution. `type=int/float/categorical`; `low/high` — границы; `step` — шаг; `log=True` — логарифмическая шкала; `choices` — варианты. Это схема каждого записанного ниже словаря. `tuning.enabled=False` оставляет все эти пространства неактивными.

Для `LinearRegression` `tuning.py` оставляет `alpha` только в режимах `ridge`/`lasso`/`elasticnet`, а `l1_ratio` — только в `elasticnet`. При `none` без иных параметров выполняется один trial для оценки baseline с пустым `best_params` (обычно tuning лучше выключить). `regularization` сама не подбирается: выберите режим перед запуском и новое имя эксперимента.

| Модель / параметр | Возможные значения trial | Сейчас | Что означает |
| --- | --- | --- | --- |
| `LogisticRegression.C` | `float: 0.0001…1000.0 (log=True)` | `active only in tuning` | Обратная сила L2/L1 регуляризации: `∞` = нет штрафа, меньше C = сильнее штраф. |
| `KNeighborsClassifier.n_neighbors` | `int: 3…35 (step=2)` | `active only in tuning` | Нечётное число соседей. |
| `KNeighborsClassifier.weights` | `categorical: ['uniform', 'distance']` | `active only in tuning` | Равные веса или веса по расстоянию. |
| `KNeighborsClassifier.p` | `categorical: [1, 2]` | `active only in tuning` | Manhattan или Euclidean при `metric='minkowski'`. |
| `DecisionTreeClassifier.max_depth` | `int: 2…30` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `DecisionTreeClassifier.min_samples_split` | `int: 2…30` | `active only in tuning` | Минимум объектов для разделения узла. |
| `DecisionTreeClassifier.min_samples_leaf` | `int: 1…20` | `active only in tuning` | Минимум объектов в листе. |
| `DecisionTreeClassifier.max_features` | `categorical: [None, 'sqrt', 'log2']` | `active only in tuning` | Доля/число признаков для split/bagging. |
| `DecisionTreeClassifier.ccp_alpha` | `float: 0.0…0.05` | `active only in tuning` | Сила cost-complexity pruning дерева (0 выключает). |
| `RandomForestClassifier.n_estimators` | `int: 100…1000 (step=50)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `RandomForestClassifier.max_depth` | `int: 3…30` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `RandomForestClassifier.min_samples_split` | `int: 2…20` | `active only in tuning` | Минимум объектов для разделения узла. |
| `RandomForestClassifier.min_samples_leaf` | `int: 1…10` | `active only in tuning` | Минимум объектов в листе. |
| `RandomForestClassifier.max_features` | `categorical: ['sqrt', 'log2', None]` | `active only in tuning` | Доля/число признаков для split/bagging. |
| `GradientBoostingClassifier.n_estimators` | `int: 50…500 (step=25)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `GradientBoostingClassifier.learning_rate` | `float: 0.01…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `GradientBoostingClassifier.max_depth` | `int: 2…8` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `GradientBoostingClassifier.min_samples_split` | `int: 2…20` | `active only in tuning` | Минимум объектов для разделения узла. |
| `GradientBoostingClassifier.min_samples_leaf` | `int: 1…10` | `active only in tuning` | Минимум объектов в листе. |
| `GradientBoostingClassifier.subsample` | `float: 0.5…1.0` | `active only in tuning` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `GradientBoostingClassifier.max_features` | `categorical: [None, 'sqrt', 'log2']` | `active only in tuning` | Доля/число признаков для split/bagging. |
| `XGBClassifier.n_estimators` | `int: 100…1200 (step=50)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `XGBClassifier.learning_rate` | `float: 0.005…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `XGBClassifier.max_depth` | `int: 2…12` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `XGBClassifier.min_child_weight` | `float: 0.5…20.0 (log=True)` | `active only in tuning` | Минимальная сумма весов в дочернем узле XGBoost. |
| `XGBClassifier.gamma` | `float: 0.0…10.0` | `active only in tuning` | Минимальное улучшение для split XGBoost. |
| `XGBClassifier.subsample` | `float: 0.5…1.0` | `active only in tuning` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `XGBClassifier.colsample_bytree` | `float: 0.5…1.0` | `active only in tuning` | Доля признаков на дерево; 1 выключает column sampling. |
| `XGBClassifier.reg_alpha` | `float: 1e-08…10.0 (log=True)` | `active only in tuning` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `XGBClassifier.reg_lambda` | `float: 0.001…100.0 (log=True)` | `active only in tuning` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMClassifier.n_estimators` | `int: 100…1200 (step=50)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `LGBMClassifier.learning_rate` | `float: 0.005…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `LGBMClassifier.num_leaves` | `int: 8…128` | `active only in tuning` | Ограничение числа листьев LightGBM. |
| `LGBMClassifier.max_depth` | `int: 3…15` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `LGBMClassifier.min_child_samples` | `int: 5…100` | `active only in tuning` | Минимум наблюдений в листе LightGBM. |
| `LGBMClassifier.subsample` | `float: 0.5…1.0` | `active only in tuning` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `LGBMClassifier.colsample_bytree` | `float: 0.5…1.0` | `active only in tuning` | Доля признаков на дерево; 1 выключает column sampling. |
| `LGBMClassifier.reg_alpha` | `float: 1e-08…10.0 (log=True)` | `active only in tuning` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMClassifier.reg_lambda` | `float: 1e-08…10.0 (log=True)` | `active only in tuning` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `CatBoostClassifier.iterations` | `int: 100…1200 (step=50)` | `active only in tuning` | Число boosting iterations CatBoost. |
| `CatBoostClassifier.learning_rate` | `float: 0.005…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `CatBoostClassifier.depth` | `int: 4…10` | `active only in tuning` | Глубина деревьев CatBoost. |
| `CatBoostClassifier.l2_leaf_reg` | `float: 0.001…100.0 (log=True)` | `active only in tuning` | L2 на листья CatBoost; 0 выключает. |
| `CatBoostClassifier.random_strength` | `float: 0.0…10.0` | `active only in tuning` | Шум score при выборе split в CatBoost; 0 выключает. |
| `CatBoostClassifier.bagging_temperature` | `float: 0.0…10.0` | `active only in tuning` | Параметр bagging_temperature выбранной модели. |
| `LinearRegression.alpha` | `float: 0.0001…1000.0 (log=True)` | `ridge` / `lasso` / `elasticnet` | Сила соответствующего штрафа; `none` не подбирает. |
| `LinearRegression.l1_ratio` | `float: 0.0…1.0` | только `elasticnet` | Доля L1; `ridge` и `lasso` её не подбирают. |
| `KNeighborsRegressor.n_neighbors` | `int: 3…35 (step=2)` | `active only in tuning` | Нечётное число соседей. |
| `KNeighborsRegressor.weights` | `categorical: ['uniform', 'distance']` | `active only in tuning` | Равные веса или веса по расстоянию. |
| `KNeighborsRegressor.p` | `categorical: [1, 2]` | `active only in tuning` | Manhattan или Euclidean при `metric='minkowski'`. |
| `DecisionTreeRegressor.max_depth` | `int: 2…30` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `DecisionTreeRegressor.min_samples_split` | `int: 2…30` | `active only in tuning` | Минимум объектов для разделения узла. |
| `DecisionTreeRegressor.min_samples_leaf` | `int: 1…20` | `active only in tuning` | Минимум объектов в листе. |
| `DecisionTreeRegressor.max_features` | `categorical: [None, 'sqrt', 'log2']` | `active only in tuning` | Доля/число признаков для split/bagging. |
| `DecisionTreeRegressor.ccp_alpha` | `float: 0.0…0.05` | `active only in tuning` | Сила cost-complexity pruning дерева (0 выключает). |
| `RandomForestRegressor.n_estimators` | `int: 100…1000 (step=50)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `RandomForestRegressor.max_depth` | `int: 3…30` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `RandomForestRegressor.min_samples_split` | `int: 2…20` | `active only in tuning` | Минимум объектов для разделения узла. |
| `RandomForestRegressor.min_samples_leaf` | `int: 1…10` | `active only in tuning` | Минимум объектов в листе. |
| `RandomForestRegressor.max_features` | `categorical: ['sqrt', 'log2', None]` | `active only in tuning` | Доля/число признаков для split/bagging. |
| `GradientBoostingRegressor.n_estimators` | `int: 50…500 (step=25)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `GradientBoostingRegressor.learning_rate` | `float: 0.01…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `GradientBoostingRegressor.max_depth` | `int: 2…8` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `GradientBoostingRegressor.min_samples_split` | `int: 2…20` | `active only in tuning` | Минимум объектов для разделения узла. |
| `GradientBoostingRegressor.min_samples_leaf` | `int: 1…10` | `active only in tuning` | Минимум объектов в листе. |
| `GradientBoostingRegressor.subsample` | `float: 0.5…1.0` | `active only in tuning` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `GradientBoostingRegressor.max_features` | `categorical: [None, 'sqrt', 'log2']` | `active only in tuning` | Доля/число признаков для split/bagging. |
| `XGBRegressor.n_estimators` | `int: 100…1200 (step=50)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `XGBRegressor.learning_rate` | `float: 0.005…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `XGBRegressor.max_depth` | `int: 2…12` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `XGBRegressor.min_child_weight` | `float: 0.5…20.0 (log=True)` | `active only in tuning` | Минимальная сумма весов в дочернем узле XGBoost. |
| `XGBRegressor.gamma` | `float: 0.0…10.0` | `active only in tuning` | Минимальное улучшение для split XGBoost. |
| `XGBRegressor.subsample` | `float: 0.5…1.0` | `active only in tuning` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `XGBRegressor.colsample_bytree` | `float: 0.5…1.0` | `active only in tuning` | Доля признаков на дерево; 1 выключает column sampling. |
| `XGBRegressor.reg_alpha` | `float: 1e-08…10.0 (log=True)` | `active only in tuning` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `XGBRegressor.reg_lambda` | `float: 0.001…100.0 (log=True)` | `active only in tuning` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMRegressor.n_estimators` | `int: 100…1200 (step=50)` | `active only in tuning` | Число деревьев/итераций ансамбля (до optional early stopping). |
| `LGBMRegressor.learning_rate` | `float: 0.005…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `LGBMRegressor.num_leaves` | `int: 8…128` | `active only in tuning` | Ограничение числа листьев LightGBM. |
| `LGBMRegressor.max_depth` | `int: 3…15` | `active only in tuning` | Максимальная глубина дерева; None или −1 означает без данного ограничения. |
| `LGBMRegressor.min_child_samples` | `int: 5…100` | `active only in tuning` | Минимум наблюдений в листе LightGBM. |
| `LGBMRegressor.subsample` | `float: 0.5…1.0` | `active only in tuning` | Доля строк на дерево boosting; 1 выключает row sampling. |
| `LGBMRegressor.colsample_bytree` | `float: 0.5…1.0` | `active only in tuning` | Доля признаков на дерево; 1 выключает column sampling. |
| `LGBMRegressor.reg_alpha` | `float: 1e-08…10.0 (log=True)` | `active only in tuning` | L1 штраф XGBoost/LightGBM; 0 выключает. |
| `LGBMRegressor.reg_lambda` | `float: 1e-08…10.0 (log=True)` | `active only in tuning` | L2 штраф XGBoost/LightGBM; 0 выключает. |
| `CatBoostRegressor.iterations` | `int: 100…1200 (step=50)` | `active only in tuning` | Число boosting iterations CatBoost. |
| `CatBoostRegressor.learning_rate` | `float: 0.005…0.3 (log=True)` | `active only in tuning` | Шаг обновления параметров/деревьев. |
| `CatBoostRegressor.depth` | `int: 4…10` | `active only in tuning` | Глубина деревьев CatBoost. |
| `CatBoostRegressor.l2_leaf_reg` | `float: 0.001…100.0 (log=True)` | `active only in tuning` | L2 на листья CatBoost; 0 выключает. |
| `CatBoostRegressor.random_strength` | `float: 0.0…10.0` | `active only in tuning` | Шум score при выборе split в CatBoost; 0 выключает. |
| `CatBoostRegressor.bagging_temperature` | `float: 0.0…10.0` | `active only in tuning` | Параметр bagging_temperature выбранной модели. |

### `postprocessing`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `status` | строка / допустимые варианты блока | `placeholder; implement in postprocessing.py before enabling` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |

### `visualization`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `save_validation_plot` | `True` / `False` | `True` | Сохранять confusion matrix либо true-vs-predicted + residuals. |
| `save_cv_scores` | `True` / `False` | `True` | Сохранять fold bar plot со средним и std. |
| `save_feature_importance` | `True` / `False` | `True` | Сохранять top coef/feature_importances_, если есть. |
| `show_shap` | `True` / `False` | `False` | Строить optional SHAP summary. |
| `shap_max_samples` | число; ограничения задаёт библиотека/алгоритм | `500` | Предел строк SHAP для контроля стоимости. |

### `logging`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `prints` | `True` / `False` | `True` | Печатать прогресс в терминал. |
| `txt_file` | `True` / `False` | `True` | Сохранять текстовый журнал. |
| `csv_file` | `True` / `False` | `True` | Сохранять CSV историю экспериментов. |
| `telegram` | `True` / `False` | `False` | Отправлять уведомления Telegram при наличии локального credentials файла. |

### `tracking`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `wandb` | `True` / `False` | `False` | Синхронизировать с Weights & Biases. |
| `wandb_project` | строка / допустимые варианты блока | `classic-ml` | Имя W&B проекта. |
| `wandb_entity` | `null` либо значение соответствующего типа | `null` | W&B team/user namespace. |
| `wandb_mode` | online / offline / disabled | `online` | online/offline/disabled режим W&B. |
| `wandb_tags` | список совместимых значений | `['classic-ml']` | Теги W&B run. |
| `log_artifacts` | `True` / `False` | `True` | Загружать checkpoint папку в W&B. |
| `log_plots` | `True` / `False` | `True` | Загружать plots в W&B. |
| `raise_on_error` | `True` / `False` | `False` | Прерывать run при ошибке W&B вместо предупреждения. |

### Значимые library defaults и выбранные значения

Для `LogisticRegression` явно заданы `C=∞`, `l1_ratio=0`, `solver=lbfgs`, `class_weight=None`, `fit_intercept=True`, `tol=1e-4`; `C=∞` — документированный способ sklearn 1.8+ убрать регуляризацию. sklearn 1.8 печатает ошибочное предупреждение для этого случая; код фильтрует только его точный текст. `liblinear` не подходит для multinomial multiclass; `saga` поддерживает L1/ElasticNet. В регрессии `LinearRegression.regularization=none` создаёт OLS, а `ridge`/`lasso`/`elasticnet` — соответствующий estimator с явно заданным `alpha`. `RandomForest` по определению строит bootstrap деревья; `bootstrap=True` явно задан. LightGBM строит histogram деревья по своей архитектуре даже при `optimization.enabled=False`. Для остальных незаданных технических defaults используйте `get_model(config).get_params()` и зафиксированную версию пакета. Корневой `ensemble.py` отдельно использует Ridge как пример meta-model; его `ensemble_config.py` не связан с Classic `models.LinearRegression`.

## Deep Learning — `dl_pipeline_v1/config.py`

### `general`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `experiment_name` | строка / допустимые варианты блока | `dl_baseline_v2` | Имя изолированного run и каталога артефактов. |
| `seed` | число; ограничения задаёт библиотека/алгоритм | `1027309` | Seed Python/NumPy/sklearn/PyTorch/Optuna. |
| `task` | classification / regression | `classification` | Тип целевой переменной; влияет на модель, loss, метрику и форму вероятностей. |
| `mode` | train / inference | `train` | Выбирает обучение либо запуск только inference. |
| `num_classes` | число; ограничения задаёт библиотека/алгоритм | `3` | Число классов; Classic вычисляет по y, DL должен совпадать с метками 0..K−1. |
| `overwrite_experiment` | `True` / `False` | `False` | При True удаляет существующий каталог этого run. |

### `paths`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `path_to_train_features` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/data/train_features.npy` | DL входные train features. |
| `path_to_train_labels` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/data/train_labels.npy` | DL входные train labels. |
| `path_to_test_features` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/data/test_features.npy` | DL test features. |
| `path_to_train_ids` | `null` либо значение соответствующего типа | `null` | DL явные ID train в .npy/.npz/.pt. |
| `path_to_test_ids` | `null` либо значение соответствующего типа | `null` | DL явные ID test в .npy/.npz/.pt. |
| `path_to_unlabeled_features` | `null` либо значение соответствующего типа | `null` | Features без labels для self-training. |
| `path_to_groups` | `null` либо значение соответствующего типа | `null` | Массив групп для group CV. |
| `path_to_folds` | `null` либо значение соответствующего типа | `null` | Готовые номера folds. |
| `path_to_checkpoints` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/checkpoints/${general.experiment_name}` | Каталог данного run. |
| `path_to_fold_checkpoints` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/folds` | DL best/last checkpoints каждого fold. |
| `path_to_oof` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/oof_predictions.csv` | Совместимый OOF CSV с id/target/prediction/fold. |
| `path_to_oof_raw` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/oof_raw_outputs.npz` | Дополнительные DL logits/embeddings NPZ. |
| `path_to_predictions` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/predictions.csv` | Совместимый test probability CSV. |
| `path_to_plots` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/plots` | Каталог компактных графиков. |
| `path_to_exports` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/exports` | Экспортированные DL модели. |
| `path_to_config_snapshot` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/config.yaml` | Путь к разрешённому config.yaml. |
| `path_to_metadata` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/metadata.json` | Run metadata с классами, ID namespace, score. |
| `path_to_environment` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/environment.json` | Версии интерпретатора, библиотек, среды. |
| `path_to_self_training` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/self_training` | Отдельные checkpoints и manifest self-training. |
| `path_to_logs` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/logs` | Общий каталог локальной истории. |
| `path_to_optuna_root` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/optuna` | Каталог SQLite Optuna. |
| `path_to_optuna_db` | путь к файлу/каталогу | `${paths.path_to_optuna_root}/${general.experiment_name}.db` | SQLite Optuna study. |
| `path_to_tuning` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/tuning` | Каталог CSV и YAML результатов Optuna. |
| `path_to_tuning_trials` | путь к файлу/каталогу | `${paths.path_to_tuning}/trials.csv` | CSV всех Optuna trials. |
| `path_to_best_params` | путь к файлу/каталогу | `${paths.path_to_tuning}/best_params.yaml` | YAML найденных гиперпараметров. |
| `path_to_results_csv` | путь к файлу/каталогу | `${paths.path_to_logs}/results.csv` | CSV истории runs. |
| `path_to_results_txt` | путь к файлу/каталогу | `${paths.path_to_logs}/results.txt` | Текст истории runs. |
| `telegram_credentials` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/telegram_credits.json` | Локальный JSON с токеном; не публикуйте. |
| `tensorboard_dir` | путь к файлу/каталогу | `${paths.path_to_checkpoints}/tensorboard` | Локальные события TensorBoard. |

### `reproducibility`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `save_config_snapshot` | `True` / `False` | `True` | Сохранять разрешённый config.yaml. |
| `save_environment` | `True` / `False` | `True` | Сохранять Python/OS/package versions. |
| `save_data_info` | `True` / `False` | `True` | Сохранять схему и размеры данных. |
| `calculate_dataset_hash` | `True` / `False` | `False` | Считать SHA-256 исходных данных; на больших файлах медленно. |
| `deterministic` | `True` / `False` | `True` | Strict torch deterministic algorithms, cuDNN benchmark=False и CUBLAS config. |

### `data`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `feature_key` | `null` либо значение соответствующего типа | `null` | Ключ массива признаков внутри NPZ/словаря. |
| `label_key` | `null` либо значение соответствующего типа | `null` | Ключ labels внутри NPZ/словаря. |
| `id_namespace` | строка / допустимые варианты блока | `row_position` | Семантика ID; должна совпасть у членов ансамбля. Для надёжности используйте явные ID. |

### `preprocessing`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `status` | строка / допустимые варианты блока | `placeholder; implement in preprocessing.py` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |

### `augmentations`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `status` | строка / допустимые варианты блока | `placeholder; implement in augmentations.py` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |

### `split`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `strategy` | KFold / StratifiedKFold / GroupKFold / StratifiedGroupKFold | `StratifiedKFold` | Выбор способа разбиения/вычисления. Значения указаны в строке. |
| `n_splits` | число; ограничения задаёт библиотека/алгоритм | `5` | Количество CV folds; для ансамбля требуются все folds. |
| `folds_to_train` | список совместимых значений | `[0, 1, 2, 3, 4]` | Номера folds для обучения. Частичный список делает OOF неполным. |
| `folds_to_inference` | список совместимых значений | `[0, 1, 2, 3, 4]` | Номера fold checkpoints для усреднения DL test predictions. |
| `shuffle` | `True` / `False` | `True` | Перемешивание перед CV или DataLoader; seed фиксирует порядок. |
| `already_split` | `True` / `False` | `False` | Использовать готовый массив fold IDs вместо sklearn splitter. |
| `all_data_train` | `True` / `False` | `False` | Обучить DL на всём train без CV; OOF недоступен. |

### `training`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `num_epochs` | число; ограничения задаёт библиотека/алгоритм | `30` | Максимальное число эпох одного DL run без optional early stopping. |
| `device` | строка / допустимые варианты блока | `auto` | PyTorch устройство; auto выбирает CUDA → MPS → CPU. |
| `debug` | `True` / `False` | `False` | Включает сокращённый набор данных для быстрой проверки. |
| `number_of_train_debug_samples` | число; ограничения задаёт библиотека/алгоритм | `512` | Размер train подвыборки одного DL fold в debug. |
| `number_of_val_debug_samples` | число; ограничения задаёт библиотека/алгоритм | `256` | Размер validation подвыборки одного DL fold в debug. |
| `save_best` | `True` / `False` | `True` | Сохранять checkpoint эпохи с лучшей validation metric. |
| `save_last` | `True` / `False` | `True` | Сохранять checkpoint последней эпохи и состояние optimizer/scheduler. |
| `resume_from_latest_checkpoint` | `True` / `False` | `False` | Продолжить существующий DL run с last.pt; учтите ограничения workers/RNG. |

### `dataloader_params`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `batch_size` | число; ограничения задаёт библиотека/алгоритм | `64` | Размер minibatch; если включён balanced sampler, размер задаёт sampler. |
| `shuffle` | `True` / `False` | `True` | Перемешивание перед CV или DataLoader; seed фиксирует порядок. |
| `drop_last` | `True` / `False` | `False` | Убирает последний неполный train batch. |

### `regularization`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает регуляризацию; дочерний флаг требует regularization.enabled. |
| `dropout.enabled` | `True` / `False` | `False` | Включает регуляризацию; дочерний флаг требует regularization.enabled. |
| `dropout.p` | число в [0, 1] | `0.2` | Вероятность dropout при явном regularization flag. |
| `weight_decay.enabled` | `True` / `False` | `False` | Включает регуляризацию; дочерний флаг требует regularization.enabled. |
| `weight_decay.value` | число; ограничения задаёт библиотека/алгоритм | `0.0001` | Значение веса/параметра в соответствующей опции. |

### `optimization`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `speed.amp.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `speed.compile.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `speed.compile.backend` | строка / допустимые варианты блока | `inductor` | Backend torch.compile. |
| `speed.compile.mode` | строка / допустимые варианты блока | `default` | Режим работы конкретного метода; значения в строке. |
| `speed.fused_optimizer.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `speed.optimized_dataloader.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `speed.optimized_dataloader.num_workers` | число; ограничения задаёт библиотека/алгоритм | `2` | Процессы DataLoader; 0 без multiprocessing. |
| `speed.optimized_dataloader.pin_memory` | `True` / `False` | `True` | Pinned CPU memory при CUDA. |
| `speed.optimized_dataloader.persistent_workers` | `True` / `False` | `True` | Повторно использовать workers между эпохами. |
| `speed.optimized_dataloader.prefetch_factor` | число; ограничения задаёт библиотека/алгоритм | `2` | Число заранее приготовленных batches на worker. |
| `memory.gradient_accumulation.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `memory.gradient_accumulation.steps` | число; ограничения задаёт библиотека/алгоритм | `4` | Число microbatches на один optimizer step. |
| `memory.gradient_checkpointing.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `memory.gradient_checkpointing.status` | строка / допустимые варианты блока | `placeholder` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |
| `training_control.early_stopping.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `training_control.early_stopping.patience` | число; ограничения задаёт библиотека/алгоритм | `7` | Сколько эпох без улучшения ждать перед остановкой. |
| `training_control.gradient_clipping.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `training_control.gradient_clipping.max_norm` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Максимальная L2 норма градиента для clipping. |
| `training_control.scheduler.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `training_control.warmup.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `training_control.ema.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `training_control.ema.decay` | число в [0, 1] | `0.999` | Коэффициент экспоненциального усреднения весов EMA. |
| `training_control.ema.update_after_step` | число; ограничения задаёт библиотека/алгоритм | `0` | Номер optimizer step, после которого EMA начинает усреднение. |
| `model_compression.quantization.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `model_compression.quantization.status` | строка / допустимые варианты блока | `placeholder` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |
| `model_compression.quantization.mode` | PTQ / QAT (оба placeholder) | `PTQ` | Режим работы конкретного метода; значения в строке. |
| `model_compression.quantization.api` | строка / допустимые варианты блока | `torchao.quantization.quantize_` | Предполагаемый API расширения; текущая функция placeholder. |
| `model_compression.pruning.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `model_compression.pruning.status` | строка / допустимые варианты блока | `placeholder` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |
| `model_compression.knowledge_distillation.enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |
| `model_compression.knowledge_distillation.status` | строка / допустимые варианты блока | `placeholder` | Статус hook: без предметной реализации включение вызывает понятную ошибку. |

### `tuning`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает Optuna или pruning; от optimization.enabled не зависит. |
| `n_trials` | число; ограничения задаёт библиотека/алгоритм | `20` | Количество Optuna trials. |
| `direction` | auto / maximize / minimize | `auto` | Оптимизировать максимум либо минимум metric. |
| `sampler` | tpe / random | `tpe` | Алгоритм Optuna TPE или Random. |
| `study_name` | строка / допустимые варианты блока | `${general.experiment_name}_tuning` | Имя Optuna study в SQLite. |
| `resume_study` | `True` / `False` | `False` | Продолжать старое Optuna study; используйте только с теми же данными и config. |
| `trial_stage` | строка / допустимые варианты блока | `base_model` | Что оптимизируется; реализован base_model без self-training/export. |
| `folds_to_use` | список совместимых значений | `[0]` | Номера folds, по которым Optuna усредняет score. |
| `pruning.enabled` | `True` / `False` | `False` | Включает Optuna или pruning; от optimization.enabled не зависит. |
| `pruning.n_startup_trials` | число; ограничения задаёт библиотека/алгоритм | `5` | Сколько trial выполнить до MedianPruner. |
| `pruning.n_warmup_steps` | число; ограничения задаёт библиотека/алгоритм | `3` | Сколько эпох не применять pruning. |

#### `tuning.search_space`

`path` указывает изменяемый config key; `type=int/float/categorical`, `low/high`, `step`, `log`, `choices` работают как в Classic. Набор по умолчанию изменяет только LR и batch size; dropout/weight decay не настраиваются, пока они выключены.

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `learning_rate` | `float: 1e-05…0.01 (log)` | `optimizer.params.lr` | Шаг обновления параметров/деревьев. |
| `batch_size` | `categorical: [32, 64, 128]` | `dataloader_params.batch_size` | Размер minibatch; если включён balanced sampler, размер задаёт sampler. |

### `model`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | имя из блока models либо MLP / TransformerMLP | `MLP` | Название выбранной модели/метрики/оптимизатора/плана. |
| `input_shape` | список совместимых значений | `[20]` | Форма одного входного образца без batch dimension. |
| `params.hidden_dims` | список совместимых значений | `[128, 64]` | Размеры скрытых слоёв MLP. |
| `params.activation` | torch.nn: ReLU / GELU / SiLU / Tanh и др. | `ReLU` | Название torch.nn activation. |
| `params.d_model` | число; ограничения задаёт библиотека/алгоритм | `64` | Размер embedding Transformer. |
| `params.nhead` | число; ограничения задаёт библиотека/алгоритм | `4` | Количество attention heads; d_model делится на nhead. |
| `params.num_layers` | число; ограничения задаёт библиотека/алгоритм | `2` | Число TransformerEncoderLayer. |
| `params.dim_feedforward` | число; ограничения задаёт библиотека/алгоритм | `128` | Размер внутренней feedforward сети Transformer. |
| `params.head_hidden_dims` | список совместимых значений | `[64]` | Размеры скрытых слоёв head TransformerMLP. |

### `strategies`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `finetuning.enabled` | `True` / `False` | `False` | Включает отдельную стратегию обучения; от optimization.enabled не зависит. |
| `finetuning.source` | checkpoint / model_builtin | `checkpoint` | Источник pretrained модели или семейство сохранённого эксперимента. |
| `finetuning.checkpoint_path` | `null` либо значение соответствующего типа | `null` | Путь к весам pretrained модели. |
| `finetuning.strict_load` | `True` / `False` | `False` | Требовать точного совпадения всех ключей state_dict. |
| `finetuning.freeze_backbone` | `True` / `False` | `True` | Заморозить backbone на начальных эпохах. |
| `finetuning.freeze_batchnorms` | `True` / `False` | `False` | Оставить BatchNorm в eval и заморозить параметры при finetuning. |
| `finetuning.unfreeze_after_epoch` | число; ограничения задаёт библиотека/алгоритм | `3` | Эпоха разморозки backbone, отсчёт с 0. |
| `finetuning.backbone_lr` | число; ограничения задаёт библиотека/алгоритм | `1e-05` | Отдельный learning rate backbone при finetuning. |
| `finetuning.head_lr` | число; ограничения задаёт библиотека/алгоритм | `0.001` | Отдельный learning rate head при finetuning. |
| `metric_learning.enabled` | `True` / `False` | `False` | Включает отдельную стратегию обучения; от optimization.enabled не зависит. |
| `metric_learning.embedding_dim` | число; ограничения задаёт библиотека/алгоритм | `64` | Размер embedding vector. |
| `metric_learning.margin` | число; ограничения задаёт библиотека/алгоритм | `0.3` | Запас между позитивным и негативным расстояниями TripletLoss. |
| `metric_learning.metric_loss_weight` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Вес metric loss в общей функции потерь. |
| `metric_learning.supervised_loss_weight` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Вес обычного supervised loss. |
| `metric_learning.normalize_embeddings` | `True` / `False` | `True` | L2 нормировка embeddings. |
| `metric_learning.balanced_batches` | `True` / `False` | `True` | Включает class-balanced batch sampler. |
| `metric_learning.classes_per_batch` | число; ограничения задаёт библиотека/алгоритм | `4` | Число классов в balanced batch. |
| `metric_learning.samples_per_class` | число; ограничения задаёт библиотека/алгоритм | `4` | Число образцов на класс в balanced batch. |
| `metric_learning.label_head` | `null` либо значение соответствующего типа | `null` | Multi-head источник классов для metric loss; None = primary. |
| `hard_negative_mining.enabled` | `True` / `False` | `False` | Включает отдельную стратегию обучения; от optimization.enabled не зависит. |
| `hard_negative_mining.strategy` | batch_hard (единственный реализованный) | `batch_hard` | Выбор способа разбиения/вычисления. Значения указаны в строке. |
| `self_training.enabled` | `True` / `False` | `False` | Включает отдельную стратегию обучения; от optimization.enabled не зависит. |
| `self_training.confidence_threshold` | число в [0, 1] | `0.95` | Порог max class probability для pseudo-label. |
| `self_training.max_rounds` | число; ограничения задаёт библиотека/алгоритм | `1` | Максимум циклов self-training. |
| `self_training.min_pseudo_samples` | число; ограничения задаёт библиотека/алгоритм | `1` | Минимум принятых pseudo-labels для нового retrain. |
| `self_training.save_pseudo_labels` | `True` / `False` | `True` | Сохранять полученные pseudo-labels и confidence. |
| `multi_head.enabled` | `True` / `False` | `False` | Включает отдельную стратегию обучения; от optimization.enabled не зависит. |
| `multi_head.primary_head` | строка / допустимые варианты блока | `main` | Голова, задающая split и итоговую метрику. |
| `multi_head.heads.main.task` | строка / допустимые варианты блока | `classification` | Тип целевой переменной; влияет на модель, loss, метрику и форму вероятностей. |
| `multi_head.heads.main.num_outputs` | число; ограничения задаёт библиотека/алгоритм | `3` | Число выходов головы. |
| `multi_head.heads.main.loss_name` | строка / допустимые варианты блока | `CrossEntropyLoss` | torch.nn loss выбранной головы. |
| `multi_head.heads.main.loss_params` | словарь аргументов (ключи зависят от выбранной функции) | `{}` | Аргументы конструктора loss. |
| `multi_head.heads.main.loss_weight` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Вес loss этой головы. |
| `multi_head.heads.main.metric_name` | строка / допустимые варианты блока | `accuracy_score` | sklearn metric этой головы. |
| `multi_head.heads.main.metric_params` | словарь аргументов (ключи зависят от выбранной функции) | `{}` | Аргументы metric. |
| `multi_head.heads.main.metric_direction` | строка / допустимые варианты блока | `maximize` | maximize/minimize для отбора best epoch. |

### `optimizer`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | torch.optim: SGD / Adam / AdamW / RMSprop и др. | `AdamW` | Название выбранной модели/метрики/оптимизатора/плана. |
| `params.lr` | число; ограничения задаёт библиотека/алгоритм | `0.001` | Настройка `optimizer.params.lr` для соответствующего этапа. |
| `params.weight_decay` | число; ограничения задаёт библиотека/алгоритм | `0.0` | Настройка `optimizer.params.weight_decay` для соответствующего этапа. |

### `scheduler`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | torch.optim.lr_scheduler: CosineAnnealingLR / ReduceLROnPlateau / OneCycleLR / StepLR и др. | `CosineAnnealingLR` | Название выбранной модели/метрики/оптимизатора/плана. |
| `interval` | epoch / step | `epoch` | Частота scheduler.step: epoch или optimizer step. |
| `params.T_max` | число; ограничения задаёт библиотека/алгоритм | `25` | Длительность cosine цикла (в единицах scheduler.step). |
| `params.eta_min` | число; ограничения задаёт библиотека/алгоритм | `1e-06` | Минимальный LR cosine scheduler. |
| `warmup.epochs` | число; ограничения задаёт библиотека/алгоритм | `3` | Длина warmup в эпохах. |
| `warmup.start_factor` | число в [0, 1] | `0.1` | Начальная доля LR в LinearLR warmup. |

### `loss`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | строка / допустимые варианты блока | `CrossEntropyLoss` | Название выбранной модели/метрики/оптимизатора/плана. |
| `params` | словарь аргументов (ключи зависят от выбранной функции) | `{}` | Дополнительные keyword arguments выбранной функции; пустой словарь не добавляет аргументов. |

### `metric`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | имя функции sklearn.metrics; accuracy_score, f1_score, roc_auc_score, log_loss, root_mean_squared_error, r2_score и др. | `accuracy_score` | Название выбранной модели/метрики/оптимизатора/плана. |
| `direction` | maximize / minimize | `maximize` | Оптимизировать максимум либо минимум metric. |
| `params` | словарь аргументов (ключи зависят от выбранной функции) | `{}` | Дополнительные keyword arguments выбранной функции; пустой словарь не добавляет аргументов. |

### `postprocessing`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `threshold` | число в [0, 1] | `0.5` | Порог binary class 1 в отдельном submission.csv. |

### `visualization`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `save_training_curves` | `True` / `False` | `True` | Сохранять один график loss/metric/LR по эпохам. |
| `save_validation_plot` | `True` / `False` | `True` | Сохранять confusion matrix либо true-vs-predicted + residuals. |
| `save_cv_scores` | `True` / `False` | `True` | Сохранять fold bar plot со средним и std. |

### `conversion`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `formats` | torch_export / onnx | `['torch_export']` | Форматы экспорта: torch_export/onnx. |
| `opset_version` | число; ограничения задаёт библиотека/алгоритм | `18` | Номер ONNX opset. |

### `tracking`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `wandb` | `True` / `False` | `False` | Синхронизировать с Weights & Biases. |
| `tensorboard` | `True` / `False` | `False` | Писать события TensorBoard. |
| `wandb_project_name` | строка / допустимые варианты блока | `DL-Pipeline` | Имя W&B проекта. |
| `wandb_username` | `null` либо значение соответствующего типа | `null` | W&B team/user namespace. |
| `wandb_log_config` | `True` / `False` | `True` | Сохранять config в W&B. |

### `logging`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `prints` | `True` / `False` | `True` | Печатать прогресс в терминал. |
| `txt_file` | `True` / `False` | `True` | Сохранять текстовый журнал. |
| `csv_file` | `True` / `False` | `True` | Сохранять CSV историю экспериментов. |
| `telegram` | `True` / `False` | `False` | Отправлять уведомления Telegram при наличии локального credentials файла. |

### Как читать основные варианты

- **Optimizer:** `SGD` — градиентные шаги, `Adam` — адаптивные моменты, `AdamW` — decoupled weight decay; в поставке `AdamW` имеет явно заданный `weight_decay=0` до включения regularization.
- **Scheduler:** `CosineAnnealingLR` — косинусный спад, `ReduceLROnPlateau` — снижение при стагнации метрики, `OneCycleLR` — один LR цикл (обычно `interval=step`), `StepLR` — ступенчатое снижение. Warmup создаётся через `LinearLR`; текущий router запрещает соединять `ReduceLROnPlateau` с `SequentialLR` warmup.
- **Loss:** `CrossEntropyLoss` — logits многоклассовой классификации; `MSELoss`/`L1Loss` — регрессия. `metric.name` принимает функции `sklearn.metrics`. Для `roc_auc_score`, `average_precision_score`, `log_loss` и `brier_score_loss` код передаёт probabilities.
- **Quantization:** `PTQ` и `QAT` только архитектурные placeholders. Будущее расширение ориентировано на `torchao.quantization.quantize_`; обучение обычной модели они сейчас не меняют.

## Универсальный ансамбль — `ensemble_config.py`

### `general`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `experiment_name` | строка / допустимые варианты блока | `ensemble_v1` | Имя изолированного run и каталога артефактов. |
| `overwrite_experiment` | `True` / `False` | `False` | При True удаляет существующий каталог этого run. |
| `seed` | число; ограничения задаёт библиотека/алгоритм | `1027309` | Seed Python/NumPy/sklearn/PyTorch/Optuna. |

### `paths`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `classic_checkpoints` | путь к файлу/каталогу | `<PROJECT_ROOT>/classic_ml_pipeline_v8/checkpoints` | Каталог Classic experiments, читаемый ансамблем. |
| `dl_checkpoints` | путь к файлу/каталогу | `<PROJECT_ROOT>/dl_pipeline_v1/checkpoints` | Каталог DL experiments, читаемый ансамблем. |
| `ensembles_root` | путь к файлу/каталогу | `<PROJECT_ROOT>/checkpoints/ensembles` | Каталог готовых ensemble experiments. |

### `metric`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `name` | имя функции sklearn.metrics; accuracy_score, f1_score, roc_auc_score, log_loss, root_mean_squared_error, r2_score и др. | `accuracy_score` | Название выбранной модели/метрики/оптимизатора/плана. |
| `direction` | maximize / minimize | `maximize` | Оптимизировать максимум либо минимум metric. |
| `params` | словарь аргументов (ключи зависят от выбранной функции) | `{}` | Дополнительные keyword arguments выбранной функции; пустой словарь не добавляет аргументов. |

### `optimization`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает оптимизацию; дочерний флаг действует только вместе с optimization.enabled. |

### `ensemble`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `preset` | ключ из ensemble.presets | `boosting_dl` | Имя выбранного набора ансамбля. |
| `voting` | hard / soft | `soft` | hard/soft метод классификационного voting. |
| `weight_optimization.enabled` | `True` / `False` | `False` | Включает этот блок независимо от других категорий, если не указано иное. |
| `weight_optimization.method` | optuna | `optuna` | Алгоритм подбора весов. |
| `weight_optimization.n_trials` | число; ограничения задаёт библиотека/алгоритм | `100` | Количество Optuna trials. |
| `meta_model.classification` | строка / допустимые варианты блока | `LogisticRegression` | Meta estimator/модели для classification. |
| `meta_model.regression` | строка / допустимые варианты блока | `Ridge` | Meta estimator/модели для regression. |
| `meta_model.params.classification.C` | число; ограничения задаёт библиотека/алгоритм | `∞` | Обратная сила L2/L1 регуляризации: `∞` = нет штрафа, меньше C = сильнее штраф. |
| `meta_model.params.classification.l1_ratio` | число; ограничения задаёт библиотека/алгоритм | `0.0` | 0=L2, 1=L1, между 0 и 1=ElasticNet; для L1/ElasticNet нужен saga (для L1 допустим liblinear). |
| `meta_model.params.classification.solver` | строка / допустимые варианты блока | `lbfgs` | Численный алгоритм для LogisticRegression/Ridge. |
| `meta_model.params.classification.max_iter` | число; ограничения задаёт библиотека/алгоритм | `1000` | Предел итераций оптимизатора LogisticRegression. |
| `meta_model.params.regression.alpha` | число; ограничения задаёт библиотека/алгоритм | `1.0` | Сила L2 Ridge; 0 убирает штраф, >0 усиливает. |

#### `ensemble.presets`

Каждый preset состоит из `type` (`average`, `weighted_average`, `voting`, `stacking`) и `members`. У участника `name` — уникальная подпись, `source` — `classic`/`dl`, `experiment` — папка исходного run, `weight` — неотрицательный коэффициент для weighted average.

| Preset | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `boostings` | `type=average` | `xgboost:classic/xgb_v1, catboost:classic/cat_v1` | Выберите через `ensemble.preset`; замените примеры на существующие experiments. |
| `neural_networks` | `type=average` | `mlp1:dl/mlp_v1, mlp2:dl/mlp_v2` | Выберите через `ensemble.preset`; замените примеры на существующие experiments. |
| `boosting_dl` | `type=average` | `xgboost:classic/xgb_v1, mlp:dl/mlp_v1` | Выберите через `ensemble.preset`; замените примеры на существующие experiments. |
| `boosting_dl_weighted` | `type=weighted_average` | `xgboost:classic/xgb_v1 w=0.6, mlp:dl/mlp_v1 w=0.4` | Выберите через `ensemble.preset`; замените примеры на существующие experiments. |
| `boosting_dl_stacking` | `type=stacking` | `xgboost:classic/xgb_v1, mlp:dl/mlp_v1` | Выберите через `ensemble.preset`; замените примеры на существующие experiments. |

### `visualization`

| Параметр | Возможные значения | Сейчас | Что означает |
| --- | --- | --- | --- |
| `save_validation_plot` | `True` / `False` | `True` | Сохранять confusion matrix либо true-vs-predicted + residuals. |
| `save_cv_scores` | `True` / `False` | `True` | Сохранять fold bar plot со средним и std. |
| `save_ensemble_diagnostics` | `True` / `False` | `True` | Сохранять сравнение base/ensemble, веса/meta и корреляцию. |

## Варианты solver и алгоритмов

- **LogisticRegression:** `lbfgs` — квазиньютоновский solver; `newton-cg`/`newton-cholesky` — ньютоновские; `sag`/`saga` — стохастические. `saga` нужен для ElasticNet и multinomial L1; `liblinear` поддерживает бинарную/OvR постановку, но не полноценный multinomial. `C=∞` снимает штраф; `l1_ratio=0`, `1`, `(0,1)` задаёт L2, L1, ElasticNet.
- **Деревья:** `gini`/`entropy`/`log_loss` измеряют неоднородность классов; `squared_error` ориентируется на дисперсию, `absolute_error` на абсолютное отклонение, `poisson` на пуассоновскую девиацию для подходящего target. `max_depth=None` и `ccp_alpha=0` снимают только соответствующие ограничения.
- **Бустинг:** `subsample<1`/`colsample_bytree<1` включают sampling; `reg_alpha`/`reg_lambda` — L1/L2; `tree_method=exact` вычисляет точные splits, `hist` использует гистограмму. `CatBoost bootstrap_type=No` отключает bootstrap; Bayesian/Bernoulli/MVS задают разные sampling схемы. LightGBM `gbdt` — gradient boosting, `dart` — dropouts деревьев, `rf` — режим random forest.

## Источники API

- [sklearn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [LightGBM sklearn API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html), [CatBoost fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit).
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [DataLoader](https://docs.pytorch.org/docs/stable/data.html), [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [torch.export](https://docs.pytorch.org/docs/stable/export.html), [torchao quantization](https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html), [Optuna reproducibility FAQ](https://optuna.readthedocs.io/en/stable/faq.html).
