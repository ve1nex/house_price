## Основная информация

**Ссылка:**  https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
**Тип задачи:**  regression
**Target:**  SalePrice
**Метрика:**  RMSE
**Train:**  1460
**Test:**  1459

---

## To-do

- [x] посмотреть на целевую переменную и изучить
- [x] разделить признаки на категории и проанализировать их 
- [x] проверить признаки на мультиколлиниарность
- [x] очистить данные(обработать пропуски, аномальные значения и тд)
- [x] посмотреть Mutual Information с таргетом 
- [x] если распределения смещены поправить их
- [x] baseline
- [x] KNN
- [x] попробовать Feature Engineering(
- ручное создание новых признаков;
- арифметические комбинации и взаимодействия;
- **K-Means** как источник новых кластерных признаков;
- **PCA** как источник новых компонент;
- признаки на основе PCA;
- **Target Encoding** категориальных признаков.)
- [x] Попробовать `Lasso`. 
- [x] Попробовать `ElasticNet`. 
- [x] Попробовать `KernelRidge`. 
- [x] Попробовать `GradientBoostingRegressor`. 
- [x] RandomForest
- [x] Попробовать `XGBoost`. 
- [x] Попробовать `LightGBM`. 
- [x] Попробовать усреднение предсказаний моделей. 
- [x] Попробовать stacking с meta-model. 
- [x] DL
- [x] Попробовать weighted blending. Ансамбль.

---

## Notes

- `SalePrice` имеет выраженную положительную асимметрию.
- Наиболее подозрительные выбросы: `Id 524`, `1299`; также выделяются `458`, `1397`.
- `GarageCars` и `GarageArea` сильно коррелируют (~0.88) и несут похожую информацию.
- `YearBuilt` и `GarageYrBlt` сильно коррелируют (~0.83).
- `GrLivArea` и `TotRmsAbvGrd` сильно коррелируют (~0.83), но описывают разные характеристики.
- `TotalBsmtSF` и `1stFlrSF` сильно коррелируют (~0.82), но имеют разный смысл.
- `Exterior1st` и `Exterior2nd` сильно дублируют друг друга.
- `GarageQual` и `GarageCond` содержат во многом похожую информацию.
- Наиболее информативные признаки по EDA/MI: `OverallQual`, `Neighborhood`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `YearBuilt`, `BsmtQual`, `ExterQual`, `KitchenQual`, `1stFlrSF`.
- `Neighborhood` — один из наиболее информативных категориальных признаков.
- `MSSubClass` показывает заметную MI несмотря на неочевидную линейную зависимость с ценой.
- `OverallCond`, `MoSold`, `YrSold` показывают слабую связь с `SalePrice`.
- В датасете есть пропуски; часть `NaN` означает отсутствие объекта, а не неизвестное значение.
- Многие непрерывные признаки имеют сильную положительную асимметрию.
- Особенно сильно смещены `MiscVal`, `PoolArea`, `LotArea`, `3SsnPorch`, `LowQualFinSF`.
- 
- 

---

## Questions

- [ ] как смотреть корреляцию с категориальными признаками
- [ ] Mutual Information что это первый раз столкнулся
- [ ] 
- [ ] 

---

## Solution

| #   | Method                |     CV | CV std | Public LB | Date  | Annotation |
| --- | --------------------- | -----: | -----: | --------: | ----- | ---------- |
| 1   | Baseline              | 0.1450 | 0.0239 |   0.16062 | 28.09 | A1         |
| 2   | ElasticNet            | 0.1102 | 0.0116 |   0.13247 | 28.09 | A2         |
| 3   | Ensemble1             | 0.1076 | 0.0123 |   0.12694 | 28.09 | A3         |
| 4   | Ensemble2             | 0.1069 | 0.0122 |   0.12711 | 28.09 | A4         |
| 5   | MLP                   | 0.1163 | 0.0104 |   0.12887 | 1.10  | A5         |
| 6   | Average Classic + MLP | 0.1079 | 0.0115 |   0.12637 | 1.10  | A6         |
|     |                       |        |        |           |       |            |

---

## Annotations

**A1 — Baseline**
- Изменение: 
	- первая модель Linear Regression
- Настройки: 
	- никакие признаки не выкидываем
	- target_transform: log1p
	- numeric_imputer: median
	- categorical_imputer: most_frequent
	- scale_numeric: false
	- encode_categorical: true
	- feature_engineering: false
	- split: 5 fold Kflod
	- LinearRegression
	- regularization: none
	- alpha: 1.0
	- l1_ratio: 0.5
	- fit_intercept: true
	- positive: false
	- max_iter: 5000
	- tol: 0.0001
	- random_state: 1027309
	- solver: auto
	- selection: cyclic
	- n_jobs: 1
	- optimization: False
	- 
- Результат: 
	- CV root_mean_squared_error: 0.1450 ± 0.0239
	- PB test: 0.16062
- Вывод: 
	- стартовая точка

**A2 — ElasticNet
- Изменение: 
	- Linear Regression с лучшеми параметрами и трюками с датасетом
- Настройки: 
	- регуляризация elastic
	- "alpha": 0.0005001389415823808
	- "l1_ratio": 0.922725283130898,
	- убрали объекты Id = 524, 1299, 1397
	- убрали фичи`GarageArea` `GarageYrBlt` `Exterior2nd` `GarageCond`
	- добавили новые фичи MSSubClassCat, Spaciousness
	- остальное как у baseline/по умолчанию
- Результат: 
	- CV root_mean_squared_error: 0.1102 ± 0.0116
	- PB test: 0.13247
- Вывод: 
	- Регуляризация и подготовка данных уменьшили RMSE на CV и Public LB относительно baseline.

**A3 — Ensemble
- Изменение: 
	- ансамбль из 3 моделей
- Настройки: 
	- 3 модели тип: avarage веса [1, 1, 1]
	- 1 модель A2
	- 2 модель "LGBMRegressor": {
		"n_estimators": 1150,
		"learning_rate": 0.011008771485856902,
		"num_leaves": 24,
		"max_depth": 9,
		"min_child_samples": 5,
		"subsample": 0.5366817499994514,
		"subsample_freq": 1,
		"colsample_bytree": 0.6695500225692324,
		"reg_alpha": 6.087236765197375e-07,
		"reg_lambda": 4.5668471583483656e-07,
		"min_split_gain": 0.0,
		"boosting_type": "gbdt",
		"random_state": "${general.seed}",
		"n_jobs": 1,
		"verbosity": -1,},
		остальное по умолчанию
	- 3 модель "GradientRegressor(sklearn)": {
		"n_estimators": 450,
		"learning_rate": 0.06502534449667197,
		"max_depth": 2,
		'subsample': 0.565744639044207,
		'min_samples_split': 14,
		'min_samples_leaf': 4,
		"max_features": None,
		"loss": "squared_error",
	    остальное по умолчанию
- Результат: 
	- CV root_mean_squared_error: 0.1076 ± 0.0123
	- PB test: 0.12694
- Вывод: 
	- небольшое улучшение на CV, но на PB заметный прогресс
**A4 — Ensemble2
- Изменение: 
	- ансамбль из 3 моделей(stacking)
- Настройки: 
	- Мета-модель: ElasticNet.
	- Внутренняя кросс-валидация stacking: 5 фолдов.
	- passthrough: False.
- Результат: 
	- CV root_mean_squared_error: 0.1069 ± 0.0122
	- PB test: 0.12711
- Вывод: 
	- улучшение относительно A3 на CV но ухудшение на PB
**A5 — MLP
- Изменение: 
	- DL модель с подобранными параметрами 
- Настройки: 
	- batch_size: 16
	- dropout: p: 0.05641011715961437
	- weight_decay: value: 1.0899202432093064e-05
	- gradient_clipping:  max_norm: 1.0
	- hidden_dims: 64 32 128
	- activation: SiLU
	- optimizer: Adam lr: 0.0010697325967296643
	- scheduler
	- loss: MSE
	- scheduler:
		name: CosineAnnealingLR
		interval: epoch
		params:
		T_max: 110
		eta_min: 5.289610681565301e-06
		warmup:
		epochs: 3
		start_factor: 0.1
	- остальное по умолчанию
- Результат:
	- CV RMSE: 0.1163 ± 0.0104.
	- Public LB: 0.12887.
- Вывод:
	- MLP уступает Classic stacking по CV и показывает больший Public LB RMSE: 0.12887 против 0.12711.

**A6 — Classic + MLP
- Изменение:
  - Объединение Classic stacking и MLP.

- Настройки:
  - Усреднение предсказаний с весами 50/50.
  - Первая модель: Classic stacking.
  - Вторая модель: MLP из текущего запуска A5.
  - Усреднение выполняется в пространстве log1p(SalePrice).
  - После объединения применяется expm1.

- Результат:
  - CV RMSE: 0.1079 ± 0.0115.
  - OOF RMSE: 0.108492.
  - Public LB: 0.12637.

- Вывод:
  - Ансамбль уступает Classic stacking по CV, но улучшает Public LB относительно Classic stacking и отдельной MLP.
---

## Processed ideas

### Good

- drop `GarageArea`
- drop `GarageYrBlt`
- drop `Exterior2nd`
- drop `GarageCond`
- удалить выбросы 524,1299
- удалить выброс 1397
- `MSSubClassCat` — переводит `MSSubClass` из числового кода в категориальный признак.
- `Spaciousness` — средняя площадь жилого пространства на комнату.
- Elastick + standartscaller
- average model(classic)
- stacking model(classic)
- LightGBM

### Neutral

- сделать пропуски по смыслу
- ridge + standartscaller
- lasso + standartscaller
- sklearn gradientboosting
- average model(classic+MLP)

### Bad

- Nan_num:mean
- Nan_cat:constant
- KNN
- удалить 458
- `log1p` для skewed numeric features
- StandartScaller
- `MedNhbdArea` — медианная `GrLivArea` для каждого района (`Neighborhood`).
- `PorchTypes` — количество типов веранд/террас, которые есть у дома.
- `MSClass` — объединяет `MSSubClass` в более крупные классы домов.
- `LivLotRatio` — отношение жилой площади дома к площади участка.
- `TotalOutsideSF` — суммарная площадь веранд, террас и других наружных площадок.
- `BldgType × GrLivArea` — взаимодействие типа здания с жилой площадью.
- `Feature1` — сумма `GrLivArea` и `TotalBsmtSF`.
- `Feature2` — взаимодействие `YearRemodAdd × TotalBsmtSF`.
- XGBregressor
- RandomForest
- MLP

---

## Final ensemble

| Model | CV RMSE | Public LB | Weight |
| ----- | ------- | --------- | ------ |
| Classic ML stacking | 0.1069 ± 0.0122 | 0.12711 | 50% |
| MLP | 0.1163 ± 0.0104 | 0.12887 | 50% |

**Ensemble CV:** 0.1079 ± 0.0115.

**Ensemble OOF RMSE:** 0.108492.

**Ensemble Public LB:** 0.12637.

**Метод объединения:** усреднение предсказаний с весами 50/50 в пространстве `log1p(SalePrice)`, затем обратное преобразование `expm1`.

**Вывод:** ансамбль уступает Classic stacking по CV, но показывает меньший RMSE на Public LB: 0.12637 против 0.12711 у Classic stacking и 0.12887 у MLP. Для итогового сабмита используется ансамбль; лучший CV среди моделей последнего запуска принадлежит Classic stacking.

---

## Final submissions

**Submission 1**  
Model / Ensemble:  
CV:  
Public LB:  
Почему выбран:  

**Submission 2**  
Model / Ensemble:  
CV:  
Public LB:  
Почему выбран:  

---

## Итоги

**Лучший CV среди моделей последнего запуска:** Classic stacking — 0.1069 ± 0.0122.

**Public LB выбранного итогового ансамбля:** 0.12637.

**Что сработало:**

- Подготовка данных и регуляризация линейной модели улучшили результат относительно baseline.
- Classic stacking показал лучший CV среди моделей последнего запуска.
- Усреднение Classic stacking и MLP улучшило Public LB относительно обеих моделей.

**Что запомнить на будущее:**

- Улучшение CV не всегда сопровождается улучшением Public LB.
- Модель с худшей отдельной метрикой может быть полезна в ансамбле.
- Средний RMSE по фолдам отличается от RMSE по всем OOF-предсказаниям.