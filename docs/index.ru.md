# warm-transfer

Переносит и калибрует скоры уже обученного рекомендателя на cold-start айтемы. Plug&play,
model-agnostic, без переобучения донора.

```python
from warmtransfer.methods import LinMap

reco = LinMap(alpha=1.0).fit(inputs, seed=42).predict(user_ids, cold_item_ids)
```

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Быстрый старт__

    ---

    Перенести скоры донора на cold items за пять минут.

    [:octicons-arrow-right-24: Начать](getting-started/quickstart.md)

-   :material-puzzle:{ .lg .middle } __Работает с вашими скорами__

    ---

    Принесите warm-item скоры любой модели. warm-transfer не переобучает донора.

    [:octicons-arrow-right-24: Зачем](explanation/why.md)

-   :material-chart-line:{ .lg .middle } __Бьёт сильные бейзлайны__

    ---

    Score transfer бьёт персонализированный Grouped MP в 36 из 40 ячеек dataset-donor.

    [:octicons-arrow-right-24: Результаты](results/full_matrix.md)

-   :material-book-open-variant:{ .lg .middle } __Методы и API__

    ---

    Семнадцать методов, один контракт `ColdStartMethod.fit(...).predict(...)`.

    [:octicons-arrow-right-24: Справочник](methods.md)

</div>

## Какую проблему решает

Ваш рекомендатель уже умеет скорить warm items, но у новых айтемов нет истории взаимодействий.
warm-transfer использует warm-скоры и контент айтемов, чтобы предсказать скоры для cold-start items.
Донором может быть ALS, BPR, CatBoost, EASE, two-tower модель или закрытая production-модель.

## Главный результат

На текущей benchmark-матрице **8 dataset loaders × 5 donors** (seed=42) методы score transfer бьют
сильный персонализированный бейзлайн `grouped_most_popular_pers` по **per-user AUC в 36 из 40
dataset-donor ячеек**.

Главный контраст: наивный nearest-neighbor transfer часто наследует популярность соседей и проигрывает
Grouped MP. Калиброванный transfer, особенно `linmap` и `stacking_plus`, сохраняет персонализацию из
скоров донора и работает устойчивее. Бенчмарк покрывает matrix-factorization, GBDT, linear item-item и
neural donors; 4 промаха в основном на ML-1M, где baseline AUC уже высокий.

Эти числа получены на одном seed. Точечные multi-seed прогоны для пограничных ячеек отслеживаются на
страницах бенчмарка.

## Архитектура

- **`warmtransfer`** — лёгкое ядро: методы переноса, метрики и content similarity.
- **`warmtransfer.bench`** — optional benchmark layer: dataset loaders, splitters, donor adapters и
  runner `warmbench`.

## Быстрый вердикт: какой метод подойдёт моим данным?

Не уверены, какой метод выбрать? Принесите взаимодействия, контент айтемов и скоры своей модели —
`recommend` прогонит все применимые методы на честном псевдо-cold holdout и подскажет лучший (и
стоит ли вообще использовать трансфер):

```python
import warmtransfer as wt

result = wt.recommend(interactions, content, donor_scores)
print(result)              # лидерборд + вердикт
result.best_transfer       # лучший не-бейзлайн метод, напр. "linmap"
result.predict(users, new_item_ids)   # победитель, дофиченный на всех warm
```

Или из терминала:

```bash
warmbench try --interactions inter.parquet --content content.parquet --scores scores.parquet
```

Оценка делается на holdout из ваших тёплых айтемов; донор не переобучается — трактуйте как
слегка оптимистичный сигнал того, помогает ли трансфер. Готовый пример — в
`examples/recommend_quickstart.py`.

## Куда дальше

- Если вы впервые в проекте: начните с [Quickstart](getting-started/quickstart.md).
- Если не уверены, какой метод подойдёт: дайте `recommend()` оценить их на ваших данных (см. *Быстрый вердикт* выше).
- Если подключаете свою модель: читайте [Plug in a donor](how-to/add-donor.md).
- Если выбираете метод: используйте [capability matrix](methods.md).
- Если проверяете научную корректность: читайте [evaluation protocol](eval-protocol.md).
