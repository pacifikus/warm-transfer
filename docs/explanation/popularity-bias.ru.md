# Популярность и Grouped MP

Самый сильный простой бейзлайн в проекте — personalized Grouped Most Popular
(`grouped_most_popular_pers`). Он использует историческую affinity пользователя к content group
cold item, а не считает все cold items одинаково неизвестными.

## Почему наивный transfer проигрывает

Nearest-neighbor методы выглядят естественно: найти warm items, похожие на cold item, и усреднить
донорские скоры этих соседей. Проблема в том, что соседи часто приносят с собой global popularity.
Cold item рядом с популярными warm items получает завышенные скоры, даже если персонализация слабая.

!!! warning "Частая ловушка"
    Высокая similarity к популярным warm items — это не то же самое, что персонализированный
    cold-start score.

## Wrong / right

| Pattern | Result |
|---|---|
| Усреднить скоры похожих warm items и остановиться | Часто наследует popularity соседей |
| Сравнивать с grouped popularity baseline | Показывает, есть ли реальная польза от transfer |
| Использовать popularity как feature или anchor | Помогает `stacking` и `stacking_plus` |
| Учить content-to-score structure напрямую | Помогает `linmap` сохранить personalization донора |

Отсюда главная гипотеза проекта: наивного transfer недостаточно, но calibrated transfer может
обойти Grouped MP.
