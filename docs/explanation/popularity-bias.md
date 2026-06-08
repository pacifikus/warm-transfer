# Popularity bias & Grouped MP

The strongest simple baseline in this project is personalized Grouped Most Popular
(`grouped_most_popular_pers`). It uses the user's historical affinity to the cold item's content
group, instead of pretending all cold items are equally unknown.

## Why naive transfer loses

Nearest-neighbor methods look attractive: find warm items similar to a cold item and average donor
scores from those neighbors. The problem is that the neighbors often carry their global popularity
with them. A cold item close to popular warm items receives inflated scores, even when personalization
is weak.

!!! warning "Common pitfall"
    High similarity to popular warm items is not the same thing as a personalized cold-start score.

## Wrong / right

| Pattern | Result |
|---|---|
| Average scores from similar warm items and stop there | Often inherits neighbor popularity |
| Compare against a grouped popularity baseline | Exposes whether transfer is actually useful |
| Use popularity as a feature or anchor | Helps `stacking` and `stacking_plus` |
| Learn content-to-score structure directly | Helps `linmap` preserve donor personalization |

The main project hypothesis follows from this: naive transfer is not enough, but calibrated transfer
can beat Grouped MP.
