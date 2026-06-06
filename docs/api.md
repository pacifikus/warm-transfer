# API Reference

Автоматически из docstring'ов (mkdocstrings).

## Контракты и типы

::: coldscore.types
    options:
      members: [ItemFeatures, Dataset, SplitResult, TransferInputs]
      show_root_heading: true
      heading_level: 3

## Базовый метод

::: coldscore.methods.base.ColdStartMethod
    options:
      show_root_heading: true
      heading_level: 3

## Cold-start методы

::: coldscore.methods.linmap.LinMap
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.stacking_plus.StackingPlus
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.stacking.StackingTransfer
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.baselines.GroupedMostPopularPersonalized
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.baselines.GroupedMostPopular
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.knn.KNNScoreAggregation
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.attention_knn.AttentionKNN
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.debiased_knn.DebiasedKNN
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.logreg_calib.LogRegCalibration
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.scale_shift.ScaleShift
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.linmap_emb.LinMapEmbedding
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.embedding_avg.EmbeddingAverage
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.attention_emb.AttentionEmbedding
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.magnitude_scaling.MagnitudeScaling
    options: { show_root_heading: true, heading_level: 3 }

::: coldscore.methods.dropoutnet.DropoutNet
    options: { show_root_heading: true, heading_level: 3 }

## Конфиг и результаты

::: coldbench.config.BenchConfig
    options: { show_root_heading: true, heading_level: 3 }

::: coldbench.results
    options:
      show_root_heading: true
      heading_level: 3

## Метрики

::: coldscore.metrics
    options:
      show_root_heading: true
      heading_level: 3

## Бенчмарк

::: coldbench.runner.BenchmarkRunner
    options: { show_root_heading: true, heading_level: 3 }

::: coldbench.splitters.pseudo_cold.PseudoColdSplitter
    options: { show_root_heading: true, heading_level: 3 }
