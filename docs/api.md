# API Reference

Automatically generated from docstrings (mkdocstrings).

## Contracts and types

::: warmtransfer.types
    options:
      members: [ItemFeatures, Dataset, SplitResult, TransferInputs]
      show_root_heading: true
      heading_level: 3

## Base method

::: warmtransfer.methods.base.ColdStartMethod
    options:
      show_root_heading: true
      heading_level: 3

## Cold-start methods

::: warmtransfer.methods.linmap.LinMap
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.stacking_plus.StackingPlus
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.stacking.StackingTransfer
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.baselines.GroupedMostPopularPersonalized
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.baselines.GroupedMostPopular
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.knn.KNNScoreAggregation
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.attention_knn.AttentionKNN
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.debiased_knn.DebiasedKNN
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.logreg_calib.LogRegCalibration
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.scale_shift.ScaleShift
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.linmap_emb.LinMapEmbedding
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.embedding_avg.EmbeddingAverage
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.attention_emb.AttentionEmbedding
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.magnitude_scaling.MagnitudeScaling
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.methods.dropoutnet.DropoutNet
    options: { show_root_heading: true, heading_level: 3 }

## Config and results

::: warmtransfer.bench.config.BenchConfig
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.bench.results
    options:
      show_root_heading: true
      heading_level: 3

## Metrics

::: warmtransfer.metrics
    options:
      show_root_heading: true
      heading_level: 3

## Benchmark

::: warmtransfer.bench.runner.BenchmarkRunner
    options: { show_root_heading: true, heading_level: 3 }

::: warmtransfer.bench.splitters.pseudo_cold.PseudoColdSplitter
    options: { show_root_heading: true, heading_level: 3 }
