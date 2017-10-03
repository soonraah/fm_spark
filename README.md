# fm_spark

FactorizationMachines by Apache Spark

## Features

- Support features that have many dimensions (<= `Int.MaxValue`)
    - In training and prediction, samples and features are distributed
- Support APIs of `apache.spark.ml`
    - be able to run cross validation
- `DataFrame` based implementation
- Support only optimization by mini-batch SGD now
