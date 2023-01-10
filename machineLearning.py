import ray
from ray.data.preprocessors import StandardScaler

from ray.air.config import ScalingConfig
from ray.train.xgboost import XGBoostTrainer

from ray import tune
from ray.tune.tuner import Tuner, TuneConfig

from ray.train.batch_predictor import BatchPredictor
from ray.train.xgboost import XGBoostPredictor

#= WARNING : Library needed : pyarrow , xgboost_ray , tabulate =#

## == READ THE DATA == ##

# Load data.
dataset = ray.data.read_csv("Example_Dataset/iris.csv")

# Split data into train and validation.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# Create a test dataset by dropping the target column.
test_dataset = valid_dataset.drop_columns(cols=["variety"])

# Preprocess the data
preprocessor = StandardScaler(columns=["sepal.length", "sepal.width" , "petal.length" , "petal.width" ])

## == ----- == ##

## == SCALE THE MODEL TRAINING == ##

trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        # Number of workers to use for data parallelism.
        num_workers=2,
        # Whether to use GPU acceleration.
        use_gpu=False,
    ),
    label_column="variety",
    num_boost_round=1,
    params={
        # XGBoost specific params
        "objective": "binary:logistic",
        # "tree_method": "gpu_hist",  # uncomment this to use GPUs.
        "eval_metric": ["logloss", "error"],
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
    preprocessor=preprocessor,
)
result = trainer.fit()
print(result.metrics)

## == ----- == ##

## == TUNE THE HYPERPARAMETERS == ##

param_space = {"params": {"max_depth": tune.randint(1, 9)}}
metric = "train-logloss"


tuner = Tuner(
    trainer,
    param_space=param_space,
    tune_config=TuneConfig(num_samples=1, metric=metric, mode="min"),
)
result_grid = tuner.fit()
best_result = result_grid.get_best_result()
print("Best result:", best_result)

## == ----- == ##

## == USE THE TRAINED MODEL FOR BATCH PREDICTION == ##

checkpoint = best_result.checkpoint

batch_predictor = BatchPredictor.from_checkpoint(checkpoint, XGBoostPredictor)

predicted_probabilities = batch_predictor.predict(test_dataset)
predicted_probabilities.show()

## == ----- == ##
