import numpy as np

from data_import import get_dataset_only_time, get_dataset_with_cfs_gefs
from split import split_train_test_by_size
from models import psatitp_models, psatitp_models_names
from precipitation_model_evaluator import PrecipitationEvaluator


def psatitp_evaluation():
    plant = "PSATJIRA"
    root_path = "../evaluation_results/PSATITP/"

    evaluators = []
    for model in psatitp_models:
        evaluators.append(PrecipitationEvaluator(model))

    df = get_dataset_only_time(plant)
    X_train, X_test, y_train, y_test = split_train_test_by_size(df, 100)
    mask = np.ones(X_train.shape[1], dtype=bool)
    mask[15:45] = False  # lag 16 to 45
    X_train = X_train.loc[:, mask]
    X_test = X_test.loc[:, mask]

    for evaluator, name in zip(evaluators, psatitp_models_names):
        suffix = f"{name}_time/"
        save_path = root_path + suffix
        evaluator.cross_validation(X_train, y_train, save_path)
        evaluator.test(X_train, X_test, y_train, y_test, save_path)

    dfcg = get_dataset_with_cfs_gefs(plant)
    X_cg_train, X_cg_test, y_cg_train, y_cg_test = split_train_test_by_size(dfcg, 50)
    mask = np.ones(X_cg_train.shape[1], dtype=bool)
    mask[15:45] = False  # lag 16 to 45
    mask[58:88] = False  # lag 16 to 45 from cfs
    X_cg_train = X_cg_train.loc[:, mask]
    X_cg_test = X_cg_test.loc[:, mask]
    X_cg_train = X_cg_train.astype({f"m_{i}": bool for i in range(2, 13)})
    X_cg_test = X_cg_test.astype({f"m_{i}": bool for i in range(2, 13)})
    for evaluator, name in zip(evaluators, psatitp_models_names):
        suffix = f"{name}_cfs_gefs/"
        save_path = root_path + suffix
        evaluator.cross_validation(X_cg_train, y_cg_train, save_path)
        evaluator.test(X_cg_train, X_cg_test, y_cg_train, y_cg_test, save_path)


if __name__ == "__main__":
    psatitp_evaluation()
