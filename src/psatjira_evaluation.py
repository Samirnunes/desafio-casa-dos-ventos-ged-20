from data_import import get_dataset_only_time, get_dataset_with_cfs_gefs
from split import split_train_test_by_size
from models import psatjira_models, psatjira_models_names
from precipitation_model_evaluator import PrecipitationEvaluator


def psatjira_evaluation():
    plant = "PSATJIRA"
    root_path = "../evaluation_results/PSATJIRA/"

    evaluators = []
    for model in psatjira_models:
        evaluators.append(PrecipitationEvaluator(model))

    df = get_dataset_only_time(plant)
    X_train, X_test, y_train, y_test = split_train_test_by_size(df, 100)
    for evaluator, name in zip(evaluators, psatjira_models_names):
        suffix = f"{name}_time/"
        save_path = root_path + suffix
        evaluator.cross_validation(X_train, y_train, save_path)
        evaluator.test(X_train, X_test, y_train, y_test, save_path)

    dfcg = get_dataset_with_cfs_gefs(plant)
    X_traincg, X_testcg, y_traincg, y_testcg = split_train_test_by_size(dfcg, 50)
    for evaluator, name in zip(evaluators, psatjira_models_names):
        suffix = f"{name}_cfs_gefs/"
        save_path = root_path + suffix
        evaluator.cross_validation(X_traincg, y_traincg, save_path)
        evaluator.test(X_traincg, X_testcg, y_traincg, y_testcg, save_path)


if __name__ == "__main__":
    psatjira_evaluation()
