import pandas as pd
import required_fields as rf
import handler as h
import work_with_model as wm
import scores
import work_with_file as wf

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

test_scores = []
roc_auc_scores = []
target_numbers, lgbm_params, cat_cols = rf.get_required_fields()
delete_columns = ["id", 'branch_code', 'city', 'index_city_code']

for target_number in target_numbers:
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')

    h.bring_values_to_the_median(train_df)

    train_df.drop(delete_columns + [target_number, "total_target"], axis=1, inplace=True)
    train_df[cat_cols] = train_df[cat_cols].astype("category")

    if target_number == "target_2":
        X_target = train_df.drop("target_1", axis=1)
        y_target = train_df.target_1
    else:
        X_target = train_df.drop("target_2", axis=1)
        y_target = train_df.target_2

    x_train, x_val, y_train, y_val = train_test_split(X_target, y_target,
                                                      test_size=0.2,
                                                      random_state=42)

    y_pred, model = wm.training_model(lgbm_params, x_train, y_train, x_val)

    test_df[cat_cols] = test_df[cat_cols].astype("category")
    test_score = wm.get_predict_for_test_file(model, test_df, delete_columns)

    if target_number == "target_2": test_scores = test_score
    else: wf.write_final_scores_in_csv_file(test_df.id, scores.get_final_scores(test_scores, test_score))

    roc_auc = roc_auc_score(y_val, y_pred)
    roc_auc_scores.append(roc_auc)


average_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
print(f'Средняя точность: {average_roc_auc}')

