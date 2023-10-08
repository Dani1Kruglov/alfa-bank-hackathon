from lightgbm import LGBMClassifier


def training_model(lgbm_params, x_train, y_train, x_val):
    model = LGBMClassifier(**lgbm_params, verbose=-1, importance_type='gain')
    model.fit(x_train, y_train)
    return model.predict_proba(x_val)[:, 1], model


def get_predict_for_test_file(model, test_df, delete_columns):
    return model.predict_proba(test_df.drop(delete_columns, axis=1))[:, 1]
