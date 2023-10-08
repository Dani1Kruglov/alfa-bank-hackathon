def get_required_fields():
    return (
        ["target_2", "target_1"],
        {
            'boosting_type': 'gbdt',  # Метод бустинга (градиентный бустинг)
            'objective': 'binary',  # Тип задачи (бинарная классификация)
            'num_leaves': 31,  # Максимальное количество листьев в дереве
            'max_depth': 6,  # Максимальная глубина дерева
            'learning_rate': 0.1,  # Скорость обучения (learning rate)
            'n_estimators': 100,  # Количество деревьев
            'subsample_for_bin': 20000,  # Количество выборок для построения гистограмм
            'min_child_samples': 20,  # Минимальное количество образцов в листе
            'colsample_bytree': 0.8,  # Доля признаков для построения каждого дерева
            'reg_alpha': 0.1,  # L1 регуляризация
            'reg_lambda': 0.1,  # L2 регуляризация
            'random_state': 42  # Зафиксированный random state для воспроизводимости
        },
        [
            'channel_code', 'city_type', 'ogrn_month', 'ogrn_year', 'okved',
            'segment'])


