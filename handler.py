import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype


def bring_values_to_the_median(train_df):
    first_row = train_df.head(1)

    for index_for_check in first_row:
        if (is_numeric_dtype(first_row[
                                 index_for_check])) and index_for_check != "target_1" and index_for_check != "target_2" and index_for_check != "total_target":
            lower_bound = train_df[index_for_check].mean() - 3 * train_df[index_for_check].std()
            upper_bound = train_df[index_for_check].mean() + 3 * train_df[index_for_check].std()
            train_df[index_for_check] = np.where(train_df[index_for_check] < lower_bound, lower_bound,
                                                 train_df[index_for_check])
            train_df[index_for_check] = np.where(train_df[index_for_check] > upper_bound, upper_bound,
                                                 train_df[index_for_check])
    return 1
