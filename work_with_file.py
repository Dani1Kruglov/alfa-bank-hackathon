import pandas as pd


def write_final_scores_in_csv_file(ids_column, final_test_score):
    ids_column.to_csv('total_target.csv', index=False)
    sample_submission_df = pd.read_csv('total_target.csv')
    sample_submission_df["score"] = final_test_score
    sample_submission_df.head()
    sample_submission_df.to_csv('total_target.csv', index=False)