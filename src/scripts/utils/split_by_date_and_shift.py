def split_and_shift_data(data, date_split: str, target_col='return_raw', group_col='symbol'):
    """
    Splits the dataset into train and test based on timestamp and creates lagged target returns.

    Args:
        data (pd.DataFrame): Preprocessed dataset with timestamp index.
        date_split (str): Cutoff date for train/test split
        target_col (str): Column to be lagged as the target
        group_col (str): Group by column for lagging

    Returns:
        train_df (pd.DataFrame): Training set with target_return column.
        test_df (pd.DataFrame): Testing set with target_return column.
    """
    # Split
    train_df = data[data.index <= date_split].copy()
    test_df = data[data.index > date_split].copy()

    # Shift target column by 1 within each group
    train_df['target_return'] = train_df.groupby(group_col)[target_col].shift(-1)
    test_df['target_return'] = test_df.groupby(group_col)[target_col].shift(-1)

    # Drop resulting NA rows (at end of group)
    train_df.dropna(subset=['target_return'], inplace=True)
    test_df.dropna(subset=['target_return'], inplace=True)

    return train_df, test_df