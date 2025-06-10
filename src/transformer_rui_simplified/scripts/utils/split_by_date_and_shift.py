def split_and_shift_data(data, date_split: str, target_col='return_raw', group_col='symbol'):
    """
    Splits dataset into train/test, shifts forward returns, scales targets (×100), and normalizes them.

    Args:
        data (pd.DataFrame): Dataset with datetime index.
        date_split (str): Cutoff date for train/test split.
        target_col (str): Column containing raw returns.
        group_col (str): Column used to group time series (e.g., stock symbol).

    Returns:
        train_df (pd.DataFrame): Training data with normalized 'target_return'.
        test_df (pd.DataFrame): Testing data with normalized 'target_return'.
        target_mean (float): Mean of train target_return before normalization.
        target_std (float): Std of train target_return before normalization.
    """
    # Split
    train_df = data[data.index <= date_split].copy()
    test_df = data[data.index > date_split].copy()

    for df in [train_df, test_df]:
        group = df.groupby(group_col)[target_col]
        df['target_return'] = group.shift(-1)
        df.dropna(subset=['target_return'], inplace=True)

        # Scale by 100 (returns to percentage units)
        df['target_return'] *= 100

    # Normalize using training statistics
    target_mean = train_df['target_return'].mean()
    target_std = train_df['target_return'].std()

    train_df['target_return'] = (train_df['target_return'] - target_mean) / (target_std + 1e-8)
    test_df['target_return'] = (test_df['target_return'] - target_mean) / (target_std + 1e-8)

    return train_df, test_df, target_mean, target_std
