def aggregate_series(values, aggregate_number, show_std=True):
    import numpy as np
    import pandas as pd

    if aggregate_number <= 0:
        raise ValueError("aggregate_number must be > 0")

    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    values = pd.to_numeric(values, errors="coerce")

    window_size = aggregate_number

    if len(values) < window_size:
        raise ValueError(
            f"Not enough values ({len(values)}) "
            f"for window size {window_size}"
        )

    # Trailing rolling window (no centering)
    mean_values = values.rolling(
        window=window_size,
        min_periods=window_size
    ).mean()

    std_values = None
    if show_std:
        std_values = values.rolling(
            window=window_size,
            min_periods=window_size
        ).std()

    # valid only where full window exists
    valid_mask = mean_values.notna()

    mean_values = mean_values[valid_mask].to_numpy()

    if show_std:
        std_values = std_values[valid_mask].to_numpy()

    valid_indices = np.where(valid_mask)[0]

    return mean_values, std_values, valid_indices