from src.config import DROP_COLS, PRUNE_COLS


def get_feature_columns(df, prune_weak=False):

    excluded = set(DROP_COLS)
    if prune_weak:
        excluded |= set(PRUNE_COLS)
    return [c for c in df.columns if c not in excluded]

