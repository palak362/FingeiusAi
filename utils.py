def top_n_merchants(df, n=10):
    """Return top `n` merchants by total amount from a transactions DataFrame."""
    return df.groupby("description")["amount"].sum().sort_values(ascending=False).head(n)
