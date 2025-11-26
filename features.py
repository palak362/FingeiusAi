import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # unify date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # amount cleanup
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    # fill missing text
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].fillna("UNKNOWN")
    import pandas as pd


    def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # unify date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # amount cleanup
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

        # fill missing text
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].fillna("UNKNOWN")

        df = df.drop_duplicates()
        return df



    def txn_aggregations(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # group by customer placeholder - if no customer id, create 'single_user'
        if "customer_id" not in df.columns:
            df["customer_id"] = "single_user"

        # time features
        df["month"] = df["date"].dt.to_period("M")

        agg = (
            df.groupby("customer_id")
            .agg(
                total_txn=("amount", "sum"),
                import pandas as pd


                def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
                    df = df.copy()

                    # unify date column
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")

                    # amount cleanup
                    if "amount" in df.columns:
                        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

                    # fill missing text
                    for c in df.select_dtypes(include="object").columns:
                        df[c] = df[c].fillna("UNKNOWN")

                    df = df.drop_duplicates()
                    return df



                def txn_aggregations(df: pd.DataFrame) -> pd.DataFrame:
                    df = df.copy()

                    # group by customer placeholder - if no customer id, create 'single_user'
                    if "customer_id" not in df.columns:
                        df["customer_id"] = "single_user"

                    # time features
                    df["month"] = df["date"].dt.to_period("M")

                    agg = (
                        df.groupby("customer_id")
                        .agg(
                            total_txn=("amount", "sum"),
                            txn_count=("amount", "count"),
                            avg_txn=("amount", "mean"),
                            median_txn=("amount", "median"),
                            max_txn=("amount", "max"),
                            min_txn=("amount", "min"),
                            std_txn=("amount", "std"),
                        )
                        .reset_index()
                    )

                    # ratio features
                    agg["avg_to_median_ratio"] = agg["avg_txn"] / (agg["median_txn"] + 1e-6)

                    return agg



                def create_model_features(df: pd.DataFrame) -> pd.DataFrame:
                    # combine txn aggregates and rolling features (example)
                    df_clean = basic_clean(df)
                    agg = txn_aggregations(df_clean)

                    # flag high weekend spending
                    df_clean["weekday"] = df_clean["date"].dt.weekday
                    weekend = (
                        df_clean[df_clean["weekday"] >= 5]
                        .groupby("customer_id")["amount"]
                        .sum()
                        .reset_index()
                        .rename(columns={"amount": "weekend_spend"})
                    )

                    agg = agg.merge(weekend, on="customer_id", how="left")
                    agg["weekend_spend"] = agg["weekend_spend"].fillna(0)

                    # create simple risk proxy: high max_txn relative to avg
                    agg["max_to_avg_ratio"] = agg["max_txn"] / (agg["avg_txn"] + 1e-6)
                    agg = agg.fillna(0)

                    return agg
