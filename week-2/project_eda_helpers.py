import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

TARGET = "SalePrice"


PRETTY_NAMES = {
    "SalePrice": "Sale Price",
    "Overall Qual": "Overall Quality",
    "Gr Liv Area": "Above Ground Living Area",
    "Garage Cars": "Garage Cars",
    "Garage Area": "Garage Area",
    "Total Bsmt SF": "Total Basement Area",
    "1st Flr SF": "1st Floor Area",
    "Year Built": "Year Built",
    "Full Bath": "Full Bathrooms Above Ground",
    "Year Remod/Add": "Year Remodel",
    "Mas Vnr Area": "Masonry Veneer Area",
    "TotRms AbvGrd": "Rooms Above Ground",
    "Fireplaces": "Number of Fireplaces",
    "BsmtFin SF 1": "Finished Basement Area Type 1",
    "Garage Yr Blt": "Garage Year Built",
    "Exter Qual": "Exterior Quality",
    "Kitchen Qual": "Kitchen Quality",
    "Bsmt Qual": "Basement Quality",
    "Fireplace Qu": "Fireplace Quality",
    "Sale Condition": "Sale Condition",
    "MS Zoning": "MS Zoning",
    "Exterior 1st": "Exterior 1st",
    "Exterior 2nd": "Exterior 2nd",
    "Garage Type": "Garage Type",
    "Neighborhood": "Neighborhood",
    "Pool QC": "Pool Quality",
    "Pool Area": "Pool Area",
}

def pretty(col):
    return PRETTY_NAMES.get(col, col)


def apply_k_formatter(ax, axis="x"):
    def k_formatter(x, pos):
        return f"{int(x / 1000)}k"
    
    formatter = ticker.FuncFormatter(k_formatter)

    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)


def make_axes(n_plots, n_cols=3, figsize=(16, 8), sharey=False):
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=sharey
    )
    axes = np.array(axes).flatten()

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    return fig, axes

def plot_target_distribution(df, target=TARGET):
    plt.figure(figsize=(16, 9))

    ax = sns.histplot(
        data=df,
        x=target,
        kde=True
    )

    apply_k_formatter(ax, axis="x")

    plt.title("Distribution of Sale Price")
    plt.xlabel(pretty(target))
    plt.ylabel("Number of Houses")

    plt.show()

def get_top_correlations(df, target=TARGET, threshold=0.4):
    corr_with_target = (
        df
        .corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(ascending=False)
    )

    top_corr = corr_with_target[
        (corr_with_target > threshold) | 
        (corr_with_target < -threshold)
    ]

    top_corr_df = (
        top_corr
        .reset_index()
        .rename(columns={
            "index": "Variable",
            target: "Correlation"
        })
    )

    top_corr_df["Variable"] = top_corr_df["Variable"].map(pretty)

    return top_corr_df

def plot_top_correlations(df, target=TARGET, threshold=0.4):
    top_corr = get_top_correlations(
        df=df,
        target=target,
        threshold=threshold
    )

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=top_corr,
        y="Variable",
        x="Correlation"
    )

    ax.set(ylabel=None)

    plt.title("Top Numerical Features Correlated with Sale Price", fontsize=14)
    plt.show()

    return top_corr

def plot_numeric_distributions(
    df,
    features,
    n_cols=5,
    figsize=(20, 10),
    k_format_cols=None,
    title=None
):
    if k_format_cols is None:
        k_format_cols = []

    fig, axes = make_axes(
        n_plots=len(features),
        n_cols=n_cols,
        figsize=figsize
    )

    for ax, col in zip(axes, features):
        sns.histplot(
            data=df,
            x=col,
            ax=ax
        )

        ax.set_xlabel(pretty(col))
        ax.set_ylabel("Count")

        if col in k_format_cols:
            apply_k_formatter(ax, axis="x")

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()

def get_categorical_importance(df, categorical_cols, target=TARGET, min_count=10):
    cat_importance = []

    for col in categorical_cols:
        grouped = df.groupby(col)[target].agg(["count", "median"])

        grouped = grouped[grouped["count"] >= min_count]

        if len(grouped) > 1:
            score = grouped["median"].max() - grouped["median"].min()
        else:
            score = 0

        cat_importance.append({
            "Variable": col,
            "Unique values": df[col].nunique(),
            "Useful categories": len(grouped),
            "Median SalePrice range": score
        })

    cat_importance_df = pd.DataFrame(cat_importance).sort_values(
        "Median SalePrice range",
        ascending=False
    )

    return cat_importance_df

def plot_categorical_counts(
    df,
    features,
    n_cols=5,
    figsize=(20, 10),
    orders=None,
    horizontal_cols=None,
    title=None
):
    if orders is None:
        orders = {}

    if horizontal_cols is None:
        horizontal_cols = []

    fig, axes = make_axes(
        n_plots=len(features),
        n_cols=n_cols,
        figsize=figsize
    )

    for ax, col in zip(axes, features):
        order = orders.get(col, df[col].value_counts().index)

        if col in horizontal_cols:
            sns.countplot(
                data=df,
                y=col,
                order=order,
                ax=ax
            )

            ax.set_xlabel("Count")
            ax.set_ylabel(pretty(col))

        else:
            sns.countplot(
                data=df,
                x=col,
                order=order,
                ax=ax
            )

            ax.set_xlabel(pretty(col))
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()


def plot_numeric_vs_target(
    df,
    features,
    target=TARGET,
    n_cols=2,
    figsize=(14, 9)
):
    fig, axes = make_axes(
        n_plots=len(features),
        n_cols=n_cols,
        figsize=figsize,
        sharey=True
    )

    for i, (ax, col) in enumerate(zip(axes, features)):
        corr = df[[col, target]].corr().iloc[0, 1]

        sns.regplot(
            data=df,
            x=col,
            y=target,
            ax=ax,
            scatter_kws={
                "alpha": 0.2,
                "s": 35,
                "edgecolor": "none"
            },
            line_kws={
                "linewidth": 2
            },
            ci=None
        )

        ax.set_title(
            f"{pretty(col)} vs {pretty(target)}\nCorrelation: {corr:.2f}",
            fontsize=12
        )

        ax.set_xlabel(pretty(col), fontsize=10)
        ax.set_ylabel(pretty(target) if i % n_cols == 0 else "")

        apply_k_formatter(ax, axis="y")

    fig.suptitle(
        "Relationship Between Key Numerical Features and Sale Price",
        fontsize=16,
        y=1.02
    )

    plt.tight_layout()
    plt.show()


def plot_boxplots_vs_target(
    df,
    features,
    target=TARGET,
    n_cols=2,
    figsize=(14, 9),
    title=None
):
    fig, axes = make_axes(
        n_plots=len(features),
        n_cols=n_cols,
        figsize=figsize,
        sharey=True
    )

    for i, (ax, col) in enumerate(zip(axes, features)):
        sns.boxplot(
            data=df,
            x=col,
            y=target,
            ax=ax
        )

        ax.set_title(
            f"{pretty(target)} by {pretty(col)}",
            fontsize=12
        )

        ax.set_xlabel(pretty(col), fontsize=10)
        ax.set_ylabel(pretty(target) if i % n_cols == 0 else "")

        apply_k_formatter(ax, axis="y")

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()


def plot_pairplot_by_price_category(
    df,
    features,
    target=TARGET,
    q=3,
    labels=None
):
    if labels is None:
        labels = ["Low", "Medium", "High"]

    pairplot_df = df[features + [target]].copy()

    pairplot_df["Price Category"] = pd.qcut(
        pairplot_df[target],
        q=q,
        labels=labels
    )

    pairplot_df = pairplot_df.drop(columns=[target])

    pairplot_df = pairplot_df.rename(
        columns={col: pretty(col) for col in pairplot_df.columns}
    )

    g = sns.pairplot(
        pairplot_df,
        hue="Price Category",
        height=2.4,
        diag_kind="kde"
    )

    g.fig.suptitle(
        "Relationships Between Key Features by Price Category",
        y=1.03,
        fontsize=16
    )

    plt.show()


def plot_percentage_crosstab_heatmap(
    df,
    row_col,
    col_col,
    figsize=(14, 8),
    cmap="Blues",
    title=None
):
    cross_tab = pd.crosstab(
        df[row_col],
        df[col_col],
        normalize="index"
    ) * 100

    annot_labels = cross_tab.map(
        lambda x: f"{x:.0f}%" if x != 0 else ""
    )

    plt.figure(figsize=figsize)

    sns.heatmap(
        cross_tab,
        cmap=cmap,
        annot=annot_labels,
        fmt=""
    )

    if title is None:
        title = f"{pretty(col_col)} Distribution by {pretty(row_col)} (%)"

    plt.title(title)
    plt.xlabel(pretty(col_col))
    plt.ylabel(pretty(row_col))

    plt.show()

    return cross_tab


def plot_pos_neg_corr_heatmap(
    df,
    target=TARGET,
    n_positive=10,
    n_negative=10,
    figsize=(14, 10)
):
    corr_matrix = df.corr(numeric_only=True)
    corr_with_target = corr_matrix[target].drop(target)

    positive_features = (
        corr_with_target
        .sort_values(ascending=False)
        .head(n_positive)
        .index
    )

    negative_features = (
        corr_with_target
        .sort_values(ascending=True)
        .head(n_negative)
        .index
    )

    selected_cols = [target] + positive_features.tolist() + negative_features.tolist()

    plt.figure(figsize=figsize)

    sns.heatmap(
        df[selected_cols].corr(),
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap of Most Positive and Negative Features Related to Sale Price")
    plt.show()


def get_high_corr_pairs(df, numeric_cols, threshold=0.8):
    corr_matrix = df[numeric_cols].corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = (
        upper_triangle
        .stack()
        .sort_values(ascending=False)
    )

    return high_corr_pairs[high_corr_pairs > threshold]


def check_structured_missingness(df, missing_col, condition_col, expected_value=0):
    missing_mask = df[missing_col].isna()

    percent = (
        df.loc[missing_mask, condition_col]
        .eq(expected_value)
        .mean()
        * 100
    )

    print(
        f"{missing_col}: {percent:.2f}% of missing values correspond to "
        f"{condition_col} = {expected_value}"
    )

    return percent


def check_multiple_structured_missingness(
    df,
    missing_cols,
    condition_col,
    expected_value=0
):
    results = {}

    for col in missing_cols:
        results[col] = check_structured_missingness(
            df=df,
            missing_col=col,
            condition_col=condition_col,
            expected_value=expected_value
        )

    return pd.Series(results).sort_values(ascending=False)


def check_missingness_explanation(df, missing_col, condition, explanation):
    missing = df[missing_col].isna()
    total = missing.sum()
    explained = (missing & condition).sum()
    unexplained = total - explained

    return {
        "Variable": missing_col,
        "Missing Count": total,
        "Explained Count": explained,
        "Explained %": round(explained / total * 100, 2) if total > 0 else 100,
        "Unexplained Count": unexplained,
        "Explanation": explanation
    }


def get_missingness_explanation_table(df, conditions, missingness_groups):
    results = [
        check_missingness_explanation(
            df=df,
            missing_col=col,
            condition=conditions[explanation],
            explanation=explanation
        )
        for explanation, cols in missingness_groups.items()
        for col in cols
    ]

    return (
        pd.DataFrame(results)
        .sort_values(
            by=["Unexplained Count", "Explained %"],
            ascending=[False, True]
        )
    )


### АНАЛИЗ ВЫБРОСОВ И ОБЩЕГО КАЧЕСТВА ДАННЫХ


def get_iqr_outliers(df, cols, factor=1.5):
    results = []

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outliers = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ]

        results.append({
            "Variable": col,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": len(outliers),
            "Outlier %": round(len(outliers) / len(df) * 100, 2)
        })

    return (
        pd.DataFrame(results)
        .sort_values("Outlier Count", ascending=False)
    )


def get_unusual_year_values(df, year_cols, current_year=2026):
    results = []

    for col in year_cols:
        unusual = df[
            (df[col].notna()) &
            (
                (df[col] < 1800) |
                (df[col] > current_year)
            )
        ]

        results.append({
            "Variable": col,
            "Unusual Count": len(unusual),
            "Unusual Values": sorted(unusual[col].unique())
        })

    return pd.DataFrame(results)


def get_negative_values(df, cols):
    results = []

    for col in cols:
        negative_count = (df[col] < 0).sum()

        results.append({
            "Variable": col,
            "Negative Count": negative_count
        })

    return (
        pd.DataFrame(results)
        .sort_values("Negative Count", ascending=False)
    )
