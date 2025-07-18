import polars as pl

def expand_grid(_lazy=False, _categorical=False, **kwargs: list) -> pl.DataFrame | pl.LazyFrame:
    """
    Create a Cartesian product (grid) of input lists as a Polars DataFrame or LazyFrame.

    Each keyword argument represents a factor (column) in the grid. The function returns
    a DataFrame (by default) where each row corresponds to one combination of values from
    the input lists. A unique row identifier `row_id` is included as the first column.

    Parameters:
        _lazy (bool, default=False): If True, return a Polars LazyFrame instead of a DataFrame.
                                     This is recommended for large input lists where the full
                                     Cartesian product may exceed memory limits.
        _categorical (bool, default=False): If True, convert string columns to pl.Categorical 
                                            data type.
        **kwargs (list): Named input lists to form the grid. Each key becomes a column name.
                         All values must be of type `list`.

    Returns:
        pl.DataFrame | pl.LazyFrame: A table representing the Cartesian product of the inputs,
                                     with a `row_id` column as a unique identifier for each row.

    Raises:
        ValueError: If any keyword is named 'row_id'.
        TypeError: If any value in kwargs is not a list.

    Notes:
        The number of rows in the resulting grid is the product of the lengths of all input lists.
        If the result is expected to be large, consider setting `lazy=True` to avoid memory issues.

    Example:
        >>> from polarsgrid import expand_grid
        >>> expand_grid(a=[1, 2], b=["x", "y"])
        shape: (4, 3)
        ┌────────┬─────┬─────┐
        │ __id__ ┆ a   ┆ b   │
        │ ---    ┆ --- ┆ --- │
        │ i64    ┆ i64 ┆ str │
        ╞════════╪═════╪═════╡
        │ 0      ┆ 1   ┆ x   │
        │ 1      ┆ 2   ┆ x   │
        │ 2      ┆ 1   ┆ y   │
        │ 3      ┆ 2   ┆ y   │
        └────────┴─────┴─────┘
    """

    if any(k == "row_id" for k in kwargs):
        raise ValueError("Keyword arguments are not allowed to be named 'row_id'")
    if not all([isinstance(v, list) for v in kwargs.values()]):
        raise TypeError("All keyword arguments should be of type 'list.")
    ldf = pl.DataFrame({k: [v] for k, v in kwargs.items()}).lazy()

    # compute length (product of all)
    nrow = 1
    for v in kwargs.values():
        nrow *= len(v)

    # explode (unnest_longer) factor columns
    for k in reversed(kwargs):
        ldf = ldf.explode(pl.col(k))

    # convert str columns to categorical
    if _categorical:
        ldf = ldf.with_columns(pl.selectors.string().cast(pl.Categorical))

    # add id at the start
    ldf = ldf.with_columns(pl.int_range(nrow).alias("row_id")).select(
        pl.col.row_id, pl.exclude("row_id")
    )

    return ldf if _lazy else ldf.collect()
