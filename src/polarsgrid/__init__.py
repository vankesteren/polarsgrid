import polars as pl
from collections.abc import Iterable


def expand_grid_lazy(*, _categorical=False, _row_id=False, **kwargs: Iterable) -> pl.LazyFrame:
    """
    Create a Cartesian product (grid) of inputs as a Polars LazyFrame.

    Each keyword argument represents a factor (column) in the grid. The function returns
    a LazyFrame (by default) where each row corresponds to one combination of values from
    the inputs. A unique row identifier `row_id` can be included as the first column.

    Parameters:
        _categorical (bool, default=False): If True, convert string columns to pl.Categorical
                                            data type.
        _row_id (bool, default=False): If True, add a column "row_id" at the start.
        **kwargs (Iterable): Named inputs to form the grid. Each key becomes a column name.
                             All values must be iterables (such as lists).
    Returns:
        pl.LazyFrame: A lazy table representing the Cartesian product of the inputs, optionally
                      with a `row_id` column as a unique identifier for each row.

    Raises:
        ValueError: If row ids are requested and any keyword arg is named 'row_id'.
        TypeError: If any value in kwargs is not an iterable.

    Notes:
        The number of rows in the resulting grid is the product of the lengths of all inputs.

    Example:
        >>> from polarsgrid import expand_grid_lazy
        >>> grid = expand_grid_lazy(a=range(2), b=["x", "y"])
        >>> grid.collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 0   ┆ x   │
        │ 1   ┆ x   │
        │ 0   ┆ y   │
        │ 1   ┆ y   │
        └─────┴─────┘
    """

    if _row_id and any(k == "row_id" for k in kwargs):
        raise ValueError("Keyword arguments are not allowed to be named 'row_id'")
    if not all([isinstance(v, Iterable) for v in kwargs.values()]):
        raise TypeError("All keyword arguments should be iterable (e.g., a list).")
    ldf = pl.DataFrame({k: [list(v)] for k, v in kwargs.items()}).lazy()

    # explode (unnest_longer) factor columns
    for k in reversed(kwargs):
        ldf = ldf.explode(k)

    if _categorical:
        # convert str columns to categorical
        ldf = ldf.with_columns(pl.selectors.string().cast(pl.Categorical))

    if _row_id:
        # add id at the start
        ldf = ldf.with_row_index("row_id")

    return ldf


def expand_grid(*, _categorical=False, _row_id=False, **kwargs: Iterable) -> pl.DataFrame:
    """
    Create a Cartesian product (grid) of inputs as a Polars DataFrame.

    Each keyword argument represents a factor (column) in the grid. The function returns
    a DataFrame where each row corresponds to one combination of values from
    the inputs. A unique row identifier `row_id` can be included as the first column.

    Parameters:
        _categorical (bool, default=False): If True, convert string columns to pl.Categorical
                                            data type.
        _row_id (bool, default=False): If True, add a column "row_id" at the start.
        **kwargs (Iterable): Named inputs to form the grid. Each key becomes a column name.
                             All values must be iterables (such as lists).

    Returns:
        pl.DataFrame: A table representing the Cartesian product of the inputs, optionally
                      with a `row_id` column as a unique identifier for each row.

    Raises:
        ValueError: If row ids are requested and any keyword arg is named 'row_id'.
        TypeError: If any value in kwargs is not an Iterable.

    Notes:
        The number of rows in the resulting grid is the product of the lengths of all inputs.
        If the result is expected to be large, consider expand_grid_lazy() to avoid memory issues.

    Example:
        >>> from polarsgrid import expand_grid
        >>> expand_grid(a=range(2), b=["x", "y"])
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 0   ┆ x   │
        │ 1   ┆ x   │
        │ 0   ┆ y   │
        │ 1   ┆ y   │
        └─────┴─────┘
    """
    return expand_grid_lazy(_categorical=_categorical, _row_id=_row_id, **kwargs).collect()
