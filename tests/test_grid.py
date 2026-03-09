import pytest
import polars as pl
from polarsgrid import expand_grid, expand_grid_lazy


@pytest.mark.parametrize(
    "expand_func, type", [(expand_grid, pl.DataFrame), (expand_grid_lazy, pl.LazyFrame)]
)
def test_grid_basic(expand_func, type):
    grid: pl.DataFrame | pl.LazyFrame = expand_func(
        a=range(4), b=["red", "green", "blue"], c=(0.1, 0.5, 0.9)
    )
    assert isinstance(grid, type)
    if isinstance(grid, pl.LazyFrame):
        grid: pl.DataFrame = grid.collect()
    assert grid["b"].dtype == pl.String
    assert grid.shape == (36, 3)


@pytest.mark.parametrize(
    "expand_func, type", [(expand_grid, pl.DataFrame), (expand_grid_lazy, pl.LazyFrame)]
)
def test_grid_options(expand_func, type):
    grid: pl.DataFrame | pl.LazyFrame = expand_func(
        a=range(4), b=["red", "green", "blue"], c=(0.1, 0.5, 0.9), _categorical=True, _row_id=True
    )
    assert isinstance(grid, type)
    if isinstance(grid, pl.LazyFrame):
        grid: pl.DataFrame = grid.collect()
    assert grid["b"].dtype == pl.Categorical
    assert grid[:, 0].name == "row_id"
    assert grid.shape == (36, 4)
    with pytest.raises(ValueError):
        grid: pl.DataFrame | pl.LazyFrame = expand_func(
            row_id=range(4),
            b=["red", "green", "blue"],
            c=[0.1, 0.5, 0.9],
            _categorical=True,
            _row_id=True,
        )
