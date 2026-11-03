from abc import abstractmethod
from typing import Any, List, Callable, Type

import pandas as pd

from .backend import load_concepts


class Step():
    """Base class for a transformation step
    """
    def __init__(self, cache=False) -> None:
        self.cache = cache
        self._performed = False

    def perform(self, input: Any = None, force: bool = False):
        """Perform the specified transformation step on the provided input.

        If caching is activated, the result of the step is stored (but not copied,
        so later changes will also change the stored information.)

        Args:
            input: Object used as input to the transformation. Defaults to None.
            force: Should the result be recalculated, even if a cached result is available? Defaults to False.

        Returns:
            Transformed input
        """
        if self.cache and self._performed and not force:
            return self.res
        else: 
            res = self.do_perform(input)
            self._performed = True
        
        if self.cache:
            self.res = res
        return res
    
    @abstractmethod
    def do_perform(self, input=None):
        """Called by `perform` to apply the actual transformation.

        Must be implemented by each class inheriting from Step.

        Args:
            input: Object used as input to the transformation. Defaults to None.
        """
        pass

    def __repr__(self) -> str:
        return self.desc


class InputStep(Step):
    """_summary_

    Args:
        Step: _description_
    """
    def __init__(self, x, **kwargs) -> None:
        super().__init__(cache=False)
        self.desc = f"Use pre-loaded variable."
        self.x = x
    
    def do_perform(self, input):
        if input is not None:
            raise ValueError(f'Input step does not accept input, got {input}')
        return self.x

class LoadStep(Step):
    def __init__(self, concept, src, cache=False, **kwargs) -> None:
        super().__init__(cache)
        self.desc = f"Load clinical concept {concept} from {src}."
        self.concept = concept
        if not isinstance(self.concept, list):
            self.concept = [self.concept]
        self.src = src
        self.kwargs = kwargs

    def do_perform(self, input):
        if input is not None:
            raise ValueError(f'Load step does not accept input, got {input}')

        return load_concepts(self.concept, self.src, **self.kwargs)


class FilterStep(Step):
    def __init__(self, col: str, condition: Callable, cache=False) -> None:
        super().__init__(cache)
        self.desc = f"Filter table based on {col}."
        self.col = col
        self.condition = condition

    def do_perform(self, input: pd.DataFrame = None):
        if not isinstance(input, pd.DataFrame):
            raise ValueError(f'Filter step requires a pandas.DataFrame as input, got {input}.')

        return input[self.condition(input[self.col])]
    
class DropStep(Step):
    def __init__(self, col: str | List[str], cache=False) -> None:
        super().__init__(cache)
        self.desc = f"Drop columns {col}."
        self.col = col

    def do_perform(self, input: pd.DataFrame = None):
        if not isinstance(input, pd.DataFrame):
            raise ValueError(f'Drop step requires a pandas.DataFrame as input, got {input}.')

        return input.drop(self.col, axis=1)
    
class RenameStep(Step):
    def __init__(self, old: str | List[str], new: str | List[str], cache=False) -> None:
        super().__init__(cache)
        self.desc = f"Rename columns {old} to {new}."
        self.old = [old] if isinstance(old, str) else old
        self.new = [new] if isinstance(new, str) else new

    def do_perform(self, input: pd.DataFrame = None):
        if not isinstance(input, pd.DataFrame):
            raise ValueError(f'Rename step requires a pandas.DataFrame as input, got {input}.')
        mapper = {o: n for o, n in zip(self.old, self.new)}
        return input.rename(columns=mapper)


class AggStep(Step):
    def __init__(self, by: str | List[str], func: str | Callable, col: str | List | None = None, cache=False) -> None:
        super().__init__(cache)
        funcname = func if isinstance(func, str) else func.__name__
        self.desc = f'Aggregate table over {by} using {funcname}.'
        self.by = by
        self.col = col
        self.func = func

    def do_perform(self, input: pd.DataFrame = None):
        if not isinstance(input, pd.DataFrame):
            raise ValueError(f'Aggregation step requires a pandas.DataFrame as input, got {input}.')
        col = self.col
        if col is None: 
            col = [c for c in input.columns if c not in self.by]

        return input.groupby(self.by)[col].aggregate(self.func).reset_index()

class TransformStep(Step):
    def __init__(self, col: str | List, func: str | Callable, cache=False) -> None:
        super().__init__(cache)
        funcname = func if isinstance(func, str) else func.__name__
        self.desc = f'Tranform column(s) {col} using {funcname}.'
        self.col = [col] if isinstance(col, str) else col
        self.func = func

    def do_perform(self, input: pd.DataFrame = None):
        if not isinstance(input, pd.DataFrame):
            raise ValueError(f'Transformation step requires a pandas.DataFrame as input, got {input}.')
        # TODO: make more elegant
        res = input.copy()
        for c in self.col:
            res[c] = self.func(res[c])
        return res


class CustomStep(Step):
    def __init__(self, func: Callable, cache=False) -> None:
        super().__init__(cache)
        self.desc = f'Custom step applying {func.__name__}.'
        self.func = func
    
    def do_perform(self, input=None):
        return self.func(input)
    
class CombineStep(Step):
    def __init__(self, steps: Type["Pipeline"] | Step, func: Callable, cache=False, **kwargs) -> None:
        super().__init__(cache)
        self.desc = f'Combine steps using {func.__name__}.'
        self.steps = steps
        self.func = func
        self.kwargs = kwargs
    
    def do_perform(self, input=None):
        res = []
        for step in self.steps:
            if isinstance(step, Pipeline):
                res.append(step.apply(input))
            else: 
                res.append(step.perform(input))
        return self.func(res, **self.kwargs)
    

class Pipeline():
    def __init__(self, desc) -> None:
        self.desc = desc
        self.steps = []

    def add_step(self, x: Step | List[Step]):
        """Add one or more transformation step to the pipeline

        Transformation steps are applied sequentially

        Args:
            x: single transformation step or list of steps
        """
        if isinstance(x, Step):
            x = [x]
        self.steps += x

    def apply(self, input: Any = None) -> Any:
        """Apply all transformation steps sequentially

        Args:
            input: Optional input to the first step, if required. All subsequent steps
             receive the output of the previous step. Defaults to None.

        Returns:
            Result of the final step
        """
        res = input
        for step in self.steps:
            res = step.perform(res)
        return res
    
    def __repr__(self) -> str:
        repr = f"<Pipeline>: {self.desc}\n"
        for i, step in enumerate(self.steps):
            repr += f"   {i+1}. {str(step)}\n"
        return repr
