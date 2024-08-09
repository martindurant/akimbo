import functools
from typing import Callable

import awkward as ak
from cudf import DataFrame, Series
from cudf.core.column.string import StringMethods

from awkward_pandas.ak_from_cudf import cudf_to_awkward as from_cudf
from awkward_pandas.mixin import Accessor
from awkward_pandas.strings import StringAccessor


class CudfStringAccessor(StringAccessor):
    def decode(self, encoding: str = "utf-8"):
        raise NotImplementedError("cudf does not support bytearray type, so we can't automatically identify them")

    def __getattr__(self, attr: str) -> Callable:
        attr = StringAccessor.method_name(attr)
        fn = getattr(StringMethods(self.accessor._obj), attr)

        @functools.wraps(fn)
        def f(*args, **kwargs):
            arr = fn(self.accessor._obj, *args, **kwargs)
            if isinstance(arr, ak.Array):
                return self.accessor.to_output(arr)
            return arr

        return f


class CudfAwkwardAccessor(Accessor):
    series_type = Series
    dataframe_type = DataFrame

    @classmethod
    def _to_output(cls, arr):
        if isinstance(arr, ak.Array):
            return ak.to_cudf(arr)
        return arr

    @classmethod
    def to_array(cls, data) -> ak.Array:
        return from_cudf(data)

    @property
    def array(self) -> ak.Array:
        return self.to_array(self._obj)

    @property
    def str(self):
        """Nested string operations"""
        # need to find string ops within cudf
        return CudfStringAccessor(self)

    @property
    def dt(self):
        """Nested datetime operations"""
        # need to find datetime ops within cudf
        raise NotImplementedError

    def apply(self, fn: Callable, *args, **kwargs):
        if "CPUDispatcher" in str(fn):
            # auto wrap original function for GPU
            raise NotImplementedError
        super().apply(fn, *args, **kwargs)


@property  # type:ignore
def ak_property(self):
    return CudfAwkwardAccessor(self)


Series.ak = ak_property  # no official register function?
