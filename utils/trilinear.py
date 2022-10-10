from mindspore.ops import Primitive, prim_attr_register
from enum import Enum
from collections.abc import Iterable
import math
import numpy as np


def check_number(arg_value, value, rel, arg_type=int, arg_name=None, prim_name=None):
    """
    Check argument integer.

    Usage:
    - arg_value = check_number(arg_value, 2, Rel.GT, int, "value", None)
    """
    rel_fn = Rel.get_fns(rel)
    prim_name = f"For \'{prim_name}\', the " if prim_name else 'The '
    arg_name = f"\'{arg_name}\'" if arg_name else 'input value'
    prim_info = f'{prim_name}' + f'{arg_name}'
    if isinstance(arg_value, arg_type):
        if math.isinf(arg_value) or math.isnan(arg_value) or np.isinf(arg_value) or np.isnan(arg_value):
            raise ValueError(f"{prim_info} must be a legal value, but got '{arg_value}'.")
    else:
        raise TypeError(f"{prim_info} must be {arg_type.__name__}, but got '{type(arg_value).__name__}'")

    type_mismatch = not isinstance(arg_value, arg_type) or isinstance(arg_value, bool)
    type_except = TypeError if type_mismatch else ValueError
    if type_mismatch or not rel_fn(arg_value, value):
        rel_str = Rel.get_strs(rel).format(value)
        raise type_except(f"{prim_info} must be {arg_type.__name__} and must {rel_str}, "
                          f"but got '{arg_value}' with type '{type(arg_value).__name__}'.")

    return arg_value


class Rel(Enum):

    """Numerical relationship between variables, logical relationship enumeration definition of range."""
    # scalar compare
    EQ = 1  # ==
    NE = 2  # !=
    LT = 3  # <
    LE = 4  # <=
    GT = 5  # >
    GE = 6  # >=
    # scalar range check
    INC_NEITHER = 7  # (), include neither
    INC_LEFT = 8  # [), include left
    INC_RIGHT = 9  # (], include right
    INC_BOTH = 10  # [], include both
    # collection in, not in
    IN = 11
    NOT_IN = 12

    @staticmethod
    def get_strs(rel):
        """Get value from rel_strs."""
        return rel_strs.get(rel, "")

    @staticmethod
    def get_fns(rel):
        """Get value from rel_fns."""
        return rel_fns.get(rel, lambda *args: False)


rel_fns = {
    # scalar compare
    Rel.EQ: lambda x, y: x == y,
    Rel.NE: lambda x, y: x != y,
    Rel.LT: lambda x, y: x < y,
    Rel.LE: lambda x, y: x <= y,
    Rel.GT: lambda x, y: x > y,
    Rel.GE: lambda x, y: x >= y,
    # scalar range check
    Rel.INC_NEITHER: lambda x, lower, upper: (lower < x < upper),
    Rel.INC_LEFT: lambda x, lower, upper: (lower <= x < upper),
    Rel.INC_RIGHT: lambda x, lower, upper: (lower < x <= upper),
    Rel.INC_BOTH: lambda x, lower, upper: (lower <= x <= upper),
    # collection in, not in
    Rel.IN: lambda x, y: x in y,
    Rel.NOT_IN: lambda x, y: x not in y,
}

rel_strs = {
    # scalar compare
    Rel.EQ: "= {}",
    Rel.NE: "!= {}",
    Rel.LT: "< {}",
    Rel.LE: "<= {}",
    Rel.GT: "> {}",
    Rel.GE: ">= {}",
    # scalar range check
    Rel.INC_NEITHER: "({}, {})",
    Rel.INC_LEFT: "[{}, {})",
    Rel.INC_RIGHT: "({}, {}]",
    Rel.INC_BOTH: "[{}, {}]",
    # collection in, not in
    Rel.IN: "in {}",
    Rel.NOT_IN: "not in {}",
}


class Validator:
    """Validator for checking input parameters"""

    @staticmethod
    def check_value_type(arg_name, arg_value, valid_types, prim_name=None):
        """Checks whether a value is instance of some types."""
        valid_types = valid_types if isinstance(valid_types, Iterable) else (valid_types,)

        def raise_error_msg():
            """func for raising error message when check failed"""
            type_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in valid_types]
            num_types = len(valid_types)
            msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
            raise TypeError(f'{msg_prefix} type of \'{arg_name}\' should be {"one of " if num_types > 1 else ""}'
                            f'\'{type_names if num_types > 1 else type_names[0]}\', '
                            f'but got \'{arg_value}\' with type \'{type(arg_value).__name__}\'.')

        # Notice: bool is subclass of int, so `check_value_type('x', True, [int])` will check fail, and
        #         `check_value_type('x', True, [bool, int])` will check pass
        if isinstance(arg_value, bool) and bool not in tuple(valid_types):
            raise_error_msg()
        if not isinstance(arg_value, tuple(valid_types)):
            raise_error_msg()
        return arg_value

    @staticmethod
    def check_bool(arg_value, arg_name=None, prim_name=None):
        """
        Check argument is instance of bool.

        Usage:
        - has_bias = check_bool(has_bias)
        - has_bias = check_bool(has_bias, "has_bias")
        """
        if not isinstance(arg_value, bool):
            prim_name = f"For '{prim_name}', the" if prim_name else 'The'
            arg_name = f"'{arg_name}'" if arg_name else 'input value'
            raise TypeError(f"{prim_name} {arg_name} must be a bool, but got {type(arg_value).__name__}.")
        return arg_value

    @staticmethod
    def check_positive_int_sequence(sequence, arg_name=None, prim_name=None):
        """
        Check argument is positive sequence, which mean all element > 0 in sequence.

        Usage:
        - sequence = check_positive_int_sequence(sequence)
        - sequence = check_positive_int_sequence(sequence, "dims")
        """
        for idx, element in enumerate(sequence):
            arg_idx = '{}[{}]'.format(arg_name if arg_name else 'arg_name', idx)
            check_number(element, 0, Rel.GT, int, arg_idx, prim_name)
        return sequence

    @staticmethod
    def check_positive_float_sequence(sequence, arg_name=None, prim_name=None):
        """
		Check argument is positive float sequence, which mean all element > 0 in sequence.

		Usage:
		- sequence = check_positive_float_sequence(sequence)
		- sequence = check_positive_float_sequence(sequence, "dims")
		"""
        for idx, element in enumerate(sequence):
            arg_idx = '{}[{}]'.format(arg_name if arg_name else 'arg_name', idx)
            check_number(element, 0, Rel.GT, float, arg_idx, prim_name)
        return sequence


class UpsampleTrilinear3D(Primitive):
    r"""
    Performs upsampling with trilinear interpolation across 3dims for 5dim inputs.

    This operator scale up the volumetric input with specified `output_size` or `scales` factors,
    using trilinear upscaling algorithm.

    Note:
        One of `scales` and `output_size` MUST be specified and it is an error if both are specified.

    Args:
        output_size (Union[tuple[int], list[int]]):  A tuple or list of 3 int
            elements :math:`(output\_depth, output\_height, output\_width)`.
            Defaults to None. Only one of `scales` and `output_size` can be specified.
        scales (Union[tuple[float], list[float]]): A tuple or list of 3 float
           elements :math:`(scale\_depth, scale\_height, scale\_width)`. Defaults to None.
        align_corners (bool): An optional bool. Defaults to false.
            If true, the input and output tensors are aligned by the center points of their corner pixels,
            preserving the values at the corner pixels.
            If false, the input and output tensors are aligned by the corner points of their corner pixels,
            and the interpolation use edge value padding for out of boundary values.

    Inputs:
        - **x** (Tensor) - A 5-D input tensor of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
          Must be one of the following types: float16, float32, float64.

    Outputs:
        - **y** (Tensor) - Upsampled output with the same data type as `x`.
          Tensor of shape :math:`(N, C, D_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: When `output_size` is not none and `output_size` is not list[int] or tuple[int].
        TypeError: When `scales` is not none and `scales` is not list[float] or tuple[float].
        TypeError: If dtype of `x` is not in [float16, float32, float64].
        TypeError: If type of `align_corners` is not bool.
        ValueError: If any value of `output_size` is negative or zero when `output_size` is not empty.
        ValueError: If any value of `scales` is negative or zero when `scales` is not empty.
        ValueError: If shape of `x` is not 5D.
        ValueError: If none of `scales` and `output_size` is specified or both specified.
        ValueError: If size of `scales` is not equal 3 when `scales` is specified.
        ValueError: If size of `output_size` is not equal 3 when `output_size` is specified.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> ops = P.UpsampleTrilinear3D(output_size=[4, 64, 48])
        >>> out = ops(Tensor(input_data=np.random.randn(2, 3, 4, 512, 256)))
        >>> print(out.shape)
        (2, 3, 4, 64, 48)

        >>> ops = P.UpsampleTrilinear3D(output_size=[2, 4, 4])
        >>> in_x = Tensor(np.arange(1, 5, dtype=np.float32).reshape((1, 1, 1, 2, 2)))
        >>> out = ops(in_x)
        >>> print(out)
        [[[[[1.   1.25 1.75 2.  ]
            [1.5  1.75 2.25 2.5 ]
            [2.5  2.75 3.25 3.5 ]
            [3.   3.25 3.75 4.  ]]
           [[1.   1.25 1.75 2.  ]
            [1.5  1.75 2.25 2.5 ]
            [2.5  2.75 3.25 3.5 ]
            [3.   3.25 3.75 4.  ]]]]]
    """
    @prim_attr_register
    def __init__(self, output_size=None, scales=None, align_corners=False):
        """Initialize UpsampleTrilinear3D."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        self.output_size = [] if output_size is None else output_size
        self.scales = [] if scales is None else scales
        self.align_corners = align_corners

        Validator.check_value_type("output_size", self.output_size, [list, tuple], self.name)
        Validator.check_value_type("scales", self.scales, [list, tuple], self.name)
        Validator.check_bool(self.align_corners, "align_corners", self.name)
        if len(self.output_size) == 3:
            Validator.check_positive_int_sequence(self.output_size, "output_size", self.name)
        if len(self.scales) == 3:
            Validator.check_positive_float_sequence(self.scales, "scales", self.name)

        self.add_prim_attr('output_size', self.output_size)
        self.add_prim_attr('scales', self.scales)
        self.add_prim_attr('align_corners', self.align_corners)
        self.add_prim_attr("cust_aicpu", self.name)


if __name__ == "__main__":
    from mindspore import Tensor, context
    context.set_context(mode=0, device_target="Ascend", device_id=4)

    ops = UpsampleTrilinear3D(output_size=[4, 64, 48])
    out = ops(Tensor(input_data=np.random.randn(2, 3, 4, 512, 256)))
    print(out.shape)