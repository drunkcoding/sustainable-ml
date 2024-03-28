import os
import importlib
import pkgutil

__op_builders__ = []

op_builder_dir = "op_builder"
op_builder_module = importlib.import_module(op_builder_dir)

for _, module_name, _ in pkgutil.iter_modules(
    [os.path.dirname(op_builder_module.__file__)]):
    # avoid self references
    if module_name != 'all_ops' and module_name != 'builder':
        module = importlib.import_module("{}.{}".format(
            op_builder_dir, module_name))
        for member_name in module.__dir__():
            if member_name.endswith(
                    'Builder'
            ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder":
                # append builder to __op_builders__ list
                builder = getattr(module, member_name)()
                __op_builders__.append(builder)

ALL_OPS = {op.name: op for op in __op_builders__ if op is not None}
