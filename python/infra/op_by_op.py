# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from ttmlir.ir import Module

from .mlir_module_executor import ExecutionResult, MLIRModuleExecutor
from .mlir_module_splitter import MLIRModuleSplitter
from .pydantic_models import OpTest
from .stablehlo_executor import StableHLOExecutor
from .ttir_executor import TTIRExecutor
from .ttnn_executor import TTNNExecutor
from .utils import ModuleDialect


def _create_executor(dialect: ModuleDialect) -> MLIRModuleExecutor:
    """
    Aux executor factory.

    Based on `dialect` creates appropriate `MLIRModuleExecutor` instance.
    """
    if dialect == ModuleDialect.STABLE_HLO:
        return StableHLOExecutor()
    elif dialect == ModuleDialect.TTIR:
        return TTIRExecutor()
    elif dialect == ModuleDialect.TTNN:
        return TTNNExecutor()
    else:
        raise ValueError(f"Unkown dialect: {dialect.name}")


def split_and_execute(module: Module | str) -> List[ExecutionResult]:
    splitter = MLIRModuleSplitter()
    executor = _create_executor(ModuleDialect.detect(module))

    # TODO can be paralelized.
    return [executor.execute(sub_module) for sub_module in splitter.split(module)]


def compile_split_and_execute(module: Module | str) -> List[ExecutionResult]:
    splitter = MLIRModuleSplitter()
    executor = _create_executor(ModuleDialect.detect(module))
    ttnn_executor = TTNNExecutor()

    ttnn_module = executor.compile(module)

    # TODO can be paralelized.
    return [
        ttnn_executor.execute(sub_module) for sub_module in splitter.split(ttnn_module)
    ]


def split_compile_split_and_execute(module: Module | str) -> List[ExecutionResult]:
    splitter = MLIRModuleSplitter()
    executor = _create_executor(ModuleDialect.detect(module))
    ttnn_module_splitter = MLIRModuleSplitter()
    ttnn_executor = TTNNExecutor()

    results = []

    # TODO can be paralelized.
    for sub_module in splitter.split(module):
        ttnn_module = executor.compile(sub_module)
        # TODO this can be replaced with compile_split_and_execute call, apart from
        # replace underlying op
        for ttnn_sub_module in ttnn_module_splitter.split(ttnn_module):
            # `ttnn_sub_module` keeps track from which ttnn op it was generated.
            # Replace it with an op from the original graph.
            # TODO this is not good. Just keep track of this op, don't replace underlying.
            # ttnn_sub_module.replace_underlying_op(sub_module.generated_from_op)

            results.append(ttnn_executor.execute(ttnn_sub_module))

    return results


def convert_results_to_pydantic_models(results: List[ExecutionResult]) -> List[OpTest]:
    return [result.convert_to_pydantic_model() for result in results]
