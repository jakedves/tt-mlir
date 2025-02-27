# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional

from ttmlir.dialects import stablehlo, tt, ttir, ttnn
from ttmlir.ir import (
    Context,
    Module,
    OpAttributeMap,
    OperationList,
    OpView,
    RankedTensorType,
    Type,
)
from ttrt.common.util import Binary

from .pydantic_models import OpTest, TensorDesc


class ModuleDialect(Enum):
    """Named like this to avoid collision with builtin `Dialect`."""

    STABLE_HLO = "stablehlo"
    TTIR = "ttir"
    TTNN = "ttnn"
    TT = "tt"

    @staticmethod
    def detect(module_or_op: str | OpView | Module) -> ModuleDialect:
        """
        Factory method. Detects dialect used in the mlir module or op string
        representation.
        """
        str_repr = str(module_or_op)

        if "stablehlo." in str_repr:
            return ModuleDialect.STABLE_HLO
        elif "ttir." in str_repr:
            return ModuleDialect.TTIR
        elif "ttnn." in str_repr:
            return ModuleDialect.TTNN
        else:
            # Fallback to returning `tt` dialet if nothing else succeeds. It bundles
            # together all builtin dialects.
            return ModuleDialect.TT


@dataclass(frozen=True)
class Operand:
    """Simple dataclass representing an operand of a MLIR operation."""

    name: str
    type: Type

    @property
    def data_type(self) -> str:
        assert isinstance(self.type, RankedTensorType)
        return str(self.type.element_type)

    @property
    def shape(self) -> List[int]:
        assert isinstance(self.type, RankedTensorType)
        return self.type.shape

    def __repr__(self) -> str:
        return f"Operand({self.name}, {self.type})"


@dataclass(frozen=True)
class Result:
    """Simple dataclass representing result of a MLIR operation."""

    name: str
    type: Type

    @property
    def data_type(self) -> str:
        assert isinstance(self.type, RankedTensorType)
        return str(self.type.element_type)

    @property
    def shape(self) -> List[int]:
        assert isinstance(self.type, RankedTensorType)
        return self.type.shape


class OpWrapper:
    """Convenience wrapper around MLIR op."""

    # ----- Public methods and properties -----

    def __init__(self, op: OpView, attrs: Optional[OpAttributeMap] = None) -> None:
        self.op = op
        self.operands = [
            Operand(operand.get_name(), operand.type) for operand in op.operands
        ]
        self.result = (
            Result(op.result.get_name(), op.result.type)
            if len(op.results) > 0
            else None
        )
        self.attributes = attrs

    def __str__(self) -> str:
        return str(self.op)

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self.op.name

    def as_module_str(self) -> str:
        """Returns self wrapped in a MLIR module str."""
        return OpWrapper._wrap_in_module_str(
            self.op, self.operands, self.result, self.attributes
        )

    def as_module(self) -> ModuleWrapper:
        """
        Returns self wrapped in `ModuleWrapper`.

        Wrapper will contain original MLIR module, dialect used and op from which
        it was created (`self`).
        """
        module_wrapper = parse_module_str(self.as_module_str())
        # Store a reference to the original op.
        module_wrapper.generated_from_op = self
        return module_wrapper

    # ----- Private methods and classes -----

    @staticmethod
    def _wrap_in_module_str(
        op: OpWrapper,
        operands: List[Operand],
        result: Optional[Result] = None,
        attributes: OpAttributeMap = None,
    ) -> str:
        """
        Wraps `op` in a MLIR `func` and then in a MLIR `module` and returns string
        representation of that module.
        """
        unpacked_operands = ", ".join(
            f"{operand.name}: {operand.type}" for operand in operands
        )

        # Handle special case of ops that don't return anything.
        if result is not None:
            fn_return_type = result.type
            return_stmt = f"return {result.name} : {result.type}"
        else:
            fn_return_type = "()"
            return_stmt = "return"

        # Handle special case of modules that carry attributes.
        if attributes is not None:
            attrs = "{" + ",\n".join(f"{a.name} = {a.attr}" for a in attributes) + "}"
        else:
            attrs = "{}"

        return (
            f"module attributes {attrs} {{ \n"
            f"\tfunc.func @main({unpacked_operands}) -> {fn_return_type} {{ \n"
            f"\t\t{op} \n"
            f"\t\t{return_stmt} \n"
            f"\t}} \n"
            f"}}"
        )


class ModuleWrapper:
    """
    Convenience wrapper around MLIR module.

    Provides posibility to keep track of the op from which module was generated, useful
    in op by op processing pipeline.
    """

    def __init__(
        self,
        module: Module,
        dialect: Optional[ModuleDialect] = None,
        generated_from_op: Optional[OpWrapper] = None,
    ) -> None:
        self.module: Module = module
        self.dialect: ModuleDialect = dialect or ModuleDialect.detect(module)
        self.generated_from_op: Optional[OpWrapper] = generated_from_op

    def __repr__(self) -> str:
        s = f"ModuleWrapper(\n{self.module})"

        if self.generated_from_op:
            s += "," + (
                f"Inputs: {self.inputs}, "
                f"Output: {self.output}, "
                f"Generated from: {self.generated_from_op.name}"
            )

        return s

    @property
    def attributes(self) -> Optional[OpAttributeMap]:
        """Returns module attributes if any, otherwise None."""
        return (
            self.module.operation.attributes
            if len(self.module.operation.attributes) > 0
            else None
        )

    @property
    def operations(self) -> OperationList:
        """Returns list of operations in module's body."""
        return self.module.body.operations

    @property
    def inputs(self) -> List[Operand]:
        """
        Shorthand accessor for operands of underlying op.

        It asserts that module wrapper was generated by wrapping an op. In case of
        module with multiple ops in func body it can be made to reflect inputs of the
        func, but there was no use case for that in current state of things.
        """
        assert self.is_generated_from_op
        return self.generated_from_op.operands

    @property
    def output(self) -> Optional[Result]:
        """Shorthand accessor for result of underlying op."""
        assert self.is_generated_from_op
        return self.generated_from_op.result

    @property
    def is_generated_from_op(self) -> bool:
        """Returns True if module was generated by wrapping an op."""
        return self.generated_from_op is not None

    def copy(self) -> ModuleWrapper:
        """
        Creates new copy of self.

        Take note that copy is shallow: reference to the original op module was
        generated from stays the same.

        This is useful to not mess up the original module in compilation steps which are
        done in-place.
        """
        copy = parse_module_str(str(self.module))
        copy.generated_from_op = self.generated_from_op
        return copy


def parse_module_str(module_str: str) -> ModuleWrapper:
    """
    Within a temporary context registers necessary dialects and parses `module_str`
    returning ModuleWrapper instance.
    """

    def preprocess_module_str(module_str: str) -> str:
        """Preprocesses module string by removing `loc(...)` from it."""
        loc_pattern = re.compile(r"\s*loc\([^)]*\)")
        return re.sub(loc_pattern, "", module_str)

    def register_dialect(dialect: ModuleDialect, ctx: Context) -> None:
        """
        Detects dialect used in `module_str` and registers it with context `ctx`.
        """
        if dialect == ModuleDialect.STABLE_HLO:
            stablehlo.register_dialect(ctx)
            # TODO there must be a better way to do this. We need to register `ttir`
            # (or any other of our dialects) otherwise we'll encounter problems with
            # `func` dialect which isn't included through `stablehlo` and doesn't
            # provide `func.register_dialect(ctx)` on its own.
            ttir.register_dialect(ctx)
        elif dialect == ModuleDialect.TTIR:
            ttir.register_dialect(ctx)
        elif dialect == ModuleDialect.TTNN:
            ttnn.register_dialect(ctx)
        elif dialect == ModuleDialect.TT:
            tt.register_dialect(ctx)
        else:
            raise ValueError(f"Unknown dialect: {dialect.name}")

    with Context() as ctx:
        cleaned_module_str = preprocess_module_str(module_str)
        dialect = ModuleDialect.detect(cleaned_module_str)
        # Must register dialect in order for parsing to work.
        register_dialect(dialect, ctx)
        mlir_module = Module.parse(cleaned_module_str)
        return ModuleWrapper(mlir_module, dialect=dialect)


class ExecutionPhase(Enum):
    GENERATED_STABLE_HLO = 1
    GENERATED_TTIR = 2
    GENERATED_TTNN = 3
    GENERATED_FLATBUFFER = 4
    EXECUTED_FLATBUFFER = 5


@dataclass
class ExecutionResult:
    """
    Final result of execution.

    Holds all info necessary to determine how far down the compilation and run pipeline
    we managed to get (i.e. which ExecutionPhase we reached).
    """

    execution_phase: ExecutionPhase
    last_generated_module: ModuleWrapper
    flatbuffer: Optional[Binary] = None
    device_run_passed: bool = False
    execution_started: datetime = datetime.now()
    last_update: datetime = datetime.now()

    @property
    def execution_ended(self) -> datetime:
        return self.last_update

    @property
    def compilation_finished(self) -> bool:
        return self.execution_phase == ExecutionPhase.GENERATED_TTNN

    @property
    def flatbuffer_generated(self) -> bool:
        return (
            self.execution_phase == ExecutionPhase.GENERATED_FLATBUFFER
            and self.flatbuffer is not None
        )

    @property
    def run_finished(self) -> bool:
        return (
            self.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            and self.device_run_passed == True
        )

    def __repr__(self) -> str:
        return f"ExecutionResult({self.execution_phase.name})"

    def convert_to_pydantic_model(self) -> OpTest:
        assert self.last_generated_module.is_generated_from_op

        if not self.device_run_passed:
            error_msg = (
                f"Couldn't execute op {self.last_generated_module.generated_from_op.name}. "
                f"Last step successfully finished: {self.execution_phase.name}."
            )
        else:
            error_msg = ""

        inputs = [
            TensorDesc(shape=input.shape, data_type=input.data_type)
            for input in self.last_generated_module.inputs
        ]
        outputs = [
            TensorDesc(
                shape=self.last_generated_module.output.shape,
                data_type=self.last_generated_module.output.data_type,
            )
        ]

        pydantic_model = OpTest(
            test_start_ts=self.execution_started,
            test_end_ts=self.execution_ended,
            success=self.device_run_passed,
            skipped=False,  # TODO Never skipped in op by op infra, always ends at some step
            error_message=error_msg,
            op_name=self.last_generated_module.generated_from_op.name,
            inputs=inputs,
            outputs=outputs,
        )

        return pydantic_model


def convert_to_module_wrapper(func: Callable) -> Callable:
    """
    Decorator to ensure that the `module` argument is always of type `ModuleWrapper`.
    If it's a string, it will be parsed using `parse_module_str`. If it is a `Module`
    it will be wrapped.
    """

    def wrapper(
        self,
        module: str | Module | ModuleWrapper,
        *args,
        **kwargs,
    ) -> List[Module]:
        if isinstance(module, str):
            m = parse_module_str(module)
        elif isinstance(module, Module):
            m = ModuleWrapper(module)
        else:
            m = module

        # Call the original function with the converted module.
        return func(self, m, *args, **kwargs)

    return wrapper
