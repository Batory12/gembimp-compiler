"""
Three-Address Code (TAC) representation for the compiler.
Uses dataclass and enum approach for type safety and consistency with AST nodes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class TACOp(Enum):
    """Three-address code operations"""
    # Arithmetic operations
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    MOD = '%'
    
    # Assignment
    ASSIGN = '='
    
    # Memory operations
    LOAD = 'LOAD'
    STORE = 'STORE'
    LOAD_ARRAY = 'LOAD_ARRAY'
    STORE_ARRAY = 'STORE_ARRAY'
    
    # I/O operations
    READ = 'READ'
    WRITE = 'WRITE'
    READ_ARRAY = 'READ_ARRAY'
    
    # Control flow
    LABEL = 'LABEL'
    GOTO = 'GOTO'
    IF = 'IF'
    CALL = 'CALL'
    RET = 'RET'
    PARAM = 'PARAM'


@dataclass
class TACInstruction:
    """A single three-address code instruction"""
    op: TACOp
    result: Optional[str] = None  # Destination (left side of assignment)
    arg1: Optional[Union[str, int]] = None  # First operand
    arg2: Optional[Union[str, int]] = None  # Second operand
    label: Optional[str] = None  # For labels, jumps, and conditional branches
    
    def __str__(self) -> str:
        """Human-readable string representation of the instruction"""
        if self.op == TACOp.LABEL:
            return f"{self.label}:"
        elif self.op == TACOp.GOTO:
            return f"GOTO {self.label or self.arg1}"
        elif self.op == TACOp.CALL:
            return f"CALL {self.label or self.arg1}"
        elif self.op == TACOp.RET:
            return "RET"
        elif self.op == TACOp.ASSIGN:
            return f"{self.result} = {self.arg1}"
        elif self.op in [TACOp.READ, TACOp.WRITE]:
            return f"{self.op.value} {self.arg1}"
        elif self.op == TACOp.IF:
            return f"IF {self.arg1} {self.arg2} {self.result} GOTO {self.label}"
        elif self.op in [TACOp.ADD, TACOp.SUB, TACOp.MUL, TACOp.DIV, TACOp.MOD]:
            return f"{self.result} = {self.arg1} {self.op.value} {self.arg2}"
        elif self.op in [TACOp.LOAD, TACOp.STORE]:
            return f"{self.op.value} {self.arg1} -> {self.result}"
        elif self.op in [TACOp.LOAD_ARRAY, TACOp.STORE_ARRAY, TACOp.READ_ARRAY]:
            if self.op == TACOp.READ_ARRAY:
                return f"{self.op.value} {self.arg1}[{self.arg2}]"
            else:
                return f"{self.op.value} {self.arg1}[{self.arg2}] -> {self.result}"
        elif self.op == TACOp.PARAM:
            return f"PARAM {self.arg1}"
        else:
            # Generic format for any other operations
            parts = []
            if self.result:
                parts.append(str(self.result))
            if self.arg1 is not None:
                parts.append(str(self.arg1))
            if self.arg2 is not None:
                parts.append(str(self.arg2))
            if self.label:
                parts.append(self.label)
            return f"{self.op.value} " + " ".join(parts)


def display_tac(instructions: List[TACInstruction]) -> str:
    """Display TAC instructions in a readable format"""
    lines = []
    for i, instr in enumerate(instructions):
        lines.append(f"{i:4d}: {str(instr)}")
    return "\n".join(lines)


class TACGenerator:
    """Generator for three-address code from AST.
    
    This is a stub implementation. The actual generation logic should be
    implemented to traverse the AST and emit TAC instructions.
    """
    
    def __init__(self):
        self.instructions: List[TACInstruction] = []
        self.temp_counter = 0
        self.label_counter = 0
    
    def new_temp(self) -> str:
        """Create a new temporary variable name"""
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def new_label(self) -> str:
        """Create a new label name"""
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def emit(self, op: TACOp, result: Optional[str] = None, 
             arg1: Optional[Union[str, int]] = None,
             arg2: Optional[Union[str, int]] = None,
             label: Optional[str] = None):
        """Emit a TAC instruction"""
        self.instructions.append(TACInstruction(op, result, arg1, arg2, label))
    
    def generate(self, ast):
        """Generate TAC from AST.
        
        This is a stub - should be implemented to traverse the AST and
        generate TAC instructions.
        """
        # TODO: Implement actual TAC generation
        # For now, return empty list for debugging purposes
        self.instructions = []
        return self.instructions
