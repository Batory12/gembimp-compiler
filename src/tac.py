"""
Three-Address Code (TAC) representation for the compiler.
Uses dataclass and enum approach for type safety and consistency with AST nodes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from ast_nodes import (
    Program, Procedure, Main, Declaration,
    AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand,
    ProcCallCommand, ReadCommand, WriteCommand,
    Condition, Expression, BinaryExpression, Value, Identifier, Access
)


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
        """Generate TAC from AST."""
        self.instructions = []
        if isinstance(ast, Program):
            # Generate TAC for main program first
            self.generate_main(ast.main)
            # Then generate procedures
            for proc in ast.procedures:
                self.generate_procedure(proc)
        else:
            raise ValueError(f"Unsupported AST node type: {type(ast)}")
        return self.instructions
    
    def generate_main(self, main_node: Main):
        """Generate TAC for main program"""
        # Declarations don't generate code, they're just metadata
        for cmd in main_node.commands:
            self.generate_command(cmd)
    
    def generate_procedure(self, proc_node: Procedure):
        """Generate TAC for a procedure"""
        # Procedure label
        proc_label = f"proc_{proc_node.head.name}"
        self.emit(TACOp.LABEL, label=proc_label)
        
        # Generate procedure body
        for cmd in proc_node.commands:
            self.generate_command(cmd)
        
        # Return statement
        self.emit(TACOp.RET)
    
    def generate_command(self, cmd):
        """Generate TAC for a command"""
        if isinstance(cmd, AssignCommand):
            self.generate_assign(cmd)
        elif isinstance(cmd, IfCommand):
            self.generate_if(cmd)
        elif isinstance(cmd, WhileCommand):
            self.generate_while(cmd)
        elif isinstance(cmd, RepeatCommand):
            self.generate_repeat(cmd)
        elif isinstance(cmd, ForCommand):
            self.generate_for(cmd)
        elif isinstance(cmd, ProcCallCommand):
            self.generate_proc_call(cmd)
        elif isinstance(cmd, ReadCommand):
            self.generate_read(cmd)
        elif isinstance(cmd, WriteCommand):
            self.generate_write(cmd)
        else:
            raise ValueError(f"Unsupported command type: {type(cmd)}")
    
    def generate_assign(self, cmd: AssignCommand):
        """Generate TAC for assignment"""
        # Evaluate the expression
        expr_temp = self.generate_expression(cmd.expression)
        
        # Assign to target
        if isinstance(cmd.identifier, Identifier):
            # Simple variable assignment
            self.emit(TACOp.ASSIGN, result=cmd.identifier.name, arg1=expr_temp)
        elif isinstance(cmd.identifier, Access):
            # Array assignment
            index_temp = self.generate_value(cmd.identifier.index)
            self.emit(TACOp.STORE_ARRAY, arg1=cmd.identifier.name, arg2=index_temp, result=expr_temp)
        else:
            raise ValueError(f"Unsupported identifier type: {type(cmd.identifier)}")
    
    def generate_if(self, cmd: IfCommand):
        """Generate TAC for IF statement"""
        then_label = self.new_label()
        else_label = self.new_label() if cmd.else_commands else None
        end_label = self.new_label()
        
        # Generate condition check
        self.generate_condition_jump(cmd.condition, then_label, else_label if else_label else end_label)
        
        # THEN block
        self.emit(TACOp.LABEL, label=then_label)
        for then_cmd in cmd.then_commands:
            self.generate_command(then_cmd)
        
        if else_label:
            # Jump over ELSE block
            self.emit(TACOp.GOTO, label=end_label)
            # ELSE block
            self.emit(TACOp.LABEL, label=else_label)
            for else_cmd in cmd.else_commands:
                self.generate_command(else_cmd)
        
        # End label
        self.emit(TACOp.LABEL, label=end_label)
    
    def generate_while(self, cmd: WhileCommand):
        """Generate TAC for WHILE loop"""
        loop_label = self.new_label()
        body_label = self.new_label()
        end_label = self.new_label()
        
        # Loop start - check condition
        self.emit(TACOp.LABEL, label=loop_label)
        
        # Check condition - if true, go to body; if false, go to end
        self.generate_condition_jump(cmd.condition, body_label, end_label)
        
        # Body label
        self.emit(TACOp.LABEL, label=body_label)
        
        # Loop body
        for body_cmd in cmd.commands:
            self.generate_command(body_cmd)
        
        # Jump back to condition check
        self.emit(TACOp.GOTO, label=loop_label)
        
        # End label (reached when condition is false)
        self.emit(TACOp.LABEL, label=end_label)
    
    def generate_repeat(self, cmd: RepeatCommand):
        """Generate TAC for REPEAT loop"""
        loop_label = self.new_label()
        end_label = self.new_label()
        
        # Loop start
        self.emit(TACOp.LABEL, label=loop_label)
        
        # Loop body
        for body_cmd in cmd.commands:
            self.generate_command(body_cmd)
        
        # Check condition - continue if false, exit if true
        # REPEAT ... UNTIL condition means: repeat until condition is true
        # So if condition is false, go back to loop_label
        self.generate_condition_jump(cmd.condition, end_label, loop_label)
        
        # End label (reached when condition is true)
        self.emit(TACOp.LABEL, label=end_label)
    
    def generate_for(self, cmd: ForCommand):
        """Generate TAC for FOR loop"""
        loop_label = self.new_label()
        end_label = self.new_label()
        
        # Evaluate from and to values
        from_temp = self.generate_value(cmd.from_val)
        to_temp = self.generate_value(cmd.to_val)
        
        # Initialize loop variable
        self.emit(TACOp.ASSIGN, result=cmd.var, arg1=from_temp)
        
        # Loop start - check condition first
        self.emit(TACOp.LABEL, label=loop_label)
        
        # Compare loop variable with to value and exit if condition fails
        if cmd.downto:
            # FOR ... DOWNTO: exit if var < to_val
            cmp_temp = self.new_temp()
            self.emit(TACOp.SUB, result=cmp_temp, arg1=cmd.var, arg2=to_temp)
            # If var - to_val < 0 (i.e., var < to_val), exit
            self.emit(TACOp.IF, arg1=cmp_temp, arg2='<', result=0, label=end_label)
        else:
            # FOR ... TO: exit if var > to_val
            cmp_temp = self.new_temp()
            self.emit(TACOp.SUB, result=cmp_temp, arg1=cmd.var, arg2=to_temp)
            # If var - to_val > 0 (i.e., var > to_val), exit
            self.emit(TACOp.IF, arg1=cmp_temp, arg2='>', result=0, label=end_label)
        
        # Loop body (reached only if condition passed)
        for body_cmd in cmd.commands:
            self.generate_command(body_cmd)
        
        # Increment/decrement loop variable
        if cmd.downto:
            dec_temp = self.new_temp()
            self.emit(TACOp.SUB, result=dec_temp, arg1=cmd.var, arg2=1)
            self.emit(TACOp.ASSIGN, result=cmd.var, arg1=dec_temp)
        else:
            inc_temp = self.new_temp()
            self.emit(TACOp.ADD, result=inc_temp, arg1=cmd.var, arg2=1)
            self.emit(TACOp.ASSIGN, result=cmd.var, arg1=inc_temp)
        
        # Jump back to condition check
        self.emit(TACOp.GOTO, label=loop_label)
        
        # End label
        self.emit(TACOp.LABEL, label=end_label)
    
    def generate_proc_call(self, cmd: ProcCallCommand):
        """Generate TAC for procedure call"""
        # Push parameters (for now, just call - parameter passing may need more work)
        for arg in cmd.args:
            self.emit(TACOp.PARAM, arg1=arg)
        
        # Call procedure
        self.emit(TACOp.CALL, arg1=cmd.name)
    
    def generate_read(self, cmd: ReadCommand):
        """Generate TAC for READ statement"""
        if isinstance(cmd.identifier, Identifier):
            self.emit(TACOp.READ, arg1=cmd.identifier.name)
        elif isinstance(cmd.identifier, Access):
            index_temp = self.generate_value(cmd.identifier.index)
            self.emit(TACOp.READ_ARRAY, arg1=cmd.identifier.name, arg2=index_temp)
        else:
            raise ValueError(f"Unsupported identifier type: {type(cmd.identifier)}")
    
    def generate_write(self, cmd: WriteCommand):
        """Generate TAC for WRITE statement"""
        value_temp = self.generate_value(cmd.value)
        self.emit(TACOp.WRITE, arg1=value_temp)
    
    def generate_expression(self, expr: Expression) -> str:
        """Generate TAC for expression, return temp variable name with result"""
        if isinstance(expr, Value):
            return self.generate_value(expr)
        elif isinstance(expr, BinaryExpression):
            left_temp = self.generate_expression(expr.left)
            right_temp = self.generate_expression(expr.right)
            result_temp = self.new_temp()
            
            # Map operator to TACOp
            op_map = {
                '+': TACOp.ADD,
                '-': TACOp.SUB,
                '*': TACOp.MUL,
                '/': TACOp.DIV,
                '%': TACOp.MOD
            }
            op = op_map.get(expr.operator)
            if op is None:
                raise ValueError(f"Unsupported binary operator: {expr.operator}")
            
            self.emit(op, result=result_temp, arg1=left_temp, arg2=right_temp)
            return result_temp
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")
    
    def generate_value(self, value: Value) -> str:
        """Generate TAC for value, return variable name or constant"""
        if isinstance(value.value, int):
            # For constants, we'll just return the string representation
            # The TAC generator/optimizer can handle this later
            return str(value.value)
        elif isinstance(value.value, Identifier):
            return value.value.name
        elif isinstance(value.value, Access):
            # Array access - load array element
            index_temp = self.generate_value(value.value.index)
            result_temp = self.new_temp()
            self.emit(TACOp.LOAD_ARRAY, arg1=value.value.name, arg2=index_temp, result=result_temp)
            return result_temp
        else:
            raise ValueError(f"Unsupported value type: {type(value.value)}")
    
    def generate_condition_jump(self, condition: Condition, true_label: str, false_label: str):
        """Generate TAC for condition check and jump
        
        Args:
            condition: The condition to evaluate
            true_label: Label to jump to if condition is true
            false_label: Label to jump to if condition is false
        """
        left_temp = self.generate_value(condition.left)
        right_temp = self.generate_value(condition.right)
        
        # Direct comparison and jump using IF instruction
        # Format: IF arg1 operator result GOTO label
        # This means: if (arg1 operator result) then goto label
        op = condition.operator
        self.emit(TACOp.IF, arg1=left_temp, arg2=op, result=right_temp, label=true_label)
        # If condition is false, jump to false label
        self.emit(TACOp.GOTO, label=false_label)
