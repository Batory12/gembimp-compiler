"""
Three-Address Code (TAC) representation for the compiler.
Uses dataclass and enum approach for type safety and consistency with AST nodes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from visitor import ASTVisitor
from ast_nodes import (
    Program, Procedure, Main, Declaration,
    AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand,
    ProcCallCommand, ReadCommand, WriteCommand,
    Condition, Expression, BinaryExpression, Value, Identifier, Access
)
from symbol_table import SymbolTable


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


class TACGenerator(ASTVisitor):
    """Visitor that generates three-address code from AST."""
    
    def __init__(self, symbol_table: Optional[SymbolTable] = None):
        super().__init__()
        self.instructions: List[TACInstruction] = []
        self.temp_counter = 0
        self.label_counter = 0
        self.symbol_table = symbol_table
        self.current_procedure: Optional[str] = None  # Track current procedure for variable qualification
    
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
    
    def generate(self, ast: Program) -> List[TACInstruction]:
        """Generate TAC from AST. Main entry point."""
        self.instructions = []
        self.temp_counter = 0
        self.label_counter = 0
        self.current_procedure = None
        self.visit_program(ast)
        return self.instructions
    
    def visit_program(self, node: Program):
        """Visit Program node - generate main first, then procedures"""
        # Generate TAC for main program first
        self.visit_main(node.main)
        # Then generate procedures
        for proc in node.procedures:
            self.visit_procedure(proc)
    
    def visit_main(self, node: Main):
        """Visit Main node - declarations don't generate code"""
        # Declarations don't generate code, they're just metadata
        for cmd in node.commands:
            cmd.accept(self)
    
    def visit_procedure(self, node: Procedure):
        """Visit Procedure node"""
        # Save previous procedure context
        prev_procedure = self.current_procedure
        
        # Set current procedure context
        self.current_procedure = node.head.name
        
        # Procedure label
        proc_label = f"proc_{node.head.name}"
        self.emit(TACOp.LABEL, label=proc_label)
        
        # Generate procedure body
        for cmd in node.commands:
            cmd.accept(self)
        
        # Return statement
        self.emit(TACOp.RET)
        
        # Restore previous procedure context
        self.current_procedure = prev_procedure
    
    def visit_command(self, cmd):
        """Dispatch to specific command visitor"""
        cmd.accept(self)
    
    def visit_assign(self, cmd: AssignCommand):
        """Visit AssignCommand - generate TAC for assignment"""
        # Evaluate the expression
        expr_temp = self.visit_expression(cmd.expression)
        
        # Assign to target
        if isinstance(cmd.identifier, Identifier):
            # Simple variable assignment - qualify the variable name
            qualified_name = self.qualify_variable_name(cmd.identifier.name)
            self.emit(TACOp.ASSIGN, result=qualified_name, arg1=expr_temp)
        elif isinstance(cmd.identifier, Access):
            # Array assignment - qualify the array name
            qualified_name = self.qualify_variable_name(cmd.identifier.name)
            index_temp = self.visit_value(cmd.identifier.index)
            self.emit(TACOp.STORE_ARRAY, arg1=qualified_name, arg2=index_temp, result=expr_temp)
        else:
            raise ValueError(f"Unsupported identifier type: {type(cmd.identifier)}")
    
    def visit_if(self, cmd: IfCommand):
        """Visit IfCommand - generate TAC for IF statement"""
        then_label = self.new_label()
        else_label = self.new_label() if cmd.else_commands else None
        end_label = self.new_label()
        
        # Generate condition check
        self.generate_condition_jump(cmd.condition, then_label, else_label if else_label else end_label)
        
        # THEN block
        self.emit(TACOp.LABEL, label=then_label)
        for then_cmd in cmd.then_commands:
            then_cmd.accept(self)
        
        if else_label:
            # Jump over ELSE block
            self.emit(TACOp.GOTO, label=end_label)
            # ELSE block
            self.emit(TACOp.LABEL, label=else_label)
            for else_cmd in cmd.else_commands:
                else_cmd.accept(self)
        
        # End label
        self.emit(TACOp.LABEL, label=end_label)
    
    def visit_while(self, cmd: WhileCommand):
        """Visit WhileCommand - generate TAC for WHILE loop"""
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
            body_cmd.accept(self)
        
        # Jump back to condition check
        self.emit(TACOp.GOTO, label=loop_label)
        
        # End label (reached when condition is false)
        self.emit(TACOp.LABEL, label=end_label)
    
    def visit_repeat(self, cmd: RepeatCommand):
        """Visit RepeatCommand - generate TAC for REPEAT loop"""
        loop_label = self.new_label()
        end_label = self.new_label()
        
        # Loop start
        self.emit(TACOp.LABEL, label=loop_label)
        
        # Loop body
        for body_cmd in cmd.commands:
            body_cmd.accept(self)
        
        # Check condition - continue if false, exit if true
        # REPEAT ... UNTIL condition means: repeat until condition is true
        # So if condition is false, go back to loop_label
        self.generate_condition_jump(cmd.condition, end_label, loop_label)
        
        # End label (reached when condition is true)
        self.emit(TACOp.LABEL, label=end_label)
    
    def visit_for(self, cmd: ForCommand):
        """Visit ForCommand - generate TAC for FOR loop"""
        loop_label = self.new_label()
        end_label = self.new_label()
        
        # Evaluate from and to values
        from_temp = self.visit_value(cmd.from_val)
        to_temp = self.visit_value(cmd.to_val)
        
        # Qualify loop variable name (FOR iterators in procedures need to be qualified)
        var_name = self.qualify_variable_name(cmd.var)
        
        self.emit(TACOp.ASSIGN, result=var_name, arg1=from_temp)
        
        # Loop start - check condition first
        self.emit(TACOp.LABEL, label=loop_label)
        
        # Compare loop variable with to value and exit if condition fails
        if cmd.downto:
            # FOR ... DOWNTO: exit if var < to_val
            self.emit(TACOp.IF, arg1=var_name, arg2='<', result=to_temp, label=end_label)
        else:
            # FOR ... TO: exit if var > to_val
            self.emit(TACOp.IF, arg1=var_name, arg2='>', result=to_temp, label=end_label)
        
        # Loop body (reached only if condition passed)
        for body_cmd in cmd.commands:
            body_cmd.accept(self)
        
        # Increment/decrement loop variable
        if cmd.downto:
            self.emit(TACOp.SUB, result=var_name, arg1=var_name, arg2=1)
            self.emit(TACOp.IF, arg1=var_name, arg2='=', result=0, label=end_label)
        else:
            self.emit(TACOp.ADD, result=var_name, arg1=var_name, arg2=1)
        
        # Jump back to condition check
        self.emit(TACOp.GOTO, label=loop_label)
        
        # End label
        self.emit(TACOp.LABEL, label=end_label)
    
    def visit_proc_call(self, cmd: ProcCallCommand):
        """Visit ProcCallCommand - generate TAC for procedure call"""
        # Push parameters - qualify argument names if they're local variables/parameters
        for arg in cmd.args:
            qualified_arg = self.qualify_variable_name(arg)
            self.emit(TACOp.PARAM, arg1=qualified_arg)
        
        # Call procedure
        self.emit(TACOp.CALL, arg1=cmd.name)
    
    def visit_read(self, cmd: ReadCommand):
        """Visit ReadCommand - generate TAC for READ statement"""
        if isinstance(cmd.identifier, Identifier):
            qualified_name = self.qualify_variable_name(cmd.identifier.name)
            self.emit(TACOp.READ, arg1=qualified_name)
        elif isinstance(cmd.identifier, Access):
            qualified_name = self.qualify_variable_name(cmd.identifier.name)
            index_temp = self.visit_value(cmd.identifier.index)
            self.emit(TACOp.READ_ARRAY, arg1=qualified_name, arg2=index_temp)
        else:
            raise ValueError(f"Unsupported identifier type: {type(cmd.identifier)}")
    
    def visit_write(self, cmd: WriteCommand):
        """Visit WriteCommand - generate TAC for WRITE statement"""
        value_temp = self.visit_value(cmd.value)
        self.emit(TACOp.WRITE, arg1=value_temp)
    
    def visit_expression(self, expr: Expression) -> str:
        """Visit Expression - generate TAC, return temp variable name with result"""
        return expr.accept(self)
    
    def visit_binary_expression(self, node: BinaryExpression) -> str:
        """Visit BinaryExpression - generate TAC for binary operation"""
        left_temp = self.visit_expression(node.left)
        right_temp = self.visit_expression(node.right)
        result_temp = self.new_temp()
        
        # Map operator to TACOp
        op_map = {
            '+': TACOp.ADD,
            '-': TACOp.SUB,
            '*': TACOp.MUL,
            '/': TACOp.DIV,
            '%': TACOp.MOD
        }
        op = op_map.get(node.operator)
        if op is None:
            raise ValueError(f"Unsupported binary operator: {node.operator}")
        
        self.emit(op, result=result_temp, arg1=left_temp, arg2=right_temp)
        return result_temp
    
    def visit_value(self, value: Value) -> str:
        """Visit Value - generate TAC, return variable name or constant"""
        if isinstance(value.value, int):
            # For constants, we'll just return the string representation
            # The TAC generator/optimizer can handle this later
            return str(value.value)
        elif isinstance(value.value, Identifier):
            return self.visit_identifier(value.value)
        elif isinstance(value.value, Access):
            return self.visit_access(value.value)
        else:
            raise ValueError(f"Unsupported value type: {type(value.value)}")
    
    def visit_identifier(self, node: Identifier) -> str:
        """Visit Identifier - return qualified variable name (local variables are prefixed with procedure name)"""
        return self.qualify_variable_name(node.name)
    
    def qualify_variable_name(self, name: str) -> str:
        """Qualify variable name based on scope.
        
        Local variables in procedures are qualified with the procedure name
        (e.g., 'x' in procedure 'foo' becomes 'foo.x') to avoid conflicts
        with global variables with the same name.
        
        Args:
            name: Variable name from AST
            
        Returns:
            Qualified variable name (procedure_name.var_name for locals, var_name for globals)
        """
        # If no symbol table or not in a procedure, return name as-is
        if not self.symbol_table or not self.current_procedure:
            return name
        
        # Check if this variable is in the current procedure's symbol table
        # (local variables, parameters, or FOR iterators in procedure scope)
        if (self.current_procedure in self.symbol_table.procedure_symbols and
            name in self.symbol_table.procedure_symbols[self.current_procedure]):
            # It's a local variable or parameter - qualify it
            return f"{self.current_procedure}.{name}"
        
        # Check if it's a FOR iterator in procedure scope
        symbol = self.symbol_table.lookup(name)
        if symbol and symbol.is_for_iterator() and symbol.scope_level == 1:
            # FOR iterator in procedure scope - qualify it
            return f"{self.current_procedure}.{name}"
        
        # Global variable or not found - return as-is
        return name
    
    def visit_access(self, node: Access) -> str:
        """Visit Access - generate TAC for array access"""
        qualified_name = self.qualify_variable_name(node.name)
        index_temp = self.visit_value(node.index)
        result_temp = self.new_temp()
        self.emit(TACOp.LOAD_ARRAY, arg1=qualified_name, arg2=index_temp, result=result_temp)
        return result_temp
    
    def generate_condition_jump(self, condition: Condition, true_label: str, false_label: str):
        """Generate TAC for condition check and jump
        
        Args:
            condition: The condition to evaluate
            true_label: Label to jump to if condition is true
            false_label: Label to jump to if condition is false
        """
        left_temp = self.visit_value(condition.left)
        right_temp = self.visit_value(condition.right)
        
        # Direct comparison and jump using IF instruction
        # Format: IF arg1 operator result GOTO label
        # This means: if (arg1 operator result) then goto label
        op = condition.operator
        self.emit(TACOp.IF, arg1=left_temp, arg2=op, result=right_temp, label=true_label)
        # If condition is false, jump to false label
        self.emit(TACOp.GOTO, label=false_label)
