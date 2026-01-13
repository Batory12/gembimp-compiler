"""
Three-Address Code (TAC) representation for the compiler.
Uses dataclass and enum approach for type safety and consistency with AST nodes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict
from visitor import ASTVisitor
from ast_nodes import (
    Program, Procedure, Main, Declaration,
    AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand,
    ProcCallCommand, ReadCommand, WriteCommand,
    Condition, Expression, BinaryExpression, Value, Identifier, Access
)
from symbol_table import SymbolTable, Symbol, SymbolType


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
    ALLOC = 'ALLOC'  # Array allocation: ALLOC array_name size start_index
    
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
        elif self.op == TACOp.ALLOC:
            return f"{self.op.value} {self.result} size={self.arg1} start={self.arg2}"
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
        self.current_procedure: Optional[str] = None  # Track current procedure scope
        # Store procedure AST nodes for inlining
        self.procedure_nodes: Dict[str, 'Procedure'] = {}  # Maps proc_name -> Procedure AST node
    
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
    
    def get_qualified_name(self, var_name: str) -> str:
        """Get qualified variable name based on current scope to handle shadowing.
        
        Returns:
            Qualified name in format: 'main.varname' or 'procname.varname'
        """
        if not self.symbol_table:
            # No symbol table, use name as-is (backward compatibility)
            return var_name
        
        # Look up the symbol to determine its scope
        symbol = self.symbol_table.lookup(var_name)
        if symbol is None:
            # Symbol not found, use name as-is (shouldn't happen after semantic analysis)
            return var_name
        
        # Check if it's a FOR iterator (most local scope)
        if symbol.is_for_iterator():
            # FOR iterators are in the current procedure scope (or main if no procedure)
            if self.current_procedure:
                return f"{self.current_procedure}.{var_name}"
            else:
                return f"main.{var_name}"
        
        # Check if it's in current procedure scope (parameters or locals)
        if self.current_procedure:
            proc_symbols = self.symbol_table.procedure_symbols.get(self.current_procedure, {})
            if var_name in proc_symbols:
                symbol = proc_symbols[var_name]
                # If it's a parameter, it should have been substituted during inlining
                # If we still see it, use the argument name from param_mapping
                if symbol.is_parameter():
                    # This should only happen if param_mapping is set (during inlining)
                    if hasattr(self, 'param_mapping') and var_name in self.param_mapping:
                        return self.param_mapping[var_name]
                    # Fallback: shouldn't happen if inlining works correctly
                    return f"{self.current_procedure}.{var_name}"
                # Local variable (not a parameter)
                return f"{self.current_procedure}.{var_name}"
        
        # Check if it's a main program variable
        if var_name in self.symbol_table.global_symbols:
            return f"main.{var_name}"
        
        # Fallback: use name as-is
        return var_name
    
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
        self.visit_program(ast)
        return self.instructions
    
    def visit_program(self, node: Program):
        """Visit Program node - generate main first, procedures are inlined at call sites"""
        # Store procedure nodes for inlining at call sites
        for proc in node.procedures:
            self.procedure_nodes[proc.head.name] = proc
        
        # Generate TAC for main program first
        # Procedures will be inlined at their call sites
        self.visit_main(node.main)
    
    def visit_main(self, node: Main):
        """Visit Main node - process declarations first, then commands"""
        # Set global scope (don't call enter_global_scope as it may reset state)
        if self.symbol_table:
            self.symbol_table.current_procedure = None
            self.symbol_table.scope_level = 0
        self.current_procedure = None
        
        # Process declarations first (generate ALLOC for arrays)
        for decl in node.declarations:
            self.visit_declaration(decl)
        
        # Then process commands
        for cmd in node.commands:
            cmd.accept(self)
    
    def visit_procedure(self, node: Procedure):
        """Visit Procedure node"""
        # Set procedure scope (don't call enter_procedure_scope as procedure already exists)
        proc_name = node.head.name
        if self.symbol_table:
            # Just set current_procedure for lookup, don't try to create new scope
            self.symbol_table.current_procedure = proc_name
            self.symbol_table.scope_level = 1
        self.current_procedure = proc_name
        
        # Procedure label
        proc_label = f"proc_{proc_name}"
        self.emit(TACOp.LABEL, label=proc_label)
        
        # Process declarations first (generate ALLOC for arrays)
        for decl in node.declarations:
            self.visit_declaration(decl)
        
        # Generate procedure body
        for cmd in node.commands:
            cmd.accept(self)
        
        # Return statement
        self.emit(TACOp.RET)
        
        # Exit procedure scope
        if self.symbol_table:
            self.symbol_table.current_procedure = None
            self.symbol_table.scope_level = 0
        self.current_procedure = None
    
    def visit_command(self, cmd):
        """Dispatch to specific command visitor"""
        cmd.accept(self)
    
    def visit_assign(self, cmd: AssignCommand):
        """Visit AssignCommand - generate TAC for assignment"""
        # Evaluate the expression
        expr_temp = self.visit_expression(cmd.expression)
        
        # Assign to target
        if isinstance(cmd.identifier, Identifier):
            # Simple variable assignment - use qualified name
            qualified_name = self.get_qualified_name(cmd.identifier.name)
            self.emit(TACOp.ASSIGN, result=qualified_name, arg1=expr_temp)
        elif isinstance(cmd.identifier, Access):
            # Array assignment - use qualified name
            qualified_name = self.get_qualified_name(cmd.identifier.name)
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
        # Enter FOR scope - add iterator to symbol table's for_iterators stack
        if self.symbol_table:
            # Check if iterator already exists in current scope
            existing = self.symbol_table.lookup(cmd.var)
            if existing and existing.is_for_iterator():
                # Already in scope (nested FOR with same iterator), don't add again
                pass
            else:
                # Create and add iterator symbol
                iterator = Symbol(
                    name=cmd.var,
                    symbol_type=SymbolType.FOR_ITERATOR,
                    scope_level=self.symbol_table.scope_level,
                    declared_at_line=None
                )
                self.symbol_table.for_iterators.append(iterator)
        
        loop_label = self.new_label()
        end_label = self.new_label()
        
        # Evaluate from and to values
        from_temp = self.visit_value(cmd.from_val)
        to_temp = self.visit_value(cmd.to_val)
        
        # Initialize loop variable - use qualified name
        qualified_var = self.get_qualified_name(cmd.var)
        self.emit(TACOp.ASSIGN, result=qualified_var, arg1=from_temp)
        
        # Loop start - check condition first
        self.emit(TACOp.LABEL, label=loop_label)
        
        # Compare loop variable with to value and exit if condition fails
        if cmd.downto:
            # FOR ... DOWNTO: exit if var < to_val
            cmp_temp = self.new_temp()
            self.emit(TACOp.SUB, result=cmp_temp, arg1=qualified_var, arg2=to_temp)
            # If var - to_val < 0 (i.e., var < to_val), exit
            self.emit(TACOp.IF, arg1=cmp_temp, arg2='<', result=0, label=end_label)
        else:
            # FOR ... TO: exit if var > to_val
            cmp_temp = self.new_temp()
            self.emit(TACOp.SUB, result=cmp_temp, arg1=qualified_var, arg2=to_temp)
            # If var - to_val > 0 (i.e., var > to_val), exit
            self.emit(TACOp.IF, arg1=cmp_temp, arg2='>', result=0, label=end_label)
        
        # Loop body (reached only if condition passed)
        for body_cmd in cmd.commands:
            body_cmd.accept(self)
        
        # Increment/decrement loop variable
        if cmd.downto:
            dec_temp = self.new_temp()
            self.emit(TACOp.SUB, result=dec_temp, arg1=qualified_var, arg2=1)
            self.emit(TACOp.ASSIGN, result=qualified_var, arg1=dec_temp)
        else:
            inc_temp = self.new_temp()
            self.emit(TACOp.ADD, result=inc_temp, arg1=qualified_var, arg2=1)
            self.emit(TACOp.ASSIGN, result=qualified_var, arg1=inc_temp)
        
        # Jump back to condition check
        self.emit(TACOp.GOTO, label=loop_label)
        
        # End label
        self.emit(TACOp.LABEL, label=end_label)
        
        # Exit FOR scope - remove iterator from stack
        if self.symbol_table and self.symbol_table.for_iterators:
            # Only pop if we added one (check if last iterator matches)
            if self.symbol_table.for_iterators and self.symbol_table.for_iterators[-1].name == cmd.var:
                self.symbol_table.for_iterators.pop()
    
    def visit_proc_call(self, cmd: ProcCallCommand):
        """Visit ProcCallCommand - inline procedure body with parameter substitution"""
        proc_name = cmd.name
        
        # Get procedure node for inlining
        if proc_name not in self.procedure_nodes:
            # Procedure not found, fallback to call
            for arg in cmd.args:
                qualified_arg = self.get_qualified_name(arg)
                self.emit(TACOp.PARAM, arg1=qualified_arg)
            self.emit(TACOp.CALL, arg1=proc_name)
            return
        
        proc_node = self.procedure_nodes[proc_name]
        
        # Get procedure symbol to find parameter names
        if not self.symbol_table:
            # No symbol table, fallback
            for arg in cmd.args:
                qualified_arg = self.get_qualified_name(arg)
                self.emit(TACOp.PARAM, arg1=qualified_arg)
            self.emit(TACOp.CALL, arg1=proc_name)
            return
        
        proc_symbol = self.symbol_table.lookup(proc_name)
        if not proc_symbol or not proc_symbol.is_procedure():
            # Procedure not found, fallback
            for arg in cmd.args:
                qualified_arg = self.get_qualified_name(arg)
                self.emit(TACOp.PARAM, arg1=qualified_arg)
            self.emit(TACOp.CALL, arg1=proc_name)
            return
        
        # Get parameter names and argument qualified names
        param_names = proc_symbol.param_names or []
        arg_qualified_names = []
        for arg in cmd.args:
            qualified_arg = self.get_qualified_name(arg)
            arg_qualified_names.append(qualified_arg)
        
        # Create parameter mapping: param_name -> argument_qualified_name (pass-by-reference)
        old_mapping = getattr(self, 'param_mapping', {})
        self.param_mapping = {}
        for i, param_name in enumerate(param_names):
            if i < len(arg_qualified_names):
                # Map parameter name to argument's qualified name
                self.param_mapping[param_name] = arg_qualified_names[i]
        
        # Save current procedure and scope
        old_procedure = self.current_procedure
        old_scope_level = self.symbol_table.scope_level if self.symbol_table else None
        
        # Set procedure scope for inlining
        if self.symbol_table:
            self.symbol_table.current_procedure = proc_name
            self.symbol_table.scope_level = 1
        self.current_procedure = proc_name
        
        # Inline procedure body (parameters will be substituted via param_mapping)
        for proc_cmd in proc_node.commands:
            proc_cmd.accept(self)
        
        # Restore scope
        if self.symbol_table:
            self.symbol_table.current_procedure = old_procedure
            self.symbol_table.scope_level = old_scope_level if old_scope_level is not None else 0
        self.current_procedure = old_procedure
        self.param_mapping = old_mapping
    
    def visit_read(self, cmd: ReadCommand):
        """Visit ReadCommand - generate TAC for READ statement"""
        if isinstance(cmd.identifier, Identifier):
            qualified_name = self.get_qualified_name(cmd.identifier.name)
            self.emit(TACOp.READ, arg1=qualified_name)
        elif isinstance(cmd.identifier, Access):
            qualified_name = self.get_qualified_name(cmd.identifier.name)
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
        """Visit Identifier - return qualified variable name"""
        return self.get_qualified_name(node.name)
    
    def visit_access(self, node: Access) -> str:
        """Visit Access - generate TAC for array access"""
        qualified_name = self.get_qualified_name(node.name)
        index_temp = self.visit_value(node.index)
        result_temp = self.new_temp()
        self.emit(TACOp.LOAD_ARRAY, arg1=qualified_name, arg2=index_temp, result=result_temp)
        return result_temp
    
    def visit_declaration(self, node: Declaration):
        """Visit Declaration - generate ALLOC instruction for arrays"""
        if node.array_range is not None:
            # Array declaration
            start, end = node.array_range
            size = end - start + 1  # Array size
            
            # Get qualified name for the array
            qualified_name = self.get_qualified_name(node.name)
            
            # Emit ALLOC instruction: ALLOC array_name size start_index
            # Format: result = array_name, arg1 = size, arg2 = start_index
            self.emit(TACOp.ALLOC, result=qualified_name, arg1=size, arg2=start)
        # Regular variables don't generate code - they're allocated on first use
    
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
