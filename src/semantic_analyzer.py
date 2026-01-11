"""
Semantic analyzer for the compiler.
Validates the AST according to the language rules using the symbol table.
"""

from typing import List, Optional
from visitor import ASTVisitor
from symbol_table import (
    SymbolTable, SymbolTableError, Symbol, SymbolType, ParamType
)
from ast_nodes import (
    Program, Procedure, Main, Declaration, ArgDecl, ProcHead,
    AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand,
    ProcCallCommand, ReadCommand, WriteCommand,
    Condition, Expression, BinaryExpression, Value, Identifier, Access
)


class SemanticError(Exception):
    """Exception raised for semantic errors"""
    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line
        super().__init__(self.message)
    
    def __str__(self):
        if self.line:
            return f"Line {self.line}: {self.message}"
        return self.message


class SemanticAnalyzer(ASTVisitor):
    """Semantic analyzer that validates AST according to language rules"""
    
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []
        self.current_line: Optional[int] = None  # Can be set if line tracking is added
    
    def analyze(self, ast: Program) -> List[SemanticError]:
        """Analyze the AST and return list of errors"""
        self.errors = []
        self.symbol_table.enter_global_scope()
        
        try:
            # First pass: collect all procedure signatures
            self._collect_procedures(ast)
            
            # Second pass: analyze main and procedures
            self.visit_program(ast)
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
        except SemanticError as e:
            self.errors.append(e)
        
        return self.errors
    
    def _collect_procedures(self, ast: Program):
        """First pass: collect all procedure signatures"""
        for proc in ast.procedures:
            self._collect_procedure_signature(proc)
    
    def _collect_procedure_signature(self, proc: Procedure):
        """Collect a procedure's signature without analyzing its body"""
        proc_name = proc.head.name
        param_names = []
        param_types = []
        
        for arg_decl in proc.head.args:
            param_name = arg_decl.name
            param_names.append(param_name)
            
            # Determine parameter type
            is_array = arg_decl.arg_type == 'T'
            if arg_decl.arg_type == 'I':
                param_type = ParamType.INPUT
            elif arg_decl.arg_type == 'O':
                param_type = ParamType.OUTPUT
            elif arg_decl.arg_type == 'T':
                param_type = ParamType.ARRAY
            else:
                param_type = ParamType.NORMAL
            
            param_types.append(param_type)
        
        # Add procedure to symbol table
        try:
            self.symbol_table.add_procedure(proc_name, param_names, param_types, self.current_line)
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
    
    def visit_program(self, node: Program):
        """Visit Program node"""
        # Visit main first
        self.visit_main(node.main)
        
        # Then visit procedures (they can call each other if defined earlier)
        for proc in node.procedures:
            self.visit_procedure(proc)
    
    def visit_main(self, node: Main):
        """Visit Main node"""
        self.symbol_table.enter_global_scope()
        
        # Process declarations
        for decl in node.declarations:
            self.visit_declaration(decl)
        
        # Process commands
        for cmd in node.commands:
            self.visit_command(cmd)
    
    def visit_procedure(self, node: Procedure):
        """Visit Procedure node"""
        proc_name = node.head.name
        
        # Enter procedure scope
        try:
            self.symbol_table.enter_procedure_scope(proc_name)
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
            return
        
        # Process procedure head (parameters)
        self.visit_proc_head(node.head)
        
        # Process declarations
        for decl in node.declarations:
            self.visit_declaration(decl)
        
        # Process commands
        for cmd in node.commands:
            self.visit_command(cmd)
        
        # Exit procedure scope
        self.symbol_table.exit_procedure_scope()
    
    def visit_proc_head(self, node: ProcHead):
        """Visit ProcHead node - add parameters to symbol table"""
        for arg_decl in node.args:
            self.visit_arg_decl(arg_decl)
    
    def visit_arg_decl(self, node: ArgDecl):
        """Visit ArgDecl node - add parameter to symbol table"""
        param_name = node.name
        is_array = node.arg_type == 'T'
        
        # Determine parameter type
        if node.arg_type == 'I':
            param_type = ParamType.INPUT
        elif node.arg_type == 'O':
            param_type = ParamType.OUTPUT
        elif node.arg_type == 'T':
            param_type = ParamType.ARRAY
        else:
            param_type = ParamType.NORMAL
        
        try:
            self.symbol_table.add_procedure_parameter(
                param_name, param_type, is_array, self.current_line
            )
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
    
    def visit_declaration(self, node: Declaration):
        """Visit Declaration node - add variable or array to symbol table"""
        name = node.name
        
        if node.array_range:
            # Array declaration
            start, end = node.array_range
            try:
                self.symbol_table.add_array(name, start, end, self.current_line)
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
        else:
            # Variable declaration
            try:
                self.symbol_table.add_variable(name, self.current_line)
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
    
    def visit_assign(self, cmd: AssignCommand):
        """Visit AssignCommand - validate assignment"""
        # Check if target is valid and can be written to
        if isinstance(cmd.identifier, Identifier):
            try:
                symbol = self.symbol_table.check_variable_usage(
                    cmd.identifier.name, is_read=False, is_write=True, line=self.current_line
                )
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
                return
        elif isinstance(cmd.identifier, Access):
            # Array assignment
            try:
                # Check array exists
                array_symbol = self.symbol_table.lookup_required(
                    cmd.identifier.name, self.current_line
                )
                if not array_symbol.is_array():
                    self.errors.append(SemanticError(
                        f"'{cmd.identifier.name}' is not an array", self.current_line
                    ))
                    return
                
                # Check index is valid (if it's a constant)
                self.visit_value(cmd.identifier.index)
                
                # Check if array can be written to (parameter type constraints)
                try:
                    self.symbol_table.check_variable_usage(
                        cmd.identifier.name, is_read=False, is_write=True, line=self.current_line
                    )
                except SymbolTableError as e:
                    self.errors.append(SemanticError(e.message, e.line))
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
                return
        else:
            self.errors.append(SemanticError(
                f"Invalid assignment target type: {type(cmd.identifier)}", self.current_line
            ))
            return
        
        # Validate expression
        self.visit_expression(cmd.expression)
    
    def visit_if(self, cmd: IfCommand):
        """Visit IfCommand"""
        # Validate condition
        self.visit_condition(cmd.condition)
        
        # Visit then and else blocks
        for then_cmd in cmd.then_commands:
            self.visit_command(then_cmd)
        
        if cmd.else_commands:
            for else_cmd in cmd.else_commands:
                self.visit_command(else_cmd)
    
    def visit_while(self, cmd: WhileCommand):
        """Visit WhileCommand"""
        # Validate condition
        self.visit_condition(cmd.condition)
        
        # Visit commands
        for body_cmd in cmd.commands:
            self.visit_command(body_cmd)
    
    def visit_repeat(self, cmd: RepeatCommand):
        """Visit RepeatCommand"""
        # Visit commands first
        for body_cmd in cmd.commands:
            self.visit_command(body_cmd)
        
        # Validate condition
        self.visit_condition(cmd.condition)
    
    def visit_for(self, cmd: ForCommand):
        """Visit ForCommand - validate FOR loop"""
        iterator_name = cmd.var
        
        # Validate FROM and TO values
        self.visit_value(cmd.from_val)
        self.visit_value(cmd.to_val)
        
        # Enter FOR scope and add iterator
        try:
            self.symbol_table.enter_for_scope(iterator_name, self.current_line)
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
            return
        
        # Visit commands in FOR loop
        for body_cmd in cmd.commands:
            self.visit_command(body_cmd)
        
        # Exit FOR scope
        self.symbol_table.exit_for_scope()
    
    def visit_proc_call(self, cmd: ProcCallCommand):
        """Visit ProcCallCommand - validate procedure call"""
        proc_name = cmd.name
        arg_count = len(cmd.args)
        
        # Check procedure call validity
        try:
            self.symbol_table.check_procedure_call(proc_name, arg_count, self.current_line)
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
            return
        
        # Enter procedure call (for recursion detection)
        self.symbol_table.enter_procedure_call(proc_name)
        
        # Get procedure symbol to check parameter types
        proc_symbol = self.symbol_table.lookup_required(proc_name, self.current_line)
        
        # Validate arguments
        if proc_symbol.param_names and len(cmd.args) == len(proc_symbol.param_names):
            for i, arg_name in enumerate(cmd.args):
                param_type = proc_symbol.param_types[i]
                
                # Check if argument exists
                try:
                    arg_symbol = self.symbol_table.lookup_required(arg_name, self.current_line)
                except SymbolTableError as e:
                    self.errors.append(SemanticError(e.message, e.line))
                    continue
                
                # Rule 5: I parameters can only receive arguments from I positions
                # (arguments that are I parameters or constants)
                if param_type == ParamType.INPUT:
                    # Argument must be an I parameter or a constant (number)
                    # For now, we can't check if it's a constant from the AST structure
                    # But we can check if it's an I parameter
                    if arg_symbol.is_parameter() and arg_symbol.param_type != ParamType.INPUT:
                        self.errors.append(SemanticError(
                            f"Parameter '{proc_symbol.param_names[i]}' marked with 'I' can only receive "
                            f"arguments from 'I' positions, but '{arg_name}' is not marked with 'I'",
                            self.current_line
                        ))
                
                # Rule 5: O parameters cannot be passed to subprocedure in place marked by I
                if param_type == ParamType.INPUT and arg_symbol.is_parameter():
                    if arg_symbol.param_type == ParamType.OUTPUT:
                        self.errors.append(SemanticError(
                            f"Parameter '{proc_symbol.param_names[i]}' marked with 'I' cannot receive "
                            f"argument '{arg_name}' which is marked with 'O'",
                            self.current_line
                        ))
                
                # Check parameter passing rules (type compatibility, scope, etc.)
                try:
                    self.symbol_table.check_parameter_passing(
                        proc_symbol.param_names[i], arg_name, param_type, self.current_line
                    )
                except SymbolTableError as e:
                    self.errors.append(SemanticError(e.message, e.line))
        
        # Exit procedure call
        self.symbol_table.exit_procedure_call()
    
    def visit_read(self, cmd: ReadCommand):
        """Visit ReadCommand - validate READ statement"""
        if isinstance(cmd.identifier, Identifier):
            # Variable read
            try:
                self.symbol_table.check_variable_usage(
                    cmd.identifier.name, is_read=False, is_write=True, line=self.current_line
                )
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
        elif isinstance(cmd.identifier, Access):
            # Array read
            try:
                array_symbol = self.symbol_table.lookup_required(
                    cmd.identifier.name, self.current_line
                )
                if not array_symbol.is_array():
                    self.errors.append(SemanticError(
                        f"'{cmd.identifier.name}' is not an array", self.current_line
                    ))
                    return
                
                # Validate index
                self.visit_value(cmd.identifier.index)
                
                # Check if array can be written to
                try:
                    self.symbol_table.check_variable_usage(
                        cmd.identifier.name, is_read=False, is_write=True, line=self.current_line
                    )
                except SymbolTableError as e:
                    self.errors.append(SemanticError(e.message, e.line))
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
        else:
            self.errors.append(SemanticError(
                f"Invalid READ target type: {type(cmd.identifier)}", self.current_line
            ))
    
    def visit_write(self, cmd: WriteCommand):
        """Visit WriteCommand - validate WRITE statement"""
        # Validate value
        self.visit_value(cmd.value)
    
    def visit_binary_expression(self, node: BinaryExpression):
        """Visit BinaryExpression - validate binary operation"""
        # Validate both operands
        self.visit_expression(node.left)
        self.visit_expression(node.right)
    
    def visit_value(self, node: Value):
        """Visit Value - validate value usage"""
        if isinstance(node.value, int):
            # Constant - no validation needed
            return
        elif isinstance(node.value, Identifier):
            # Variable read
            try:
                self.symbol_table.check_variable_usage(
                    node.value.name, is_read=True, is_write=False, line=self.current_line
                )
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
        elif isinstance(node.value, Access):
            # Array access
            try:
                array_symbol = self.symbol_table.lookup_required(
                    node.value.name, self.current_line
                )
                if not array_symbol.is_array():
                    self.errors.append(SemanticError(
                        f"'{node.value.name}' is not an array", self.current_line
                    ))
                    return
                
                # Validate index
                self.visit_value(node.value.index)
                
                # Check if array can be read
                try:
                    self.symbol_table.check_variable_usage(
                        node.value.name, is_read=True, is_write=False, line=self.current_line
                    )
                except SymbolTableError as e:
                    self.errors.append(SemanticError(e.message, e.line))
            except SymbolTableError as e:
                self.errors.append(SemanticError(e.message, e.line))
        else:
            self.errors.append(SemanticError(
                f"Invalid value type: {type(node.value)}", self.current_line
            ))
    
    def visit_condition(self, node: Condition):
        """Visit Condition - validate condition"""
        # Validate both sides
        self.visit_value(node.left)
        self.visit_value(node.right)
    
    def visit_identifier(self, node: Identifier):
        """Visit Identifier - validate identifier usage"""
        try:
            self.symbol_table.check_variable_usage(
                node.name, is_read=True, is_write=False, line=self.current_line
            )
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
    
    def visit_access(self, node: Access):
        """Visit Access - validate array access"""
        try:
            array_symbol = self.symbol_table.lookup_required(
                node.name, self.current_line
            )
            if not array_symbol.is_array():
                self.errors.append(SemanticError(
                    f"'{node.name}' is not an array", self.current_line
                ))
                return
            
            # Validate index
            self.visit_value(node.index)
        except SymbolTableError as e:
            self.errors.append(SemanticError(e.message, e.line))
