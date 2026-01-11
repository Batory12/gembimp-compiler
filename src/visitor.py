"""Base visitor class for AST traversal"""
from abc import ABC
from typing import Optional, Any
from ast_nodes import (
    Program, Procedure, Main, Declaration, ArgDecl, ProcHead,
    AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand,
    ProcCallCommand, ReadCommand, WriteCommand,
    Condition, Expression, BinaryExpression, Value, Identifier, Access
)


class ASTVisitor(ABC):
    """Base class for all AST visitors.
    
    Each visitor implements specific operations by overriding visit methods.
    Subclasses only need to override methods for nodes they care about.
    """
    
    # Top-level nodes
    def visit_program(self, node: Program) -> Any:
        """Visit Program node"""
        # Visit main first, then procedures
        self.visit_main(node.main)
        for proc in node.procedures:
            self.visit_procedure(proc)
    
    def visit_procedure(self, node: Procedure) -> Any:
        """Visit Procedure node"""
        self.visit_proc_head(node.head)
        for decl in node.declarations:
            self.visit_declaration(decl)
        for cmd in node.commands:
            self.visit_command(cmd)
    
    def visit_main(self, node: Main) -> Any:
        """Visit Main node"""
        for decl in node.declarations:
            self.visit_declaration(decl)
        for cmd in node.commands:
            self.visit_command(cmd)
    
    def visit_proc_head(self, node: ProcHead) -> Any:
        """Visit ProcHead node"""
        for arg in node.args:
            self.visit_arg_decl(arg)
    
    def visit_arg_decl(self, node: ArgDecl) -> Any:
        """Visit ArgDecl node"""
        pass
    
    def visit_declaration(self, node: Declaration) -> Any:
        """Visit Declaration node"""
        pass
    
    # Command nodes
    def visit_command(self, cmd) -> Any:
        """Dispatch to specific command visitor"""
        if isinstance(cmd, AssignCommand):
            return self.visit_assign(cmd)
        elif isinstance(cmd, IfCommand):
            return self.visit_if(cmd)
        elif isinstance(cmd, WhileCommand):
            return self.visit_while(cmd)
        elif isinstance(cmd, RepeatCommand):
            return self.visit_repeat(cmd)
        elif isinstance(cmd, ForCommand):
            return self.visit_for(cmd)
        elif isinstance(cmd, ProcCallCommand):
            return self.visit_proc_call(cmd)
        elif isinstance(cmd, ReadCommand):
            return self.visit_read(cmd)
        elif isinstance(cmd, WriteCommand):
            return self.visit_write(cmd)
        else:
            raise ValueError(f"Unsupported command type: {type(cmd)}")
    
    def visit_assign(self, node: AssignCommand) -> Any:
        """Visit AssignCommand node"""
        self.visit_identifier_or_access(node.identifier)
        self.visit_expression(node.expression)
    
    def visit_if(self, node: IfCommand) -> Any:
        """Visit IfCommand node"""
        self.visit_condition(node.condition)
        for cmd in node.then_commands:
            self.visit_command(cmd)
        if node.else_commands:
            for cmd in node.else_commands:
                self.visit_command(cmd)
    
    def visit_while(self, node: WhileCommand) -> Any:
        """Visit WhileCommand node"""
        self.visit_condition(node.condition)
        for cmd in node.commands:
            self.visit_command(cmd)
    
    def visit_repeat(self, node: RepeatCommand) -> Any:
        """Visit RepeatCommand node"""
        for cmd in node.commands:
            self.visit_command(cmd)
        self.visit_condition(node.condition)
    
    def visit_for(self, node: ForCommand) -> Any:
        """Visit ForCommand node"""
        self.visit_value(node.from_val)
        self.visit_value(node.to_val)
        for cmd in node.commands:
            self.visit_command(cmd)
    
    def visit_proc_call(self, node: ProcCallCommand) -> Any:
        """Visit ProcCallCommand node"""
        pass
    
    def visit_read(self, node: ReadCommand) -> Any:
        """Visit ReadCommand node"""
        self.visit_identifier_or_access(node.identifier)
    
    def visit_write(self, node: WriteCommand) -> Any:
        """Visit WriteCommand node"""
        self.visit_value(node.value)
    
    # Expression nodes
    def visit_expression(self, expr: Expression) -> Any:
        """Dispatch to specific expression visitor"""
        if isinstance(expr, Value):
            return self.visit_value(expr)
        elif isinstance(expr, BinaryExpression):
            return self.visit_binary_expression(expr)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")
    
    def visit_binary_expression(self, node: BinaryExpression) -> Any:
        """Visit BinaryExpression node"""
        self.visit_expression(node.left)
        self.visit_expression(node.right)
    
    def visit_value(self, node: Value) -> Any:
        """Visit Value node"""
        if isinstance(node.value, int):
            return  # Constant
        elif isinstance(node.value, Identifier):
            self.visit_identifier(node.value)
        elif isinstance(node.value, Access):
            self.visit_access(node.value)
    
    # Condition nodes
    def visit_condition(self, node: Condition) -> Any:
        """Visit Condition node"""
        self.visit_value(node.left)
        self.visit_value(node.right)
    
    # Identifier and Access nodes
    def visit_identifier_or_access(self, node) -> Any:
        """Visit Identifier or Access node"""
        if isinstance(node, Identifier):
            return self.visit_identifier(node)
        elif isinstance(node, Access):
            return self.visit_access(node)
        else:
            raise ValueError(f"Unsupported identifier type: {type(node)}")
    
    def visit_identifier(self, node: Identifier) -> Any:
        """Visit Identifier node"""
        pass
    
    def visit_access(self, node: Access) -> Any:
        """Visit Access node"""
        self.visit_value(node.index)
