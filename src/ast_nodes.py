from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Union


class ASTNode(ABC):
    """Base class for all AST nodes"""
    def accept(self, visitor):
        """Accept a visitor - must be implemented by subclasses"""
        raise NotImplementedError(f"accept() not implemented for {type(self).__name__}")


@dataclass
class Identifier(ASTNode):
    name: str
    
    def accept(self, visitor):
        return visitor.visit_identifier(self)


@dataclass
class Access(ASTNode):
    """Array access: array_name[index]"""
    name: str
    index: 'Value'  # The index expression (can be a variable, number, or expression)
    
    def accept(self, visitor):
        return visitor.visit_access(self)


# Expression nodes
class Expression(ASTNode):
    """Abstract base class for expressions"""
    pass


@dataclass
class Value(Expression):
    """Expression representing a single value (number, identifier, or array access)"""
    value: Union[int, Identifier, Access]
    
    def accept(self, visitor):
        return visitor.visit_value(self)


@dataclass
class BinaryExpression(Expression):
    """Expression representing a binary operation"""
    operator: str  # '+', '-', '*', '/', '%'
    left: Expression
    right: Expression
    
    def accept(self, visitor):
        return visitor.visit_binary_expression(self)


# Condition nodes
@dataclass
class Condition(ASTNode):
    operator: str  # '=', '!=', '>', '<', '>=', '<='
    left: Value
    right: Value
    
    def accept(self, visitor):
        return visitor.visit_condition(self)


# Statement nodes
class Statement(ASTNode):
    """Abstract base class for statements"""
    pass


# Command/Statement nodes
@dataclass
class AssignCommand(Statement):
    identifier: Union[Identifier, 'Access']
    expression: Expression
    
    def accept(self, visitor):
        return visitor.visit_assign(self)


@dataclass
class IfCommand(Statement):
    condition: Condition
    then_commands: List['Command']
    else_commands: Optional[List['Command']] = None
    
    def accept(self, visitor):
        return visitor.visit_if(self)


@dataclass
class WhileCommand(Statement):
    condition: Condition
    commands: List['Command']
    
    def accept(self, visitor):
        return visitor.visit_while(self)


@dataclass
class RepeatCommand(Statement):
    commands: List['Command']
    condition: Condition
    
    def accept(self, visitor):
        return visitor.visit_repeat(self)


@dataclass
class ForCommand(Statement):
    var: str
    from_val: Value
    to_val: Value
    commands: List['Command']
    downto: bool = False
    
    def accept(self, visitor):
        return visitor.visit_for(self)


@dataclass
class ProcCallCommand(Statement):
    name: str
    args: List[str]
    
    def accept(self, visitor):
        return visitor.visit_proc_call(self)


@dataclass
class ReadCommand(Statement):
    identifier: Union[Identifier, 'Access']
    
    def accept(self, visitor):
        return visitor.visit_read(self)


@dataclass
class WriteCommand(Statement):
    value: Value
    
    def accept(self, visitor):
        return visitor.visit_write(self)


Command = Union[AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand, ProcCallCommand, ReadCommand, WriteCommand]


# Declaration nodes
@dataclass
class Declaration(ASTNode):
    name: str
    array_range: Optional[tuple] = None  # (start, end) tuple or None
    
    def accept(self, visitor):
        return visitor.visit_declaration(self)


# Procedure argument declaration
@dataclass
class ArgDecl(ASTNode):
    arg_type: Optional[str]  # type is 'T', 'I', 'O', or None (empty)
    name: str
    
    def accept(self, visitor):
        return visitor.visit_arg_decl(self)


@dataclass
class ProcHead(ASTNode):
    name: str
    args: List[ArgDecl]
    
    def accept(self, visitor):
        return visitor.visit_proc_head(self)


@dataclass
class Procedure(ASTNode):
    head: ProcHead
    declarations: List[Declaration]
    commands: List[Command]
    
    def accept(self, visitor):
        return visitor.visit_procedure(self)


@dataclass
class Main(ASTNode):
    declarations: List[Declaration]
    commands: List[Command]
    
    def accept(self, visitor):
        return visitor.visit_main(self)


@dataclass
class Program(ASTNode):
    procedures: List[Procedure]
    main: Main
    
    def accept(self, visitor):
        return visitor.visit_program(self)


def display_ast(node: ASTNode, indent: int = 0) -> str:
    """Display the AST in a tree-like format"""
    indent_str = "  " * indent
    
    if isinstance(node, Program):
        lines = ["Program:"]
        for proc in node.procedures:
            lines.append(display_ast(proc, indent + 1))
        lines.append(display_ast(node.main, indent + 1))
        return "\n".join(lines)
    
    elif isinstance(node, Procedure):
        lines = [f"{indent_str}Procedure: {node.head.name}"]
        if node.head.args:
            args_str = ", ".join([f"{'(' + a.arg_type + ')' if a.arg_type else '()'}{a.name}" 
                                  for a in node.head.args])
            lines.append(f"{indent_str}  Args: {args_str}")
        if node.declarations:
            lines.append(f"{indent_str}  Declarations:")
            for decl in node.declarations:
                lines.append(display_ast(decl, indent + 2))
        if node.commands:
            lines.append(f"{indent_str}  Commands:")
            for cmd in node.commands:
                lines.append(display_ast(cmd, indent + 2))
        return "\n".join(lines)
    
    elif isinstance(node, Main):
        lines = [f"{indent_str}Main:"]
        if node.declarations:
            lines.append(f"{indent_str}  Declarations:")
            for decl in node.declarations:
                lines.append(display_ast(decl, indent + 2))
        if node.commands:
            lines.append(f"{indent_str}  Commands:")
            for cmd in node.commands:
                lines.append(display_ast(cmd, indent + 2))
        return "\n".join(lines)
    
    elif isinstance(node, Declaration):
        if node.array_range:
            return f"{indent_str}{node.name}[{node.array_range[0]}:{node.array_range[1]}]"
        return f"{indent_str}{node.name}"
    
    elif isinstance(node, AssignCommand):
        return f"{indent_str}Assign: {display_ast(node.identifier, 0)} := {display_ast(node.expression, 0)}"
    
    elif isinstance(node, IfCommand):
        lines = [f"{indent_str}If: {display_ast(node.condition, 0)}"]
        lines.append(f"{indent_str}  Then:")
        for cmd in node.then_commands:
            lines.append(display_ast(cmd, indent + 2))
        if node.else_commands:
            lines.append(f"{indent_str}  Else:")
            for cmd in node.else_commands:
                lines.append(display_ast(cmd, indent + 2))
        return "\n".join(lines)
    
    elif isinstance(node, WhileCommand):
        lines = [f"{indent_str}While: {display_ast(node.condition, 0)}"]
        for cmd in node.commands:
            lines.append(display_ast(cmd, indent + 1))
        return "\n".join(lines)
    
    elif isinstance(node, RepeatCommand):
        lines = [f"{indent_str}Repeat:"]
        for cmd in node.commands:
            lines.append(display_ast(cmd, indent + 1))
        lines.append(f"{indent_str}Until: {display_ast(node.condition, 0)}")
        return "\n".join(lines)
    
    elif isinstance(node, ForCommand):
        direction = "DOWNTO" if node.downto else "TO"
        lines = [f"{indent_str}For: {node.var} FROM {display_ast(node.from_val, 0)} {direction} {display_ast(node.to_val, 0)}"]
        for cmd in node.commands:
            lines.append(display_ast(cmd, indent + 1))
        return "\n".join(lines)
    
    elif isinstance(node, ProcCallCommand):
        args_str = ", ".join(node.args)
        return f"{indent_str}Call: {node.name}({args_str})"
    
    elif isinstance(node, ReadCommand):
        return f"{indent_str}Read: {display_ast(node.identifier, 0)}"
    
    elif isinstance(node, WriteCommand):
        return f"{indent_str}Write: {display_ast(node.value, 0)}"
    
    elif isinstance(node, Condition):
        return f"{node.left} {node.operator} {node.right}"
    
    elif isinstance(node, BinaryExpression):
        return f"({display_ast(node.left, 0)} {node.operator} {display_ast(node.right, 0)})"
    
    elif isinstance(node, Value):
        if isinstance(node.value, int):
            return str(node.value)
        return display_ast(node.value, 0)
    
    elif isinstance(node, Identifier):
        return node.name
    
    elif isinstance(node, Access):
        return f"{node.name}[{display_ast(node.index, 0)}]"
    
    else:
        return f"{indent_str}{type(node).__name__}"