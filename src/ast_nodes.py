from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Union


class ASTNode(ABC):
    pass


# Value nodes
@dataclass
class Value(ASTNode):
    value: Union[int, 'Identifier']


@dataclass
class Identifier(ASTNode):
    name: str
    index: Optional[Union[str, int]] = None  # None for simple variable, str/int for array access


# Expression nodes
@dataclass
class Expression(ASTNode):
    operator: Optional[str]  # None for single value, '+', '-', '*', '/', '%' for operations
    left: Value
    right: Optional[Value] = None


# Condition nodes
@dataclass
class Condition(ASTNode):
    operator: str  # '=', '!=', '>', '<', '>=', '<='
    left: Value
    right: Value


# Command nodes
@dataclass
class AssignCommand(ASTNode):
    identifier: Identifier
    expression: Expression


@dataclass
class IfCommand(ASTNode):
    condition: Condition
    then_commands: List['Command']
    else_commands: Optional[List['Command']] = None


@dataclass
class WhileCommand(ASTNode):
    condition: Condition
    commands: List['Command']


@dataclass
class RepeatCommand(ASTNode):
    commands: List['Command']
    condition: Condition


@dataclass
class ForCommand(ASTNode):
    var: str
    from_val: Value
    to_val: Value
    commands: List['Command']
    downto: bool = False


@dataclass
class ProcCallCommand(ASTNode):
    name: str
    args: List[str]


@dataclass
class ReadCommand(ASTNode):
    identifier: Identifier


@dataclass
class WriteCommand(ASTNode):
    value: Value


Command = Union[AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand, ProcCallCommand, ReadCommand, WriteCommand]


# Declaration nodes
@dataclass
class Declaration(ASTNode):
    name: str
    array_range: Optional[tuple] = None  # (start, end) tuple or None


# Procedure argument declaration
@dataclass
class ArgDecl(ASTNode):
    arg_type: Optional[str]  # type is 'T', 'I', 'O', or None (empty)
    name: str


@dataclass
class ProcHead(ASTNode):
    name: str
    args: List[ArgDecl]


@dataclass
class Procedure(ASTNode):
    head: ProcHead
    declarations: List[Declaration]
    commands: List[Command]


@dataclass
class Main(ASTNode):
    declarations: List[Declaration]
    commands: List[Command]


@dataclass
class Program(ASTNode):
    procedures: List[Procedure]
    main: Main


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
    
    elif isinstance(node, Expression):
        if node.operator is None:
            return display_ast(node.left, 0)
        return f"({display_ast(node.left, 0)} {node.operator} {display_ast(node.right, 0)})"
    
    elif isinstance(node, Value):
        if isinstance(node.value, int):
            return str(node.value)
        return display_ast(node.value, 0)
    
    elif isinstance(node, Identifier):
        if node.index is None:
            return node.name
        return f"{node.name}[{node.index}]"
    
    else:
        return f"{indent_str}{type(node).__name__}"