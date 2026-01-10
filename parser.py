from sly import Parser as SlyParser
from lexer import CompilerLexer
from ast_nodes import (
    Program, Procedure, Main, Declaration, ArgDecl, ProcHead,
    AssignCommand, IfCommand, WhileCommand, RepeatCommand, ForCommand,
    ProcCallCommand, ReadCommand, WriteCommand,
    Condition, Expression, Value, Identifier
)


class Parser(SlyParser):
    tokens = CompilerLexer.tokens
    start = 'program_all'
    
    def __init__(self, lexer=None):
        # sly.Parser doesn't need lexer in constructor, but we keep it for compatibility
        self.lexer = lexer
        super().__init__()
    
    def error(self, token):
        if token:
            raise SyntaxError(f"Unexpected token '{token.value}' at line {token.lineno}")
        else:
            raise SyntaxError("Unexpected end of file")
    
    # Grammar rules
    @_('procedures main')
    def program_all(self, p):
        return Program(p.procedures, p.main)
    
    @_('procedures procedure')
    def procedures(self, p):
        return p.procedures + [p.procedure]
    
    @_('')
    def procedures(self, p):
        return []
    
    @_('PROCEDURE proc_head IS declarations IN commands END')
    def procedure(self, p):
        return Procedure(p.proc_head, p.declarations, p.commands)
    
    @_('PROCEDURE proc_head IS IN commands END')
    def procedure(self, p):
        return Procedure(p.proc_head, [], p.commands)
    
    @_('PROGRAM IS declarations IN commands END')
    def main(self, p):
        return Main(p.declarations, p.commands)
    
    @_('PROGRAM IS IN commands END')
    def main(self, p):
        return Main([], p.commands)
    
    @_('commands command')
    def commands(self, p):
        return p.commands + [p.command]
    
    @_('command')
    def commands(self, p):
        return [p.command]
    
    @_('identifier ASSIGN expression SEMICOLON')
    def command(self, p):
        return AssignCommand(p.identifier, p.expression)
    
    @_('IF condition THEN commands ELSE commands ENDIF')
    def command(self, p):
        return IfCommand(p.condition, p.commands0, p.commands1)
    
    @_('IF condition THEN commands ENDIF')
    def command(self, p):
        return IfCommand(p.condition, p.commands, None)
    
    @_('WHILE condition DO commands ENDWHILE')
    def command(self, p):
        return WhileCommand(p.condition, p.commands)
    
    @_('REPEAT commands UNTIL condition SEMICOLON')
    def command(self, p):
        return RepeatCommand(p.commands, p.condition)
    
    @_('FOR PIDENTIFIER FROM value TO value DO commands ENDFOR')
    def command(self, p):
        return ForCommand(p.PIDENTIFIER, p.value0, p.value1, p.commands, False)
    
    @_('FOR PIDENTIFIER FROM value DOWNTO value DO commands ENDFOR')
    def command(self, p):
        return ForCommand(p.PIDENTIFIER, p.value0, p.value1, p.commands, True)
    
    @_('PIDENTIFIER LPAREN args RPAREN SEMICOLON')
    def command(self, p):
        return ProcCallCommand(p.PIDENTIFIER, p.args)
    
    @_('READ identifier SEMICOLON')
    def command(self, p):
        return ReadCommand(p.identifier)
    
    @_('WRITE value SEMICOLON')
    def command(self, p):
        return WriteCommand(p.value)
    
    @_('PIDENTIFIER LPAREN RPAREN')
    def proc_head(self, p):
        return ProcHead(p.PIDENTIFIER, [])
    
    @_('PIDENTIFIER LPAREN args_decl RPAREN')
    def proc_head(self, p):
        return ProcHead(p.PIDENTIFIER, p.args_decl)
    
    @_('args_decl COMMA type PIDENTIFIER')
    def args_decl(self, p):
        return p.args_decl + [ArgDecl(p.type, p.PIDENTIFIER)]
    
    @_('type PIDENTIFIER')
    def args_decl(self, p):
        return [ArgDecl(p.type, p.PIDENTIFIER)]
    
    @_('TYPE_T')
    def type(self, p):
        return 'T'
    
    @_('TYPE_I')
    def type(self, p):
        return 'I'
    
    @_('TYPE_O')
    def type(self, p):
        return 'O'
    
    @_('')
    def type(self, p):
        return None
    
    @_('args COMMA PIDENTIFIER')
    def args(self, p):
        return p.args + [p.PIDENTIFIER]
    
    @_('PIDENTIFIER')
    def args(self, p):
        return [p.PIDENTIFIER]
    
    @_('declarations COMMA declaration_item')
    def declarations(self, p):
        return p.declarations + [p.declaration_item]
    
    @_('declaration_item')
    def declarations(self, p):
        return [p.declaration_item]
    
    @_('PIDENTIFIER')
    def declaration_item(self, p):
        return Declaration(p.PIDENTIFIER)
    
    @_('PIDENTIFIER LBRACKET NUM COLON NUM RBRACKET')
    def declaration_item(self, p):
        return Declaration(p.PIDENTIFIER, (int(p.NUM0), int(p.NUM1)))
    
    @_('value PLUS value')
    def expression(self, p):
        return Expression('+', p.value0, p.value1)
    
    @_('value MINUS value')
    def expression(self, p):
        return Expression('-', p.value0, p.value1)
    
    @_('value MUL value')
    def expression(self, p):
        return Expression('*', p.value0, p.value1)
    
    @_('value DIV value')
    def expression(self, p):
        return Expression('/', p.value0, p.value1)
    
    @_('value MOD value')
    def expression(self, p):
        return Expression('%', p.value0, p.value1)
    
    @_('value')
    def expression(self, p):
        return Expression(None, p.value, None)
    
    @_('value EQ value')
    def condition(self, p):
        return Condition('=', p.value0, p.value1)
    
    @_('value NE value')
    def condition(self, p):
        return Condition('!=', p.value0, p.value1)
    
    @_('value GT value')
    def condition(self, p):
        return Condition('>', p.value0, p.value1)
    
    @_('value LT value')
    def condition(self, p):
        return Condition('<', p.value0, p.value1)
    
    @_('value GE value')
    def condition(self, p):
        return Condition('>=', p.value0, p.value1)
    
    @_('value LE value')
    def condition(self, p):
        return Condition('<=', p.value0, p.value1)
    
    @_('NUM')
    def value(self, p):
        return Value(int(p.NUM))
    
    @_('identifier')
    def value(self, p):
        return Value(p.identifier)
    
    @_('PIDENTIFIER')
    def identifier(self, p):
        return Identifier(p.PIDENTIFIER)
    
    @_('PIDENTIFIER LBRACKET PIDENTIFIER RBRACKET')
    def identifier(self, p):
        return Identifier(p.PIDENTIFIER0, p.PIDENTIFIER1)
    
    @_('PIDENTIFIER LBRACKET NUM RBRACKET')
    def identifier(self, p):
        return Identifier(p.PIDENTIFIER, int(p.NUM))
