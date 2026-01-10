from sly import Lexer
class CompilerLexer(Lexer):
    tokens = {
        PROGRAM, PROCEDURE, IS, IN, END, IF, THEN, ELSE, ENDIF,
        WHILE, DO, ENDWHILE, REPEAT, UNTIL, FOR, FROM, TO, DOWNTO, ENDFOR,
        READ, WRITE,
        ASSIGN, EQ, NE, GT, LT, GE, LE,
        PLUS, MINUS, MUL, DIV, MOD,
        SEMICOLON, COMMA, LPAREN, RPAREN, LBRACKET, RBRACKET, COLON,
        PIDENTIFIER, NUM, TYPE_I, TYPE_O, TYPE_T
    }
    
    ignore = ' \t\n'
    ignore_comment = r'#.*'
    
    ASSIGN = r':='
    GE = r'>='
    LE = r'<='
    NE = r'!='
    GT = r'>'
    LT = r'<'
    EQ = r'='
    SEMICOLON = r';'
    COMMA = r','
    LPAREN = r'\('
    RPAREN = r'\)'
    LBRACKET = r'\['
    RBRACKET = r'\]'
    COLON = r':'
    PLUS = r'\+'
    MINUS = r'-'
    MUL = r'\*'
    DIV = r'/'
    MOD = r'%'
    
    PROGRAM = r'PROGRAM'
    PROCEDURE = r'PROCEDURE'
    IS = r'IS'
    IN = r'IN'
    ENDIF = r'ENDIF'
    ENDWHILE = r'ENDWHILE'
    ENDFOR = r'ENDFOR'
    END = r'END'
    IF = r'IF'
    THEN = r'THEN'
    ELSE = r'ELSE'
    WHILE = r'WHILE'
    DO = r'DO'
    REPEAT = r'REPEAT'
    UNTIL = r'UNTIL'
    FOR = r'FOR'
    FROM = r'FROM'
    TO = r'TO'
    DOWNTO = r'DOWNTO'
    READ = r'READ'
    WRITE = r'WRITE'
    
    TYPE_T = r'T'
    TYPE_I = r'I'
    TYPE_O = r'O'
    PIDENTIFIER = r'[_a-z]+'
    NUM = r'[0-9]+'
    
    
    def error(self, t):
        raise SyntaxError(f"Unexpected token: {t.value[0]} at line {self.lineno}")