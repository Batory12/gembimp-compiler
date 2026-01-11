from sly import Lexer

class LexicalError(Exception):
    def __init__(self, character: str, line: int = None):
        self.character = character
        self.line = line
        super().__init__(f"Illegal character: {character} at line {line}")

class CompilerLexer(Lexer):
    tokens = {
        'PROGRAM', 'PROCEDURE', 'IS', 'IN', 'END', 'IF', 'THEN', 'ELSE', 'ENDIF',
        'WHILE', 'DO', 'ENDWHILE', 'REPEAT', 'UNTIL', 'FOR', 'FROM', 'TO', 'DOWNTO', 'ENDFOR',
        'READ', 'WRITE',
        'ASSIGN', 'EQ', 'NE', 'GT', 'LT', 'GE', 'LE',
        'PLUS', 'MINUS', 'MUL', 'DIV', 'MOD',
        'SEMICOLON', 'COMMA', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'COLON',
        'PIDENTIFIER', 'NUM', 'TYPE_I', 'TYPE_O', 'TYPE_T'
    }
    
    ignore = ' \t'
    ignore_comment = r'#.*'
    
    PROCEDURE   = r'PROCEDURE'
    ENDWHILE    = r'ENDWHILE'
    PROGRAM     = r'PROGRAM'
    DOWNTO      = r'DOWNTO'
    ENDFOR      = r'ENDFOR'
    REPEAT      = r'REPEAT'
    UNTIL       = r'UNTIL'
    ENDIF       = r'ENDIF'
    WRITE       = r'WRITE'
    WHILE       = r'WHILE'
    THEN        = r'THEN'
    ELSE        = r'ELSE'
    FROM        = r'FROM'
    READ        = r'READ'
    END         = r'END'
    FOR         = r'FOR'
    IS          = r'IS'
    IF          = r'IF'
    TO          = r'TO'
    IN          = r'IN'
    DO          = r'DO'
    GE          = r'>='
    LE          = r'<='
    ASSIGN      = r':='
    NE          = r'!='
    GT          = r'>'
    LT          = r'<'
    EQ          = r'='
    SEMICOLON   = r';'
    COMMA       = r','
    LPAREN      = r'\('
    RPAREN      = r'\)'
    LBRACKET    = r'\['
    RBRACKET    = r'\]'
    COLON       = r':'
    PLUS        = r'\+'
    MINUS       = r'-'
    MUL         = r'\*'
    DIV         = r'/'
    MOD         = r'%'
    TYPE_T      = r'T'
    TYPE_I      = r'I'
    TYPE_O      = r'O'
    PIDENTIFIER = r'[_a-z]+'
    NUM         = r'[0-9]+'
    
    # Define a rule so we can track line numbers
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)
    
    def error(self, t):
        raise LexicalError(t.value[0], self.lineno)