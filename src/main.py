from lexer import CompilerLexer, LexicalError
from parser import CompilerParser
from ast_nodes import display_ast
import sys


def main():
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            text = f.read()
    else:
        # Read from stdin
        text = sys.stdin.read()
    
    lexer = CompilerLexer()
    parser = CompilerParser()
    
    try:
        ast = parser.parse(lexer.tokenize(text))
        print(display_ast(ast))
    except LexicalError as e:
        print(f"Lexical Error: {e.message}", file=sys.stderr)
        sys.exit(1)
    except SyntaxError as e:
        print(f"Syntax Error: {e.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()