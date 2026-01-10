from lexer import CompilerLexer
from parser import Parser
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
    parser = Parser()
    
    try:
        ast = parser.parse(lexer.tokenize(text))
        print(display_ast(ast))
    except SyntaxError as e:
        print(f"Syntax Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()