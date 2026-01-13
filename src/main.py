from lexer import CompilerLexer, LexicalError
from parser import CompilerParser
from ast_nodes import display_ast
from semantic_analyzer import SemanticAnalyzer
from tac import TACGenerator, display_tac
from vm_generator import VMGenerator
import sys
import os


def main():
    if len(sys.argv) > 1:
        # Read from file
        input_file = sys.argv[1]
        with open(input_file, 'r') as f:
            text = f.read()
    else:
        # Read from stdin
        input_file = None
        text = sys.stdin.read()
    
    lexer = CompilerLexer()
    parser = CompilerParser()
    analyzer = SemanticAnalyzer()
    
    try:
        ast = parser.parse(lexer.tokenize(text))
        errors = analyzer.analyze(ast)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            sys.exit(1)
        
        # Generate TAC - pass symbol table for scope-aware code generation
        tac_gen = TACGenerator(symbol_table=analyzer.symbol_table)
        tac_instructions = tac_gen.generate(ast)
        
        # Generate VM code
        vm_gen = VMGenerator()
        vm_code = vm_gen.generate(tac_instructions)
        
        # Determine output file names for debugging
        if input_file:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            ast_file = f"{base_name}.ast"
            tac_file = f"{base_name}.tac"
            vm_file = f"{base_name}.vm"
        else:
            ast_file = "output.ast"
            tac_file = "output.tac"
            vm_file = "output.vm"
        
        # Dump AST to file
        with open(ast_file, 'w') as f:
            f.write(display_ast(ast))
        
        # Dump TAC to file
        with open(tac_file, 'w') as f:
            f.write(display_tac(tac_instructions))
        
        # Dump VM code to file
        with open(vm_file, 'w') as f:
            f.write(vm_code)
        
    except LexicalError as e:
        print(f"Lexical Error: {e}", file=sys.stderr)
        sys.exit(1)
    except SyntaxError as e:
        print(f"Syntax Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()