from lexer import CompilerLexer, LexicalError
from parser import CompilerParser
from ast_nodes import display_ast
from semantic_analyzer import SemanticAnalyzer
from tac import TACGenerator, display_tac
from vm_generator import VMGenerator
import sys
import os

DEBUG = True
def main(debug: bool, input_file: str, output_file: str):
    with open(input_file, 'r') as f:
        text = f.read()
    
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
        
        # Generate TAC (pass symbol table for variable name qualification)
        tac_gen = TACGenerator(symbol_table=analyzer.symbol_table)
        tac_instructions = tac_gen.generate(ast)
        
        # Generate VM code (pass symbol table for array size information)
        vm_gen = VMGenerator(symbol_table=analyzer.symbol_table)
        vm_code = vm_gen.generate(tac_instructions)
    
        
        if debug:
            with open("debug.ast", 'w') as f:
                f.write(display_ast(ast))
            
            # Dump TAC to file
            with open("debug.tac", 'w') as f:
                f.write(display_tac(tac_instructions))
        
        # Dump VM code to file
        with open(output_file, 'w') as f:
            f.write(vm_code)
        
    except LexicalError as e:
        print(f"Lexical Error: {e}", file=sys.stderr)
        sys.exit(1)
    except SyntaxError as e:
        print(f"Syntax Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    main(debug=DEBUG, input_file=sys.argv[1], output_file=sys.argv[2])