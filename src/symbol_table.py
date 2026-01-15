"""
Symbol table implementation for the compiler.
Tracks variables, arrays, procedures, and their scopes according to the language rules.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class SymbolType(Enum):
    """Type of symbol"""
    VARIABLE = "variable"
    ARRAY = "array"
    PROCEDURE = "procedure"
    FOR_ITERATOR = "for_iterator"  # Special type for FOR loop iterators


class ParamType(Enum):
    """Parameter type for procedure arguments"""
    NORMAL = None  # Default: IN-OUT (by reference)
    INPUT = "I"    # Input only (constant, cannot modify)
    OUTPUT = "O"   # Output only (undefined, cannot read before assignment)
    ARRAY = "T"    # Array parameter (must be prefixed with T)


@dataclass
class Symbol:
    """Represents a symbol in the symbol table"""
    name: str
    symbol_type: SymbolType
    # For arrays
    array_start: Optional[int] = None
    array_end: Optional[int] = None
    # For procedures
    param_types: Optional[List[ParamType]] = None  # Types of formal parameters
    param_names: Optional[List[str]] = None  # Names of formal parameters
    # For procedure parameters
    param_type: Optional[ParamType] = None  # Type of this parameter (if this is a parameter)
    # For OUTPUT parameters - track if they've been assigned
    output_assigned: bool = False  # True if OUTPUT parameter has been assigned
    # Scope information
    scope_level: int = 0  # 0 = global, 1+ = nested scopes
    declared_at_line: Optional[int] = None  # For error reporting
    
    def is_array(self) -> bool:
        """Check if this symbol is an array"""
        return self.symbol_type == SymbolType.ARRAY
    
    def is_procedure(self) -> bool:
        """Check if this symbol is a procedure"""
        return self.symbol_type == SymbolType.PROCEDURE
    
    def is_for_iterator(self) -> bool:
        """Check if this symbol is a FOR loop iterator"""
        return self.symbol_type == SymbolType.FOR_ITERATOR
    
    def is_parameter(self) -> bool:
        """Check if this symbol is a procedure parameter"""
        return self.param_type is not None


class SymbolTableError(Exception):
    """Exception raised for symbol table errors"""
    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line
        super().__init__(self.message)


class SymbolTable:
    """Symbol table with scope management"""
    
    def __init__(self):
        # Global scope symbols
        self.global_symbols: Dict[str, Symbol] = {}
        # Procedure symbols: procedure_name -> {param_name -> Symbol, local_name -> Symbol}
        self.procedure_symbols: Dict[str, Dict[str, Symbol]] = {}
        # FOR loop iterators: stack of iterators (for nested FOR loops)
        self.for_iterators: List[Symbol] = []
        # Current procedure name (None if in global scope)
        self.current_procedure: Optional[str] = None
        # Scope level tracking
        self.scope_level: int = 0
        # Procedure definitions in order (for checking forward references)
        self.procedure_order: List[str] = []
        # Current procedure call stack (for detecting recursion)
        self.call_stack: List[str] = []
    
    def enter_global_scope(self):
        """Enter global scope"""
        self.current_procedure = None
        self.scope_level = 0
    
    def enter_procedure_scope(self, proc_name: str):
        """Enter procedure scope"""
        if proc_name in self.procedure_symbols:
            raise SymbolTableError(f"Procedure '{proc_name}' already defined")
        
        self.current_procedure = proc_name
        self.procedure_symbols[proc_name] = {}
        # Note: procedure_order is now populated in add_procedure, not here
        self.scope_level = 1
    
    def exit_procedure_scope(self):
        """Exit procedure scope"""
        self.current_procedure = None
        self.scope_level = 0
    
    def enter_for_scope(self, iterator_name: str, line: Optional[int] = None):
        """Enter FOR loop scope - add iterator to current scope"""
        # Check if iterator already exists in current scope
        existing = self.lookup(iterator_name)
        if existing and existing.is_for_iterator():
            raise SymbolTableError(
                f"FOR loop iterator '{iterator_name}' already declared in this scope",
                line
            )
        
        iterator = Symbol(
            name=iterator_name,
            symbol_type=SymbolType.FOR_ITERATOR,
            scope_level=self.scope_level,
            declared_at_line=line
        )
        self.for_iterators.append(iterator)
        return iterator
    
    def exit_for_scope(self):
        """Exit FOR loop scope - remove iterator"""
        if self.for_iterators:
            self.for_iterators.pop()
    
    def add_variable(self, name: str, line: Optional[int] = None) -> Symbol:
        """Add a variable to the current scope
        
        Variables declared in procedures are local and shadow (hide) any global variables
        with the same name. This allows procedures to have local variables with the same
        name as global variables.
        
        Args:
            name: Variable name
            line: Line number for error reporting
            
        Returns:
            The created symbol
            
        Raises:
            SymbolTableError: If variable already declared in current scope
        """
        if self.current_procedure is None:
            # Global variable
            if name in self.global_symbols:
                raise SymbolTableError(f"Variable '{name}' already declared in global scope", line)
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.VARIABLE,
                scope_level=0,
                declared_at_line=line
            )
            self.global_symbols[name] = symbol
            return symbol
        else:
            # Local variable in procedure (shadows global variables with same name)
            if name in self.procedure_symbols[self.current_procedure]:
                raise SymbolTableError(
                    f"Variable '{name}' already declared in procedure '{self.current_procedure}'",
                    line
                )
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.VARIABLE,
                scope_level=1,
                declared_at_line=line
            )
            self.procedure_symbols[self.current_procedure][name] = symbol
            return symbol
    
    def add_array(self, name: str, start: int, end: int, line: Optional[int] = None) -> Symbol:
        """Add an array to the current scope
        
        Arrays declared in procedures are local and shadow (hide) any global arrays
        with the same name. This allows procedures to have local arrays with the same
        name as global arrays.
        
        Args:
            name: Array name
            start: Start index
            end: End index
            line: Line number for error reporting
            
        Raises:
            SymbolTableError: If start > end or array already declared
        """
        # Rule 2: Check that start <= end
        if start > end:
            raise SymbolTableError(
                f"Array '{name}' declaration error: start index ({start}) must be <= end index ({end})",
                line
            )
        
        if self.current_procedure is None:
            # Global array
            if name in self.global_symbols:
                raise SymbolTableError(f"Array '{name}' already declared in global scope", line)
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.ARRAY,
                array_start=start,
                array_end=end,
                scope_level=0,
                declared_at_line=line
            )
            self.global_symbols[name] = symbol
            return symbol
        else:
            # Local array in procedure (shadows global arrays with same name)
            if name in self.procedure_symbols[self.current_procedure]:
                raise SymbolTableError(
                    f"Array '{name}' already declared in procedure '{self.current_procedure}'",
                    line
                )
            symbol = Symbol(
                name=name,
                symbol_type=SymbolType.ARRAY,
                array_start=start,
                array_end=end,
                scope_level=1,
                declared_at_line=line
            )
            self.procedure_symbols[self.current_procedure][name] = symbol
            return symbol
    
    def add_procedure_parameter(self, name: str, param_type: Optional[ParamType] = None,
                                is_array: bool = False, line: Optional[int] = None) -> Symbol:
        """Add a procedure parameter to the current procedure
        
        Args:
            name: Parameter name
            param_type: Parameter type (I, O, T, or None for normal)
            is_array: Whether this is an array parameter (should have T prefix)
            line: Line number for error reporting
            
        Raises:
            SymbolTableError: If parameter already declared or array name doesn't have T prefix
        """
        if self.current_procedure is None:
            raise SymbolTableError("Cannot add procedure parameter outside procedure scope", line)
        
        # Rule 3: Array names in formal parameters should be prefixed with T
        if is_array and param_type != ParamType.ARRAY:
            raise SymbolTableError(
                f"Array parameter '{name}' must be prefixed with 'T' in procedure '{self.current_procedure}'",
                line
            )
        
        if name in self.procedure_symbols[self.current_procedure]:
            raise SymbolTableError(
                f"Parameter '{name}' already declared in procedure '{self.current_procedure}'",
                line
            )
        
        # Determine the actual param_type
        if param_type is None:
            param_type = ParamType.NORMAL
        
        symbol = Symbol(
            name=name,
            symbol_type=SymbolType.ARRAY if is_array else SymbolType.VARIABLE,
            param_type=param_type,
            scope_level=1,
            declared_at_line=line
        )
        self.procedure_symbols[self.current_procedure][name] = symbol
        return symbol
    
    def add_procedure(self, name: str, param_names: List[str], 
                     param_types: List[ParamType], line: Optional[int] = None) -> Symbol:
        """Add a procedure definition
        
        Args:
            name: Procedure name
            param_names: List of parameter names
            param_types: List of parameter types (corresponding to param_names)
            line: Line number for error reporting
        """
        if name in self.global_symbols:
            raise SymbolTableError(f"Procedure '{name}' already declared", line)
        
        symbol = Symbol(
            name=name,
            symbol_type=SymbolType.PROCEDURE,
            param_names=param_names,
            param_types=param_types,
            scope_level=0,
            declared_at_line=line
        )
        self.global_symbols[name] = symbol
        # Add to procedure_order to track declaration order for forward reference checking
        self.procedure_order.append(name)
        return symbol
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in the current scope
        
        Search order (implements variable shadowing):
        1. FOR loop iterators (most local)
        2. Current procedure parameters and locals (shadows globals)
        3. Global symbols
        
        Variables declared in procedures shadow (hide) global variables with the same name.
        Local variables are always found before global variables.
        
        Returns:
            Symbol if found, None otherwise
        """
        # First check FOR loop iterators (most local scope)
        for iterator in reversed(self.for_iterators):
            if iterator.name == name:
                return iterator
        
        # Then check current procedure scope (local variables and parameters shadow globals)
        if self.current_procedure and name in self.procedure_symbols[self.current_procedure]:
            return self.procedure_symbols[self.current_procedure][name]
        
        # Finally check global scope (only if not shadowed by local)
        if name in self.global_symbols:
            return self.global_symbols[name]
        
        return None
    
    def lookup_required(self, name: str, line: Optional[int] = None) -> Symbol:
        """Look up a symbol, raising an error if not found"""
        symbol = self.lookup(name)
        if symbol is None:
            scope = f"procedure '{self.current_procedure}'" if self.current_procedure else "global scope"
            raise SymbolTableError(f"Symbol '{name}' not found in {scope}", line)
        return symbol
    
    def check_procedure_call(self, proc_name: str, arg_count: int, line: Optional[int] = None):
        """Check if a procedure call is valid
        
        Args:
            proc_name: Name of procedure to call
            arg_count: Number of arguments provided
            line: Line number for error reporting
            
        Raises:
            SymbolTableError: If procedure doesn't exist, wrong argument count, or recursive call
        """
        # Rule 4: Can only call procedures defined earlier
        if proc_name not in self.global_symbols:
            raise SymbolTableError(f"Procedure '{proc_name}' not found", line)
        
        proc_symbol = self.global_symbols[proc_name]
        if not proc_symbol.is_procedure():
            raise SymbolTableError(f"'{proc_name}' is not a procedure", line)
        
        # Rule 4: Check if procedure was defined before current procedure
        if self.current_procedure:
            try:
                current_idx = self.procedure_order.index(self.current_procedure)
                proc_idx = self.procedure_order.index(proc_name)
                # A procedure can only call procedures defined earlier (proc_idx < current_idx)
                # If proc_idx == current_idx, it's a direct self-call (recursion)
                # If proc_idx > current_idx, it's defined later (error)
                if proc_idx == current_idx:
                    # Direct recursion (procedure calling itself)
                    raise SymbolTableError(
                        f"Recursive call to procedure '{proc_name}' is not allowed",
                        line
                    )
                elif proc_idx > current_idx:
                    raise SymbolTableError(
                        f"Cannot call procedure '{proc_name}' - it is defined after current procedure '{self.current_procedure}'",
                        line
                    )
            except ValueError:
                # Should not happen if procedures are properly registered
                pass
        
        # Rule 3: Check for indirect recursion (through other procedures)
        if proc_name in self.call_stack:
            raise SymbolTableError(
                f"Recursive call to procedure '{proc_name}' is not allowed",
                line
            )
        
        # Check argument count
        expected_count = len(proc_symbol.param_names) if proc_symbol.param_names else 0
        if arg_count != expected_count:
            raise SymbolTableError(
                f"Procedure '{proc_name}' expects {expected_count} arguments, got {arg_count}",
                line
            )
    
    def enter_procedure_call(self, proc_name: str):
        """Enter a procedure call (for recursion detection)"""
        self.call_stack.append(proc_name)
    
    def exit_procedure_call(self):
        """Exit a procedure call"""
        if self.call_stack:
            self.call_stack.pop()
    
    def check_variable_usage(self, name: str, is_read: bool, is_write: bool,
                            line: Optional[int] = None) -> Symbol:
        """Check if a variable can be used in the current context
        
        Implements variable shadowing: local variables in procedures shadow global variables.
        Variables used in procedures must be parameters, local variables, or FOR iterators.
        Global variables cannot be accessed from procedures (they are shadowed by locals).
        
        Args:
            name: Variable name
            is_read: Whether variable is being read
            is_write: Whether variable is being written
            line: Line number for error reporting
            
        Returns:
            The symbol if usage is valid
            
        Raises:
            SymbolTableError: If variable cannot be used as requested
        """
        symbol = self.lookup_required(name, line)
        
        # Rule 3: Variables used in procedure must be parameters or local (not global)
        # Since lookup() checks procedure scope before global scope, if we're in a procedure
        # and the symbol found is global (scope_level == 0 and not a procedure definition),
        # it means there's no local variable shadowing it, which is an error.
        if self.current_procedure:
            # Check if symbol is a global variable (not a procedure definition)
            is_global_var = (symbol.scope_level == 0 and 
                           not symbol.is_procedure() and 
                           name in self.global_symbols and
                           name not in self.procedure_symbols[self.current_procedure])
            
            if is_global_var:
                # Global variable accessed from procedure - not allowed
                # (local variables with same name would shadow it)
                raise SymbolTableError(
                    f"Variable '{name}' used in procedure '{self.current_procedure}' must be a parameter or local variable",
                    line
                )
        
        # Rule 5: Check parameter type constraints
        if symbol.is_parameter():
            if symbol.param_type == ParamType.INPUT and is_write:
                raise SymbolTableError(
                    f"Parameter '{name}' marked with 'I' cannot be modified",
                    line
                )
            if symbol.param_type == ParamType.OUTPUT and is_read and not symbol.output_assigned:
                raise SymbolTableError(
                    f"Parameter '{name}' marked with 'O' cannot be read before assignment",
                    line
                )
        
        # Rule 6: FOR loop iterator cannot be modified
        if symbol.is_for_iterator() and is_write:
            raise SymbolTableError(
                f"FOR loop iterator '{name}' cannot be modified",
                line
            )
        
        return symbol
    
    def mark_output_assigned(self, name: str, line: Optional[int] = None):
        """Mark a parameter as assigned (for OUTPUT propagation)
        
        Args:
            name: Variable name
            line: Line number for error reporting (optional)
            
        Raises:
            SymbolTableError: If variable is not found
        """
        symbol = self.lookup_required(name, line)
        
        if symbol.is_parameter():
            # Mark as assigned - this tracks assignment for OUTPUT propagation
            symbol.output_assigned = True
    
    def check_array_access(self, name: str, index: int, line: Optional[int] = None) -> Symbol:
        """Check if an array access is valid
        
        Args:
            name: Array name
            index: Index being accessed
            line: Line number for error reporting
            
        Returns:
            The array symbol if access is valid
            
        Raises:
            SymbolTableError: If array doesn't exist or index is out of bounds
        """
        symbol = self.lookup_required(name, line)
        
        if not symbol.is_array():
            raise SymbolTableError(f"'{name}' is not an array", line)
        
        # Check bounds
        if symbol.array_start is not None and symbol.array_end is not None:
            if index < symbol.array_start or index > symbol.array_end:
                raise SymbolTableError(
                    f"Array '{name}' index {index} out of bounds [{symbol.array_start}:{symbol.array_end}]",
                    line
                )
        
        return symbol
    
    def check_parameter_passing(self, param_name: str, arg_name: str,
                                param_type: ParamType, line: Optional[int] = None):
        """Check if an argument can be passed to a parameter
        
        Args:
            param_name: Name of the formal parameter
            arg_name: Name of the actual argument
            param_type: Type of the formal parameter (I, O, T, or None)
            line: Line number for error reporting
            
        Raises:
            SymbolTableError: If argument cannot be passed to parameter
        """
        arg_symbol = self.lookup_required(arg_name, line)
        
        # Rule 5: I parameters can only receive arguments from I positions
        # Rule 5: O parameters cannot be passed to I positions
        # (This is checked at the call site, not here)
        
        # Check if argument is accessible in current scope
        if self.current_procedure:
            # Argument must be a parameter or local variable of current procedure
            if not arg_symbol.is_parameter() and arg_name not in self.procedure_symbols[self.current_procedure]:
                if arg_name in self.global_symbols:
                    raise SymbolTableError(
                        f"Cannot pass global variable '{arg_name}' as argument from procedure '{self.current_procedure}'",
                        line
                    )
        
        # Type compatibility
        if param_type == ParamType.ARRAY and not arg_symbol.is_array():
            raise SymbolTableError(
                f"Array parameter '{param_name}' expects an array, got '{arg_name}'",
                line
            )
        if param_type != ParamType.ARRAY and arg_symbol.is_array():
            raise SymbolTableError(
                f"Parameter '{param_name}' expects a variable, got array '{arg_name}'",
                line
            )
    
    def get_all_symbols(self) -> Dict[str, Symbol]:
        """Get all symbols (global + all procedure symbols)"""
        all_symbols = self.global_symbols.copy()
        for proc_name, proc_syms in self.procedure_symbols.items():
            for sym_name, sym in proc_syms.items():
                # Use qualified name for procedure symbols
                qualified_name = f"{proc_name}.{sym_name}"
                all_symbols[qualified_name] = sym
        return all_symbols
