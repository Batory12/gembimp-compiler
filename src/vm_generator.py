"""
VM Code Generator: Converts Three-Address Code (TAC) to VM instructions.
Based on the VM specification in vm.md.
"""

from enum import Enum, StrEnum
from typing import List, Dict, Optional, Union
from tac import TACInstruction, TACOp
from dataclasses import dataclass


class Register(Enum):
    """VM register identifiers"""
    A = 'a'
    B = 'b'
    C = 'c'
    D = 'd'
    E = 'e'
    F = 'f'
    G = 'g'
    H = 'h'
    
    def __str__(self) -> str:
        return self.value


class VMInstructionType(StrEnum):
    READ = 'READ'
    WRITE = 'WRITE'
    LOAD = 'LOAD'
    STORE = 'STORE'
    RLOAD = 'RLOAD'
    RSTORE = 'RSTORE'
    ADD = 'ADD'
    SUB = 'SUB'
    SWP = 'SWP'
    RST = 'RST'
    INC = 'INC'
    DEC = 'DEC'
    SHL = 'SHL'
    SHR = 'SHR'
    JUMP = 'JUMP'
    JPOS = 'JPOS'
    JZERO = 'JZERO'
    CALL = 'CALL'
    RTRN = 'RTRN'
    HALT = 'HALT'


@dataclass
class VMInstruction:
    """A single VM instruction with optional argument and comment"""
    instruction: Optional[VMInstructionType] = None
    arg: Optional[Union[str, int, Register]] = None
    comment: Optional[str] = None
    
    def __str__(self) -> str:
        """Convert instruction to string representation"""
        # If only comment, return just the comment
        if self.instruction is None and self.comment:
            return f"# {self.comment}"
        
        # If no instruction type, skip (shouldn't happen)
        if self.instruction is None:
            return ""
        
        text = str(self.instruction.value)
        if self.arg is not None:
            # Convert Register enum to string, or use arg as-is
            if isinstance(self.arg, Register):
                text += f" {self.arg.value}"
            else:
                text += f" {self.arg}"
        if self.comment:
            text += f" # {self.comment}"
        return text

class VMGenerator:
    """Converts TAC instructions to VM assembly code."""
    
    def __init__(self):
        # Variable/register to memory location mapping
        self.variable_map: Dict[str, int] = {}
        self.next_memory = 0
        
        # Label to instruction number mapping (filled in first pass)
        self.label_map: Dict[str, int] = {}
        # Generated VM instructions
        self.instructions: List[VMInstruction] = []
        
        # Track current instruction number
        self.instruction_count = 0

        self.label_counter = 0
        
    def get_memory_location(self, var_name: str) -> int:
        """Get memory location for a variable, allocating if needed."""
        if var_name not in self.variable_map:
            self.variable_map[var_name] = self.next_memory
            self.next_memory += 1
        return self.variable_map[var_name]
    
    def is_constant(self, value: Union[str, int]) -> bool:
        """Check if a value is a numeric constant."""
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            try:
                int(value)
                return True
            except ValueError:
                return False
        return False
    
    def get_constant_value(self, value: Union[str, int]) -> int:
        """Get integer value from a constant."""
        if isinstance(value, int):
            return value
        return int(value)
    
    def emit(self, instruction: VMInstruction):
        """Emit a VM instruction."""
        self.instructions.append(instruction)
        self.instruction_count += 1
    
    def build_constant(self, reg: Register, value: int) -> List[VMInstruction]:
        """Build a constant value in a register using INC, DEC, SHL, SHR.
        Returns list of instructions."""
        insts = []
        
        # Reset register first
        insts.append(VMInstruction(VMInstructionType.RST, reg))
        if value == 0:
            return insts
        
        binary = bin(value)[3:]  # Skip '0b1'
        insts.append(VMInstruction(VMInstructionType.INC, reg))
        for bit in binary:
            insts.append(VMInstruction(VMInstructionType.SHL, reg))
            if bit == '1':
                insts.append(VMInstruction(VMInstructionType.INC, reg))
        return insts
    
    def load_to_ra(self, source: Union[str, int]):
        """Load a value (variable or constant) into ra register."""
        if self.is_constant(source):
            const_val = self.get_constant_value(source)
            for inst in self.build_constant(Register.A, const_val):
                self.emit(inst)
        else:
            # Variable - load from memory
            mem_loc = self.get_memory_location(source)
            self.emit(VMInstruction(VMInstructionType.LOAD, mem_loc))
    
    def store_from_ra(self, dest: str):
        """Store ra register to a variable."""
        mem_loc = self.get_memory_location(dest)
        self.emit(VMInstruction(VMInstructionType.STORE, mem_loc))
    
    def load_to_rb(self, source: Union[str, int]):
        """Load a value into rb register."""
        if self.is_constant(source):
            const_val = self.get_constant_value(source)
            for inst in self.build_constant(Register.B, const_val):
                self.emit(inst)
        else:
            mem_loc = self.get_memory_location(source)
            self.emit(VMInstruction(VMInstructionType.LOAD, mem_loc))
            self.emit(VMInstruction(VMInstructionType.SWP, Register.B))  # Move to rb
    
    def generate(self, tac_instructions: List[TACInstruction]) -> str:
        """Generate VM code from TAC instructions."""
        # First pass: generate VM code, track labels and jumps
        self.instructions = []
        self.instruction_count = 0
        self.variable_map = {}
        self.next_memory = 0
        self.label_map = {}
        self.jump_patches = []  # List of (instruction_index, label_name) tuples
        
        for instr in tac_instructions:
            if instr.op == TACOp.LABEL:
                # Record label at current instruction count (comments don't count as instructions)
                if self.label_map.get(instr.label) is not None:
                    raise ValueError(f"Label {instr.label} already defined")
                self.label_map[instr.label] = self.instruction_count
            else:
                self.generate_instruction(instr)
        
        # Patch all jump instructions with correct addresses
        # At this point, addresses and list positions match exactly (no comments inserted yet)
        for instr_addr, label_name in self.jump_patches:
            if label_name not in self.label_map:
                raise ValueError(f"Label {label_name} not found")
            self.instructions[instr_addr].arg = self.label_map[label_name]
        
        # Add HALT at the end if not present
        if not self.instructions or self.instructions[-1].instruction != VMInstructionType.HALT:
            self.emit(VMInstruction(VMInstructionType.HALT))
        
        return '\n'.join(str(inst) for inst in self.instructions)
    
    def generate_instruction(self, instr: TACInstruction):
        """Generate VM code for a single TAC instruction."""
        if instr.op == TACOp.ASSIGN:
            # result = arg1
            self.load_to_ra(instr.arg1)
            self.store_from_ra(instr.result)
            
        elif instr.op == TACOp.ADD:
            # result = arg1 + arg2
            self.load_to_rb(instr.arg2)
            self.load_to_ra(instr.arg1)
            self.emit(VMInstruction(VMInstructionType.ADD, 'b'))  # ra = ra + rb
            self.store_from_ra(instr.result)
            
        elif instr.op == TACOp.SUB:
            # result = arg1 - arg2
            self.load_to_rb(instr.arg2)
            self.load_to_ra(instr.arg1)
            self.emit(VMInstruction(VMInstructionType.SUB, 'b'))  # ra = max(ra - rb, 0)
            self.store_from_ra(instr.result)
            
        elif instr.op == TACOp.MUL:
            # result = arg1 * arg2 (using repeated addition)
            self.generate_multiplication(instr.arg1, instr.arg2, instr.result)
            
        elif instr.op == TACOp.DIV:
            # result = arg1 / arg2 (integer division)
            self.generate_divmod(instr.arg1, instr.arg2, q_result=instr.result, r_result=None)
            
        elif instr.op == TACOp.MOD:
            # result = arg1 % arg2
            self.generate_divmod(instr.arg1, instr.arg2, q_result=None, r_result=instr.result)
            
        elif instr.op == TACOp.READ:
            # READ arg1 (variable name)
            self.emit(VMInstruction(VMInstructionType.READ))
            self.store_from_ra(instr.arg1)
            
        elif instr.op == TACOp.WRITE:
            # WRITE arg1 (variable name or temp)
            self.load_to_ra(instr.arg1)
            self.emit(VMInstruction(VMInstructionType.WRITE))
            
        elif instr.op == TACOp.GOTO:
            # GOTO label - use placeholder, will patch later
            label_name = instr.label or instr.arg1
            self.emit(VMInstruction(VMInstructionType.JUMP, 0))  # Placeholder, will patch
            # Record instruction address (instruction_count was incremented by emit)
            self.add_jump_patch(label_name)
            
        elif instr.op == TACOp.IF:
            # IF arg1 op arg2 GOTO label
            # Format: IF arg1 op result GOTO label means: if (arg1 op result) goto label
            op = instr.arg2  # The comparison operator: <, >, =, <=, >=, !=
            left = instr.arg1
            right = instr.result  # In TAC IF format, result is the right operand
            label = instr.label
            
            self.generate_conditional_jump(left, op, right, label)
            
        elif instr.op == TACOp.CALL:
            # CALL procedure_name
            proc_label = f'proc_{instr.arg1}'
            # CALL j means: ra <- k + 1, k <- j
            self.emit(VMInstruction(VMInstructionType.CALL, 0))  # Placeholder, will patch
            # Record instruction address
            self.add_jump_patch(proc_label)
            
        elif instr.op == TACOp.RET:
            # RET (return from procedure)
            # RTRN means: k <- ra
            self.emit(VMInstruction(VMInstructionType.RTRN))
            
        elif instr.op == TACOp.LOAD_ARRAY:
            # LOAD_ARRAY array_name[index] -> result
            # Use RLOAD for indirect memory access
            self.generate_array_load(instr.arg1, instr.arg2, instr.result)
            
        elif instr.op == TACOp.STORE_ARRAY:
            # STORE_ARRAY array_name[index] -> value
            # Use RSTORE for indirect memory access
            self.generate_array_store(instr.arg1, instr.arg2, instr.result)
            
        elif instr.op == TACOp.PARAM:
            # PARAM arg1 - parameter passing
            # Parameters are passed via variables, so no code needed here
            pass
            
        else:
            raise ValueError(f"Unsupported TAC operation: {instr.op}")
    
    def generate_conditional_jump(self, left: Union[str, int], op: str, right: Union[str, int], target_label: str):
        """Generate code for conditional jump: if (left op right) goto target_label."""
        if op == '<':
            # if left < right: compute (right - left) > 0
            self.load_to_rb(left)
            self.load_to_ra(right)
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = right - left
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(target_label)
            
        elif op == '>':
            # if left > right: compute (left - right) > 0
            self.load_to_rb(right)
            self.load_to_ra(left)
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = left - right
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(target_label)
            
        elif op == '=':
            # reverse of !=

            neq = self.make_label("not_eq")
            self.load_to_rb(left)
            self.load_to_ra(right)
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = right - left
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(neq)

            self.load_to_rb(right)
            self.load_to_ra(left)
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = left - right
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(neq)
            self.emit(VMInstruction(VMInstructionType.JUMP, 0))
            self.add_jump_patch(target_label)
            self.emit_label(neq)
            
        elif op == '<=':
            # if left <= right: (right - left) >= 0 -> (right - left + 1) > 0
            self.load_to_rb(left)
            self.load_to_ra(right)
            self.emit(VMInstruction(VMInstructionType.INC, Register.A))
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = right + 1 - left
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(target_label)
            
        elif op == '>=':
             # if left >= right: compute (left - right + 1) > 0
            self.load_to_rb(right)
            self.load_to_ra(left)
            self.emit(VMInstruction(VMInstructionType.INC, Register.A))
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = left + 1- right
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(target_label)
            
        elif op == '!=':
            # check if left > right or right > left
            self.load_to_rb(left)
            self.load_to_ra(right)
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = right - left
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(target_label)

            self.load_to_rb(right)
            self.load_to_ra(left)
            self.emit(VMInstruction(VMInstructionType.SUB, Register.B))  # ra = left - right
            self.emit(VMInstruction(VMInstructionType.JPOS, 0))  # Placeholder, will patch
            self.add_jump_patch(target_label)
            
        else:
            raise ValueError(f"Unsupported comparison operator: {op}")
    def gen_reg_copy(self, source: Register, dest: Register):
        """Copy the value of source register to dest register - cost 6 or 7 (hehe)
        Register A gets dirty, source != Register.A
        """
        if source == dest:
            return
        self.emit(VMInstruction(VMInstructionType.RST, Register.A))
        self.emit(VMInstruction(VMInstructionType.ADD, source))
        if dest != Register.A:
            self.emit(VMInstruction(VMInstructionType.SWP, dest))

    def make_label(self, label: str):
        """Make a label a unique label name in place"""
        full_label = f"{label}_{self.label_counter}"
        self.label_counter += 1
        return full_label

    def emit_label(self, label: str):
        """Emit a label"""
        self.label_map[label] = self.instruction_count
        if not self.instructions[-1].comment:
            self.instructions[-1].comment = ""
        self.instructions[-1].comment += f" Label: {label}"

    def add_jump_patch(self, label: str):
        """Add a jump patch to the jump_patches list for last instruction"""
        self.jump_patches.append((self.instruction_count - 1, label))
        if not self.instructions[-1].comment:
            self.instructions[-1].comment = ""
        self.instructions[-1].comment += f" Jump: {label}"


    def generate_multiplication(self, arg1: Union[str, int], arg2: Union[str, int], result: str):
        """Generate code for multiplication: result = arg1 * arg2.
        Alogrithm: logarithmitic multiplication
        mask = 1
        i = 1
        res = 0
        
        while b - mask > 0:
            mask <<= 1
            i += 1
        
        while i > 0:
            res <<= 1
            
            if (b - mask) >= 0:
                res += a
                b -= mask
            
            mask >>= 1
            i -= 1

        Save registers A, B, C, D, E, F
        """
        acc = Register.A

        mask = Register.B
        self.emit(VMInstruction(VMInstructionType.RST, mask, f"{result} = {arg1} * {arg2}"))
        self.emit(VMInstruction(VMInstructionType.INC, mask))
        
        i = Register.C
        self.emit(VMInstruction(VMInstructionType.RST, i))
        self.emit(VMInstruction(VMInstructionType.INC, i))
        
        a = Register.D
        self.gen_load_to_register(arg1, a)
        
        b = Register.E
        self.gen_load_to_register(arg2, b)
        
        res = Register.F
        self.emit(VMInstruction(VMInstructionType.RST, res))

        # Loop: while b - mask > 0
        begin_mask_loop = self.make_label(f"_begin_mul_mask_loop")
        self.emit_label(begin_mask_loop)
        self.gen_reg_copy(b, acc)
        self.emit(VMInstruction(VMInstructionType.SUB, mask))
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        end_mask_loop = self.make_label(f"_end_mul_mask_loop")
        self.add_jump_patch(end_mask_loop)

        # mask <<= 1
        self.emit(VMInstruction(VMInstructionType.SHL, mask))

        # i++
        self.emit(VMInstruction(VMInstructionType.INC, i))

        # end while
        self.emit(VMInstruction(VMInstructionType.JUMP, 0))
        self.add_jump_patch(begin_mask_loop)
        self.emit_label(end_mask_loop)
        # while i > 0
        begin_loop = self.make_label(f"_begin_mul_main_loop")
        self.emit_label(begin_loop)
        self.gen_reg_copy(i, acc)
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        end_loop = self.make_label(f"_end_mul_main_loop")
        self.add_jump_patch(end_loop)

        # res <<= 1
        self.emit(VMInstruction(VMInstructionType.SHL, res))

        # if (b - mask) >= 0: (eq to b + 1- mask > 0)
        self.gen_reg_copy(b, acc)
        self.emit(VMInstruction(VMInstructionType.INC, acc))
        self.emit(VMInstruction(VMInstructionType.SUB, mask))
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        end_if = self.make_label(f"_end_mul_if")
        self.add_jump_patch(end_if)

        # res += a
        self.gen_reg_copy(a, acc)
        self.emit(VMInstruction(VMInstructionType.ADD, res))
        self.emit(VMInstruction(VMInstructionType.SWP, res))

        # b -= mask
        self.gen_reg_copy(b, acc)
        self.emit(VMInstruction(VMInstructionType.SUB, mask))
        self.emit(VMInstruction(VMInstructionType.SWP, b))

        # end if
        self.emit_label(end_if)

        # mask >>= 1
        self.emit(VMInstruction(VMInstructionType.SHR, mask))

        # i -= 1
        self.emit(VMInstruction(VMInstructionType.DEC, i))
        
        self.emit(VMInstruction(VMInstructionType.JUMP, 0))
        self.add_jump_patch(begin_loop)
        # end while
        self.emit_label(end_loop)

        # Save result
        self.emit(VMInstruction(VMInstructionType.SWP, res))
        self.emit(VMInstruction(VMInstructionType.STORE, self.get_memory_location(result)))





        
    def generate_divmod(self, arg1: Union[str, int], arg2: Union[str, int], q_result: Optional[str] = None, r_result: Optional[str] = None):
        """Generate code for div-mod algorithm. Logarithmitic division. Follows alogrithm:
        def divmod_vm(a, b):
        ```python
            mask = 1
            i = 1
            q, r = 0, 0
            
            if b == 0:
                return 0, 0
            while a - mask > 0:
                mask <<= 1
                i += 1
            
            while i > 0:
                r <<= 1
                q <<= 1
                if a - mask >= 0:
                    r += 1
                    a -= mask
                
                if r >= b:
                    q += 1
                    r -= b
                
                mask >>= 1
                i -= 1
            
            return q, r
        ```

        """

        
        acc = Register.A
        # q, r = 0, 0
        q = Register.F
        self.emit(VMInstruction(VMInstructionType.RST, q))
        r = Register.G
        self.emit(VMInstruction(VMInstructionType.RST, r))

        b = Register.E
        self.gen_load_to_register(arg2, b)
        
        # if b == 0 return 0, 0
        self.emit(VMInstruction(VMInstructionType.SWP, b))
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        end_loop = self.make_label(f"_end_div_main_loop")
        self.add_jump_patch(end_loop)
        self.emit(VMInstruction(VMInstructionType.SWP, b))
        
        a = Register.D
        self.gen_load_to_register(arg1, a)
        # mask = 1
        mask = Register.B
        self.emit(VMInstruction(VMInstructionType.RST, mask, f"begin division of {arg1} by {arg2}"))
        self.emit(VMInstruction(VMInstructionType.INC, mask))
        
        # i = 1
        i = Register.C
        self.emit(VMInstruction(VMInstructionType.RST, i))
        self.emit(VMInstruction(VMInstructionType.INC, i))
        
        
        

        # Loop: while a - mask > 0
        begin_mask_loop = self.make_label(f"_begin_div_mask_loop")
        self.emit_label(begin_mask_loop)
        self.gen_reg_copy(a, acc)
        self.emit(VMInstruction(VMInstructionType.SUB, mask))
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        end_mask_loop = self.make_label(f"_end_div_mask_loop")
        self.add_jump_patch(end_mask_loop)

        # mask <<= 1
        self.emit(VMInstruction(VMInstructionType.SHL, mask))

        # i++
        self.emit(VMInstruction(VMInstructionType.INC, i))

        # end while
        self.emit(VMInstruction(VMInstructionType.JUMP, 0))
        self.add_jump_patch(begin_mask_loop)
        self.emit_label(end_mask_loop)
        
        # while i > 0:
        begin_loop = self.make_label(f"_begin_div_main_loop")
        self.emit_label(begin_loop)
        self.gen_reg_copy(i, acc)
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        self.add_jump_patch(end_loop)

        # r <<= 1
        self.emit(VMInstruction(VMInstructionType.SHL, r))
        # q <<= 1
        self.emit(VMInstruction(VMInstructionType.SHL, q))
        # if a - mask >= 0:
        self.gen_reg_copy(a, acc)
        self.emit(VMInstruction(VMInstructionType.INC, acc))
        self.emit(VMInstruction(VMInstructionType.SUB, mask))
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        end_if1 = self.make_label(f"_end_div_if1")
        self.add_jump_patch(end_if1)
        # r += 1
        self.emit(VMInstruction(VMInstructionType.INC, r))
        # a -= mask
        self.emit(VMInstruction(VMInstructionType.SWP, a))
        self.emit(VMInstruction(VMInstructionType.SUB, mask))
        self.emit(VMInstruction(VMInstructionType.SWP, a))
        # end if 
        self.emit_label(end_if1)
        # if r >= b:
        self.gen_reg_copy(r, acc)
        self.emit(VMInstruction(VMInstructionType.INC, acc))
        self.emit(VMInstruction(VMInstructionType.SUB, b))
        self.emit(VMInstruction(VMInstructionType.JZERO, 0))
        end_if2 = self.make_label(f"_end_div_if2")
        self.add_jump_patch(end_if2)
        # q += 1
        self.emit(VMInstruction(VMInstructionType.INC, q))
        # r -= b
        self.emit(VMInstruction(VMInstructionType.SWP, r))
        self.emit(VMInstruction(VMInstructionType.SUB, b))
        self.emit(VMInstruction(VMInstructionType.SWP, r))
        # end if
        self.emit_label(end_if2)
        # mask >>= 1
        self.emit(VMInstruction(VMInstructionType.SHR, mask))
        # i -= 1
        self.emit(VMInstruction(VMInstructionType.DEC, i))
        # end while
        self.emit(VMInstruction(VMInstructionType.JUMP, 0))
        self.add_jump_patch(begin_loop)
        self.emit_label(end_loop)
        # end while
        if q_result is not None:
            self.emit(VMInstruction(VMInstructionType.SWP, q))
            self.emit(VMInstruction(VMInstructionType.STORE, self.get_memory_location(q_result)))
        if r_result is not None:
            self.emit(VMInstruction(VMInstructionType.SWP, r))
            self.emit(VMInstruction(VMInstructionType.STORE, self.get_memory_location(r_result)))
    
    def generate_array_load(self, array_name: str, index: Union[str, int], result: str):
        """Generate code for array load: result = array_name[index]."""
        # Compute address: array_base + index
        # Load index into rb, compute address in rc, use RLOAD
        
        # Get array base address (assume arrays start at a fixed offset)
        # For simplicity, use variable_map to track array base addresses
        array_base_key = f'{array_name}_base'
        if array_base_key not in self.variable_map:
            self.variable_map[array_base_key] = self.next_memory
            self.next_memory += 10  # Reserve space for array
        
        array_base_addr = self.variable_map[array_base_key]
        
        # Load index into rb
        self.load_to_rb(index)
        
        # Compute address: array_base_addr + index in rc
        # First load base address into rc
        for inst in self.build_constant(Register.C, array_base_addr):
            self.emit(inst)
        # Now add index: rc = rc + rb
        self.emit(VMInstruction(VMInstructionType.SWP, Register.A))  # Save ra
        self.emit(VMInstruction(VMInstructionType.SWP, Register.C))  # rc -> ra
        self.emit(VMInstruction(VMInstructionType.ADD, Register.B))  # ra = rc + rb
        self.emit(VMInstruction(VMInstructionType.SWP, Register.C))  # ra -> rc
        self.emit(VMInstruction(VMInstructionType.SWP, Register.A))  # Restore ra
        
        # Now use RLOAD: ra <- prc
        self.emit(VMInstruction(VMInstructionType.RLOAD, Register.C))
        self.store_from_ra(result)
    
    def generate_array_store(self, array_name: str, index: Union[str, int], value: Union[str, int]):
        """Generate code for array store: array_name[index] = value."""
        # Compute address: array_base + index
        # Load value into ra, compute address in rc, use RSTORE
        
        # Get array base address
        array_base_key = f'{array_name}_base'
        if array_base_key not in self.variable_map:
            self.variable_map[array_base_key] = self.next_memory
            self.next_memory += 10  # Reserve space for array
        
        array_base_addr = self.variable_map[array_base_key]
        
        # Load index into rb
        self.load_to_rb(index)
        
        # Compute address: array_base_addr + index in rc
        for inst in self.build_constant(Register.C, array_base_addr):
            self.emit(inst)
        self.emit(VMInstruction(VMInstructionType.SWP, Register.A))  # Save ra
        self.emit(VMInstruction(VMInstructionType.SWP, Register.C))  # rc -> ra
        self.emit(VMInstruction(VMInstructionType.ADD, Register.B))  # ra = rc + rb
        self.emit(VMInstruction(VMInstructionType.SWP, Register.C))  # ra -> rc
        self.emit(VMInstruction(VMInstructionType.SWP, Register.A))  # Restore ra
        
        # Load value into ra
        self.load_to_ra(value)
        
        # Use RSTORE: prc <- ra
        self.emit(VMInstruction(VMInstructionType.RSTORE, Register.C))
    
    def gen_load_to_register(self, value: Union[str, int], register: Register):
        if self.is_constant(value):
            self.build_constant(register, self.get_constant_value(value))
        else:
            self.load_to_ra(value)
            self.emit(VMInstruction(VMInstructionType.SWP, register))
