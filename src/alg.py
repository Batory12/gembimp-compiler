def multiply(a, b):
    mask = 1
    i = 1
    result = 0
    
    while b - mask > 0:
        mask <<= 1
        i += 1
    
    while i > 0:
        result <<= 1
        
        if (b - mask) >= 0:
            result += a
            b -= mask
        
        mask >>= 1
        i -= 1
    
    return result



def divmod(a, b):
    if b == 0: 
        raise ValueError("Division by zero")
    quotient = 0
    remainder = 0
    print(list(enumerate(map(int, bin(a)[2:]))))
    for i, bit in enumerate(map(int, bin(a)[2:])):
        remainder = (remainder << 1) | bit
        if remainder >= b:
            quotient |= (1 << (len(bin(a)) - i-3))
            remainder -= b
    return quotient, remainder  # quotient = a // b, remainder = a % b


def divmod_vm(a, b):
    mask = 1
    i = 1
    
    while a - mask > 0:
        mask <<= 1
        i += 1
    
    q, r = 0, 0
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

# test accross all possible values of a and b
print(divmod_vm(10, 0))