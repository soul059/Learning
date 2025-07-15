# 🧮 Mathematical Foundations & History of Cryptography

## 📖 Table of Contents
1. [Historical Overview](#historical-overview)
2. [Mathematical Prerequisites](#mathematical-prerequisites)
3. [Number Theory Fundamentals](#number-theory-fundamentals)
4. [Group Theory](#group-theory)
5. [Probability and Information Theory](#probability-and-information-theory)
6. [Complexity Theory](#complexity-theory)
7. [Practical Examples](#practical-examples)

---

## 🏛️ Historical Overview

### Ancient Cryptography (3000 BCE - 400 CE)

#### Egyptian Hieroglyphs (3000 BCE)
- **Purpose**: Ceremonial secrecy, not military
- **Method**: Substitution of symbols
- **Example**: Non-standard hieroglyphs in tomb inscriptions

#### Spartan Scytale (7th Century BCE)
```
Original: ATTACKATDAWN
Scytale (4 strips):
A T T A
C K A T
D A W N
Read down: ACDT TKAW TAAN
```

#### Caesar Cipher (50 BCE)
- **Mathematical Model**: `C = (P + k) mod 26`
- **Key Space**: 26 possible keys
- **Weakness**: Frequency analysis

```python
def caesar_encrypt(plaintext, key):
    result = ""
    for char in plaintext:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            result += chr((ord(char) - ascii_offset + key) % 26 + ascii_offset)
        else:
            result += char
    return result

# Example
plaintext = "HELLO WORLD"
key = 3
ciphertext = caesar_encrypt(plaintext, key)  # "KHOOR ZRUOG"
```

#### Polybius Square (200 BCE)
```
   1 2 3 4 5
1  A B C D E
2  F G H I/J K
3  L M N O P
4  Q R S T U
5  V W X Y Z

HELLO = 23 15 31 31 34
```

### Medieval Cryptography (400 - 1400 CE)

#### Al-Kindi's Frequency Analysis (9th Century)
- **Breakthrough**: First systematic cryptanalysis
- **Method**: Statistical analysis of letter frequencies
- **Impact**: Broke substitution ciphers for 500 years

```python
def frequency_analysis(text):
    """Al-Kindi's frequency analysis method"""
    freq = {}
    text = text.upper().replace(' ', '')
    
    for char in text:
        if char.isalpha():
            freq[char] = freq.get(char, 0) + 1
    
    # Sort by frequency
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_freq

# English letter frequencies
english_freq = ['E', 'T', 'A', 'O', 'I', 'N', 'S', 'H', 'R', 'D']
```

#### Alberti Cipher Disk (1467)
- **Innovation**: First polyalphabetic cipher
- **Key Advancement**: Multiple substitution alphabets
- **Mathematical Concept**: Periodic key application

### Renaissance Cryptography (1400 - 1700)

#### Vigenère Cipher (1553)
- **Formula**: `C[i] = (P[i] + K[i mod len(K)]) mod 26`
- **Strength**: Resisted frequency analysis
- **Weakness**: Periodic key repetition

```python
def vigenere_encrypt(plaintext, key):
    result = ""
    key = key.upper()
    key_index = 0
    
    for char in plaintext.upper():
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            encrypted_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            result += encrypted_char
            key_index += 1
        else:
            result += char
    
    return result

# Example
plaintext = "ATTACKATDAWN"
key = "LEMON"
ciphertext = vigenere_encrypt(plaintext, key)  # "LXFOPVEFRNHR"
```

#### Great Cipher (1626)
- **Innovation**: Homophonic substitution
- **Security**: Multiple symbols per letter
- **Historical Use**: Louis XIV's diplomatic correspondence

---

## 🔢 Mathematical Prerequisites

### Basic Arithmetic Operations

#### Modular Arithmetic
The foundation of modern cryptography.

**Definition**: `a ≡ b (mod n)` if `n | (a - b)`

**Properties**:
1. **Addition**: `(a + b) mod n = ((a mod n) + (b mod n)) mod n`
2. **Multiplication**: `(a × b) mod n = ((a mod n) × (b mod n)) mod n`
3. **Exponentiation**: `a^b mod n` (requires efficient algorithms)

```python
def mod_exp(base, exp, mod):
    """Efficient modular exponentiation using square-and-multiply"""
    result = 1
    base = base % mod
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    
    return result

# Example: 2^10 mod 1000
result = mod_exp(2, 10, 1000)  # 24
```

#### Extended Euclidean Algorithm
Essential for finding modular inverses.

```python
def extended_gcd(a, b):
    """Extended Euclidean Algorithm"""
    if a == 0:
        return b, 0, 1
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd, x, y

def mod_inverse(a, m):
    """Find modular inverse of a modulo m"""
    gcd, x, y = extended_gcd(a, m)
    
    if gcd != 1:
        raise ValueError("Modular inverse does not exist")
    
    return (x % m + m) % m

# Example: Find inverse of 3 modulo 11
inverse = mod_inverse(3, 11)  # 4, because (3 * 4) mod 11 = 1
```

### Prime Numbers and Factorization

#### Primality Testing

```python
def is_prime_trial_division(n):
    """Basic primality test using trial division"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True

def miller_rabin(n, k=5):
    """Miller-Rabin probabilistic primality test"""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as d * 2^r
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Perform k rounds of testing
    import random
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = mod_exp(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = mod_exp(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True
```

#### Prime Generation

```python
def generate_prime(bits):
    """Generate a random prime number with specified bit length"""
    import random
    
    while True:
        # Generate random odd number
        candidate = random.getrandbits(bits)
        candidate |= (1 << bits - 1) | 1  # Set MSB and LSB
        
        if miller_rabin(candidate):
            return candidate

# Generate 1024-bit prime
prime = generate_prime(1024)
```

---

## 🔢 Number Theory Fundamentals

### Greatest Common Divisor (GCD)

#### Euclidean Algorithm
```python
def gcd(a, b):
    """Euclidean algorithm for GCD"""
    while b:
        a, b = b, a % b
    return a

# Properties:
# gcd(a, b) = gcd(b, a mod b)
# gcd(a, 0) = a
```

#### Bézout's Identity
For any integers a and b, there exist integers x and y such that:
`ax + by = gcd(a, b)`

### Euler's Totient Function φ(n)

**Definition**: Number of integers from 1 to n that are coprime to n.

**Formula for prime powers**:
- If p is prime: `φ(p) = p - 1`
- If p is prime: `φ(p^k) = p^k - p^(k-1) = p^(k-1)(p - 1)`

**Formula for general n**:
`φ(n) = n × ∏(1 - 1/p)` for all prime factors p of n

```python
def euler_totient(n):
    """Calculate Euler's totient function φ(n)"""
    result = n
    p = 2
    
    while p * p <= n:
        if n % p == 0:
            # Remove all factors p
            while n % p == 0:
                n //= p
            # Multiply result by (1 - 1/p)
            result -= result // p
        p += 1
    
    # If n > 1, then it's a prime factor
    if n > 1:
        result -= result // n
    
    return result

# Examples:
# φ(9) = φ(3²) = 3²⁻¹(3-1) = 3×2 = 6
# φ(15) = φ(3×5) = φ(3)×φ(5) = 2×4 = 8
```

### Fermat's Little Theorem

**Theorem**: If p is prime and a is not divisible by p, then:
`a^(p-1) ≡ 1 (mod p)`

**Corollary**: `a^p ≡ a (mod p)` for any integer a

```python
def fermat_test(a, p):
    """Test if p might be prime using Fermat's Little Theorem"""
    return mod_exp(a, p - 1, p) == 1

# Note: This is not a definitive primality test due to Carmichael numbers
```

### Euler's Theorem

**Theorem**: If gcd(a, n) = 1, then:
`a^φ(n) ≡ 1 (mod n)`

This generalizes Fermat's Little Theorem.

### Chinese Remainder Theorem (CRT)

**Problem**: Solve system of congruences:
```
x ≡ a₁ (mod n₁)
x ≡ a₂ (mod n₂)
...
x ≡ aₖ (mod nₖ)
```

**Solution** (when n₁, n₂, ..., nₖ are pairwise coprime):

```python
def chinese_remainder_theorem(remainders, moduli):
    """Solve system of congruences using CRT"""
    total = 0
    prod = 1
    
    for m in moduli:
        prod *= m
    
    for r, m in zip(remainders, moduli):
        p = prod // m
        total += r * mod_inverse(p, m) * p
    
    return total % prod

# Example: Solve x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
remainders = [2, 3, 2]
moduli = [3, 5, 7]
solution = chinese_remainder_theorem(remainders, moduli)  # x = 23
```

---

## 🌐 Group Theory

### Basic Definitions

#### Group
A set G with operation ∘ is a group if:
1. **Closure**: ∀a,b ∈ G, a∘b ∈ G
2. **Associativity**: ∀a,b,c ∈ G, (a∘b)∘c = a∘(b∘c)
3. **Identity**: ∃e ∈ G such that ∀a ∈ G, a∘e = e∘a = a
4. **Inverse**: ∀a ∈ G, ∃a⁻¹ ∈ G such that a∘a⁻¹ = a⁻¹∘a = e

#### Multiplicative Group (ℤ/nℤ)*
The set of integers modulo n that are coprime to n, under multiplication.

```python
def multiplicative_group(n):
    """Generate multiplicative group (Z/nZ)*"""
    group = []
    for i in range(1, n):
        if gcd(i, n) == 1:
            group.append(i)
    return group

# Example: (Z/8Z)* = {1, 3, 5, 7}
group = multiplicative_group(8)  # [1, 3, 5, 7]
```

#### Order of an Element
The smallest positive integer k such that a^k ≡ 1 (mod n).

```python
def element_order(a, n):
    """Find the order of element a in (Z/nZ)*"""
    if gcd(a, n) != 1:
        raise ValueError("Element must be coprime to modulus")
    
    order = 1
    current = a % n
    
    while current != 1:
        current = (current * a) % n
        order += 1
    
    return order

# The order divides φ(n)
```

#### Generator (Primitive Root)
An element g ∈ (ℤ/pℤ)* is a generator if its order is φ(p) = p-1.

```python
def is_primitive_root(g, p):
    """Check if g is a primitive root modulo p"""
    if not is_prime_trial_division(p):
        return False
    
    return element_order(g, p) == p - 1

def find_primitive_root(p):
    """Find a primitive root modulo prime p"""
    for g in range(2, p):
        if is_primitive_root(g, p):
            return g
    return None

# Example: Find primitive root of 7
root = find_primitive_root(7)  # 3 is a primitive root of 7
```

### Elliptic Curves

#### Elliptic Curve Definition
An elliptic curve over finite field 𝔽ₚ is defined by:
`y² ≡ x³ + ax + b (mod p)`

where `4a³ + 27b² ≠ 0 (mod p)`

```python
class EllipticCurve:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
        
        # Check discriminant
        discriminant = (4 * a**3 + 27 * b**2) % p
        if discriminant == 0:
            raise ValueError("Invalid elliptic curve parameters")
    
    def is_on_curve(self, x, y):
        """Check if point (x, y) is on the curve"""
        left = (y * y) % self.p
        right = (x**3 + self.a * x + self.b) % self.p
        return left == right
    
    def point_addition(self, P, Q):
        """Add two points on the elliptic curve"""
        if P is None:  # P is point at infinity
            return Q
        if Q is None:  # Q is point at infinity
            return P
        
        x1, y1 = P
        x2, y2 = Q
        
        if x1 == x2:
            if y1 == y2:
                # Point doubling
                s = (3 * x1 * x1 + self.a) * mod_inverse(2 * y1, self.p) % self.p
            else:
                # Points are inverses
                return None  # Point at infinity
        else:
            # Point addition
            s = (y2 - y1) * mod_inverse(x2 - x1, self.p) % self.p
        
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_multiplication(self, k, P):
        """Compute k*P using double-and-add algorithm"""
        if k == 0:
            return None  # Point at infinity
        
        result = None
        addend = P
        
        while k:
            if k & 1:
                result = self.point_addition(result, addend)
            addend = self.point_addition(addend, addend)
            k >>= 1
        
        return result

# Example: secp256k1 curve (used in Bitcoin)
# y² = x³ + 7 (mod p) where p = 2²⁵⁶ - 2³² - 977
```

---

## 📊 Probability and Information Theory

### Information Theory Basics

#### Entropy
Measure of uncertainty or information content.

**Shannon Entropy**: `H(X) = -∑ P(x) log₂ P(x)`

```python
import math

def shannon_entropy(probabilities):
    """Calculate Shannon entropy"""
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# Example: Fair coin has entropy = 1 bit
fair_coin = [0.5, 0.5]
entropy = shannon_entropy(fair_coin)  # 1.0

# Biased coin has lower entropy
biased_coin = [0.9, 0.1]
entropy = shannon_entropy(biased_coin)  # ~0.47
```

#### Perfect Secrecy (Shannon's Theorem)
A cipher has perfect secrecy if:
`P(M = m | C = c) = P(M = m)` for all m, c

**One-Time Pad** achieves perfect secrecy:
- Key length ≥ message length
- Key is truly random
- Key is used only once

```python
def one_time_pad_encrypt(message, key):
    """One-time pad encryption (perfect secrecy)"""
    if len(key) < len(message):
        raise ValueError("Key must be at least as long as message")
    
    ciphertext = []
    for i, char in enumerate(message):
        # XOR with key
        encrypted_char = ord(char) ^ ord(key[i])
        ciphertext.append(encrypted_char)
    
    return bytes(ciphertext)

def one_time_pad_decrypt(ciphertext, key):
    """One-time pad decryption"""
    message = []
    for i, byte_val in enumerate(ciphertext):
        # XOR with key
        decrypted_char = byte_val ^ ord(key[i])
        message.append(chr(decrypted_char))
    
    return ''.join(message)
```

### Cryptographic Randomness

#### True Random vs. Pseudorandom
- **True Random**: From physical processes (thermal noise, quantum effects)
- **Pseudorandom**: Deterministic algorithms with random-like output

#### Random Number Generation

```python
import os
import hashlib

def secure_random_bytes(n):
    """Generate cryptographically secure random bytes"""
    return os.urandom(n)

class LinearCongruentialGenerator:
    """Simple PRNG (NOT cryptographically secure)"""
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
        self.current = seed
    
    def next(self):
        self.current = (self.a * self.current + self.c) % self.m
        return self.current

class CryptographicPRNG:
    """Cryptographically secure PRNG using hash functions"""
    def __init__(self, seed):
        self.state = hashlib.sha256(seed).digest()
        self.counter = 0
    
    def next_bytes(self, n):
        result = b''
        while len(result) < n:
            # Hash state || counter
            hash_input = self.state + self.counter.to_bytes(8, 'big')
            block = hashlib.sha256(hash_input).digest()
            result += block
            self.counter += 1
        
        return result[:n]
```

---

## ⚡ Complexity Theory

### Computational Complexity Classes

#### P vs NP
- **P**: Problems solvable in polynomial time
- **NP**: Problems verifiable in polynomial time
- **NP-Complete**: Hardest problems in NP

#### Cryptographic Assumptions

```python
def discrete_log_problem(g, h, p):
    """
    Discrete Logarithm Problem: Given g, h, p, find x such that g^x ≡ h (mod p)
    
    This is believed to be hard for large primes p
    """
    # Brute force approach (exponential time)
    for x in range(1, p):
        if mod_exp(g, x, p) == h:
            return x
    return None

def integer_factorization(n):
    """
    Integer Factorization: Given n, find prime factors
    
    No known polynomial-time algorithm for large n
    """
    factors = []
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    
    if n > 1:
        factors.append(n)
    
    return factors
```

### Hardness Assumptions

#### Discrete Logarithm Problem (DLP)
Given g, h, p where p is prime and g is a generator of (ℤ/pℤ)*, find x such that g^x ≡ h (mod p).

#### Computational Diffie-Hellman (CDH)
Given g, g^a, g^b, compute g^ab.

#### Decisional Diffie-Hellman (DDH)
Given g, g^a, g^b, g^c, decide if c = ab.

#### RSA Problem
Given n = pq, e, and c ≡ m^e (mod n), find m.

---

## 🎯 Practical Examples

### Caesar Cipher Cryptanalysis

```python
def caesar_cryptanalysis(ciphertext):
    """Break Caesar cipher using frequency analysis"""
    # English letter frequencies
    english_freq = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97,
        'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
        'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
        'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
        'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
    }
    
    best_key = 0
    best_score = float('inf')
    
    for key in range(26):
        decrypted = caesar_decrypt(ciphertext, key)
        score = calculate_chi_squared(decrypted, english_freq)
        
        if score < best_score:
            best_score = score
            best_key = key
    
    return best_key, caesar_decrypt(ciphertext, best_key)

def caesar_decrypt(ciphertext, key):
    """Decrypt Caesar cipher"""
    return caesar_encrypt(ciphertext, -key)

def calculate_chi_squared(text, expected_freq):
    """Calculate chi-squared statistic"""
    observed = {}
    text = text.upper().replace(' ', '')
    
    for char in text:
        if char.isalpha():
            observed[char] = observed.get(char, 0) + 1
    
    chi_squared = 0
    total_chars = sum(observed.values())
    
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        observed_count = observed.get(char, 0)
        expected_count = expected_freq.get(char, 0) * total_chars / 100
        
        if expected_count > 0:
            chi_squared += (observed_count - expected_count) ** 2 / expected_count
    
    return chi_squared
```

### Vigenère Cipher Cryptanalysis

```python
def vigenere_cryptanalysis(ciphertext):
    """Break Vigenère cipher using Kasiski examination and frequency analysis"""
    
    def find_key_length(ciphertext):
        """Find likely key length using Kasiski examination"""
        coincidences = {}
        
        # Find repeated sequences
        for length in range(3, min(20, len(ciphertext) // 4)):
            for i in range(len(ciphertext) - length):
                substring = ciphertext[i:i+length]
                for j in range(i + length, len(ciphertext) - length):
                    if ciphertext[j:j+length] == substring:
                        distance = j - i
                        for factor in range(2, distance + 1):
                            if distance % factor == 0:
                                coincidences[factor] = coincidences.get(factor, 0) + 1
        
        # Return most likely key length
        if coincidences:
            return max(coincidences.items(), key=lambda x: x[1])[0]
        else:
            return 1
    
    def solve_single_substitution(ciphertext):
        """Solve single substitution cipher using frequency analysis"""
        best_key = 0
        best_score = float('inf')
        
        for key in range(26):
            decrypted = ''.join([chr((ord(c) - ord('A') - key) % 26 + ord('A')) 
                               for c in ciphertext if c.isalpha()])
            score = calculate_chi_squared(decrypted, english_freq)
            
            if score < best_score:
                best_score = score
                best_key = key
        
        return best_key
    
    ciphertext = ciphertext.upper().replace(' ', '')
    key_length = find_key_length(ciphertext)
    
    # Split into columns based on key length
    columns = [''] * key_length
    for i, char in enumerate(ciphertext):
        if char.isalpha():
            columns[i % key_length] += char
    
    # Solve each column as a Caesar cipher
    key = []
    for column in columns:
        key_char = solve_single_substitution(column)
        key.append(chr(key_char + ord('A')))
    
    recovered_key = ''.join(key)
    return recovered_key, vigenere_decrypt(ciphertext, recovered_key)
```

### Diffie-Hellman Key Exchange

```python
def diffie_hellman_demo():
    """Demonstrate Diffie-Hellman key exchange"""
    # Public parameters
    p = 23  # Prime modulus (small for demo)
    g = 5   # Generator
    
    print(f"Public parameters: p = {p}, g = {g}")
    
    # Alice's private key
    a = 6
    A = mod_exp(g, a, p)  # g^a mod p
    print(f"Alice: private key = {a}, public key = {A}")
    
    # Bob's private key
    b = 15
    B = mod_exp(g, b, p)  # g^b mod p
    print(f"Bob: private key = {b}, public key = {B}")
    
    # Shared secret computation
    shared_secret_alice = mod_exp(B, a, p)  # B^a mod p = g^(ba) mod p
    shared_secret_bob = mod_exp(A, b, p)    # A^b mod p = g^(ab) mod p
    
    print(f"Alice computes: {B}^{a} mod {p} = {shared_secret_alice}")
    print(f"Bob computes: {A}^{b} mod {p} = {shared_secret_bob}")
    print(f"Shared secret: {shared_secret_alice}")
    
    assert shared_secret_alice == shared_secret_bob
    return shared_secret_alice

# Demo
shared_secret = diffie_hellman_demo()
```

---

## 📚 Further Reading

### Essential Papers
1. **Shannon (1949)**: "Communication Theory of Secrecy Systems"
2. **Diffie & Hellman (1976)**: "New Directions in Cryptography"
3. **Rivest, Shamir & Adleman (1978)**: "A Method for Obtaining Digital Signatures"
4. **Goldwasser & Micali (1984)**: "Probabilistic Encryption"

### Recommended Books
1. **"Introduction to Modern Cryptography"** - Katz & Lindell
2. **"Applied Cryptography"** - Bruce Schneier
3. **"A Course in Number Theory and Cryptography"** - Neal Koblitz
4. **"The Mathematics of Ciphers"** - S.C. Coutinho

### Online Resources
- **NIST Cryptographic Standards**: https://csrc.nist.gov/
- **IACR Cryptology ePrint Archive**: https://eprint.iacr.org/
- **Crypto101**: https://www.crypto101.io/

---

## 🎯 Exercises

### Beginner Exercises
1. Implement Caesar cipher with key 13 (ROT13)
2. Break a monoalphabetic substitution cipher using frequency analysis
3. Compute gcd(1071, 462) using Euclidean algorithm
4. Find all primitive roots modulo 13

### Intermediate Exercises
1. Implement Vigenère cipher encryption and decryption
2. Break a Vigenère cipher with known key length
3. Compute 3^644 mod 645 efficiently
4. Implement basic RSA key generation

### Advanced Exercises
1. Implement Pollard's rho algorithm for integer factorization
2. Implement baby-step giant-step algorithm for discrete logarithm
3. Analyze the security of a custom cipher design
4. Implement elliptic curve point addition and scalar multiplication

---

*Next: [Classical Cryptography →](02_classical_cryptography.md)*
