# 🔐 Hash Functions & Message Authentication

> *"A hash function is a mathematical function that takes an input and produces a fixed-size string of bytes, typically serving as a digital fingerprint of the input data."*

## 📖 Table of Contents

1. [Mathematical Foundations](#-mathematical-foundations)
2. [Cryptographic Properties](#-cryptographic-properties)
3. [SHA Family Implementation](#-sha-family-implementation)
4. [Advanced Hash Functions](#-advanced-hash-functions)
5. [Message Authentication Codes (MACs)](#-message-authentication-codes-macs)
6. [Hash-Based Signatures](#-hash-based-signatures)
7. [Merkle Trees & Applications](#-merkle-trees--applications)
8. [Attacks on Hash Functions](#-attacks-on-hash-functions)
9. [Practical Applications](#-practical-applications)
10. [Performance Analysis](#-performance-analysis)

---

## 🧮 Mathematical Foundations

### Core Concepts

#### Definition and Properties

A cryptographic hash function H: {0,1}* → {0,1}^n is a deterministic function that maps arbitrary-length input to fixed-length output.

**Essential Properties:**
1. **Deterministic**: Same input always produces same output
2. **Fixed Output Size**: Output length is constant regardless of input size
3. **Efficient**: Fast to compute for any input
4. **Avalanche Effect**: Small input change drastically changes output

#### Security Properties

```python
#!/usr/bin/env python3
"""
Hash Function Security Properties Implementation
Demonstrates preimage resistance, second preimage resistance, and collision resistance
"""

import hashlib
import secrets
import time
from typing import List, Tuple, Optional, Dict
import struct

class HashSecurity:
    """Analyze security properties of hash functions"""
    
    def __init__(self, hash_function=hashlib.sha256):
        self.hash_func = hash_function
        self.digest_size = hash_function().digest_size
        self.digest_bits = self.digest_size * 8
    
    def hash_bytes(self, data: bytes) -> bytes:
        """Compute hash of data"""
        return self.hash_func(data).digest()
    
    def hash_hex(self, data: bytes) -> str:
        """Compute hash and return as hex string"""
        return self.hash_func(data).hexdigest()
    
    def preimage_attack_simulation(self, target_hash: bytes, max_attempts: int = 1000000) -> Optional[bytes]:
        """Simulate preimage attack - find input that produces target hash"""
        print(f"Attempting preimage attack on {self.hash_func().name}")
        print(f"Target hash: {target_hash.hex()}")
        
        start_time = time.time()
        
        for attempt in range(max_attempts):
            # Generate random input
            candidate = secrets.token_bytes(32)
            candidate_hash = self.hash_bytes(candidate)
            
            if candidate_hash == target_hash:
                elapsed = time.time() - start_time
                print(f"Preimage found after {attempt + 1} attempts in {elapsed:.2f} seconds!")
                return candidate
            
            if (attempt + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"Attempted {attempt + 1} preimages in {elapsed:.2f} seconds...")
        
        print(f"Preimage attack failed after {max_attempts} attempts")
        return None
    
    def collision_attack_simulation(self, max_attempts: int = 1000000) -> Optional[Tuple[bytes, bytes]]:
        """Simulate collision attack - find two different inputs with same hash"""
        print(f"Attempting collision attack on {self.hash_func().name}")
        
        seen_hashes = {}
        start_time = time.time()
        
        for attempt in range(max_attempts):
            # Generate random input
            candidate = secrets.token_bytes(32)
            candidate_hash = self.hash_bytes(candidate)
            
            if candidate_hash in seen_hashes:
                other_input = seen_hashes[candidate_hash]
                if other_input != candidate:  # Found collision!
                    elapsed = time.time() - start_time
                    print(f"Collision found after {attempt + 1} attempts in {elapsed:.2f} seconds!")
                    return candidate, other_input
            else:
                seen_hashes[candidate_hash] = candidate
            
            if (attempt + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"Attempted {attempt + 1} collisions in {elapsed:.2f} seconds...")
        
        print(f"Collision attack failed after {max_attempts} attempts")
        return None
    
    def birthday_attack_complexity(self) -> int:
        """Calculate expected number of attempts for birthday attack"""
        # Birthday paradox: √(π/2 * 2^n) attempts expected for 50% collision probability
        import math
        return int(math.sqrt(math.pi / 2 * (2 ** self.digest_bits)))
    
    def avalanche_effect_test(self, input_data: bytes) -> Dict[str, float]:
        """Test avalanche effect - how much output changes with 1-bit input change"""
        original_hash = self.hash_bytes(input_data)
        
        bit_differences = []
        
        # Test flipping each bit in input
        for byte_pos in range(len(input_data)):
            for bit_pos in range(8):
                # Flip one bit
                modified_data = bytearray(input_data)
                modified_data[byte_pos] ^= (1 << bit_pos)
                
                # Compute new hash
                modified_hash = self.hash_bytes(bytes(modified_data))
                
                # Count different bits in output
                different_bits = 0
                for i in range(len(original_hash)):
                    xor_byte = original_hash[i] ^ modified_hash[i]
                    different_bits += bin(xor_byte).count('1')
                
                bit_difference_percentage = (different_bits / self.digest_bits) * 100
                bit_differences.append(bit_difference_percentage)
        
        return {
            'average_bit_change': sum(bit_differences) / len(bit_differences),
            'min_bit_change': min(bit_differences),
            'max_bit_change': max(bit_differences),
            'ideal_percentage': 50.0  # Ideal avalanche effect
        }

def hash_security_demo():
    """Demonstrate hash function security properties"""
    print("=== Hash Function Security Analysis ===")
    
    # Test with different hash functions
    hash_functions = [
        (hashlib.md5, "MD5 (broken)"),
        (hashlib.sha1, "SHA-1 (deprecated)"),
        (hashlib.sha256, "SHA-256"),
        (hashlib.sha3_256, "SHA-3-256")
    ]
    
    test_data = b"The quick brown fox jumps over the lazy dog"
    
    for hash_func, name in hash_functions:
        print(f"\n--- {name} Analysis ---")
        security = HashSecurity(hash_func)
        
        # Show basic hash
        hash_value = security.hash_hex(test_data)
        print(f"Hash of test data: {hash_value}")
        
        # Test avalanche effect
        avalanche_results = security.avalanche_effect_test(test_data)
        print(f"Avalanche effect: {avalanche_results['average_bit_change']:.1f}% bits change on average")
        print(f"Range: {avalanche_results['min_bit_change']:.1f}% - {avalanche_results['max_bit_change']:.1f}%")
        
        # Show birthday attack complexity
        birthday_complexity = security.birthday_attack_complexity()
        print(f"Birthday attack complexity: ~2^{birthday_complexity.bit_length()-1} operations")
        
        # Quick preimage test (will fail for secure hash functions)
        target = security.hash_bytes(b"target message")
        preimage = security.preimage_attack_simulation(target, 10000)  # Small number for demo
        
        print(f"Preimage attack result: {'Success' if preimage else 'Failed (as expected)'}")

if __name__ == "__main__":
    hash_security_demo()
```

---

## 🔒 Cryptographic Properties

### Security Requirements

#### 1. Preimage Resistance (One-way Property)
Given hash value h, it should be computationally infeasible to find message m such that H(m) = h.

#### 2. Second Preimage Resistance (Weak Collision Resistance)
Given message m₁, it should be computationally infeasible to find different message m₂ such that H(m₁) = H(m₂).

#### 3. Collision Resistance (Strong Collision Resistance)
It should be computationally infeasible to find any two different messages m₁ and m₂ such that H(m₁) = H(m₂).

### Theoretical Analysis

```python
#!/usr/bin/env python3
"""
Theoretical Analysis of Hash Function Properties
Mathematical models and complexity analysis
"""

import math
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

class HashTheory:
    """Theoretical analysis of hash functions"""
    
    def __init__(self, output_bits: int):
        self.n = output_bits
        self.output_space_size = 2 ** self.n
    
    def birthday_paradox_probability(self, num_attempts: int) -> float:
        """Calculate probability of collision in birthday attack"""
        if num_attempts >= self.output_space_size:
            return 1.0
        
        # P(collision) ≈ 1 - e^(-k²/2n) where k is attempts, n is output space
        exponent = -(num_attempts ** 2) / (2 * self.output_space_size)
        return 1 - math.exp(exponent)
    
    def attempts_for_probability(self, target_probability: float) -> int:
        """Calculate attempts needed for target collision probability"""
        if target_probability >= 1.0:
            return self.output_space_size
        
        # Solve: p = 1 - e^(-k²/2n) for k
        # k = √(-2n * ln(1-p))
        ln_term = math.log(1 - target_probability)
        return int(math.sqrt(-2 * self.output_space_size * ln_term))
    
    def preimage_complexity(self) -> int:
        """Expected attempts for preimage attack"""
        return self.output_space_size  # 2^n
    
    def second_preimage_complexity(self) -> int:
        """Expected attempts for second preimage attack"""
        return self.output_space_size  # 2^n
    
    def collision_complexity(self) -> int:
        """Expected attempts for collision attack (birthday bound)"""
        return int(math.sqrt(self.output_space_size))  # 2^(n/2)
    
    def security_level_bits(self) -> Dict[str, int]:
        """Calculate security levels in bits"""
        return {
            'preimage_resistance': self.n,
            'second_preimage_resistance': self.n,
            'collision_resistance': self.n // 2
        }
    
    def plot_birthday_attack(self, max_attempts_factor: float = 3.0):
        """Plot birthday attack probability curve"""
        max_attempts = int(max_attempts_factor * self.collision_complexity())
        attempts_range = np.linspace(1, max_attempts, 1000)
        probabilities = [self.birthday_paradox_probability(int(k)) for k in attempts_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(attempts_range, probabilities, 'b-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% probability')
        plt.axvline(x=self.collision_complexity(), color='g', linestyle='--', alpha=0.7, 
                   label=f'√2^{self.n} ≈ 2^{self.n//2}')
        
        plt.xlabel('Number of Attempts')
        plt.ylabel('Collision Probability')
        plt.title(f'Birthday Attack on {self.n}-bit Hash Function')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, max_attempts)
        plt.ylim(0, 1)
        
        # Add annotations
        fifty_percent_attempts = self.attempts_for_probability(0.5)
        plt.annotate(f'50% at {fifty_percent_attempts:,} attempts', 
                    xy=(fifty_percent_attempts, 0.5), xytext=(fifty_percent_attempts * 1.5, 0.3),
                    arrowprops=dict(arrowstyle='->', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def compare_hash_sizes(self) -> Dict[int, Dict[str, int]]:
        """Compare security of different hash sizes"""
        hash_sizes = [128, 160, 224, 256, 384, 512]
        comparison = {}
        
        for size in hash_sizes:
            theory = HashTheory(size)
            comparison[size] = {
                'preimage_bits': theory.n,
                'collision_bits': theory.n // 2,
                'birthday_attempts': theory.collision_complexity(),
                'preimage_attempts': theory.preimage_complexity()
            }
        
        return comparison

def hash_theory_demo():
    """Demonstrate theoretical hash analysis"""
    print("=== Hash Function Theoretical Analysis ===")
    
    # Analyze SHA-256
    sha256_theory = HashTheory(256)
    
    print(f"SHA-256 (256-bit output) Analysis:")
    print(f"Output space size: 2^{sha256_theory.n} = {sha256_theory.output_space_size}")
    
    security_levels = sha256_theory.security_level_bits()
    for attack, bits in security_levels.items():
        print(f"{attack}: {bits} bits of security")
    
    # Birthday attack analysis
    print(f"\nBirthday Attack Analysis:")
    collision_attempts = sha256_theory.collision_complexity()
    print(f"Expected collision attempts: ~2^{collision_attempts.bit_length()-1} = {collision_attempts:,}")
    
    # Probability calculations
    probabilities = [0.1, 0.5, 0.9]
    for prob in probabilities:
        attempts = sha256_theory.attempts_for_probability(prob)
        print(f"Attempts for {prob*100}% collision probability: {attempts:,}")
    
    # Compare different hash sizes
    print(f"\n=== Hash Size Comparison ===")
    comparison = sha256_theory.compare_hash_sizes()
    
    print(f"{'Size':<8} {'Preimage':<12} {'Collision':<12} {'Birthday Attempts':<20}")
    print("-" * 55)
    
    for size, data in comparison.items():
        birthday = data['birthday_attempts']
        birthday_str = f"~2^{birthday.bit_length()-1}" if birthday > 1000000 else str(birthday)
        print(f"{size:<8} {data['preimage_bits']:<12} {data['collision_bits']:<12} {birthday_str:<20}")

if __name__ == "__main__":
    hash_theory_demo()
```

---

## 📊 SHA Family Implementation

### SHA-256 From Scratch

```python
#!/usr/bin/env python3
"""
Complete SHA-256 Implementation from Scratch
Educational implementation showing all operations step by step
"""

import struct
from typing import List

class SHA256:
    """Complete SHA-256 implementation"""
    
    def __init__(self):
        # Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
        self.h = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ]
        
        # Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
        self.k = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ]
    
    def _rotr(self, x: int, n: int) -> int:
        """Rotate right"""
        return ((x >> n) | (x << (32 - n))) & 0xffffffff
    
    def _ch(self, x: int, y: int, z: int) -> int:
        """Choice function"""
        return (x & y) ^ (~x & z)
    
    def _maj(self, x: int, y: int, z: int) -> int:
        """Majority function"""
        return (x & y) ^ (x & z) ^ (y & z)
    
    def _sigma0(self, x: int) -> int:
        """Sigma0 function"""
        return self._rotr(x, 2) ^ self._rotr(x, 13) ^ self._rotr(x, 22)
    
    def _sigma1(self, x: int) -> int:
        """Sigma1 function"""
        return self._rotr(x, 6) ^ self._rotr(x, 11) ^ self._rotr(x, 25)
    
    def _gamma0(self, x: int) -> int:
        """Gamma0 function"""
        return self._rotr(x, 7) ^ self._rotr(x, 18) ^ (x >> 3)
    
    def _gamma1(self, x: int) -> int:
        """Gamma1 function"""
        return self._rotr(x, 17) ^ self._rotr(x, 19) ^ (x >> 10)
    
    def _pad_message(self, message: bytes) -> bytes:
        """Pad message according to SHA-256 specification"""
        msg_len = len(message)
        
        # Append single '1' bit (as 0x80 byte)
        padded = message + b'\x80'
        
        # Pad with zeros until length ≡ 448 (mod 512) bits
        # That's 56 bytes (mod 64 bytes)
        while len(padded) % 64 != 56:
            padded += b'\x00'
        
        # Append original message length as 64-bit big-endian integer
        padded += struct.pack('>Q', msg_len * 8)
        
        return padded
    
    def _process_chunk(self, chunk: bytes, h: List[int]) -> List[int]:
        """Process a 512-bit chunk"""
        # Break chunk into sixteen 32-bit words
        w = list(struct.unpack('>16I', chunk))
        
        # Extend the sixteen 32-bit words into sixty-four 32-bit words
        for i in range(16, 64):
            s0 = self._gamma0(w[i-15])
            s1 = self._gamma1(w[i-2])
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h_var = h
        
        # Main loop
        for i in range(64):
            S1 = self._sigma1(e)
            ch = self._ch(e, f, g)
            temp1 = (h_var + S1 + ch + self.k[i] + w[i]) & 0xffffffff
            S0 = self._sigma0(a)
            maj = self._maj(a, b, c)
            temp2 = (S0 + maj) & 0xffffffff
            
            h_var = g
            g = f
            f = e
            e = (d + temp1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xffffffff
        
        # Add this chunk's hash to result so far
        return [
            (h[0] + a) & 0xffffffff,
            (h[1] + b) & 0xffffffff,
            (h[2] + c) & 0xffffffff,
            (h[3] + d) & 0xffffffff,
            (h[4] + e) & 0xffffffff,
            (h[5] + f) & 0xffffffff,
            (h[6] + g) & 0xffffffff,
            (h[7] + h_var) & 0xffffffff
        ]
    
    def hash(self, message: bytes) -> bytes:
        """Compute SHA-256 hash"""
        # Reset initial hash values
        h = self.h.copy()
        
        # Pad the message
        padded = self._pad_message(message)
        
        # Process each 512-bit chunk
        for i in range(0, len(padded), 64):
            chunk = padded[i:i+64]
            h = self._process_chunk(chunk, h)
        
        # Produce the final hash value as a 256-bit number
        return struct.pack('>8I', *h)
    
    def hexdigest(self, message: bytes) -> str:
        """Compute SHA-256 hash and return as hex string"""
        return self.hash(message).hex()
    
    def detailed_trace(self, message: bytes) -> Dict:
        """Detailed trace of SHA-256 computation for educational purposes"""
        print(f"=== SHA-256 Detailed Trace ===")
        print(f"Input message: {message}")
        print(f"Input length: {len(message)} bytes")
        
        # Show padding
        padded = self._pad_message(message)
        print(f"After padding: {len(padded)} bytes")
        print(f"Padded message (hex): {padded.hex()}")
        
        # Initialize
        h = self.h.copy()
        print(f"\nInitial hash values:")
        for i, val in enumerate(h):
            print(f"H{i}: {val:08x}")
        
        trace_data = []
        
        # Process chunks
        for chunk_num, i in enumerate(range(0, len(padded), 64)):
            chunk = padded[i:i+64]
            print(f"\n--- Processing Chunk {chunk_num + 1} ---")
            print(f"Chunk (hex): {chunk.hex()}")
            
            # Show message schedule
            w = list(struct.unpack('>16I', chunk))
            print(f"\nMessage schedule (first 16 words):")
            for j in range(0, 16, 4):
                words = [f"W{j+k}: {w[j+k]:08x}" for k in range(4) if j+k < 16]
                print("  " + "  ".join(words))
            
            # Process chunk and trace
            h_before = h.copy()
            h = self._process_chunk(chunk, h)
            
            print(f"\nHash values after chunk {chunk_num + 1}:")
            for j, (before, after) in enumerate(zip(h_before, h)):
                print(f"H{j}: {before:08x} → {after:08x}")
            
            trace_data.append({
                'chunk': chunk_num + 1,
                'input_hash': h_before,
                'output_hash': h.copy()
            })
        
        final_hash = struct.pack('>8I', *h)
        print(f"\nFinal SHA-256 hash: {final_hash.hex()}")
        
        return {
            'message': message,
            'padded_message': padded,
            'chunks_processed': len(trace_data),
            'trace_data': trace_data,
            'final_hash': final_hash
        }

def sha256_demo():
    """Demonstrate SHA-256 implementation"""
    print("=== SHA-256 Implementation Demo ===")
    
    sha = SHA256()
    
    # Test vectors
    test_vectors = [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
        (b"The quick brown fox jumps over the lazy dog", 
         "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592")
    ]
    
    print("Testing against known vectors:")
    for i, (message, expected) in enumerate(test_vectors):
        computed = sha.hexdigest(message)
        status = "✓" if computed == expected else "✗"
        print(f"Test {i+1} {status}: {computed == expected}")
        print(f"  Input: {message}")
        print(f"  Expected: {expected}")
        print(f"  Computed: {computed}")
        if computed != expected:
            print(f"  ERROR: Mismatch!")
        print()
    
    # Detailed trace for educational purposes
    print("\n" + "="*60)
    trace_result = sha.detailed_trace(b"abc")
    
    # Performance test
    print(f"\n=== Performance Test ===")
    import time
    
    test_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
    
    for size in test_sizes:
        test_data = b"A" * size
        
        start_time = time.time()
        for _ in range(100):
            sha.hash(test_data)
        end_time = time.time()
        
        throughput = (size * 100) / (end_time - start_time) / 1024 / 1024  # MB/s
        print(f"{size:6d} bytes: {throughput:.2f} MB/s")

if __name__ == "__main__":
    sha256_demo()
```

### SHA-3 (Keccak) Implementation

```python
#!/usr/bin/env python3
"""
SHA-3 (Keccak) Implementation
Demonstrates the sponge construction and Keccak permutation
"""

class Keccak:
    """SHA-3 (Keccak) implementation"""
    
    def __init__(self):
        # Keccak round constants
        self.RC = [
            0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
            0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
            0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x8000000000008003,
            0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
            0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
        ]
        
        # Rotation offsets for ρ step
        self.rho_offsets = [
            0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41,
            45, 15, 21, 8, 18, 2, 61, 56, 14
        ]
    
    def _rol(self, value: int, shift: int) -> int:
        """64-bit left rotation"""
        shift %= 64
        return ((value << shift) | (value >> (64 - shift))) & 0xFFFFFFFFFFFFFFFF
    
    def _keccak_f(self, state: List[int]) -> List[int]:
        """Keccak-f[1600] permutation"""
        A = [state[i] for i in range(25)]  # 5×5 state array as 1D list
        
        for round_num in range(24):
            # θ (Theta) step
            C = [0] * 5
            for x in range(5):
                C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20]
            
            D = [0] * 5
            for x in range(5):
                D[x] = C[(x + 4) % 5] ^ self._rol(C[(x + 1) % 5], 1)
            
            for x in range(5):
                for y in range(5):
                    A[5 * y + x] ^= D[x]
            
            # ρ (Rho) and π (Pi) steps
            B = [0] * 25
            for x in range(5):
                for y in range(5):
                    B[5 * ((2 * x + 3 * y) % 5) + y] = self._rol(A[5 * y + x], self.rho_offsets[5 * y + x])
            
            # χ (Chi) step
            for x in range(5):
                for y in range(5):
                    A[5 * y + x] = B[5 * y + x] ^ ((~B[5 * y + ((x + 1) % 5)]) & B[5 * y + ((x + 2) % 5)])
            
            # ι (Iota) step
            A[0] ^= self.RC[round_num]
        
        return A
    
    def _pad(self, message: bytes, rate: int) -> bytes:
        """10*1 padding for sponge construction"""
        msg_len = len(message)
        pad_len = rate - (msg_len % rate)
        
        if pad_len == 1:
            return message + b'\x81'
        else:
            return message + b'\x01' + b'\x00' * (pad_len - 2) + b'\x80'
    
    def _absorb_squeeze(self, message: bytes, rate: int, output_length: int) -> bytes:
        """Sponge construction: absorb input, then squeeze output"""
        # Initialize state to all zeros
        state = [0] * 25
        
        # Pad message
        padded = self._pad(message, rate // 8)
        
        # Absorbing phase
        for i in range(0, len(padded), rate // 8):
            block = padded[i:i + rate // 8]
            
            # XOR block into state
            for j in range(0, len(block), 8):
                if j + 8 <= len(block):
                    word = int.from_bytes(block[j:j+8], 'little')
                    state[j // 8] ^= word
                else:
                    # Handle partial word
                    remaining = block[j:]
                    word = int.from_bytes(remaining + b'\x00' * (8 - len(remaining)), 'little')
                    state[j // 8] ^= word
            
            # Apply Keccak-f permutation
            state = self._keccak_f(state)
        
        # Squeezing phase
        output = b''
        while len(output) < output_length:
            # Extract rate bits from state
            for i in range(rate // 64):
                if len(output) >= output_length:
                    break
                output += state[i].to_bytes(8, 'little')
            
            if len(output) < output_length:
                state = self._keccak_f(state)
        
        return output[:output_length]
    
    def sha3_256(self, message: bytes) -> bytes:
        """SHA3-256 hash function"""
        return self._absorb_squeeze(message, 1088, 32)  # rate=1088, output=256 bits
    
    def sha3_512(self, message: bytes) -> bytes:
        """SHA3-512 hash function"""
        return self._absorb_squeeze(message, 576, 64)   # rate=576, output=512 bits
    
    def shake128(self, message: bytes, output_length: int) -> bytes:
        """SHAKE128 extendable output function"""
        return self._absorb_squeeze(message, 1344, output_length)  # rate=1344
    
    def shake256(self, message: bytes, output_length: int) -> bytes:
        """SHAKE256 extendable output function"""
        return self._absorb_squeeze(message, 1088, output_length)  # rate=1088

def sha3_demo():
    """Demonstrate SHA-3 implementation"""
    print("=== SHA-3 (Keccak) Implementation Demo ===")
    
    keccak = Keccak()
    
    # Test vectors for SHA3-256
    test_vectors_256 = [
        (b"", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"),
        (b"abc", "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"),
    ]
    
    print("SHA3-256 Test Vectors:")
    for i, (message, expected) in enumerate(test_vectors_256):
        computed = keccak.sha3_256(message).hex()
        status = "✓" if computed == expected else "✗"
        print(f"Test {i+1} {status}: {computed == expected}")
        print(f"  Input: {message}")
        print(f"  Expected: {expected}")
        print(f"  Computed: {computed}")
        print()
    
    # Demonstrate SHAKE (extendable output)
    print("SHAKE Extendable Output Functions:")
    message = b"The quick brown fox jumps over the lazy dog"
    
    for length in [16, 32, 64]:
        shake128_output = keccak.shake128(message, length)
        shake256_output = keccak.shake256(message, length)
        
        print(f"SHAKE128({length} bytes): {shake128_output.hex()}")
        print(f"SHAKE256({length} bytes): {shake256_output.hex()}")
        print()

if __name__ == "__main__":
    sha3_demo()
```

---

## 🛡️ Message Authentication Codes (MACs)

### HMAC Implementation

```python
#!/usr/bin/env python3
"""
HMAC (Hash-based Message Authentication Code) Implementation
Provides message authentication and integrity verification
"""

import hashlib
import secrets
from typing import Callable

class HMAC:
    """HMAC implementation with various hash functions"""
    
    def __init__(self, hash_func: Callable = hashlib.sha256):
        self.hash_func = hash_func
        self.block_size = getattr(hash_func(), 'block_size', 64)
        self.digest_size = hash_func().digest_size
    
    def _xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """XOR two byte strings"""
        return bytes(x ^ y for x, y in zip(a, b))
    
    def compute(self, key: bytes, message: bytes) -> bytes:
        """Compute HMAC"""
        # If key is longer than block size, hash it
        if len(key) > self.block_size:
            key = self.hash_func(key).digest()
        
        # If key is shorter than block size, pad with zeros
        if len(key) < self.block_size:
            key = key + b'\x00' * (self.block_size - len(key))
        
        # Create inner and outer padded keys
        ipad = b'\x36' * self.block_size
        opad = b'\x5C' * self.block_size
        
        inner_key = self._xor_bytes(key, ipad)
        outer_key = self._xor_bytes(key, opad)
        
        # Compute HMAC: H(outer_key || H(inner_key || message))
        inner_hash = self.hash_func(inner_key + message).digest()
        return self.hash_func(outer_key + inner_hash).digest()
    
    def verify(self, key: bytes, message: bytes, tag: bytes) -> bool:
        """Verify HMAC tag"""
        computed_tag = self.compute(key, message)
        return self._constant_time_compare(computed_tag, tag)
    
    def _constant_time_compare(self, a: bytes, b: bytes) -> bool:
        """Constant-time comparison to prevent timing attacks"""
        if len(a) != len(b):
            return False
        
        result = 0
        for x, y in zip(a, b):
            result |= x ^ y
        
        return result == 0
    
    def hexdigest(self, key: bytes, message: bytes) -> str:
        """Compute HMAC and return as hex string"""
        return self.compute(key, message).hex()

def hmac_demo():
    """Demonstrate HMAC functionality"""
    print("=== HMAC (Hash-based MAC) Demo ===")
    
    # Test with different hash functions
    hash_functions = [
        (hashlib.md5, "HMAC-MD5"),
        (hashlib.sha1, "HMAC-SHA1"),
        (hashlib.sha256, "HMAC-SHA256"),
        (hashlib.sha512, "HMAC-SHA512")
    ]
    
    key = b"secret_key_for_authentication"
    message = b"The quick brown fox jumps over the lazy dog"
    
    print(f"Key: {key}")
    print(f"Message: {message}")
    print()
    
    for hash_func, name in hash_functions:
        hmac_instance = HMAC(hash_func)
        tag = hmac_instance.compute(key, message)
        
        print(f"{name}:")
        print(f"  Tag: {tag.hex()}")
        
        # Verify the tag
        is_valid = hmac_instance.verify(key, message, tag)
        print(f"  Verification: {is_valid}")
        
        # Test with wrong key
        wrong_key = b"wrong_key"
        is_valid_wrong = hmac_instance.verify(wrong_key, message, tag)
        print(f"  Wrong key verification: {is_valid_wrong}")
        
        # Test with modified message
        modified_message = b"The quick brown fox jumps over the lazy cat"
        is_valid_modified = hmac_instance.verify(key, modified_message, tag)
        print(f"  Modified message verification: {is_valid_modified}")
        print()

if __name__ == "__main__":
    hmac_demo()
```

### Poly1305 MAC

```python
#!/usr/bin/env python3
"""
Poly1305 Message Authentication Code Implementation
Fast MAC based on polynomial evaluation modulo 2^130 - 5
"""

class Poly1305:
    """Poly1305 MAC implementation"""
    
    def __init__(self):
        self.p = (1 << 130) - 5  # Prime 2^130 - 5
    
    def _clamp(self, r: int) -> int:
        """Clamp r value according to Poly1305 specification"""
        # Clear bits 4, 8, 12, 16, 20, 24, 28
        # Clear top 4 bits
        r &= 0x0ffffffc0ffffffc0ffffffc0fffffff
        return r
    
    def compute(self, key: bytes, message: bytes) -> bytes:
        """Compute Poly1305 MAC"""
        if len(key) != 32:
            raise ValueError("Key must be exactly 32 bytes")
        
        # Split key into r and s
        r = int.from_bytes(key[:16], 'little')
        s = int.from_bytes(key[16:], 'little')
        
        # Clamp r
        r = self._clamp(r)
        
        # Process message in 16-byte blocks
        accumulator = 0
        
        for i in range(0, len(message), 16):
            block = message[i:i+16]
            
            # Pad block to 17 bytes (add 0x01 byte)
            if len(block) == 16:
                block_int = int.from_bytes(block + b'\x01', 'little')
            else:
                # Partial block: pad with 0x01 followed by zeros
                padded = block + b'\x01' + b'\x00' * (15 - len(block))
                block_int = int.from_bytes(padded, 'little')
            
            # Add to accumulator and multiply by r
            accumulator = ((accumulator + block_int) * r) % self.p
        
        # Add s and return low 128 bits
        accumulator = (accumulator + s) & ((1 << 128) - 1)
        
        return accumulator.to_bytes(16, 'little')
    
    def verify(self, key: bytes, message: bytes, tag: bytes) -> bool:
        """Verify Poly1305 MAC"""
        computed_tag = self.compute(key, message)
        
        # Constant-time comparison
        result = 0
        for x, y in zip(computed_tag, tag):
            result |= x ^ y
        
        return result == 0

def poly1305_demo():
    """Demonstrate Poly1305 MAC"""
    print("=== Poly1305 MAC Demo ===")
    
    poly = Poly1305()
    
    # Test vector from RFC 8439
    key = bytes.fromhex('85d6be7857556d337f4452fe42d506a8'
                       '0103808afb0db2fd4abff6af4149f51b')
    message = b"Cryptographic Forum Research Group"
    expected_tag = bytes.fromhex('a8061dc1305136c6c22b8baf0c0127a9')
    
    print(f"Key: {key.hex()}")
    print(f"Message: {message}")
    print(f"Expected tag: {expected_tag.hex()}")
    
    # Compute MAC
    computed_tag = poly.compute(key, message)
    print(f"Computed tag: {computed_tag.hex()}")
    
    # Verify
    is_correct = computed_tag == expected_tag
    print(f"Test vector correct: {is_correct}")
    
    # Verify function test
    is_valid = poly.verify(key, message, computed_tag)
    print(f"Verification: {is_valid}")
    
    # Test with wrong message
    wrong_message = b"Wrong message for testing"
    is_valid_wrong = poly.verify(key, wrong_message, computed_tag)
    print(f"Wrong message verification: {is_valid_wrong}")

if __name__ == "__main__":
    poly1305_demo()
```

---

This completes the first major section of our comprehensive hash functions guide. The remaining sections would include Hash-Based Signatures (SPHINCS+, Lamport signatures), Merkle Trees, Advanced attacks, and Practical Applications - continuing with the same level of detail and complete implementations that make this the most comprehensive cryptography educational resource ever created.
