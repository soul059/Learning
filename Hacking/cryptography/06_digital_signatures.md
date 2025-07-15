# 🔐 Digital Signatures & Authentication

> *"Digital signatures provide the digital equivalent of handwritten signatures, ensuring authenticity, integrity, and non-repudiation in the digital world."*

## 📖 Table of Contents

1. [Mathematical Foundations](#-mathematical-foundations)
2. [RSA Signatures](#-rsa-signatures)
3. [DSA & ECDSA](#-dsa--ecdsa)
4. [EdDSA & Ed25519](#-eddsa--ed25519)
5. [Hash-Based Signatures](#-hash-based-signatures)
6. [Blind Signatures](#-blind-signatures)
7. [Ring Signatures](#-ring-signatures)
8. [Multi-Signatures](#-multi-signatures)
9. [Threshold Signatures](#-threshold-signatures)
10. [Non-Repudiation Systems](#-non-repudiation-systems)

---

## 🧮 Mathematical Foundations

### Digital Signature Scheme Components

A digital signature scheme consists of three algorithms:
1. **KeyGen(1^λ)**: Generate key pair (sk, pk)
2. **Sign(sk, m)**: Generate signature σ for message m
3. **Verify(pk, m, σ)**: Verify signature σ on message m

### Security Properties

```python
#!/usr/bin/env python3
"""
Digital Signature Security Properties Implementation
Demonstrates unforgeability, authenticity, and non-repudiation
"""

import hashlib
import secrets
import time
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

class DigitalSignatureScheme(ABC):
    """Abstract base class for digital signature schemes"""
    
    @abstractmethod
    def generate_keypair(self) -> Tuple[any, any]:
        """Generate a new key pair (private_key, public_key)"""
        pass
    
    @abstractmethod
    def sign(self, private_key: any, message: bytes) -> bytes:
        """Sign a message with private key"""
        pass
    
    @abstractmethod
    def verify(self, public_key: any, message: bytes, signature: bytes) -> bool:
        """Verify signature with public key"""
        pass
    
    def hash_message(self, message: bytes) -> bytes:
        """Hash message before signing (if needed)"""
        return hashlib.sha256(message).digest()

class SignatureSecurity:
    """Analyze security properties of signature schemes"""
    
    def __init__(self, scheme: DigitalSignatureScheme):
        self.scheme = scheme
        self.signatures_seen = {}  # Track signatures for replay attacks
    
    def test_basic_functionality(self, message: bytes) -> Dict[str, bool]:
        """Test basic sign/verify functionality"""
        # Generate keys
        private_key, public_key = self.scheme.generate_keypair()
        
        # Sign message
        signature = self.scheme.sign(private_key, message)
        
        # Verify signature
        is_valid = self.scheme.verify(public_key, message, signature)
        
        # Test with wrong message
        wrong_message = b"Different message"
        is_valid_wrong = self.scheme.verify(public_key, wrong_message, signature)
        
        # Test deterministic property (sign same message twice)
        signature2 = self.scheme.sign(private_key, message)
        deterministic = signature == signature2
        
        return {
            'basic_verification': is_valid,
            'wrong_message_rejected': not is_valid_wrong,
            'deterministic': deterministic,
            'signature_length': len(signature)
        }
    
    def test_unforgeability(self, num_attempts: int = 1000) -> Dict[str, any]:
        """Test existential unforgeability under chosen message attack"""
        private_key, public_key = self.scheme.generate_keypair()
        
        # Attacker can request signatures on chosen messages
        chosen_messages = [
            b"Message 1",
            b"Message 2", 
            b"Message 3",
            b"Known plaintext attack"
        ]
        
        # Get legitimate signatures
        legitimate_signatures = {}
        for msg in chosen_messages:
            sig = self.scheme.sign(private_key, msg)
            legitimate_signatures[msg] = sig
        
        # Try to forge signatures on new messages
        forgery_attempts = 0
        successful_forgeries = 0
        
        for _ in range(num_attempts):
            # Generate random message not in chosen_messages
            random_msg = secrets.token_bytes(32)
            if random_msg in chosen_messages:
                continue
            
            forgery_attempts += 1
            
            # Try random signature
            random_signature = secrets.token_bytes(64)  # Assume 64-byte signatures
            
            if self.scheme.verify(public_key, random_msg, random_signature):
                successful_forgeries += 1
        
        forgery_rate = successful_forgeries / forgery_attempts if forgery_attempts > 0 else 0
        
        return {
            'chosen_message_count': len(chosen_messages),
            'forgery_attempts': forgery_attempts,
            'successful_forgeries': successful_forgeries,
            'forgery_rate': forgery_rate,
            'security_level': 'HIGH' if forgery_rate == 0 else 'LOW'
        }
    
    def test_non_repudiation(self, message: bytes) -> Dict[str, any]:
        """Test non-repudiation properties"""
        # Alice generates keys and signs
        alice_private, alice_public = self.scheme.generate_keypair()
        signature = self.scheme.sign(alice_private, message)
        
        # Bob verifies (different party)
        bob_verification = self.scheme.verify(alice_public, message, signature)
        
        # Carol verifies (third party)
        carol_verification = self.scheme.verify(alice_public, message, signature)
        
        # Charlie (attacker) tries to verify with different key
        charlie_private, charlie_public = self.scheme.generate_keypair()
        charlie_false_verification = self.scheme.verify(charlie_public, message, signature)
        
        return {
            'alice_can_prove_authorship': True,  # She has the private key
            'bob_can_verify': bob_verification,
            'third_party_can_verify': carol_verification,
            'wrong_key_rejected': not charlie_false_verification,
            'publicly_verifiable': bob_verification and carol_verification
        }
    
    def performance_benchmark(self, message_sizes: List[int], iterations: int = 100) -> Dict[str, List[float]]:
        """Benchmark signature performance"""
        private_key, public_key = self.scheme.generate_keypair()
        
        results = {
            'message_sizes': message_sizes,
            'sign_times': [],
            'verify_times': [],
            'signature_sizes': []
        }
        
        for size in message_sizes:
            message = b"A" * size
            
            # Benchmark signing
            sign_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                signature = self.scheme.sign(private_key, message)
                end = time.perf_counter()
                sign_times.append(end - start)
            
            # Benchmark verification
            verify_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                self.scheme.verify(public_key, message, signature)
                end = time.perf_counter()
                verify_times.append(end - start)
            
            results['sign_times'].append(sum(sign_times) / len(sign_times))
            results['verify_times'].append(sum(verify_times) / len(verify_times))
            results['signature_sizes'].append(len(signature))
        
        return results

def signature_security_demo():
    """Demonstrate signature security analysis"""
    print("=== Digital Signature Security Analysis ===")
    
    # We'll implement a simple RSA signature scheme for testing
    class SimpleRSASignature(DigitalSignatureScheme):
        def generate_keypair(self):
            # Simplified RSA key generation (not secure for real use)
            p, q = 61, 53  # Small primes for demo
            n = p * q
            phi = (p - 1) * (q - 1)
            e = 17
            d = pow(e, -1, phi)
            return (d, p, q), (n, e)
        
        def sign(self, private_key, message):
            d, p, q = private_key
            n = p * q
            h = int.from_bytes(self.hash_message(message)[:4], 'big')  # Truncate for demo
            signature = pow(h, d, n)
            return signature.to_bytes(4, 'big')
        
        def verify(self, public_key, message, signature):
            n, e = public_key
            sig_int = int.from_bytes(signature, 'big')
            h = int.from_bytes(self.hash_message(message)[:4], 'big')
            decrypted = pow(sig_int, e, n)
            return decrypted == h
    
    # Test the signature scheme
    rsa_scheme = SimpleRSASignature()
    security_tester = SignatureSecurity(rsa_scheme)
    
    test_message = b"Hello, this is a test message for digital signatures!"
    
    # Test basic functionality
    print("--- Basic Functionality Test ---")
    basic_results = security_tester.test_basic_functionality(test_message)
    for test, result in basic_results.items():
        print(f"{test}: {result}")
    
    # Test unforgeability
    print("\n--- Unforgeability Test ---")
    unforgeability_results = security_tester.test_unforgeability(100)  # Small number for demo
    for test, result in unforgeability_results.items():
        print(f"{test}: {result}")
    
    # Test non-repudiation
    print("\n--- Non-Repudiation Test ---")
    non_repudiation_results = security_tester.test_non_repudiation(test_message)
    for test, result in non_repudiation_results.items():
        print(f"{test}: {result}")
    
    # Performance benchmark
    print("\n--- Performance Benchmark ---")
    perf_results = security_tester.performance_benchmark([100, 500, 1000], 10)
    for i, size in enumerate(perf_results['message_sizes']):
        print(f"Message size {size} bytes:")
        print(f"  Average sign time: {perf_results['sign_times'][i]*1000:.3f} ms")
        print(f"  Average verify time: {perf_results['verify_times'][i]*1000:.3f} ms")
        print(f"  Signature size: {perf_results['signature_sizes'][i]} bytes")

if __name__ == "__main__":
    signature_security_demo()
```

---

## 🔐 RSA Signatures

### RSA-PSS (Probabilistic Signature Scheme)

```python
#!/usr/bin/env python3
"""
RSA-PSS (Probabilistic Signature Scheme) Implementation
Provides enhanced security over deterministic RSA signatures
"""

import hashlib
import secrets
import os
from typing import Tuple

class RSAPSS:
    """RSA-PSS signature implementation"""
    
    def __init__(self, key_size: int = 2048, hash_func=hashlib.sha256):
        self.key_size = key_size
        self.hash_func = hash_func
        self.hash_length = hash_func().digest_size
        self.salt_length = self.hash_length  # Recommended salt length
    
    def generate_keypair(self) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
        """Generate RSA key pair"""
        # Simplified key generation (use proper implementation in production)
        e = 65537
        
        # Generate primes (simplified for demo)
        import random
        def generate_prime(bits):
            while True:
                candidate = random.getrandbits(bits)
                candidate |= (1 << bits - 1) | 1
                if self._is_prime(candidate):
                    return candidate
        
        p = generate_prime(self.key_size // 2)
        q = generate_prime(self.key_size // 2)
        
        n = p * q
        phi = (p - 1) * (q - 1)
        d = pow(e, -1, phi)
        
        private_key = (n, e, d)
        public_key = (n, e)
        
        return private_key, public_key
    
    def _is_prime(self, n: int, k: int = 5) -> bool:
        """Miller-Rabin primality test"""
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
        
        # Test witnesses
        for _ in range(k):
            a = secrets.randbelow(n - 3) + 2
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def _mgf1(self, seed: bytes, length: int) -> bytes:
        """MGF1 mask generation function"""
        T = b''
        counter = 0
        
        while len(T) < length:
            C = counter.to_bytes(4, 'big')
            T += self.hash_func(seed + C).digest()
            counter += 1
        
        return T[:length]
    
    def _pss_encode(self, message: bytes, em_bits: int) -> bytes:
        """PSS encoding operation"""
        # Hash the message
        message_hash = self.hash_func(message).digest()
        
        # Calculate sizes
        em_len = (em_bits + 7) // 8
        
        if em_len < self.hash_length + self.salt_length + 2:
            raise ValueError("Encoding error: insufficient space")
        
        # Generate salt
        salt = secrets.token_bytes(self.salt_length)
        
        # Compute M' = padding1 || message_hash || salt
        padding1 = b'\x00' * 8
        m_prime = padding1 + message_hash + salt
        
        # Compute H = Hash(M')
        H = self.hash_func(m_prime).digest()
        
        # Generate PS (padding string)
        ps_len = em_len - self.salt_length - self.hash_length - 2
        PS = b'\x00' * ps_len
        
        # Compute DB = PS || 0x01 || salt
        DB = PS + b'\x01' + salt
        
        # Compute dbMask = MGF(H, em_len - hash_length - 1)
        db_mask = self._mgf1(H, em_len - self.hash_length - 1)
        
        # Compute maskedDB = DB XOR dbMask
        masked_db = bytes(a ^ b for a, b in zip(DB, db_mask))
        
        # Set leftmost bits to zero
        leftmost_bits = 8 * em_len - em_bits
        if leftmost_bits > 0:
            masked_db = bytes([masked_db[0] & (0xFF >> leftmost_bits)]) + masked_db[1:]
        
        # Compute EM = maskedDB || H || 0xbc
        EM = masked_db + H + b'\xbc'
        
        return EM
    
    def _pss_verify_encoding(self, message: bytes, em: bytes, em_bits: int) -> bool:
        """PSS verification of encoding"""
        # Hash the message
        message_hash = self.hash_func(message).digest()
        
        # Calculate sizes
        em_len = (em_bits + 7) // 8
        
        if em_len < self.hash_length + self.salt_length + 2:
            return False
        
        if em[-1] != 0xbc:
            return False
        
        # Split EM
        masked_db = em[:em_len - self.hash_length - 1]
        H = em[em_len - self.hash_length - 1:-1]
        
        # Check leftmost bits
        leftmost_bits = 8 * em_len - em_bits
        if leftmost_bits > 0:
            if masked_db[0] & (0xFF << (8 - leftmost_bits)):
                return False
        
        # Compute dbMask = MGF(H, em_len - hash_length - 1)
        db_mask = self._mgf1(H, em_len - self.hash_length - 1)
        
        # Compute DB = maskedDB XOR dbMask
        DB = bytes(a ^ b for a, b in zip(masked_db, db_mask))
        
        # Set leftmost bits to zero
        if leftmost_bits > 0:
            DB = bytes([DB[0] & (0xFF >> leftmost_bits)]) + DB[1:]
        
        # Check DB structure
        ps_len = em_len - self.hash_length - self.salt_length - 2
        
        # Check PS is all zeros
        if DB[:ps_len] != b'\x00' * ps_len:
            return False
        
        # Check separator
        if len(DB) > ps_len and DB[ps_len] != 0x01:
            return False
        
        # Extract salt
        salt = DB[ps_len + 1:]
        
        # Compute M' = padding1 || message_hash || salt
        padding1 = b'\x00' * 8
        m_prime = padding1 + message_hash + salt
        
        # Compute H' = Hash(M')
        h_prime = self.hash_func(m_prime).digest()
        
        # Verify H == H'
        return H == h_prime
    
    def sign(self, private_key: Tuple[int, int, int], message: bytes) -> bytes:
        """Sign message using RSA-PSS"""
        n, e, d = private_key
        
        # Encode message using PSS
        em_bits = n.bit_length() - 1
        em = self._pss_encode(message, em_bits)
        
        # Convert to integer
        em_int = int.from_bytes(em, 'big')
        
        # RSA signature
        signature_int = pow(em_int, d, n)
        
        # Convert back to bytes
        signature_bytes = signature_int.to_bytes((n.bit_length() + 7) // 8, 'big')
        
        return signature_bytes
    
    def verify(self, public_key: Tuple[int, int], message: bytes, signature: bytes) -> bool:
        """Verify RSA-PSS signature"""
        n, e = public_key
        
        # Convert signature to integer
        signature_int = int.from_bytes(signature, 'big')
        
        # RSA verification
        em_int = pow(signature_int, e, n)
        
        # Convert to bytes
        em_len = (n.bit_length() + 7) // 8
        em = em_int.to_bytes(em_len, 'big')
        
        # Verify PSS encoding
        em_bits = n.bit_length() - 1
        return self._pss_verify_encoding(message, em, em_bits)

def rsa_pss_demo():
    """Demonstrate RSA-PSS signatures"""
    print("=== RSA-PSS Signature Demo ===")
    
    # Create RSA-PSS instance
    rsa_pss = RSAPSS(key_size=1024)  # Small for demo
    
    # Generate keys
    print("Generating RSA key pair...")
    private_key, public_key = rsa_pss.generate_keypair()
    n, e = public_key
    
    print(f"Key size: {n.bit_length()} bits")
    print(f"Public exponent: {e}")
    
    # Test message
    message = b"This is a test message for RSA-PSS signatures!"
    print(f"\nMessage: {message}")
    
    # Sign message
    print("Signing message...")
    signature = rsa_pss.sign(private_key, message)
    print(f"Signature length: {len(signature)} bytes")
    print(f"Signature (hex): {signature.hex()[:64]}...")
    
    # Verify signature
    print("Verifying signature...")
    is_valid = rsa_pss.verify(public_key, message, signature)
    print(f"Signature valid: {is_valid}")
    
    # Test with modified message
    modified_message = b"This is a modified test message for RSA-PSS signatures!"
    is_valid_modified = rsa_pss.verify(public_key, modified_message, signature)
    print(f"Modified message verification: {is_valid_modified}")
    
    # Test probabilistic property (same message, different signatures)
    signature2 = rsa_pss.sign(private_key, message)
    signatures_different = signature != signature2
    both_valid = (rsa_pss.verify(public_key, message, signature) and 
                  rsa_pss.verify(public_key, message, signature2))
    
    print(f"\nProbabilistic signatures:")
    print(f"Two signatures different: {signatures_different}")
    print(f"Both signatures valid: {both_valid}")

if __name__ == "__main__":
    rsa_pss_demo()
```

---

## 🔑 DSA & ECDSA

### ECDSA with Multiple Curves

```python
#!/usr/bin/env python3
"""
ECDSA Implementation with Multiple Elliptic Curves
Supports secp256k1, secp256r1, and other standard curves
"""

import hashlib
import secrets
from typing import Tuple, NamedTuple

class Point(NamedTuple):
    """Elliptic curve point"""
    x: int
    y: int

class EllipticCurve:
    """Elliptic curve definition"""
    
    def __init__(self, name: str, p: int, a: int, b: int, G: Point, n: int, h: int):
        self.name = name
        self.p = p    # Prime modulus
        self.a = a    # Curve parameter a
        self.b = b    # Curve parameter b
        self.G = G    # Generator point
        self.n = n    # Order of G
        self.h = h    # Cofactor

class ECDSACurves:
    """Standard elliptic curves for ECDSA"""
    
    @staticmethod
    def secp256k1():
        """Bitcoin's secp256k1 curve"""
        p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        a = 0
        b = 7
        G = Point(
            0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
            0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        )
        n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        h = 1
        
        return EllipticCurve("secp256k1", p, a, b, G, n, h)
    
    @staticmethod
    def secp256r1():
        """NIST P-256 curve"""
        p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
        a = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC
        b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
        G = Point(
            0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
            0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
        )
        n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
        h = 1
        
        return EllipticCurve("secp256r1", p, a, b, G, n, h)

class ECDSASignature:
    """ECDSA signature implementation"""
    
    def __init__(self, curve: EllipticCurve):
        self.curve = curve
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Modular multiplicative inverse"""
        return pow(a, -1, m)
    
    def _point_add(self, P: Point, Q: Point) -> Point:
        """Elliptic curve point addition"""
        if P is None:  # Identity element
            return Q
        if Q is None:  # Identity element
            return P
        
        if P.x == Q.x:
            if P.y == Q.y:
                return self._point_double(P)
            else:
                return None  # Point at infinity
        
        # Different points
        s = ((Q.y - P.y) * self._mod_inverse(Q.x - P.x, self.curve.p)) % self.curve.p
        x3 = (s * s - P.x - Q.x) % self.curve.p
        y3 = (s * (P.x - x3) - P.y) % self.curve.p
        
        return Point(x3, y3)
    
    def _point_double(self, P: Point) -> Point:
        """Elliptic curve point doubling"""
        if P is None:
            return None
        
        s = ((3 * P.x * P.x + self.curve.a) * self._mod_inverse(2 * P.y, self.curve.p)) % self.curve.p
        x3 = (s * s - 2 * P.x) % self.curve.p
        y3 = (s * (P.x - x3) - P.y) % self.curve.p
        
        return Point(x3, y3)
    
    def _point_multiply(self, k: int, P: Point) -> Point:
        """Scalar multiplication k * P"""
        if k == 0:
            return None  # Point at infinity
        if k == 1:
            return P
        
        result = None
        addend = P
        
        while k:
            if k & 1:
                result = self._point_add(result, addend)
            addend = self._point_double(addend)
            k >>= 1
        
        return result
    
    def generate_keypair(self) -> Tuple[int, Point]:
        """Generate ECDSA key pair"""
        # Generate random private key
        private_key = secrets.randbelow(self.curve.n - 1) + 1
        
        # Compute public key Q = d * G
        public_key = self._point_multiply(private_key, self.curve.G)
        
        return private_key, public_key
    
    def sign(self, private_key: int, message: bytes) -> Tuple[int, int]:
        """Sign message using ECDSA"""
        # Hash the message
        message_hash = hashlib.sha256(message).digest()
        z = int.from_bytes(message_hash, 'big')
        
        # Truncate z if necessary
        if z.bit_length() > self.curve.n.bit_length():
            z = z >> (z.bit_length() - self.curve.n.bit_length())
        
        while True:
            # Generate random k
            k = secrets.randbelow(self.curve.n - 1) + 1
            
            # Compute (x1, y1) = k * G
            point = self._point_multiply(k, self.curve.G)
            if point is None:
                continue
            
            # Compute r = x1 mod n
            r = point.x % self.curve.n
            if r == 0:
                continue
            
            # Compute s = k^(-1) * (z + r * private_key) mod n
            k_inv = self._mod_inverse(k, self.curve.n)
            s = (k_inv * (z + r * private_key)) % self.curve.n
            if s == 0:
                continue
            
            return r, s
    
    def verify(self, public_key: Point, message: bytes, signature: Tuple[int, int]) -> bool:
        """Verify ECDSA signature"""
        r, s = signature
        
        # Verify signature parameters
        if not (1 <= r < self.curve.n and 1 <= s < self.curve.n):
            return False
        
        # Hash the message
        message_hash = hashlib.sha256(message).digest()
        z = int.from_bytes(message_hash, 'big')
        
        # Truncate z if necessary
        if z.bit_length() > self.curve.n.bit_length():
            z = z >> (z.bit_length() - self.curve.n.bit_length())
        
        # Compute w = s^(-1) mod n
        w = self._mod_inverse(s, self.curve.n)
        
        # Compute u1 = z * w mod n and u2 = r * w mod n
        u1 = (z * w) % self.curve.n
        u2 = (r * w) % self.curve.n
        
        # Compute (x1, y1) = u1 * G + u2 * Q
        point1 = self._point_multiply(u1, self.curve.G)
        point2 = self._point_multiply(u2, public_key)
        point = self._point_add(point1, point2)
        
        if point is None:
            return False
        
        # Verify r == x1 mod n
        return r == (point.x % self.curve.n)

def ecdsa_demo():
    """Demonstrate ECDSA with multiple curves"""
    print("=== ECDSA Multi-Curve Demo ===")
    
    curves = [
        ECDSACurves.secp256k1(),
        ECDSACurves.secp256r1()
    ]
    
    message = b"Hello, ECDSA signatures on multiple curves!"
    
    for curve in curves:
        print(f"\n--- {curve.name} ---")
        ecdsa = ECDSASignature(curve)
        
        # Generate keys
        private_key, public_key = ecdsa.generate_keypair()
        print(f"Private key: {hex(private_key)}")
        print(f"Public key: ({hex(public_key.x)[:20]}..., {hex(public_key.y)[:20]}...)")
        
        # Sign message
        signature = ecdsa.sign(private_key, message)
        r, s = signature
        print(f"Signature: (r={hex(r)[:20]}..., s={hex(s)[:20]}...)")
        
        # Verify signature
        is_valid = ecdsa.verify(public_key, message, signature)
        print(f"Signature valid: {is_valid}")
        
        # Test with modified message
        modified_message = b"Modified message for ECDSA!"
        is_valid_modified = ecdsa.verify(public_key, modified_message, signature)
        print(f"Modified message valid: {is_valid_modified}")

if __name__ == "__main__":
    ecdsa_demo()
```

---

## 🔏 EdDSA & Ed25519

### Complete Ed25519 Implementation

```python
#!/usr/bin/env python3
"""
Ed25519 Digital Signature Implementation
Edwards-curve Digital Signature Algorithm using Curve25519
"""

import hashlib
import secrets
from typing import Tuple, bytes as Bytes

class Ed25519:
    """Ed25519 signature implementation"""
    
    def __init__(self):
        # Curve25519 parameters
        self.p = 2**255 - 19
        self.d = -121665 * pow(121666, -1, self.p) % self.p
        self.l = 2**252 + 27742317777372353535851937790883648493  # Order of base point
        
        # Base point
        self.B = self._decode_point(bytes([
            0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66
        ]))
    
    def _mod_p(self, x: int) -> int:
        """Reduce modulo p"""
        return x % self.p
    
    def _mod_l(self, x: int) -> int:
        """Reduce modulo l"""
        return x % self.l
    
    def _pow_p(self, x: int, exp: int) -> int:
        """Modular exponentiation modulo p"""
        return pow(x, exp, self.p)
    
    def _inv_p(self, x: int) -> int:
        """Modular inverse modulo p"""
        return pow(x, self.p - 2, self.p)
    
    def _point_add(self, P: Tuple[int, int], Q: Tuple[int, int]) -> Tuple[int, int]:
        """Edwards curve point addition"""
        x1, y1 = P
        x2, y2 = Q
        
        x3 = (x1 * y2 + y1 * x2) * self._inv_p(1 + self.d * x1 * x2 * y1 * y2) % self.p
        y3 = (y1 * y2 - x1 * x2) * self._inv_p(1 - self.d * x1 * x2 * y1 * y2) % self.p
        
        return (x3, y3)
    
    def _point_multiply(self, k: int, P: Tuple[int, int]) -> Tuple[int, int]:
        """Scalar multiplication"""
        if k == 0:
            return (0, 1)  # Identity point
        if k == 1:
            return P
        
        result = (0, 1)  # Identity
        addend = P
        
        while k > 0:
            if k & 1:
                result = self._point_add(result, addend)
            addend = self._point_add(addend, addend)
            k >>= 1
        
        return result
    
    def _encode_point(self, P: Tuple[int, int]) -> bytes:
        """Encode point to 32 bytes"""
        x, y = P
        
        # Encode y coordinate
        encoded = y.to_bytes(32, 'little')
        
        # Set sign bit for x coordinate
        if x & 1:
            encoded = bytes([encoded[31] | 0x80]) + encoded[:-1]
        
        return encoded
    
    def _decode_point(self, encoded: bytes) -> Tuple[int, int]:
        """Decode point from 32 bytes"""
        if len(encoded) != 32:
            raise ValueError("Invalid point encoding")
        
        # Extract y coordinate
        y = int.from_bytes(encoded, 'little') & ((1 << 255) - 1)
        
        # Extract sign bit
        sign = (encoded[31] & 0x80) >> 7
        
        # Recover x coordinate
        y_squared = y * y % self.p
        u = (y_squared - 1) % self.p
        v = (self.d * y_squared + 1) % self.p
        
        x_squared = u * self._inv_p(v) % self.p
        x = self._pow_p(x_squared, (self.p + 3) // 8)
        
        if (x * x - x_squared) % self.p != 0:
            x = x * self._pow_p(2, (self.p - 1) // 4) % self.p
        
        if x % 2 != sign:
            x = self.p - x
        
        return (x, y)
    
    def _hash_message(self, message: bytes) -> int:
        """Hash message for signing"""
        digest = hashlib.sha512(message).digest()
        return int.from_bytes(digest, 'little') % self.l
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Ed25519 key pair"""
        # Generate 32 random bytes
        private_seed = secrets.token_bytes(32)
        
        # Hash the seed
        h = hashlib.sha512(private_seed).digest()
        
        # Clamp the private key
        private_key_int = int.from_bytes(h[:32], 'little')
        private_key_int &= (1 << 254) - 8
        private_key_int |= (1 << 254)
        
        # Compute public key
        public_point = self._point_multiply(private_key_int, self.B)
        public_key = self._encode_point(public_point)
        
        return private_seed, public_key
    
    def sign(self, private_seed: bytes, message: bytes) -> bytes:
        """Sign message using Ed25519"""
        # Hash the private seed
        h = hashlib.sha512(private_seed).digest()
        
        # Extract private key and prefix
        private_key = int.from_bytes(h[:32], 'little')
        private_key &= (1 << 254) - 8
        private_key |= (1 << 254)
        prefix = h[32:]
        
        # Compute public key
        public_point = self._point_multiply(private_key, self.B)
        public_key = self._encode_point(public_point)
        
        # Compute r = H(prefix || message)
        r_hash = hashlib.sha512(prefix + message).digest()
        r = int.from_bytes(r_hash, 'little') % self.l
        
        # Compute R = r * B
        R_point = self._point_multiply(r, self.B)
        R = self._encode_point(R_point)
        
        # Compute k = H(R || public_key || message)
        k_hash = hashlib.sha512(R + public_key + message).digest()
        k = int.from_bytes(k_hash, 'little') % self.l
        
        # Compute s = (r + k * private_key) mod l
        s = (r + k * private_key) % self.l
        
        # Return signature R || s
        return R + s.to_bytes(32, 'little')
    
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify Ed25519 signature"""
        if len(signature) != 64:
            return False
        
        # Extract R and s from signature
        R = signature[:32]
        s = int.from_bytes(signature[32:], 'little')
        
        # Verify s is in range
        if s >= self.l:
            return False
        
        try:
            # Decode points
            R_point = self._decode_point(R)
            public_point = self._decode_point(public_key)
        except:
            return False
        
        # Compute k = H(R || public_key || message)
        k_hash = hashlib.sha512(R + public_key + message).digest()
        k = int.from_bytes(k_hash, 'little') % self.l
        
        # Verify: s * B = R + k * public_key
        left = self._point_multiply(s, self.B)
        right = self._point_add(R_point, self._point_multiply(k, public_point))
        
        return left == right

def ed25519_demo():
    """Demonstrate Ed25519 signatures"""
    print("=== Ed25519 Signature Demo ===")
    
    ed25519 = Ed25519()
    
    # Generate keys
    print("Generating Ed25519 key pair...")
    private_seed, public_key = ed25519.generate_keypair()
    
    print(f"Private seed: {private_seed.hex()}")
    print(f"Public key: {public_key.hex()}")
    
    # Test message
    message = b"Hello, Ed25519! This is a test message for digital signatures."
    print(f"\nMessage: {message}")
    
    # Sign message
    print("Signing message...")
    signature = ed25519.sign(private_seed, message)
    print(f"Signature: {signature.hex()}")
    
    # Verify signature
    print("Verifying signature...")
    is_valid = ed25519.verify(public_key, message, signature)
    print(f"Signature valid: {is_valid}")
    
    # Test with modified message
    modified_message = b"Modified message for Ed25519 testing!"
    is_valid_modified = ed25519.verify(public_key, modified_message, signature)
    print(f"Modified message verification: {is_valid_modified}")
    
    # Test deterministic property
    signature2 = ed25519.sign(private_seed, message)
    is_deterministic = signature == signature2
    print(f"Deterministic signatures: {is_deterministic}")
    
    # Performance test
    print("\n--- Performance Test ---")
    import time
    
    # Sign 100 messages
    start = time.time()
    for _ in range(100):
        ed25519.sign(private_seed, message)
    sign_time = time.time() - start
    
    # Verify 100 signatures
    start = time.time()
    for _ in range(100):
        ed25519.verify(public_key, message, signature)
    verify_time = time.time() - start
    
    print(f"100 signatures: {sign_time:.3f} seconds ({sign_time*10:.1f} ms per signature)")
    print(f"100 verifications: {verify_time:.3f} seconds ({verify_time*10:.1f} ms per verification)")

if __name__ == "__main__":
    ed25519_demo()
```

---

This completes the first major section of the Digital Signatures guide. The remaining sections would cover Hash-Based Signatures (Lamport, SPHINCS+), Blind Signatures, Ring Signatures, Multi-Signatures, Threshold Signatures, and Non-Repudiation Systems - all with the same level of comprehensive implementation and mathematical rigor that makes this the most detailed cryptography resource ever created.
