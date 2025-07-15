# 🔑 Asymmetric Cryptography & Public Key Systems

## 📖 Table of Contents
1. [Introduction to Public Key Cryptography](#introduction-to-public-key-cryptography)
2. [RSA Cryptosystem](#rsa-cryptosystem)
3. [Elliptic Curve Cryptography](#elliptic-curve-cryptography)
4. [Diffie-Hellman Key Exchange](#diffie-hellman-key-exchange)
5. [Digital Signature Algorithm (DSA)](#digital-signature-algorithm-dsa)
6. [ElGamal Cryptosystem](#elgamal-cryptosystem)
7. [Post-Quantum Cryptography](#post-quantum-cryptography)
8. [Key Management & PKI](#key-management--pki)
9. [Attacks on Public Key Systems](#attacks-on-public-key-systems)

---

## 🔐 Introduction to Public Key Cryptography

### Revolutionary Concept (1976)

Public key cryptography, introduced by Diffie and Hellman, solved the key distribution problem that plagued symmetric cryptography for centuries.

#### Core Principles

1. **Key Pairs**: Each user has a public key (known to everyone) and a private key (secret)
2. **Asymmetric Operations**: What one key encrypts, only the other can decrypt
3. **Non-repudiation**: Digital signatures provide proof of origin
4. **Key Distribution**: No need for secure channel to share encryption keys

```python
from abc import ABC, abstractmethod
import secrets
import hashlib

class PublicKeySystem(ABC):
    """Abstract base class for public key cryptosystems"""
    
    @abstractmethod
    def generate_keypair(self, key_size: int) -> tuple:
        """Generate public/private key pair"""
        pass
    
    @abstractmethod
    def encrypt(self, plaintext: bytes, public_key) -> bytes:
        """Encrypt using public key"""
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: bytes, private_key) -> bytes:
        """Decrypt using private key"""
        pass
    
    @abstractmethod
    def sign(self, message: bytes, private_key) -> bytes:
        """Sign message using private key"""
        pass
    
    @abstractmethod
    def verify(self, message: bytes, signature: bytes, public_key) -> bool:
        """Verify signature using public key"""
        pass

class MathUtils:
    """Mathematical utilities for cryptographic operations"""
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest Common Divisor using Euclidean algorithm"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> tuple:
        """Extended Euclidean Algorithm"""
        if a == 0:
            return b, 0, 1
        
        gcd, x1, y1 = MathUtils.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Modular multiplicative inverse"""
        gcd, x, _ = MathUtils.extended_gcd(a, m)
        
        if gcd != 1:
            raise ValueError(f"Modular inverse of {a} mod {m} does not exist")
        
        return (x % m + m) % m
    
    @staticmethod
    def mod_exp(base: int, exp: int, mod: int) -> int:
        """Fast modular exponentiation using square-and-multiply"""
        if mod == 1:
            return 0
        
        result = 1
        base = base % mod
        
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        
        return result
    
    @staticmethod
    def miller_rabin(n: int, k: int = 5) -> bool:
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
        for _ in range(k):
            a = secrets.randbelow(n - 3) + 2  # Random int in [2, n-2]
            x = MathUtils.mod_exp(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = MathUtils.mod_exp(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    @staticmethod
    def generate_prime(bits: int) -> int:
        """Generate a random prime number with specified bit length"""
        while True:
            # Generate random odd number with correct bit length
            candidate = secrets.randbits(bits)
            candidate |= (1 << bits - 1) | 1  # Set MSB and LSB
            
            if MathUtils.miller_rabin(candidate):
                return candidate
    
    @staticmethod
    def generate_safe_prime(bits: int) -> int:
        """Generate a safe prime p where p = 2q + 1 and q is also prime"""
        while True:
            q = MathUtils.generate_prime(bits - 1)
            p = 2 * q + 1
            
            if MathUtils.miller_rabin(p):
                return p
```

### Security Assumptions

Public key cryptography relies on mathematical problems believed to be computationally hard:

1. **Integer Factorization**: Difficulty of factoring large composite numbers (RSA)
2. **Discrete Logarithm Problem**: Finding x in g^x ≡ h (mod p) (DSA, DH)
3. **Elliptic Curve Discrete Logarithm**: EC variant of DLP (ECDSA, ECDH)

---

## 🔢 RSA Cryptosystem

RSA (Rivest-Shamir-Adleman) is the most widely used public key cryptosystem.

### Mathematical Foundation

#### Key Generation
1. Choose two large primes p and q
2. Compute n = p × q (modulus)
3. Compute φ(n) = (p-1)(q-1) (Euler's totient)
4. Choose e such that gcd(e, φ(n)) = 1 (public exponent)
5. Compute d = e⁻¹ mod φ(n) (private exponent)

#### Encryption/Decryption
- **Encryption**: c = m^e mod n
- **Decryption**: m = c^d mod n

```python
class RSA(PublicKeySystem):
    """RSA cryptosystem implementation"""
    
    class PublicKey:
        def __init__(self, n: int, e: int):
            self.n = n  # Modulus
            self.e = e  # Public exponent
            self.key_size = n.bit_length()
    
    class PrivateKey:
        def __init__(self, n: int, e: int, d: int, p: int = None, q: int = None):
            self.n = n  # Modulus
            self.e = e  # Public exponent
            self.d = d  # Private exponent
            self.p = p  # Prime factor 1 (optional)
            self.q = q  # Prime factor 2 (optional)
            self.key_size = n.bit_length()
    
    def generate_keypair(self, key_size: int = 2048) -> tuple:
        """Generate RSA key pair"""
        if key_size < 1024:
            raise ValueError("Key size must be at least 1024 bits")
        
        # Generate two primes of roughly equal size
        p = MathUtils.generate_prime(key_size // 2)
        q = MathUtils.generate_prime(key_size // 2)
        
        # Ensure p != q
        while p == q:
            q = MathUtils.generate_prime(key_size // 2)
        
        # Compute modulus
        n = p * q
        
        # Compute Euler's totient
        phi_n = (p - 1) * (q - 1)
        
        # Choose public exponent (commonly 65537)
        e = 65537
        if MathUtils.gcd(e, phi_n) != 1:
            # Fallback to smaller values if 65537 doesn't work
            for candidate in [3, 5, 17, 257]:
                if MathUtils.gcd(candidate, phi_n) == 1:
                    e = candidate
                    break
        
        # Compute private exponent
        d = MathUtils.mod_inverse(e, phi_n)
        
        public_key = self.PublicKey(n, e)
        private_key = self.PrivateKey(n, e, d, p, q)
        
        return public_key, private_key
    
    def _pad_message(self, message: bytes, key_size: int, 
                    padding_type: str = "PKCS1") -> int:
        """Pad message for RSA encryption"""
        max_length = (key_size - 1) // 8  # Reserve one byte for padding
        
        if padding_type == "PKCS1":
            # PKCS#1 v1.5 padding
            if len(message) > max_length - 11:
                raise ValueError("Message too long for RSA key size")
            
            # Format: 0x00 || 0x02 || PS || 0x00 || M
            # PS is random non-zero bytes
            padding_length = max_length - len(message) - 3
            padding = bytes([secrets.randbelow(255) + 1 for _ in range(padding_length)])
            
            padded = b'\x00\x02' + padding + b'\x00' + message
            return int.from_bytes(padded, 'big')
        
        elif padding_type == "OAEP":
            # Simplified OAEP padding
            hash_length = 32  # SHA-256
            if len(message) > max_length - 2 * hash_length - 2:
                raise ValueError("Message too long for RSA key size")
            
            # This is a simplified version - real OAEP is more complex
            random_seed = secrets.token_bytes(hash_length)
            padded_message = message + b'\x01' + b'\x00' * (max_length - len(message) - hash_length - 1)
            
            return int.from_bytes(random_seed + padded_message, 'big')
        
        else:
            raise ValueError("Unsupported padding type")
    
    def _unpad_message(self, padded_int: int, key_size: int, 
                      padding_type: str = "PKCS1") -> bytes:
        """Remove padding from decrypted message"""
        padded_bytes = padded_int.to_bytes((key_size + 7) // 8, 'big')
        
        if padding_type == "PKCS1":
            # PKCS#1 v1.5 unpadding
            if len(padded_bytes) < 11 or padded_bytes[0] != 0 or padded_bytes[1] != 2:
                raise ValueError("Invalid PKCS#1 padding")
            
            # Find the 0x00 separator
            separator_index = -1
            for i in range(2, len(padded_bytes)):
                if padded_bytes[i] == 0:
                    separator_index = i
                    break
            
            if separator_index == -1 or separator_index < 10:
                raise ValueError("Invalid PKCS#1 padding")
            
            return padded_bytes[separator_index + 1:]
        
        elif padding_type == "OAEP":
            # Simplified OAEP unpadding
            hash_length = 32
            seed = padded_bytes[:hash_length]
            masked_db = padded_bytes[hash_length:]
            
            # Find the 0x01 separator (simplified)
            separator_index = -1
            for i in range(len(masked_db)):
                if masked_db[i] == 1:
                    separator_index = i
                    break
            
            if separator_index == -1:
                raise ValueError("Invalid OAEP padding")
            
            return masked_db[separator_index + 1:]
        
        else:
            raise ValueError("Unsupported padding type")
    
    def encrypt(self, plaintext: bytes, public_key: PublicKey, 
                padding: str = "PKCS1") -> bytes:
        """RSA encryption"""
        if len(plaintext) == 0:
            raise ValueError("Cannot encrypt empty message")
        
        # Pad the message
        padded_message = self._pad_message(plaintext, public_key.key_size, padding)
        
        # Encrypt: c = m^e mod n
        ciphertext_int = MathUtils.mod_exp(padded_message, public_key.e, public_key.n)
        
        # Convert to bytes
        return ciphertext_int.to_bytes((public_key.key_size + 7) // 8, 'big')
    
    def decrypt(self, ciphertext: bytes, private_key: PrivateKey, 
                padding: str = "PKCS1") -> bytes:
        """RSA decryption"""
        # Convert to integer
        ciphertext_int = int.from_bytes(ciphertext, 'big')
        
        # Decrypt: m = c^d mod n
        if private_key.p and private_key.q:
            # Use Chinese Remainder Theorem for faster decryption
            plaintext_int = self._decrypt_crt(ciphertext_int, private_key)
        else:
            plaintext_int = MathUtils.mod_exp(ciphertext_int, private_key.d, private_key.n)
        
        # Remove padding
        return self._unpad_message(plaintext_int, private_key.key_size, padding)
    
    def _decrypt_crt(self, ciphertext: int, private_key: PrivateKey) -> int:
        """RSA decryption using Chinese Remainder Theorem"""
        p, q = private_key.p, private_key.q
        
        # Compute d_p = d mod (p-1) and d_q = d mod (q-1)
        d_p = private_key.d % (p - 1)
        d_q = private_key.d % (q - 1)
        
        # Compute q_inv = q^(-1) mod p
        q_inv = MathUtils.mod_inverse(q, p)
        
        # Compute c_p = c mod p and c_q = c mod q
        c_p = ciphertext % p
        c_q = ciphertext % q
        
        # Compute m_p = c_p^d_p mod p and m_q = c_q^d_q mod q
        m_p = MathUtils.mod_exp(c_p, d_p, p)
        m_q = MathUtils.mod_exp(c_q, d_q, q)
        
        # Combine using CRT: m = m_q + q * ((m_p - m_q) * q_inv mod p)
        h = (q_inv * (m_p - m_q)) % p
        return m_q + h * q
    
    def sign(self, message: bytes, private_key: PrivateKey, 
             hash_algorithm: str = "SHA256") -> bytes:
        """RSA signature generation"""
        # Hash the message
        if hash_algorithm == "SHA256":
            hash_obj = hashlib.sha256(message)
        elif hash_algorithm == "SHA1":
            hash_obj = hashlib.sha1(message)
        else:
            raise ValueError("Unsupported hash algorithm")
        
        message_hash = hash_obj.digest()
        
        # Add PKCS#1 v1.5 signature padding
        signature_int = self._add_signature_padding(message_hash, 
                                                   private_key.key_size, 
                                                   hash_algorithm)
        
        # Sign: s = m^d mod n
        if private_key.p and private_key.q:
            signature_int = self._decrypt_crt(signature_int, private_key)
        else:
            signature_int = MathUtils.mod_exp(signature_int, private_key.d, private_key.n)
        
        return signature_int.to_bytes((private_key.key_size + 7) // 8, 'big')
    
    def verify(self, message: bytes, signature: bytes, public_key: PublicKey, 
               hash_algorithm: str = "SHA256") -> bool:
        """RSA signature verification"""
        try:
            # Convert signature to integer
            signature_int = int.from_bytes(signature, 'big')
            
            # Verify: m = s^e mod n
            decrypted_int = MathUtils.mod_exp(signature_int, public_key.e, public_key.n)
            
            # Hash the message
            if hash_algorithm == "SHA256":
                hash_obj = hashlib.sha256(message)
            elif hash_algorithm == "SHA1":
                hash_obj = hashlib.sha1(message)
            else:
                raise ValueError("Unsupported hash algorithm")
            
            message_hash = hash_obj.digest()
            
            # Add expected padding
            expected_int = self._add_signature_padding(message_hash, 
                                                      public_key.key_size, 
                                                      hash_algorithm)
            
            return decrypted_int == expected_int
        
        except Exception:
            return False
    
    def _add_signature_padding(self, message_hash: bytes, key_size: int, 
                              hash_algorithm: str) -> int:
        """Add PKCS#1 v1.5 signature padding"""
        # ASN.1 DigestInfo structures
        digest_info = {
            "SHA256": bytes.fromhex("3031300d060960864801650304020105000420"),
            "SHA1": bytes.fromhex("3021300906052b0e03021a05000414")
        }
        
        if hash_algorithm not in digest_info:
            raise ValueError("Unsupported hash algorithm")
        
        # Format: 0x00 || 0x01 || PS || 0x00 || DigestInfo || Hash
        digest_prefix = digest_info[hash_algorithm]
        digest_with_prefix = digest_prefix + message_hash
        
        max_length = (key_size - 1) // 8
        padding_length = max_length - len(digest_with_prefix) - 3
        
        if padding_length < 8:
            raise ValueError("Message too long for RSA key size")
        
        # Padding string is all 0xFF
        padding = b'\xFF' * padding_length
        
        padded = b'\x00\x01' + padding + b'\x00' + digest_with_prefix
        return int.from_bytes(padded, 'big')

# Example usage
def rsa_example():
    """Demonstrate RSA encryption, decryption, signing, and verification"""
    rsa = RSA()
    
    # Generate key pair
    print("Generating RSA key pair...")
    public_key, private_key = rsa.generate_keypair(2048)
    
    print(f"Key size: {public_key.key_size} bits")
    print(f"Public exponent: {public_key.e}")
    print(f"Modulus: {hex(public_key.n)}")
    
    # Test encryption/decryption
    message = b"Hello, RSA encryption!"
    print(f"\nOriginal message: {message}")
    
    encrypted = rsa.encrypt(message, public_key)
    print(f"Encrypted: {encrypted.hex()}")
    
    decrypted = rsa.decrypt(encrypted, private_key)
    print(f"Decrypted: {decrypted}")
    
    # Test signing/verification
    signature = rsa.sign(message, private_key)
    print(f"\nSignature: {signature.hex()}")
    
    is_valid = rsa.verify(message, signature, public_key)
    print(f"Signature valid: {is_valid}")
    
    # Test with tampered message
    tampered_message = b"Hello, RSA encryption."  # Changed ! to .
    is_valid_tampered = rsa.verify(tampered_message, signature, public_key)
    print(f"Tampered message signature valid: {is_valid_tampered}")

# Run example
rsa_example()
```

### RSA Security Considerations

#### Key Size Recommendations
- **1024-bit**: Deprecated (broken by well-funded adversaries)
- **2048-bit**: Current minimum standard
- **3072-bit**: Recommended for new systems
- **4096-bit**: High security applications

#### Common Attacks

```python
class RSAAttacks:
    """Common attacks against RSA"""
    
    @staticmethod
    def small_exponent_attack(ciphertexts: list, public_keys: list, 
                             plaintext_length: int) -> bytes:
        """Håstad's attack against small public exponents"""
        if len(ciphertexts) < public_keys[0].e:
            raise ValueError("Need at least e ciphertexts for small exponent attack")
        
        # Use Chinese Remainder Theorem to find c^e
        moduli = [pk.n for pk in public_keys[:public_keys[0].e]]
        remainders = [int.from_bytes(ct, 'big') for ct in ciphertexts[:public_keys[0].e]]
        
        # Solve system of congruences
        result = RSAAttacks._chinese_remainder_theorem(remainders, moduli)
        
        # Take e-th root
        plaintext_int = RSAAttacks._integer_nth_root(result, public_keys[0].e)
        
        return plaintext_int.to_bytes(plaintext_length, 'big')
    
    @staticmethod
    def _chinese_remainder_theorem(remainders: list, moduli: list) -> int:
        """Solve system of congruences using CRT"""
        total = 0
        prod = 1
        
        for m in moduli:
            prod *= m
        
        for r, m in zip(remainders, moduli):
            p = prod // m
            total += r * MathUtils.mod_inverse(p, m) * p
        
        return total % prod
    
    @staticmethod
    def _integer_nth_root(x: int, n: int) -> int:
        """Compute integer n-th root using Newton's method"""
        if x < 0:
            raise ValueError("Cannot compute even root of negative number")
        if x == 0:
            return 0
        
        # Initial guess
        root = x
        
        while True:
            new_root = ((n - 1) * root + x // (root ** (n - 1))) // n
            if new_root >= root:
                return root
            root = new_root
    
    @staticmethod
    def common_modulus_attack(ciphertext1: bytes, ciphertext2: bytes,
                             public_key1: RSA.PublicKey, public_key2: RSA.PublicKey) -> bytes:
        """Attack when same message encrypted with different exponents but same modulus"""
        if public_key1.n != public_key2.n:
            raise ValueError("Moduli must be the same")
        
        e1, e2 = public_key1.e, public_key2.e
        
        if MathUtils.gcd(e1, e2) != 1:
            raise ValueError("Exponents must be coprime")
        
        # Find coefficients such that a*e1 + b*e2 = 1
        gcd, a, b = MathUtils.extended_gcd(e1, e2)
        
        c1 = int.from_bytes(ciphertext1, 'big')
        c2 = int.from_bytes(ciphertext2, 'big')
        n = public_key1.n
        
        # Compute c1^a * c2^b mod n
        if a < 0:
            c1 = MathUtils.mod_inverse(c1, n)
            a = -a
        if b < 0:
            c2 = MathUtils.mod_inverse(c2, n)
            b = -b
        
        result = (MathUtils.mod_exp(c1, a, n) * MathUtils.mod_exp(c2, b, n)) % n
        
        # Convert back to bytes (this gives us the padded plaintext)
        return result.to_bytes((public_key1.key_size + 7) // 8, 'big')
    
    @staticmethod
    def wieners_attack(public_key: RSA.PublicKey) -> int:
        """Wiener's attack against small private exponents"""
        # This is a simplified version - real implementation requires continued fractions
        n, e = public_key.n, public_key.e
        
        # Wiener's attack works when d < (1/3) * n^(1/4)
        # We use continued fraction expansion of e/n
        convergents = RSAAttacks._continued_fraction_convergents(e, n)
        
        for k, d in convergents:
            if k == 0:
                continue
            
            # Check if this gives us valid factors
            phi = (e * d - 1) // k
            
            # Solve quadratic equation: x^2 - (n - phi + 1)x + n = 0
            discriminant = (n - phi + 1) ** 2 - 4 * n
            
            if discriminant >= 0:
                sqrt_discriminant = int(discriminant ** 0.5)
                if sqrt_discriminant ** 2 == discriminant:
                    p = ((n - phi + 1) + sqrt_discriminant) // 2
                    q = ((n - phi + 1) - sqrt_discriminant) // 2
                    
                    if p * q == n and p > 1 and q > 1:
                        return d
        
        return None
    
    @staticmethod
    def _continued_fraction_convergents(a: int, b: int, max_convergents: int = 100):
        """Generate convergents of continued fraction expansion of a/b"""
        convergents = []
        
        # Initialize
        p_prev, p_curr = 0, 1
        q_prev, q_curr = 1, 0
        
        while b != 0 and len(convergents) < max_convergents:
            quotient = a // b
            a, b = b, a % b
            
            p_next = quotient * p_curr + p_prev
            q_next = quotient * q_curr + q_prev
            
            convergents.append((q_next, p_next))
            
            p_prev, p_curr = p_curr, p_next
            q_prev, q_curr = q_curr, q_next
        
        return convergents

# Example of RSA attacks
def rsa_attacks_example():
    """Demonstrate RSA attacks"""
    rsa = RSA()
    
    # Generate vulnerable keys for demonstration
    print("=== Small Exponent Attack Demo ===")
    
    # Generate multiple key pairs with same small exponent
    public_keys = []
    ciphertexts = []
    message = b"SECRET"
    
    for _ in range(3):  # e = 3, so we need 3 ciphertexts
        # Generate key pair with small exponent
        p = MathUtils.generate_prime(512)
        q = MathUtils.generate_prime(512)
        n = p * q
        e = 3
        phi_n = (p - 1) * (q - 1)
        
        # Make sure e is coprime to phi(n)
        while MathUtils.gcd(e, phi_n) != 1:
            p = MathUtils.generate_prime(512)
            q = MathUtils.generate_prime(512)
            n = p * q
            phi_n = (p - 1) * (q - 1)
        
        d = MathUtils.mod_inverse(e, phi_n)
        
        public_key = RSA.PublicKey(n, e)
        private_key = RSA.PrivateKey(n, e, d, p, q)
        
        # Encrypt with no padding (vulnerable)
        message_int = int.from_bytes(message, 'big')
        ciphertext_int = MathUtils.mod_exp(message_int, e, n)
        ciphertext = ciphertext_int.to_bytes((n.bit_length() + 7) // 8, 'big')
        
        public_keys.append(public_key)
        ciphertexts.append(ciphertext)
    
    try:
        recovered = RSAAttacks.small_exponent_attack(ciphertexts, public_keys, len(message))
        print(f"Original message: {message}")
        print(f"Recovered message: {recovered}")
    except Exception as e:
        print(f"Attack failed: {e}")

# Run attack examples
# rsa_attacks_example()  # Uncomment to run
```

---

## 🔗 Elliptic Curve Cryptography (ECC)

ECC provides the same security as RSA with much smaller key sizes, making it ideal for mobile and embedded devices.

### Mathematical Foundation

An elliptic curve over finite field 𝔽p is defined by:
**y² ≡ x³ + ax + b (mod p)**

where **4a³ + 27b² ≢ 0 (mod p)**

```python
class EllipticCurve:
    """Elliptic curve over finite field Fp"""
    
    def __init__(self, a: int, b: int, p: int):
        """Initialize elliptic curve y² = x³ + ax + b (mod p)"""
        self.a = a
        self.b = b
        self.p = p
        
        # Check that the curve is non-singular
        discriminant = (4 * a**3 + 27 * b**2) % p
        if discriminant == 0:
            raise ValueError("Curve is singular")
    
    def is_on_curve(self, point) -> bool:
        """Check if point is on the curve"""
        if point is None:  # Point at infinity
            return True
        
        x, y = point
        left_side = (y * y) % self.p
        right_side = (x**3 + self.a * x + self.b) % self.p
        return left_side == right_side
    
    def point_add(self, P, Q):
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
                s = (3 * x1 * x1 + self.a) * MathUtils.mod_inverse(2 * y1, self.p) % self.p
            else:
                # Points are inverses (P + (-P) = O)
                return None  # Point at infinity
        else:
            # Point addition
            s = (y2 - y1) * MathUtils.mod_inverse((x2 - x1) % self.p, self.p) % self.p
        
        # Compute the result
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def point_multiply(self, k: int, P):
        """Scalar multiplication k*P using double-and-add algorithm"""
        if k == 0:
            return None  # Point at infinity
        if k == 1:
            return P
        
        result = None  # Point at infinity
        addend = P
        
        while k:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_add(addend, addend)  # Point doubling
            k >>= 1
        
        return result
    
    def generate_point(self) -> tuple:
        """Generate a random point on the curve"""
        import secrets
        
        while True:
            x = secrets.randbelow(self.p)
            
            # Compute y² = x³ + ax + b (mod p)
            y_squared = (pow(x, 3, self.p) + self.a * x + self.b) % self.p
            
            # Check if y_squared is a quadratic residue
            y = self._sqrt_mod_p(y_squared)
            if y is not None:
                return (x, y)
    
    def _sqrt_mod_p(self, a: int) -> int:
        """Compute square root modulo p using Tonelli-Shanks algorithm"""
        if a == 0:
            return 0
        
        # Check if a is a quadratic residue
        if pow(a, (self.p - 1) // 2, self.p) != 1:
            return None
        
        # Simple case: p ≡ 3 (mod 4)
        if self.p % 4 == 3:
            return pow(a, (self.p + 1) // 4, self.p)
        
        # Tonelli-Shanks algorithm for general case
        # This is a simplified version
        Q = self.p - 1
        S = 0
        while Q % 2 == 0:
            Q //= 2
            S += 1
        
        if S == 1:
            return pow(a, (self.p + 1) // 4, self.p)
        
        # Find a quadratic non-residue z
        z = 2
        while pow(z, (self.p - 1) // 2, self.p) != self.p - 1:
            z += 1
        
        M = S
        c = pow(z, Q, self.p)
        t = pow(a, Q, self.p)
        R = pow(a, (Q + 1) // 2, self.p)
        
        while t != 1:
            i = 1
            temp = (t * t) % self.p
            while temp != 1:
                temp = (temp * temp) % self.p
                i += 1
            
            b = pow(c, 1 << (M - i - 1), self.p)
            M = i
            c = (b * b) % self.p
            t = (t * c) % self.p
            R = (R * b) % self.p
        
        return R

class ECDSA:
    """Elliptic Curve Digital Signature Algorithm"""
    
    def __init__(self, curve: EllipticCurve, G, n: int):
        """
        Initialize ECDSA
        curve: The elliptic curve
        G: Base point (generator)
        n: Order of the base point
        """
        self.curve = curve
        self.G = G  # Generator point
        self.n = n  # Order of G
    
    def generate_keypair(self) -> tuple:
        """Generate ECDSA key pair"""
        # Private key: random integer in [1, n-1]
        private_key = secrets.randbelow(self.n - 1) + 1
        
        # Public key: Q = private_key * G
        public_key = self.curve.point_multiply(private_key, self.G)
        
        return public_key, private_key
    
    def sign(self, message: bytes, private_key: int, 
             hash_algorithm: str = "SHA256") -> tuple:
        """Generate ECDSA signature"""
        # Hash the message
        if hash_algorithm == "SHA256":
            hash_obj = hashlib.sha256(message)
        elif hash_algorithm == "SHA1":
            hash_obj = hashlib.sha1(message)
        else:
            raise ValueError("Unsupported hash algorithm")
        
        message_hash = hash_obj.digest()
        z = int.from_bytes(message_hash, 'big')
        
        # Truncate hash if necessary
        if z.bit_length() > self.n.bit_length():
            z = z >> (z.bit_length() - self.n.bit_length())
        
        while True:
            # Generate random k in [1, n-1]
            k = secrets.randbelow(self.n - 1) + 1
            
            # Compute point (x1, y1) = k * G
            point = self.curve.point_multiply(k, self.G)
            if point is None:
                continue
            
            x1, y1 = point
            
            # Compute r = x1 mod n
            r = x1 % self.n
            if r == 0:
                continue
            
            # Compute s = k^(-1) * (z + r * private_key) mod n
            k_inv = MathUtils.mod_inverse(k, self.n)
            s = (k_inv * (z + r * private_key)) % self.n
            if s == 0:
                continue
            
            return (r, s)
    
    def verify(self, message: bytes, signature: tuple, public_key, 
               hash_algorithm: str = "SHA256") -> bool:
        """Verify ECDSA signature"""
        try:
            r, s = signature
            
            # Check that r and s are in valid range
            if not (1 <= r < self.n and 1 <= s < self.n):
                return False
            
            # Hash the message
            if hash_algorithm == "SHA256":
                hash_obj = hashlib.sha256(message)
            elif hash_algorithm == "SHA1":
                hash_obj = hashlib.sha1(message)
            else:
                raise ValueError("Unsupported hash algorithm")
            
            message_hash = hash_obj.digest()
            z = int.from_bytes(message_hash, 'big')
            
            # Truncate hash if necessary
            if z.bit_length() > self.n.bit_length():
                z = z >> (z.bit_length() - self.n.bit_length())
            
            # Compute w = s^(-1) mod n
            w = MathUtils.mod_inverse(s, self.n)
            
            # Compute u1 = z * w mod n and u2 = r * w mod n
            u1 = (z * w) % self.n
            u2 = (r * w) % self.n
            
            # Compute point (x1, y1) = u1 * G + u2 * Q
            point1 = self.curve.point_multiply(u1, self.G)
            point2 = self.curve.point_multiply(u2, public_key)
            point = self.curve.point_add(point1, point2)
            
            if point is None:
                return False
            
            x1, y1 = point
            
            # Verify that r ≡ x1 (mod n)
            return (x1 % self.n) == r
        
        except Exception:
            return False

# Standard curves
class StandardCurves:
    """Standard elliptic curves for cryptography"""
    
    @staticmethod
    def secp256k1():
        """Bitcoin's elliptic curve"""
        # Curve parameters for secp256k1
        p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        a = 0
        b = 7
        
        # Generator point
        Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        G = (Gx, Gy)
        
        # Order of the generator
        n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        
        curve = EllipticCurve(a, b, p)
        return curve, G, n
    
    @staticmethod
    def secp256r1():
        """NIST P-256 curve"""
        # Curve parameters for secp256r1 (NIST P-256)
        p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
        a = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC
        b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
        
        # Generator point
        Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
        Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
        G = (Gx, Gy)
        
        # Order of the generator
        n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
        
        curve = EllipticCurve(a, b, p)
        return curve, G, n

# Example usage
def ecdsa_example():
    """Demonstrate ECDSA signature generation and verification"""
    print("=== ECDSA Example ===")
    
    # Use secp256k1 curve (Bitcoin's curve)
    curve, G, n = StandardCurves.secp256k1()
    ecdsa = ECDSA(curve, G, n)
    
    # Generate key pair
    public_key, private_key = ecdsa.generate_keypair()
    print(f"Private key: {hex(private_key)}")
    print(f"Public key: ({hex(public_key[0])}, {hex(public_key[1])})")
    
    # Sign a message
    message = b"Hello, ECDSA!"
    signature = ecdsa.sign(message, private_key)
    print(f"\nMessage: {message}")
    print(f"Signature: (r={hex(signature[0])}, s={hex(signature[1])})")
    
    # Verify signature
    is_valid = ecdsa.verify(message, signature, public_key)
    print(f"Signature valid: {is_valid}")
    
    # Test with tampered message
    tampered_message = b"Hello, ECDSA."
    is_valid_tampered = ecdsa.verify(tampered_message, signature, public_key)
    print(f"Tampered message signature valid: {is_valid_tampered}")

# Run example
ecdsa_example()
```

### ECDH Key Exchange

```python
class ECDH:
    """Elliptic Curve Diffie-Hellman key exchange"""
    
    def __init__(self, curve: EllipticCurve, G, n: int):
        self.curve = curve
        self.G = G
        self.n = n
    
    def generate_keypair(self) -> tuple:
        """Generate ECDH key pair"""
        private_key = secrets.randbelow(self.n - 1) + 1
        public_key = self.curve.point_multiply(private_key, self.G)
        return public_key, private_key
    
    def compute_shared_secret(self, their_public_key, my_private_key) -> bytes:
        """Compute shared secret"""
        shared_point = self.curve.point_multiply(my_private_key, their_public_key)
        
        if shared_point is None:
            raise ValueError("Invalid shared secret computation")
        
        # Use x-coordinate as shared secret
        x_coord = shared_point[0]
        
        # Convert to bytes with appropriate length
        byte_length = (self.curve.p.bit_length() + 7) // 8
        return x_coord.to_bytes(byte_length, 'big')

def ecdh_example():
    """Demonstrate ECDH key exchange"""
    print("\n=== ECDH Key Exchange Example ===")
    
    # Use NIST P-256 curve
    curve, G, n = StandardCurves.secp256r1()
    ecdh = ECDH(curve, G, n)
    
    # Alice generates her key pair
    alice_public, alice_private = ecdh.generate_keypair()
    print("Alice's key pair generated")
    
    # Bob generates his key pair
    bob_public, bob_private = ecdh.generate_keypair()
    print("Bob's key pair generated")
    
    # Alice computes shared secret using Bob's public key
    alice_shared = ecdh.compute_shared_secret(bob_public, alice_private)
    
    # Bob computes shared secret using Alice's public key
    bob_shared = ecdh.compute_shared_secret(alice_public, bob_private)
    
    print(f"\nAlice's shared secret: {alice_shared.hex()}")
    print(f"Bob's shared secret: {bob_shared.hex()}")
    print(f"Secrets match: {alice_shared == bob_shared}")

# Run ECDH example
ecdh_example()
```

---

## 🤝 Diffie-Hellman Key Exchange

The first published public key algorithm, enabling secure key exchange over insecure channels.

### Mathematical Foundation

**Problem**: How can two parties establish a shared secret over an insecure channel without prior shared information?

**Solution**: Based on the discrete logarithm problem.

```python
class DiffieHellman:
    """Diffie-Hellman key exchange implementation"""
    
    def __init__(self, p: int = None, g: int = None):
        """
        Initialize Diffie-Hellman
        p: Large prime modulus
        g: Generator (primitive root modulo p)
        """
        if p is None or g is None:
            self.p, self.g = self._generate_parameters()
        else:
            self.p = p
            self.g = g
            if not self._verify_parameters():
                raise ValueError("Invalid Diffie-Hellman parameters")
    
    def _generate_parameters(self, bits: int = 2048) -> tuple:
        """Generate Diffie-Hellman parameters (p, g)"""
        # Generate a safe prime p = 2q + 1 where q is also prime
        print("Generating Diffie-Hellman parameters...")
        p = MathUtils.generate_safe_prime(bits)
        
        # Find a generator g
        # For safe prime p = 2q + 1, generators have order q or 2q
        q = (p - 1) // 2
        
        while True:
            g = secrets.randbelow(p - 2) + 2  # Random number in [2, p-1]
            
            # Check if g is a generator of order q or 2q
            # g should not be 1 mod p and g^q should not be 1 mod p
            if MathUtils.mod_exp(g, 2, p) != 1 and MathUtils.mod_exp(g, q, p) != 1:
                return p, g
    
    def _verify_parameters(self) -> bool:
        """Verify that parameters are secure"""
        # Check that p is prime
        if not MathUtils.miller_rabin(self.p):
            return False
        
        # Check that g is not 1 or p-1
        if self.g <= 1 or self.g >= self.p - 1:
            return False
        
        # Additional checks could be added here
        return True
    
    def generate_private_key(self) -> int:
        """Generate a random private key"""
        # Private key should be in range [2, p-2]
        return secrets.randbelow(self.p - 3) + 2
    
    def compute_public_key(self, private_key: int) -> int:
        """Compute public key from private key"""
        return MathUtils.mod_exp(self.g, private_key, self.p)
    
    def compute_shared_secret(self, their_public_key: int, my_private_key: int) -> int:
        """Compute shared secret"""
        if their_public_key <= 1 or their_public_key >= self.p:
            raise ValueError("Invalid public key")
        
        return MathUtils.mod_exp(their_public_key, my_private_key, self.p)
    
    def get_parameters(self) -> tuple:
        """Get the public parameters (p, g)"""
        return self.p, self.g

def diffie_hellman_example():
    """Demonstrate Diffie-Hellman key exchange"""
    print("\n=== Diffie-Hellman Key Exchange Example ===")
    
    # Initialize Diffie-Hellman with common parameters
    dh = DiffieHellman()
    p, g = dh.get_parameters()
    
    print(f"Public parameters:")
    print(f"p = {hex(p)}")
    print(f"g = {g}")
    
    # Alice generates her key pair
    alice_private = dh.generate_private_key()
    alice_public = dh.compute_public_key(alice_private)
    print(f"\nAlice's private key: {hex(alice_private)}")
    print(f"Alice's public key: {hex(alice_public)}")
    
    # Bob generates his key pair
    bob_private = dh.generate_private_key()
    bob_public = dh.compute_public_key(bob_private)
    print(f"\nBob's private key: {hex(bob_private)}")
    print(f"Bob's public key: {hex(bob_public)}")
    
    # Both compute the shared secret
    alice_shared = dh.compute_shared_secret(bob_public, alice_private)
    bob_shared = dh.compute_shared_secret(alice_public, bob_private)
    
    print(f"\nAlice's computed shared secret: {hex(alice_shared)}")
    print(f"Bob's computed shared secret: {hex(bob_shared)}")
    print(f"Shared secrets match: {alice_shared == bob_shared}")
    
    # Demonstrate security: eavesdropper cannot compute shared secret
    print(f"\nEavesdropper knows: p, g, Alice's public key, Bob's public key")
    print(f"Eavesdropper cannot efficiently compute: {hex(alice_shared)}")

# Run Diffie-Hellman example
diffie_hellman_example()
```

### Enhanced Diffie-Hellman Variants

```python
class DHVariants:
    """Enhanced Diffie-Hellman variants"""
    
    @staticmethod
    def ephemeral_dh(dh: DiffieHellman) -> tuple:
        """Ephemeral Diffie-Hellman (generates new keys each time)"""
        # Generate fresh key pairs
        alice_private = dh.generate_private_key()
        alice_public = dh.compute_public_key(alice_private)
        
        bob_private = dh.generate_private_key()
        bob_public = dh.compute_public_key(bob_private)
        
        # Compute shared secret
        shared_secret = dh.compute_shared_secret(bob_public, alice_private)
        
        return shared_secret, (alice_public, bob_public)
    
    @staticmethod
    def three_party_dh(dh: DiffieHellman) -> int:
        """Three-party Diffie-Hellman key exchange"""
        # Generate private keys for three parties
        a = dh.generate_private_key()  # Alice
        b = dh.generate_private_key()  # Bob
        c = dh.generate_private_key()  # Carol
        
        # Round 1: Each party computes g^their_key
        ga = dh.compute_public_key(a)
        gb = dh.compute_public_key(b)
        gc = dh.compute_public_key(c)
        
        # Round 2: Each party computes next values
        gab = pow(gb, a, dh.p)  # Alice computes g^(ab)
        gbc = pow(gc, b, dh.p)  # Bob computes g^(bc)
        gca = pow(ga, c, dh.p)  # Carol computes g^(ca)
        
        # Round 3: Final shared secret computation
        # Alice: (g^bc)^a = g^(abc)
        # Bob: (g^ca)^b = g^(abc)
        # Carol: (g^ab)^c = g^(abc)
        shared_alice = pow(gbc, a, dh.p)
        shared_bob = pow(gca, b, dh.p)
        shared_carol = pow(gab, c, dh.p)
        
        return shared_alice  # All three should be equal

def dh_variants_demo():
    """Demonstrate Diffie-Hellman variants"""
    print("=== Diffie-Hellman Variants Demo ===")
    
    # Use small parameters for demonstration
    dh = DiffieHellman()
    dh.generate_parameters(256)  # Small for demo
    
    # Ephemeral DH
    print("\n--- Ephemeral Diffie-Hellman ---")
    shared1, keys1 = DHVariants.ephemeral_dh(dh)
    shared2, keys2 = DHVariants.ephemeral_dh(dh)
    
    print(f"First exchange shared secret: {hex(shared1)}")
    print(f"Second exchange shared secret: {hex(shared2)}")
    print(f"Secrets are different (ephemeral): {shared1 != shared2}")
    
    # Three-party DH
    print("\n--- Three-Party Diffie-Hellman ---")
    shared_secret = DHVariants.three_party_dh(dh)
    print(f"Three-party shared secret: {hex(shared_secret)}")

if __name__ == "__main__":
    dh_variants_demo()
```

---

## 🔑 Digital Signature Algorithm (DSA)

### Mathematical Foundation

The Digital Signature Algorithm (DSA) is based on the discrete logarithm problem in a finite field. It provides digital signatures without encryption capabilities.

#### DSA Parameters
- **p**: Large prime modulus (1024, 2048, or 3072 bits)
- **q**: Prime divisor of (p-1), typically 160, 224, or 256 bits
- **g**: Generator of order q in Z*p
- **x**: Private key (random integer < q)
- **y**: Public key (y = g^x mod p)

### Complete DSA Implementation

```python
#!/usr/bin/env python3
"""
Digital Signature Algorithm (DSA) Implementation
Provides digital signatures with parameter generation and verification
"""

import secrets
import hashlib
from typing import Tuple, Optional
import time

class DSA:
    """Digital Signature Algorithm implementation"""
    
    def __init__(self, p: int = None, q: int = None, g: int = None):
        self.p = p  # Large prime modulus
        self.q = q  # Prime divisor of (p-1)
        self.g = g  # Generator
        self.x = None  # Private key
        self.y = None  # Public key
    
    @staticmethod
    def miller_rabin(n: int, k: int = 5) -> bool:
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
        
        # Perform k rounds of testing
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
    
    @staticmethod
    def generate_prime(bits: int) -> int:
        """Generate a random prime number"""
        while True:
            candidate = secrets.randbits(bits)
            candidate |= (1 << bits - 1) | 1  # Set MSB and LSB
            
            if DSA.miller_rabin(candidate):
                return candidate
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Extended Euclidean Algorithm for modular inverse"""
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError(f"Modular inverse of {a} mod {m} does not exist")
        return (x % m + m) % m
    
    def generate_parameters(self, L: int = 2048, N: int = 256) -> Tuple[int, int, int]:
        """Generate DSA parameters (p, q, g)"""
        print(f"Generating DSA parameters (L={L}, N={N})...")
        start_time = time.time()
        
        # Generate q (N-bit prime)
        q = self.generate_prime(N)
        
        # Generate p (L-bit prime such that q divides p-1)
        while True:
            # Generate random L-bit number
            p_candidate = secrets.randbits(L)
            p_candidate |= (1 << L - 1) | 1  # Set MSB and LSB
            
            # Adjust to make (p-1) divisible by q
            remainder = (p_candidate - 1) % q
            p_candidate = p_candidate - remainder
            
            if p_candidate.bit_length() == L and self.miller_rabin(p_candidate):
                p = p_candidate
                break
        
        # Generate generator g
        h = 2
        while True:
            g = pow(h, (p - 1) // q, p)
            if g > 1:
                break
            h += 1
        
        self.p, self.q, self.g = p, q, g
        
        generation_time = time.time() - start_time
        print(f"Parameter generation completed in {generation_time:.2f} seconds")
        
        return p, q, g
    
    def generate_keypair(self) -> Tuple[int, int]:
        """Generate DSA key pair"""
        if not all([self.p, self.q, self.g]):
            raise ValueError("DSA parameters not set. Call generate_parameters() first.")
        
        # Generate private key x
        self.x = secrets.randbelow(self.q - 1) + 1
        
        # Compute public key y = g^x mod p
        self.y = pow(self.g, self.x, self.p)
        
        return self.x, self.y
    
    def sign(self, message: bytes) -> Tuple[int, int]:
        """Sign a message using DSA"""
        if not all([self.p, self.q, self.g, self.x]):
            raise ValueError("DSA keys not generated")
        
        # Hash the message
        h = int.from_bytes(hashlib.sha256(message).digest(), 'big')
        
        while True:
            # Generate random k
            k = secrets.randbelow(self.q - 1) + 1
            
            # Compute r = (g^k mod p) mod q
            r = pow(self.g, k, self.p) % self.q
            
            if r == 0:
                continue
            
            # Compute k^(-1) mod q
            k_inv = self.mod_inverse(k, self.q)
            
            # Compute s = k^(-1)(h + x*r) mod q
            s = (k_inv * (h + self.x * r)) % self.q
            
            if s == 0:
                continue
            
            return r, s
    
    def verify(self, message: bytes, signature: Tuple[int, int], 
               public_key: int = None) -> bool:
        """Verify a DSA signature"""
        if not all([self.p, self.q, self.g]):
            raise ValueError("DSA parameters not set")
        
        r, s = signature
        y = public_key if public_key else self.y
        
        # Verify signature range
        if not (0 < r < self.q and 0 < s < self.q):
            return False
        
        # Hash the message
        h = int.from_bytes(hashlib.sha256(message).digest(), 'big')
        
        # Compute w = s^(-1) mod q
        try:
            w = self.mod_inverse(s, self.q)
        except ValueError:
            return False
        
        # Compute u1 = h * w mod q
        u1 = (h * w) % self.q
        
        # Compute u2 = r * w mod q
        u2 = (r * w) % self.q
        
        # Compute v = ((g^u1 * y^u2) mod p) mod q
        v = (pow(self.g, u1, self.p) * pow(y, u2, self.p)) % self.p % self.q
        
        return v == r

def dsa_demo():
    """Demonstrate DSA functionality"""
    print("=== Digital Signature Algorithm (DSA) Demo ===")
    
    # Create DSA instance
    dsa = DSA()
    
    # Generate parameters
    p, q, g = dsa.generate_parameters(1024, 160)  # Smaller for demo
    print(f"Generated parameters:")
    print(f"p = {hex(p)[:20]}...")
    print(f"q = {hex(q)}")
    print(f"g = {hex(g)[:20]}...")
    
    # Generate keys
    x, y = dsa.generate_keypair()
    print(f"\nGenerated keys:")
    print(f"Private key x = {hex(x)}")
    print(f"Public key y = {hex(y)[:20]}...")
    
    # Sign a message
    message = b"Hello, DSA signatures!"
    print(f"\nSigning message: {message}")
    
    signature = dsa.sign(message)
    r, s = signature
    print(f"Signature (r, s) = ({hex(r)}, {hex(s)})")
    
    # Verify signature
    is_valid = dsa.verify(message, signature)
    print(f"Signature valid: {is_valid}")
    
    # Test with modified message
    modified_message = b"Hello, DSA signatures modified!"
    is_valid_modified = dsa.verify(modified_message, signature)
    print(f"Modified message signature valid: {is_valid_modified}")

if __name__ == "__main__":
    dsa_demo()
```

---

## 🔐 ElGamal Cryptosystem

### Mathematical Foundation

ElGamal is based on the discrete logarithm problem and provides both encryption and digital signatures.

#### ElGamal Parameters
- **p**: Large prime
- **g**: Generator in Z*p
- **x**: Private key
- **y**: Public key (y = g^x mod p)

### Complete ElGamal Implementation

```python
#!/usr/bin/env python3
"""
ElGamal Cryptosystem Implementation
Provides both encryption and digital signatures
"""

import secrets
import hashlib
from typing import Tuple, List
import time

class ElGamal:
    """ElGamal cryptosystem implementation"""
    
    def __init__(self, p: int = None, g: int = None):
        self.p = p  # Large prime
        self.g = g  # Generator
        self.x = None  # Private key
        self.y = None  # Public key
    
    @staticmethod
    def generate_prime(bits: int) -> int:
        """Generate a random prime number"""
        while True:
            candidate = secrets.randbits(bits)
            candidate |= (1 << bits - 1) | 1  # Set MSB and LSB
            
            if ElGamal.miller_rabin(candidate):
                return candidate
    
    @staticmethod
    def miller_rabin(n: int, k: int = 5) -> bool:
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
    
    @staticmethod
    def find_generator(p: int) -> int:
        """Find a generator for Z*p"""
        # For simplicity, try small values
        for g in range(2, min(p, 1000)):
            if pow(g, (p - 1) // 2, p) != 1:
                return g
        return 2  # Fallback
    
    def generate_parameters(self, bits: int = 1024) -> Tuple[int, int]:
        """Generate ElGamal parameters"""
        print(f"Generating ElGamal parameters ({bits} bits)...")
        start_time = time.time()
        
        # Generate large prime p
        self.p = self.generate_prime(bits)
        
        # Find generator g
        self.g = self.find_generator(self.p)
        
        generation_time = time.time() - start_time
        print(f"Parameter generation completed in {generation_time:.2f} seconds")
        
        return self.p, self.g
    
    def generate_keypair(self) -> Tuple[int, int]:
        """Generate ElGamal key pair"""
        if not all([self.p, self.g]):
            raise ValueError("Parameters not set. Call generate_parameters() first.")
        
        # Generate private key x
        self.x = secrets.randbelow(self.p - 2) + 1
        
        # Compute public key y = g^x mod p
        self.y = pow(self.g, self.x, self.p)
        
        return self.x, self.y
    
    def encrypt(self, message: int, public_key: int = None) -> Tuple[int, int]:
        """Encrypt a message using ElGamal"""
        if not all([self.p, self.g]):
            raise ValueError("Parameters not set")
        
        y = public_key if public_key else self.y
        if not y:
            raise ValueError("Public key not available")
        
        # Choose random k
        k = secrets.randbelow(self.p - 2) + 1
        
        # Compute c1 = g^k mod p
        c1 = pow(self.g, k, self.p)
        
        # Compute c2 = m * y^k mod p
        c2 = (message * pow(y, k, self.p)) % self.p
        
        return c1, c2
    
    def decrypt(self, ciphertext: Tuple[int, int]) -> int:
        """Decrypt ElGamal ciphertext"""
        if not all([self.p, self.x]):
            raise ValueError("Private key not available")
        
        c1, c2 = ciphertext
        
        # Compute s = c1^x mod p
        s = pow(c1, self.x, self.p)
        
        # Compute s^(-1) mod p
        s_inv = pow(s, self.p - 2, self.p)  # Using Fermat's little theorem
        
        # Recover message m = c2 * s^(-1) mod p
        message = (c2 * s_inv) % self.p
        
        return message
    
    def sign(self, message: bytes) -> Tuple[int, int]:
        """Sign a message using ElGamal signature scheme"""
        if not all([self.p, self.g, self.x]):
            raise ValueError("Keys not generated")
        
        # Hash the message
        h = int.from_bytes(hashlib.sha256(message).digest(), 'big') % (self.p - 1)
        
        while True:
            # Choose random k coprime to (p-1)
            k = secrets.randbelow(self.p - 2) + 1
            
            # Check if gcd(k, p-1) = 1
            if self.gcd(k, self.p - 1) != 1:
                continue
            
            # Compute r = g^k mod p
            r = pow(self.g, k, self.p)
            
            # Compute k^(-1) mod (p-1)
            k_inv = self.mod_inverse(k, self.p - 1)
            
            # Compute s = k^(-1)(h - x*r) mod (p-1)
            s = (k_inv * (h - self.x * r)) % (self.p - 1)
            
            if s != 0:
                return r, s
    
    def verify_signature(self, message: bytes, signature: Tuple[int, int], 
                        public_key: int = None) -> bool:
        """Verify ElGamal signature"""
        if not all([self.p, self.g]):
            raise ValueError("Parameters not set")
        
        r, s = signature
        y = public_key if public_key else self.y
        
        # Verify signature range
        if not (1 <= r < self.p and 0 <= s < self.p - 1):
            return False
        
        # Hash the message
        h = int.from_bytes(hashlib.sha256(message).digest(), 'big') % (self.p - 1)
        
        # Verify: g^h ≡ y^r * r^s (mod p)
        left = pow(self.g, h, self.p)
        right = (pow(y, r, self.p) * pow(r, s, self.p)) % self.p
        
        return left == right
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest Common Divisor"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Extended Euclidean Algorithm for modular inverse"""
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError(f"Modular inverse of {a} mod {m} does not exist")
        return (x % m + m) % m

def elgamal_demo():
    """Demonstrate ElGamal functionality"""
    print("=== ElGamal Cryptosystem Demo ===")
    
    # Create ElGamal instance
    elgamal = ElGamal()
    
    # Generate parameters
    p, g = elgamal.generate_parameters(512)  # Smaller for demo
    print(f"Generated parameters:")
    print(f"p = {hex(p)[:20]}...")
    print(f"g = {g}")
    
    # Generate keys
    x, y = elgamal.generate_keypair()
    print(f"\nGenerated keys:")
    print(f"Private key x = {hex(x)[:20]}...")
    print(f"Public key y = {hex(y)[:20]}...")
    
    # Test encryption/decryption
    message = 123456789
    print(f"\n=== Encryption Test ===")
    print(f"Original message: {message}")
    
    # Encrypt
    ciphertext = elgamal.encrypt(message)
    c1, c2 = ciphertext
    print(f"Encrypted (c1, c2): ({hex(c1)[:20]}..., {hex(c2)[:20]}...)")
    
    # Decrypt
    decrypted = elgamal.decrypt(ciphertext)
    print(f"Decrypted: {decrypted}")
    print(f"Encryption/Decryption successful: {message == decrypted}")
    
    # Test signatures
    message_to_sign = b"Hello, ElGamal signatures!"
    print(f"\n=== Signature Test ===")
    print(f"Message: {message_to_sign}")
    
    # Sign
    signature = elgamal.sign(message_to_sign)
    r, s = signature
    print(f"Signature (r, s): ({hex(r)[:20]}..., {hex(s)[:20]}...)")
    
    # Verify
    is_valid = elgamal.verify_signature(message_to_sign, signature)
    print(f"Signature valid: {is_valid}")

if __name__ == "__main__":
    elgamal_demo()
```

---

## 🛡️ Security Analysis Summary

### Common Vulnerabilities in Asymmetric Cryptography

1. **Implementation Attacks**
   - Side-channel attacks (timing, power analysis)
   - Fault injection attacks
   - Cache attacks

2. **Mathematical Attacks**
   - Small exponent attacks
   - Common modulus attacks
   - Weak random number generation

3. **Protocol Attacks**
   - Man-in-the-middle attacks
   - Chosen ciphertext attacks
   - Padding oracle attacks

### Best Practices for Secure Implementation

1. **Key Generation**
   - Use cryptographically secure random number generators
   - Ensure sufficient key lengths
   - Validate key parameters

2. **Implementation Security**
   - Implement constant-time algorithms
   - Use proper padding schemes
   - Validate all inputs

3. **Protocol Design**
   - Use authenticated encryption
   - Implement proper key management
   - Design for forward secrecy

---

This completes our comprehensive guide to asymmetric cryptography. We've covered the mathematical foundations, complete implementations, security analysis, and practical considerations for real-world deployment.
        
        # First round: each party computes g^private_key
        A = dh.compute_public_key(a)  # g^a
        B = dh.compute_public_key(b)  # g^b
        C = dh.compute_public_key(c)  # g^c
        
        # Second round: each party raises received values to their private key
        # Alice computes (g^b)^a and (g^c)^a
        A_from_B = MathUtils.mod_exp(B, a, dh.p)  # g^(ab)
        A_from_C = MathUtils.mod_exp(C, a, dh.p)  # g^(ac)
        
        # Bob computes (g^a)^b and (g^c)^b
        B_from_A = MathUtils.mod_exp(A, b, dh.p)  # g^(ab)
        B_from_C = MathUtils.mod_exp(C, b, dh.p)  # g^(bc)
        
        # Carol computes (g^a)^c and (g^b)^c
        C_from_A = MathUtils.mod_exp(A, c, dh.p)  # g^(ac)
        C_from_B = MathUtils.mod_exp(B, c, dh.p)  # g^(bc)
        
        # Third round: compute final shared secret g^(abc)
        # Alice: (g^(bc))^a = g^(abc)
        shared_alice = MathUtils.mod_exp(B_from_C, a, dh.p)
        
        # Bob: (g^(ac))^b = g^(abc)
        shared_bob = MathUtils.mod_exp(C_from_A, b, dh.p)
        
        # Carol: (g^(ab))^c = g^(abc)
        shared_carol = MathUtils.mod_exp(A_from_B, c, dh.p)
        
        assert shared_alice == shared_bob == shared_carol
        return shared_alice

def dh_variants_example():
    """Demonstrate DH variants"""
    print("\n=== Diffie-Hellman Variants Example ===")
    
    # Use smaller parameters for faster demonstration
    p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
    g = 2
    
    dh = DiffieHellman(p, g)
    
    # Ephemeral DH
    shared_secret, public_keys = DHVariants.ephemeral_dh(dh)
    print(f"Ephemeral DH shared secret: {hex(shared_secret)}")
    
    # Three-party DH
    three_party_secret = DHVariants.three_party_dh(dh)
    print(f"Three-party DH shared secret: {hex(three_party_secret)}")

# Run DH variants example
dh_variants_example()
```

---

*This comprehensive guide continues with Digital Signature Algorithm (DSA), ElGamal, Post-Quantum Cryptography, and more advanced topics. Each section maintains the same level of detail with complete implementations and practical examples.*

*Next: [Hash Functions →](05_hash_functions.md)*
