# 🔨 Comprehensive Cryptographic Tools & Analysis Suite

## 📖 Table of Contents
1. [Caesar Cipher Tools](#caesar-cipher-tools)
2. [Vigenère Cipher Analysis](#vigenère-cipher-analysis)
3. [RSA Implementation & Attacks](#rsa-implementation--attacks)
4. [AES Encryption Suite](#aes-encryption-suite)
5. [Hash Function Tools](#hash-function-tools)
6. [Random Number Generators](#random-number-generators)
7. [Frequency Analysis Tools](#frequency-analysis-tools)
8. [Modern Cipher Implementations](#modern-cipher-implementations)

---

## 🏛️ Caesar Cipher Tools

```python
#!/usr/bin/env python3
"""
Advanced Caesar Cipher Implementation & Analysis Tools
Includes brute force, frequency analysis, and statistical attacks
"""

import string
import collections
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class CaesarCipher:
    """Advanced Caesar cipher with multiple analysis methods"""
    
    def __init__(self):
        self.alphabet = string.ascii_uppercase
        self.english_freq = {
            'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97,
            'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
            'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
            'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
            'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
        }
    
    def encrypt(self, plaintext: str, key: int) -> str:
        """Encrypt text using Caesar cipher"""
        result = []
        key = key % 26  # Normalize key
        
        for char in plaintext.upper():
            if char in self.alphabet:
                shifted = (self.alphabet.index(char) + key) % 26
                result.append(self.alphabet[shifted])
            else:
                result.append(char)
        
        return ''.join(result)
    
    def decrypt(self, ciphertext: str, key: int) -> str:
        """Decrypt text using Caesar cipher"""
        return self.encrypt(ciphertext, -key)
    
    def brute_force_attack(self, ciphertext: str) -> Dict[int, str]:
        """Try all possible keys"""
        results = {}
        for key in range(26):
            results[key] = self.decrypt(ciphertext, key)
        return results
    
    def frequency_analysis_attack(self, ciphertext: str) -> Tuple[int, str, float]:
        """Attack using frequency analysis"""
        best_key = 0
        best_score = float('inf')
        best_text = ""
        
        for key in range(26):
            decrypted = self.decrypt(ciphertext, key)
            score = self._chi_squared_score(decrypted)
            
            if score < best_score:
                best_score = score
                best_key = key
                best_text = decrypted
        
        return best_key, best_text, best_score
    
    def _chi_squared_score(self, text: str) -> float:
        """Calculate chi-squared statistic against English frequencies"""
        text = ''.join([c for c in text.upper() if c.isalpha()])
        if not text:
            return float('inf')
        
        observed = collections.Counter(text)
        total_chars = len(text)
        
        chi_squared = 0
        for char in self.alphabet:
            observed_count = observed.get(char, 0)
            expected_count = self.english_freq.get(char, 0) * total_chars / 100
            
            if expected_count > 0:
                chi_squared += (observed_count - expected_count) ** 2 / expected_count
        
        return chi_squared
    
    def index_of_coincidence(self, text: str) -> float:
        """Calculate index of coincidence"""
        text = ''.join([c for c in text.upper() if c.isalpha()])
        n = len(text)
        
        if n <= 1:
            return 0
        
        frequencies = collections.Counter(text)
        ic = sum(f * (f - 1) for f in frequencies.values()) / (n * (n - 1))
        return ic
    
    def plot_frequency_analysis(self, text: str, title: str = "Frequency Analysis"):
        """Plot letter frequency distribution"""
        text = ''.join([c for c in text.upper() if c.isalpha()])
        frequencies = collections.Counter(text)
        
        # Calculate percentages
        total = sum(frequencies.values())
        percentages = {char: (frequencies.get(char, 0) / total) * 100 for char in self.alphabet}
        
        # Plot
        plt.figure(figsize=(12, 6))
        chars = list(self.alphabet)
        observed = [percentages[char] for char in chars]
        expected = [self.english_freq[char] for char in chars]
        
        x = range(len(chars))
        plt.bar([i - 0.2 for i in x], observed, 0.4, label='Observed', alpha=0.7)
        plt.bar([i + 0.2 for i in x], expected, 0.4, label='Expected English', alpha=0.7)
        
        plt.xlabel('Letters')
        plt.ylabel('Frequency (%)')
        plt.title(title)
        plt.xticks(x, chars)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def statistical_analysis(self, text: str) -> Dict[str, float]:
        """Comprehensive statistical analysis"""
        text_clean = ''.join([c for c in text.upper() if c.isalpha()])
        
        return {
            'length': len(text_clean),
            'unique_chars': len(set(text_clean)),
            'index_of_coincidence': self.index_of_coincidence(text),
            'chi_squared': self._chi_squared_score(text),
            'most_common_char': collections.Counter(text_clean).most_common(1)[0] if text_clean else ('', 0),
            'entropy': self._calculate_entropy(text_clean)
        }
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        if not text:
            return 0
        
        frequencies = collections.Counter(text)
        total = len(text)
        
        entropy = 0
        for count in frequencies.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * (probability).bit_length()
        
        return entropy

def caesar_demo():
    """Demonstrate Caesar cipher tools"""
    caesar = CaesarCipher()
    
    # Original plaintext
    plaintext = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    key = 13  # ROT13
    
    print("=== Caesar Cipher Demonstration ===")
    print(f"Original: {plaintext}")
    
    # Encrypt
    ciphertext = caesar.encrypt(plaintext, key)
    print(f"Encrypted (key={key}): {ciphertext}")
    
    # Decrypt
    decrypted = caesar.decrypt(ciphertext, key)
    print(f"Decrypted: {decrypted}")
    
    # Brute force attack
    print("\n=== Brute Force Attack ===")
    brute_results = caesar.brute_force_attack(ciphertext)
    for k, text in brute_results.items():
        print(f"Key {k:2d}: {text}")
    
    # Frequency analysis attack
    print("\n=== Frequency Analysis Attack ===")
    best_key, best_text, score = caesar.frequency_analysis_attack(ciphertext)
    print(f"Best key: {best_key}")
    print(f"Best decryption: {best_text}")
    print(f"Chi-squared score: {score:.2f}")
    
    # Statistical analysis
    print("\n=== Statistical Analysis ===")
    stats = caesar.statistical_analysis(ciphertext)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Plot frequency analysis
    # caesar.plot_frequency_analysis(plaintext, "Original Text Frequency")
    # caesar.plot_frequency_analysis(ciphertext, "Encrypted Text Frequency")

if __name__ == "__main__":
    caesar_demo()
```

---

## 🔄 Vigenère Cipher Analysis

```python
#!/usr/bin/env python3
"""
Advanced Vigenère Cipher Implementation & Cryptanalysis
Includes Kasiski examination, index of coincidence, and Friedman test
"""

import string
import math
import collections
from typing import List, Dict, Tuple, Optional
import itertools

class VigenereCipher:
    """Advanced Vigenère cipher with comprehensive cryptanalysis"""
    
    def __init__(self):
        self.alphabet = string.ascii_uppercase
        self.english_freq = {
            'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97,
            'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
            'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
            'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
            'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
        }
        self.english_ic = 0.067  # Expected IC for English
    
    def encrypt(self, plaintext: str, key: str) -> str:
        """Encrypt using Vigenère cipher"""
        result = []
        key = key.upper()
        key_index = 0
        
        for char in plaintext.upper():
            if char in self.alphabet:
                shift = self.alphabet.index(key[key_index % len(key)])
                encrypted_char = self.alphabet[(self.alphabet.index(char) + shift) % 26]
                result.append(encrypted_char)
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def decrypt(self, ciphertext: str, key: str) -> str:
        """Decrypt using Vigenère cipher"""
        result = []
        key = key.upper()
        key_index = 0
        
        for char in ciphertext.upper():
            if char in self.alphabet:
                shift = self.alphabet.index(key[key_index % len(key)])
                decrypted_char = self.alphabet[(self.alphabet.index(char) - shift) % 26]
                result.append(decrypted_char)
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def kasiski_examination(self, ciphertext: str, min_length: int = 3, 
                           max_length: int = 20) -> List[Tuple[int, int]]:
        """Perform Kasiski examination to find likely key lengths"""
        ciphertext = ''.join([c for c in ciphertext.upper() if c.isalpha()])
        
        # Find repeated sequences
        sequences = {}
        for length in range(min_length, min(max_length, len(ciphertext) // 3)):
            for i in range(len(ciphertext) - length + 1):
                sequence = ciphertext[i:i + length]
                if sequence in sequences:
                    sequences[sequence].append(i)
                else:
                    sequences[sequence] = [i]
        
        # Calculate distances and their factors
        factor_counts = collections.defaultdict(int)
        
        for sequence, positions in sequences.items():
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    distance = positions[i + 1] - positions[i]
                    
                    # Find all factors of the distance
                    for factor in range(2, distance + 1):
                        if distance % factor == 0:
                            factor_counts[factor] += 1
        
        # Sort by frequency
        likely_lengths = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        return likely_lengths[:10]
    
    def index_of_coincidence(self, text: str) -> float:
        """Calculate index of coincidence"""
        text = ''.join([c for c in text.upper() if c.isalpha()])
        n = len(text)
        
        if n <= 1:
            return 0
        
        frequencies = collections.Counter(text)
        ic = sum(f * (f - 1) for f in frequencies.values()) / (n * (n - 1))
        return ic
    
    def friedman_test(self, ciphertext: str, max_key_length: int = 20) -> List[Tuple[int, float]]:
        """Use Friedman test to estimate key length"""
        ciphertext = ''.join([c for c in ciphertext.upper() if c.isalpha()])
        results = []
        
        for key_length in range(1, max_key_length + 1):
            # Split text into groups based on key position
            groups = [''] * key_length
            for i, char in enumerate(ciphertext):
                groups[i % key_length] += char
            
            # Calculate average IC for all groups
            total_ic = sum(self.index_of_coincidence(group) for group in groups if group)
            avg_ic = total_ic / key_length if key_length > 0 else 0
            
            results.append((key_length, avg_ic))
        
        # Sort by IC closest to English IC
        results.sort(key=lambda x: abs(x[1] - self.english_ic))
        return results
    
    def chi_squared_test(self, text: str) -> float:
        """Calculate chi-squared statistic against English frequencies"""
        text = ''.join([c for c in text.upper() if c.isalpha()])
        if not text:
            return float('inf')
        
        observed = collections.Counter(text)
        total = len(text)
        
        chi_squared = 0
        for char in self.alphabet:
            observed_count = observed.get(char, 0)
            expected_count = self.english_freq.get(char, 0) * total / 100
            
            if expected_count > 0:
                chi_squared += (observed_count - expected_count) ** 2 / expected_count
        
        return chi_squared
    
    def find_key_character(self, ciphertext_group: str) -> str:
        """Find the most likely key character for a group using frequency analysis"""
        best_char = 'A'
        best_score = float('inf')
        
        for shift in range(26):
            # Decrypt the group with this shift
            decrypted = ''
            for char in ciphertext_group:
                if char in self.alphabet:
                    decrypted += self.alphabet[(self.alphabet.index(char) - shift) % 26]
            
            # Calculate fitness using chi-squared test
            score = self.chi_squared_test(decrypted)
            if score < best_score:
                best_score = score
                best_char = self.alphabet[shift]
        
        return best_char
    
    def cryptanalysis(self, ciphertext: str) -> Tuple[Optional[str], Optional[str], Dict[str, any]]:
        """Comprehensive Vigenère cryptanalysis"""
        ciphertext_clean = ''.join([c for c in ciphertext.upper() if c.isalpha()])
        
        if len(ciphertext_clean) < 50:
            return None, None, {"error": "Text too short for reliable analysis"}
        
        # Step 1: Determine key length using multiple methods
        kasiski_results = self.kasiski_examination(ciphertext)
        friedman_results = self.friedman_test(ciphertext)
        
        # Combine results to find most likely key length
        key_length_scores = collections.defaultdict(float)
        
        # Score from Kasiski examination
        for length, count in kasiski_results[:5]:
            key_length_scores[length] += count * 2
        
        # Score from Friedman test
        for length, ic in friedman_results[:5]:
            # Higher score for IC closer to English IC
            score = 1 / (1 + abs(ic - self.english_ic))
            key_length_scores[length] += score * 10
        
        # Find best key length
        if not key_length_scores:
            return None, None, {"error": "Could not determine key length"}
        
        best_key_length = max(key_length_scores.items(), key=lambda x: x[1])[0]
        
        # Step 2: Determine key characters
        key_chars = []
        groups = [''] * best_key_length
        
        # Split ciphertext into groups
        for i, char in enumerate(ciphertext_clean):
            groups[i % best_key_length] += char
        
        # Find key character for each group
        for group in groups:
            if group:
                key_char = self.find_key_character(group)
                key_chars.append(key_char)
            else:
                key_chars.append('A')  # Default
        
        recovered_key = ''.join(key_chars)
        
        # Step 3: Decrypt and verify
        decrypted_text = self.decrypt(ciphertext, recovered_key)
        
        # Calculate confidence metrics
        decrypted_clean = ''.join([c for c in decrypted_text.upper() if c.isalpha()])
        confidence_metrics = {
            'key_length': best_key_length,
            'recovered_key': recovered_key,
            'decrypted_ic': self.index_of_coincidence(decrypted_clean),
            'chi_squared': self.chi_squared_test(decrypted_clean),
            'kasiski_top_lengths': kasiski_results[:3],
            'friedman_top_lengths': friedman_results[:3]
        }
        
        return recovered_key, decrypted_text, confidence_metrics
    
    def mutual_index_of_coincidence(self, text1: str, text2: str) -> float:
        """Calculate mutual index of coincidence between two texts"""
        text1 = ''.join([c for c in text1.upper() if c.isalpha()])
        text2 = ''.join([c for c in text2.upper() if c.isalpha()])
        
        if not text1 or not text2:
            return 0
        
        freq1 = collections.Counter(text1)
        freq2 = collections.Counter(text2)
        
        mic = 0
        for char in self.alphabet:
            mic += freq1.get(char, 0) * freq2.get(char, 0)
        
        return mic / (len(text1) * len(text2))

def vigenere_demo():
    """Demonstrate Vigenère cipher cryptanalysis"""
    vigenere = VigenereCipher()
    
    # Test message
    plaintext = """THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. THIS IS A LONGER MESSAGE TO DEMONSTRATE 
    VIGENERE CIPHER CRYPTANALYSIS. THE ALGORITHM USES FREQUENCY ANALYSIS AND STATISTICAL METHODS 
    TO BREAK THE CIPHER WITHOUT KNOWING THE KEY."""
    
    key = "CRYPTOGRAPHY"
    
    print("=== Vigenère Cipher Cryptanalysis ===")
    print(f"Original key: {key}")
    print(f"Original text: {plaintext[:50]}...")
    
    # Encrypt
    ciphertext = vigenere.encrypt(plaintext, key)
    print(f"Encrypted: {ciphertext[:50]}...")
    
    # Perform cryptanalysis
    print("\n=== Cryptanalysis Results ===")
    recovered_key, decrypted_text, metrics = vigenere.cryptanalysis(ciphertext)
    
    if recovered_key:
        print(f"Recovered key: {recovered_key}")
        print(f"Original key:  {key}")
        print(f"Key correct: {recovered_key == key}")
        print(f"\nDecrypted text: {decrypted_text[:100]}...")
        
        print(f"\n=== Confidence Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    else:
        print("Cryptanalysis failed")
        print(f"Error: {metrics.get('error', 'Unknown error')}")

if __name__ == "__main__":
    vigenere_demo()
```

---

## 🔐 RSA Implementation & Attacks

```python
#!/usr/bin/env python3
"""
Comprehensive RSA Implementation with Security Analysis
Includes key generation, encryption/decryption, signatures, and common attacks
"""

import secrets
import hashlib
import math
from typing import Tuple, Optional, List
import time

class RSAMath:
    """Mathematical utilities for RSA"""
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest Common Divisor"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean Algorithm"""
        if a == 0:
            return b, 0, 1
        
        gcd, x1, y1 = RSAMath.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Modular multiplicative inverse"""
        gcd, x, _ = RSAMath.extended_gcd(a, m)
        
        if gcd != 1:
            raise ValueError(f"Modular inverse of {a} mod {m} does not exist")
        
        return (x % m + m) % m
    
    @staticmethod
    def mod_exp(base: int, exp: int, mod: int) -> int:
        """Fast modular exponentiation"""
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
            x = RSAMath.mod_exp(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = RSAMath.mod_exp(x, 2, n)
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
            
            if RSAMath.miller_rabin(candidate):
                return candidate
    
    @staticmethod
    def pollards_rho(n: int, max_iterations: int = 100000) -> Optional[int]:
        """Pollard's rho algorithm for integer factorization"""
        if n % 2 == 0:
            return 2
        
        x = 2
        y = 2
        d = 1
        
        def f(x):
            return (x * x + 1) % n
        
        for _ in range(max_iterations):
            x = f(x)
            y = f(f(y))
            d = RSAMath.gcd(abs(x - y), n)
            
            if 1 < d < n:
                return d
        
        return None

class RSAKey:
    """RSA key pair representation"""
    
    def __init__(self, n: int, e: int, d: int = None, p: int = None, q: int = None):
        self.n = n  # Modulus
        self.e = e  # Public exponent
        self.d = d  # Private exponent
        self.p = p  # Prime factor 1
        self.q = q  # Prime factor 2
        self.key_size = n.bit_length()
    
    def public_key(self) -> Tuple[int, int]:
        """Return public key components"""
        return (self.n, self.e)
    
    def private_key(self) -> Tuple[int, int, int]:
        """Return private key components"""
        return (self.n, self.e, self.d)

class RSACipher:
    """RSA encryption and digital signatures"""
    
    @staticmethod
    def generate_keypair(key_size: int = 2048) -> RSAKey:
        """Generate RSA key pair"""
        print(f"Generating {key_size}-bit RSA key pair...")
        start_time = time.time()
        
        # Generate two primes
        p = RSAMath.generate_prime(key_size // 2)
        q = RSAMath.generate_prime(key_size // 2)
        
        # Ensure p != q
        while p == q:
            q = RSAMath.generate_prime(key_size // 2)
        
        # Compute modulus
        n = p * q
        
        # Compute Euler's totient
        phi_n = (p - 1) * (q - 1)
        
        # Choose public exponent
        e = 65537
        if RSAMath.gcd(e, phi_n) != 1:
            for candidate in [3, 5, 17, 257]:
                if RSAMath.gcd(candidate, phi_n) == 1:
                    e = candidate
                    break
        
        # Compute private exponent
        d = RSAMath.mod_inverse(e, phi_n)
        
        generation_time = time.time() - start_time
        print(f"Key generation completed in {generation_time:.2f} seconds")
        
        return RSAKey(n, e, d, p, q)
    
    @staticmethod
    def encrypt(message: int, public_key: Tuple[int, int]) -> int:
        """RSA encryption: c = m^e mod n"""
        n, e = public_key
        return RSAMath.mod_exp(message, e, n)
    
    @staticmethod
    def decrypt(ciphertext: int, private_key: RSAKey) -> int:
        """RSA decryption: m = c^d mod n"""
        if private_key.p and private_key.q:
            # Use Chinese Remainder Theorem for faster decryption
            return RSACipher._decrypt_crt(ciphertext, private_key)
        else:
            return RSAMath.mod_exp(ciphertext, private_key.d, private_key.n)
    
    @staticmethod
    def _decrypt_crt(ciphertext: int, private_key: RSAKey) -> int:
        """RSA decryption using Chinese Remainder Theorem"""
        p, q = private_key.p, private_key.q
        
        # Compute exponents
        d_p = private_key.d % (p - 1)
        d_q = private_key.d % (q - 1)
        
        # Compute inverse
        q_inv = RSAMath.mod_inverse(q, p)
        
        # Compute partial results
        m_p = RSAMath.mod_exp(ciphertext % p, d_p, p)
        m_q = RSAMath.mod_exp(ciphertext % q, d_q, q)
        
        # Combine using CRT
        h = (q_inv * (m_p - m_q)) % p
        return m_q + h * q
    
    @staticmethod
    def sign(message_hash: int, private_key: RSAKey) -> int:
        """RSA signature: s = h^d mod n"""
        return RSACipher.decrypt(message_hash, private_key)
    
    @staticmethod
    def verify(signature: int, message_hash: int, public_key: Tuple[int, int]) -> bool:
        """RSA signature verification"""
        recovered_hash = RSACipher.encrypt(signature, public_key)
        return recovered_hash == message_hash

class RSAAttacks:
    """Common attacks against RSA"""
    
    @staticmethod
    def factor_small_n(n: int, max_factor: int = 1000000) -> Optional[Tuple[int, int]]:
        """Trial division for small factors"""
        for i in range(2, min(int(math.sqrt(n)) + 1, max_factor)):
            if n % i == 0:
                return i, n // i
        return None
    
    @staticmethod
    def pollards_rho_attack(n: int) -> Optional[Tuple[int, int]]:
        """Use Pollard's rho to factor n"""
        factor = RSAMath.pollards_rho(n)
        if factor and factor != n:
            return factor, n // factor
        return None
    
    @staticmethod
    def fermat_factorization(n: int, max_iterations: int = 1000000) -> Optional[Tuple[int, int]]:
        """Fermat's factorization method"""
        a = int(math.ceil(math.sqrt(n)))
        
        for _ in range(max_iterations):
            b_squared = a * a - n
            if b_squared >= 0:
                b = int(math.sqrt(b_squared))
                if b * b == b_squared:
                    p = a + b
                    q = a - b
                    if p * q == n and p > 1 and q > 1:
                        return p, q
            a += 1
        
        return None
    
    @staticmethod
    def common_modulus_attack(c1: int, c2: int, e1: int, e2: int, n: int) -> Optional[int]:
        """Attack when same message encrypted with different exponents"""
        if RSAMath.gcd(e1, e2) != 1:
            return None
        
        # Find coefficients such that a*e1 + b*e2 = 1
        gcd, a, b = RSAMath.extended_gcd(e1, e2)
        
        if a < 0:
            c1 = RSAMath.mod_inverse(c1, n)
            a = -a
        if b < 0:
            c2 = RSAMath.mod_inverse(c2, n)
            b = -b
        
        # Compute c1^a * c2^b mod n
        result = (RSAMath.mod_exp(c1, a, n) * RSAMath.mod_exp(c2, b, n)) % n
        return result
    
    @staticmethod
    def hastad_attack(ciphertexts: List[int], public_keys: List[Tuple[int, int]], e: int) -> Optional[int]:
        """Håstad's attack for small public exponents"""
        if len(ciphertexts) < e:
            return None
        
        # Use Chinese Remainder Theorem
        moduli = [pk[0] for pk in public_keys[:e]]
        remainders = ciphertexts[:e]
        
        # Solve system of congruences
        total = 0
        prod = 1
        for m in moduli:
            prod *= m
        
        for r, m in zip(remainders, moduli):
            p = prod // m
            total += r * RSAMath.mod_inverse(p, m) * p
        
        result = total % prod
        
        # Take e-th root
        return RSAAttacks._integer_nth_root(result, e)
    
    @staticmethod
    def _integer_nth_root(x: int, n: int) -> Optional[int]:
        """Compute integer n-th root"""
        if x < 0:
            return None
        
        if x == 0:
            return 0
        
        # Newton's method
        root = x
        while True:
            new_root = ((n - 1) * root + x // (root ** (n - 1))) // n
            if new_root >= root:
                return root
            root = new_root
    
    @staticmethod
    def timing_attack_simulation(private_key: RSAKey, samples: int = 100) -> Dict[str, float]:
        """Simulate timing attack on RSA decryption"""
        times_with_crt = []
        times_without_crt = []
        
        for _ in range(samples):
            # Generate random ciphertext
            ciphertext = secrets.randbelow(private_key.n)
            
            # Time CRT decryption
            start = time.perf_counter()
            RSACipher._decrypt_crt(ciphertext, private_key)
            times_with_crt.append(time.perf_counter() - start)
            
            # Time regular decryption
            start = time.perf_counter()
            RSAMath.mod_exp(ciphertext, private_key.d, private_key.n)
            times_without_crt.append(time.perf_counter() - start)
        
        return {
            'avg_time_crt': sum(times_with_crt) / len(times_with_crt),
            'avg_time_regular': sum(times_without_crt) / len(times_without_crt),
            'speedup_factor': sum(times_without_crt) / sum(times_with_crt)
        }

def rsa_demo():
    """Comprehensive RSA demonstration"""
    print("=== RSA Cryptosystem Demonstration ===")
    
    # Generate key pair
    key = RSACipher.generate_keypair(1024)  # Smaller key for demo
    
    print(f"\nKey Details:")
    print(f"Key size: {key.key_size} bits")
    print(f"n = {hex(key.n)}")
    print(f"e = {key.e}")
    print(f"d = {hex(key.d)}")
    
    # Test encryption/decryption
    message = 123456789
    print(f"\n=== Encryption/Decryption Test ===")
    print(f"Original message: {message}")
    
    # Encrypt
    ciphertext = RSACipher.encrypt(message, key.public_key())
    print(f"Encrypted: {hex(ciphertext)}")
    
    # Decrypt
    decrypted = RSACipher.decrypt(ciphertext, key)
    print(f"Decrypted: {decrypted}")
    print(f"Decryption successful: {message == decrypted}")
    
    # Test digital signatures
    print(f"\n=== Digital Signature Test ===")
    message_to_sign = b"Hello, RSA signatures!"
    message_hash = int.from_bytes(hashlib.sha256(message_to_sign).digest(), 'big')
    
    # Sign
    signature = RSACipher.sign(message_hash, key)
    print(f"Message: {message_to_sign}")
    print(f"Signature: {hex(signature)}")
    
    # Verify
    is_valid = RSACipher.verify(signature, message_hash, key.public_key())
    print(f"Signature valid: {is_valid}")
    
    # Test performance
    print(f"\n=== Performance Analysis ===")
    timing_results = RSAAttacks.timing_attack_simulation(key, 50)
    for metric, value in timing_results.items():
        print(f"{metric}: {value:.6f}")
    
    # Test attacks
    print(f"\n=== Security Analysis ===")
    
    # Try to factor n (should fail for proper key sizes)
    print("Attempting factorization attacks...")
    
    # Small factor attack
    small_factors = RSAAttacks.factor_small_n(key.n, 10000)
    if small_factors:
        print(f"Small factors found: {small_factors}")
    else:
        print("No small factors found")
    
    # Pollard's rho (will likely timeout for secure keys)
    print("Attempting Pollard's rho factorization...")
    rho_result = RSAAttacks.pollards_rho_attack(key.n)
    if rho_result:
        print(f"Pollard's rho factors: {rho_result}")
    else:
        print("Pollard's rho attack failed")

if __name__ == "__main__":
    rsa_demo()
```

---

*This is just the beginning of the comprehensive practical examples. The remaining tools (AES Suite, Hash Functions, Random Number Generators, Frequency Analysis Tools, and Modern Cipher Implementations) would continue with the same level of detail and functionality.*

*Each tool includes:*
- *Complete implementations*
- *Security analysis capabilities*
- *Attack simulations*
- *Performance benchmarking*
- *Educational demonstrations*
- *Real-world usage examples*
