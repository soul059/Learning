# 🔐 Modern Symmetric Cryptography

## 📖 Table of Contents
1. [Introduction to Symmetric Cryptography](#introduction-to-symmetric-cryptography)
2. [Stream Ciphers](#stream-ciphers)
3. [Block Ciphers](#block-ciphers)
4. [Advanced Encryption Standard (AES)](#advanced-encryption-standard-aes)
5. [Block Cipher Modes of Operation](#block-cipher-modes-of-operation)
6. [Authenticated Encryption](#authenticated-encryption)
7. [Key Derivation and Management](#key-derivation-and-management)
8. [Side-Channel Attacks](#side-channel-attacks)
9. [Post-Quantum Considerations](#post-quantum-considerations)

---

## 🔐 Introduction to Symmetric Cryptography

### Core Principles

Symmetric cryptography uses the same key for both encryption and decryption. The security relies on keeping the key secret while the algorithm can be public.

#### Perfect Secrecy (Shannon's Theorem)
A cipher has perfect secrecy if the ciphertext reveals no information about the plaintext.

**Requirements for Perfect Secrecy:**
1. Key length ≥ message length
2. Key must be truly random
3. Key must never be reused

```python
import os
import secrets
from typing import bytes

class OneTimePad:
    """Implementation of the only provably secure cipher"""
    
    @staticmethod
    def generate_key(length: int) -> bytes:
        """Generate a truly random key"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def encrypt(plaintext: bytes, key: bytes) -> bytes:
        """Encrypt using one-time pad"""
        if len(key) < len(plaintext):
            raise ValueError("Key must be at least as long as plaintext")
        
        ciphertext = bytearray()
        for i in range(len(plaintext)):
            ciphertext.append(plaintext[i] ^ key[i])
        
        return bytes(ciphertext)
    
    @staticmethod
    def decrypt(ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt using one-time pad (XOR is self-inverse)"""
        return OneTimePad.encrypt(ciphertext, key)

# Example
otp = OneTimePad()
message = b"ATTACK AT DAWN"
key = otp.generate_key(len(message))
encrypted = otp.encrypt(message, key)
decrypted = otp.decrypt(encrypted, key)

print(f"Original: {message}")
print(f"Key: {key.hex()}")
print(f"Encrypted: {encrypted.hex()}")
print(f"Decrypted: {decrypted}")
```

### Security Definitions

#### Computational Security
Security based on computational complexity assumptions.

#### Semantic Security (IND-CPA)
An adversary cannot distinguish between encryptions of two messages of their choice.

#### Chosen Plaintext Attack (CPA) Security
Adversary can obtain encryptions of plaintexts of their choice.

#### Chosen Ciphertext Attack (CCA) Security
Adversary can obtain decryptions of ciphertexts of their choice (except the challenge).

---

## 🌊 Stream Ciphers

Stream ciphers encrypt data one bit or byte at a time using a keystream.

### Linear Feedback Shift Registers (LFSRs)

```python
class LFSR:
    """Linear Feedback Shift Register - building block of many stream ciphers"""
    
    def __init__(self, seed: int, taps: list, length: int):
        """
        Initialize LFSR
        seed: initial state
        taps: feedback tap positions (1-indexed)
        length: register length in bits
        """
        self.state = seed
        self.taps = [length - tap for tap in taps]  # Convert to 0-indexed from right
        self.length = length
        self.mask = (1 << length) - 1
    
    def step(self) -> int:
        """Generate next bit and update state"""
        # Calculate feedback bit
        feedback = 0
        for tap in self.taps:
            feedback ^= (self.state >> tap) & 1
        
        # Shift state and insert feedback
        self.state = ((self.state << 1) | feedback) & self.mask
        
        # Return the rightmost bit as output
        return self.state & 1
    
    def generate_keystream(self, length: int) -> list:
        """Generate keystream of specified length"""
        keystream = []
        for _ in range(length):
            keystream.append(self.step())
        return keystream
    
    def get_period(self) -> int:
        """Calculate the period of the LFSR"""
        initial_state = self.state
        period = 0
        
        while True:
            self.step()
            period += 1
            if self.state == initial_state:
                break
            
            # Prevent infinite loop
            if period > 2**self.length:
                return -1
        
        return period

# Example: 4-bit LFSR with maximal period
lfsr = LFSR(seed=0b1111, taps=[4, 3], length=4)
keystream = lfsr.generate_keystream(20)
print(f"Keystream: {keystream}")
print(f"Period: {lfsr.get_period()}")
```

### RC4 Stream Cipher

```python
class RC4:
    """RC4 stream cipher implementation"""
    
    def __init__(self, key: bytes):
        """Initialize RC4 with key"""
        self.key = key
        self.S = list(range(256))
        self.i = 0
        self.j = 0
        
        # Key Scheduling Algorithm (KSA)
        j = 0
        for i in range(256):
            j = (j + self.S[i] + key[i % len(key)]) % 256
            self.S[i], self.S[j] = self.S[j], self.S[i]
    
    def generate_keystream_byte(self) -> int:
        """Generate one byte of keystream"""
        self.i = (self.i + 1) % 256
        self.j = (self.j + self.S[self.i]) % 256
        self.S[self.i], self.S[self.j] = self.S[self.j], self.S[self.i]
        
        K = self.S[(self.S[self.i] + self.S[self.j]) % 256]
        return K
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt plaintext"""
        ciphertext = bytearray()
        for byte in plaintext:
            keystream_byte = self.generate_keystream_byte()
            ciphertext.append(byte ^ keystream_byte)
        return bytes(ciphertext)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext (same as encrypt for stream cipher)"""
        return self.encrypt(ciphertext)

# Example
rc4 = RC4(b"SECRET_KEY")
plaintext = b"Hello, World!"
ciphertext = rc4.encrypt(plaintext)

# Need new instance for decryption (or reset state)
rc4_decrypt = RC4(b"SECRET_KEY")
decrypted = rc4_decrypt.decrypt(ciphertext)

print(f"Plaintext: {plaintext}")
print(f"Ciphertext: {ciphertext.hex()}")
print(f"Decrypted: {decrypted}")
```

### ChaCha20 Stream Cipher

```python
import struct

class ChaCha20:
    """ChaCha20 stream cipher (simplified implementation)"""
    
    def __init__(self, key: bytes, nonce: bytes, counter: int = 0):
        """Initialize ChaCha20"""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        if len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes")
        
        self.key = key
        self.nonce = nonce
        self.counter = counter
    
    def _quarter_round(self, a: int, b: int, c: int, d: int, state: list):
        """ChaCha20 quarter round function"""
        state[a] = (state[a] + state[b]) & 0xffffffff
        state[d] ^= state[a]
        state[d] = ((state[d] << 16) | (state[d] >> 16)) & 0xffffffff
        
        state[c] = (state[c] + state[d]) & 0xffffffff
        state[b] ^= state[c]
        state[b] = ((state[b] << 12) | (state[b] >> 20)) & 0xffffffff
        
        state[a] = (state[a] + state[b]) & 0xffffffff
        state[d] ^= state[a]
        state[d] = ((state[d] << 8) | (state[d] >> 24)) & 0xffffffff
        
        state[c] = (state[c] + state[d]) & 0xffffffff
        state[b] ^= state[c]
        state[b] = ((state[b] << 7) | (state[b] >> 25)) & 0xffffffff
    
    def _chacha20_block(self, counter: int) -> bytes:
        """Generate one 64-byte block of keystream"""
        # Initialize state
        state = [0] * 16
        
        # Constants
        state[0] = 0x61707865
        state[1] = 0x3320646e
        state[2] = 0x79622d32
        state[3] = 0x6b206574
        
        # Key
        key_words = struct.unpack('<8I', self.key)
        for i in range(8):
            state[4 + i] = key_words[i]
        
        # Counter and nonce
        state[12] = counter
        nonce_words = struct.unpack('<3I', self.nonce)
        for i in range(3):
            state[13 + i] = nonce_words[i]
        
        # Save initial state
        initial_state = state[:]
        
        # 20 rounds (10 double rounds)
        for _ in range(10):
            # Column rounds
            self._quarter_round(0, 4, 8, 12, state)
            self._quarter_round(1, 5, 9, 13, state)
            self._quarter_round(2, 6, 10, 14, state)
            self._quarter_round(3, 7, 11, 15, state)
            
            # Diagonal rounds
            self._quarter_round(0, 5, 10, 15, state)
            self._quarter_round(1, 6, 11, 12, state)
            self._quarter_round(2, 7, 8, 13, state)
            self._quarter_round(3, 4, 9, 14, state)
        
        # Add initial state
        for i in range(16):
            state[i] = (state[i] + initial_state[i]) & 0xffffffff
        
        # Convert to bytes
        return b''.join(struct.pack('<I', word) for word in state)
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt plaintext"""
        ciphertext = bytearray()
        counter = self.counter
        
        for i in range(0, len(plaintext), 64):
            keystream_block = self._chacha20_block(counter)
            block = plaintext[i:i + 64]
            
            for j in range(len(block)):
                ciphertext.append(block[j] ^ keystream_block[j])
            
            counter += 1
        
        return bytes(ciphertext)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext"""
        return self.encrypt(ciphertext)

# Example
key = secrets.token_bytes(32)
nonce = secrets.token_bytes(12)
chacha20 = ChaCha20(key, nonce)

plaintext = b"This is a test message for ChaCha20 encryption!"
ciphertext = chacha20.encrypt(plaintext)

# Decrypt
chacha20_decrypt = ChaCha20(key, nonce)
decrypted = chacha20_decrypt.decrypt(ciphertext)

print(f"Plaintext: {plaintext}")
print(f"Ciphertext: {ciphertext.hex()}")
print(f"Decrypted: {decrypted}")
```

---

## 🧱 Block Ciphers

Block ciphers encrypt fixed-size blocks of data.

### Data Encryption Standard (DES)

```python
class DES:
    """Simplified DES implementation for educational purposes"""
    
    # Initial Permutation
    IP = [
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6,
        64, 56, 48, 40, 32, 24, 16, 8,
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7
    ]
    
    # Final Permutation (inverse of IP)
    FP = [
        40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25
    ]
    
    # S-Boxes (simplified - only S1 shown)
    S1 = [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ]
    
    def __init__(self, key: bytes):
        """Initialize DES with 8-byte key"""
        if len(key) != 8:
            raise ValueError("DES key must be 8 bytes")
        self.key = key
        self.subkeys = self._generate_subkeys()
    
    def _permute(self, data: int, table: list, input_bits: int) -> int:
        """Apply permutation table to data"""
        result = 0
        for i, pos in enumerate(table):
            if data & (1 << (input_bits - pos)):
                result |= 1 << (len(table) - 1 - i)
        return result
    
    def _generate_subkeys(self) -> list:
        """Generate 16 subkeys from main key (simplified)"""
        # This is a simplified version - real DES has complex key schedule
        subkeys = []
        key_int = int.from_bytes(self.key, 'big')
        
        for round_num in range(16):
            # Rotate key for each round (simplified)
            key_int = ((key_int << 1) | (key_int >> 55)) & ((1 << 56) - 1)
            # Extract 48 bits for subkey (simplified)
            subkey = key_int & ((1 << 48) - 1)
            subkeys.append(subkey)
        
        return subkeys
    
    def _f_function(self, right: int, subkey: int) -> int:
        """DES f-function (simplified)"""
        # Expansion (32 -> 48 bits) - simplified
        expanded = ((right << 16) | right) & ((1 << 48) - 1)
        
        # XOR with subkey
        xored = expanded ^ subkey
        
        # S-box substitution (simplified - only using S1)
        output = 0
        for i in range(8):
            # Extract 6 bits
            six_bits = (xored >> (42 - 6 * i)) & 0x3f
            row = ((six_bits & 0x20) >> 4) | (six_bits & 0x01)
            col = (six_bits >> 1) & 0x0f
            
            # Use S1 for all (simplified)
            s_output = self.S1[row][col]
            output |= s_output << (28 - 4 * i)
        
        return output & ((1 << 32) - 1)
    
    def _des_round(self, left: int, right: int, subkey: int) -> tuple:
        """One round of DES"""
        new_left = right
        new_right = left ^ self._f_function(right, subkey)
        return new_left, new_right
    
    def encrypt_block(self, block: bytes) -> bytes:
        """Encrypt 8-byte block"""
        if len(block) != 8:
            raise ValueError("Block must be 8 bytes")
        
        # Convert to integer
        data = int.from_bytes(block, 'big')
        
        # Initial permutation
        data = self._permute(data, self.IP, 64)
        
        # Split into left and right halves
        left = (data >> 32) & ((1 << 32) - 1)
        right = data & ((1 << 32) - 1)
        
        # 16 rounds
        for round_num in range(16):
            left, right = self._des_round(left, right, self.subkeys[round_num])
        
        # Combine halves (note: right first in DES)
        combined = (right << 32) | left
        
        # Final permutation
        result = self._permute(combined, self.FP, 64)
        
        return result.to_bytes(8, 'big')
    
    def decrypt_block(self, block: bytes) -> bytes:
        """Decrypt 8-byte block"""
        if len(block) != 8:
            raise ValueError("Block must be 8 bytes")
        
        # Same as encryption but with reversed subkey order
        data = int.from_bytes(block, 'big')
        data = self._permute(data, self.IP, 64)
        
        left = (data >> 32) & ((1 << 32) - 1)
        right = data & ((1 << 32) - 1)
        
        # 16 rounds with reversed subkeys
        for round_num in range(15, -1, -1):
            left, right = self._des_round(left, right, self.subkeys[round_num])
        
        combined = (right << 32) | left
        result = self._permute(combined, self.FP, 64)
        
        return result.to_bytes(8, 'big')

# Example
des_key = b"YELLOW SUBMARINE"[:8]  # 8 bytes
des = DES(des_key)

plaintext_block = b"HELLO123"
encrypted_block = des.encrypt_block(plaintext_block)
decrypted_block = des.decrypt_block(encrypted_block)

print(f"Plaintext: {plaintext_block}")
print(f"Encrypted: {encrypted_block.hex()}")
print(f"Decrypted: {decrypted_block}")
```

### Triple DES (3DES)

```python
class TripleDES:
    """Triple DES implementation"""
    
    def __init__(self, key: bytes):
        """Initialize 3DES with 16 or 24 byte key"""
        if len(key) == 16:
            # Two-key 3DES: K1, K2, K1
            self.key1 = key[:8]
            self.key2 = key[8:16]
            self.key3 = self.key1
        elif len(key) == 24:
            # Three-key 3DES: K1, K2, K3
            self.key1 = key[:8]
            self.key2 = key[8:16]
            self.key3 = key[16:24]
        else:
            raise ValueError("3DES key must be 16 or 24 bytes")
        
        self.des1 = DES(self.key1)
        self.des2 = DES(self.key2)
        self.des3 = DES(self.key3)
    
    def encrypt_block(self, block: bytes) -> bytes:
        """3DES encryption: DES_K1(DES_K2^-1(DES_K1(plaintext)))"""
        # First DES encryption
        intermediate1 = self.des1.encrypt_block(block)
        
        # Second DES decryption
        intermediate2 = self.des2.decrypt_block(intermediate1)
        
        # Third DES encryption
        ciphertext = self.des3.encrypt_block(intermediate2)
        
        return ciphertext
    
    def decrypt_block(self, block: bytes) -> bytes:
        """3DES decryption: DES_K1^-1(DES_K2(DES_K3^-1(ciphertext)))"""
        # Reverse the encryption process
        intermediate1 = self.des3.decrypt_block(block)
        intermediate2 = self.des2.encrypt_block(intermediate1)
        plaintext = self.des1.decrypt_block(intermediate2)
        
        return plaintext

# Example
tdes_key = b"YELLOW SUBMARINE" + b"BLUE KEY"  # 16 bytes
tdes = TripleDES(tdes_key)

plaintext_block = b"HELLO123"
encrypted_block = tdes.encrypt_block(plaintext_block)
decrypted_block = tdes.decrypt_block(encrypted_block)

print(f"3DES Plaintext: {plaintext_block}")
print(f"3DES Encrypted: {encrypted_block.hex()}")
print(f"3DES Decrypted: {decrypted_block}")
```

---

## 🏆 Advanced Encryption Standard (AES)

AES is the current standard for symmetric encryption, supporting key sizes of 128, 192, and 256 bits.

### AES Implementation

```python
class AES:
    """Advanced Encryption Standard implementation"""
    
    # S-box for SubBytes transformation
    SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    ]
    
    # Inverse S-box for InvSubBytes
    INV_SBOX = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
    ]
    
    # Round constants for key expansion
    RCON = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
    ]
    
    def __init__(self, key: bytes):
        """Initialize AES with 128, 192, or 256-bit key"""
        if len(key) not in [16, 24, 32]:
            raise ValueError("AES key must be 16, 24, or 32 bytes")
        
        self.key_size = len(key)
        self.rounds = {16: 10, 24: 12, 32: 14}[self.key_size]
        self.round_keys = self._key_expansion(key)
    
    def _key_expansion(self, key: bytes) -> list:
        """AES key expansion algorithm"""
        key_words = []
        
        # Convert key to words (4 bytes each)
        for i in range(0, len(key), 4):
            word = list(key[i:i+4])
            key_words.append(word)
        
        # Expand key
        for i in range(len(key_words), 4 * (self.rounds + 1)):
            temp = key_words[i-1][:]
            
            if i % (self.key_size // 4) == 0:
                # RotWord
                temp = temp[1:] + [temp[0]]
                # SubWord
                temp = [self.SBOX[b] for b in temp]
                # XOR with Rcon
                temp[0] ^= self.RCON[(i // (self.key_size // 4)) - 1]
            elif self.key_size == 32 and i % 8 == 4:
                # Additional SubWord for AES-256
                temp = [self.SBOX[b] for b in temp]
            
            # XOR with word from key_size/4 positions back
            new_word = []
            for j in range(4):
                new_word.append(temp[j] ^ key_words[i - (self.key_size // 4)][j])
            
            key_words.append(new_word)
        
        # Group into round keys
        round_keys = []
        for round_num in range(self.rounds + 1):
            round_key = []
            for word_num in range(4):
                round_key.extend(key_words[round_num * 4 + word_num])
            round_keys.append(round_key)
        
        return round_keys
    
    def _sub_bytes(self, state: list) -> list:
        """SubBytes transformation"""
        return [self.SBOX[byte] for byte in state]
    
    def _inv_sub_bytes(self, state: list) -> list:
        """Inverse SubBytes transformation"""
        return [self.INV_SBOX[byte] for byte in state]
    
    def _shift_rows(self, state: list) -> list:
        """ShiftRows transformation"""
        # Convert to 4x4 matrix
        matrix = [state[i:i+4] for i in range(0, 16, 4)]
        
        # Shift rows
        for row in range(4):
            matrix[row] = matrix[row][row:] + matrix[row][:row]
        
        # Convert back to list
        return [byte for row in matrix for byte in row]
    
    def _inv_shift_rows(self, state: list) -> list:
        """Inverse ShiftRows transformation"""
        matrix = [state[i:i+4] for i in range(0, 16, 4)]
        
        # Shift rows in opposite direction
        for row in range(4):
            matrix[row] = matrix[row][-row:] + matrix[row][:-row]
        
        return [byte for row in matrix for byte in row]
    
    def _gf_multiply(self, a: int, b: int) -> int:
        """Multiplication in GF(2^8)"""
        result = 0
        for _ in range(8):
            if b & 1:
                result ^= a
            high_bit = a & 0x80
            a <<= 1
            if high_bit:
                a ^= 0x1b  # Irreducible polynomial x^8 + x^4 + x^3 + x + 1
            b >>= 1
        return result & 0xff
    
    def _mix_columns(self, state: list) -> list:
        """MixColumns transformation"""
        result = [0] * 16
        
        for col in range(4):
            # Extract column
            column = [state[row * 4 + col] for row in range(4)]
            
            # Apply MixColumns matrix
            result[0 * 4 + col] = (self._gf_multiply(0x02, column[0]) ^ 
                                  self._gf_multiply(0x03, column[1]) ^ 
                                  column[2] ^ column[3])
            result[1 * 4 + col] = (column[0] ^ 
                                  self._gf_multiply(0x02, column[1]) ^ 
                                  self._gf_multiply(0x03, column[2]) ^ 
                                  column[3])
            result[2 * 4 + col] = (column[0] ^ column[1] ^ 
                                  self._gf_multiply(0x02, column[2]) ^ 
                                  self._gf_multiply(0x03, column[3]))
            result[3 * 4 + col] = (self._gf_multiply(0x03, column[0]) ^ 
                                  column[1] ^ column[2] ^ 
                                  self._gf_multiply(0x02, column[3]))
        
        return result
    
    def _inv_mix_columns(self, state: list) -> list:
        """Inverse MixColumns transformation"""
        result = [0] * 16
        
        for col in range(4):
            column = [state[row * 4 + col] for row in range(4)]
            
            # Apply inverse MixColumns matrix
            result[0 * 4 + col] = (self._gf_multiply(0x0e, column[0]) ^ 
                                  self._gf_multiply(0x0b, column[1]) ^ 
                                  self._gf_multiply(0x0d, column[2]) ^ 
                                  self._gf_multiply(0x09, column[3]))
            result[1 * 4 + col] = (self._gf_multiply(0x09, column[0]) ^ 
                                  self._gf_multiply(0x0e, column[1]) ^ 
                                  self._gf_multiply(0x0b, column[2]) ^ 
                                  self._gf_multiply(0x0d, column[3]))
            result[2 * 4 + col] = (self._gf_multiply(0x0d, column[0]) ^ 
                                  self._gf_multiply(0x09, column[1]) ^ 
                                  self._gf_multiply(0x0e, column[2]) ^ 
                                  self._gf_multiply(0x0b, column[3]))
            result[3 * 4 + col] = (self._gf_multiply(0x0b, column[0]) ^ 
                                  self._gf_multiply(0x0d, column[1]) ^ 
                                  self._gf_multiply(0x09, column[2]) ^ 
                                  self._gf_multiply(0x0e, column[3]))
        
        return result
    
    def _add_round_key(self, state: list, round_key: list) -> list:
        """AddRoundKey transformation"""
        return [state[i] ^ round_key[i] for i in range(16)]
    
    def encrypt_block(self, block: bytes) -> bytes:
        """Encrypt 16-byte block"""
        if len(block) != 16:
            raise ValueError("Block must be 16 bytes")
        
        state = list(block)
        
        # Initial round
        state = self._add_round_key(state, self.round_keys[0])
        
        # Main rounds
        for round_num in range(1, self.rounds):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, self.round_keys[round_num])
        
        # Final round (no MixColumns)
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, self.round_keys[self.rounds])
        
        return bytes(state)
    
    def decrypt_block(self, block: bytes) -> bytes:
        """Decrypt 16-byte block"""
        if len(block) != 16:
            raise ValueError("Block must be 16 bytes")
        
        state = list(block)
        
        # Initial round
        state = self._add_round_key(state, self.round_keys[self.rounds])
        
        # Main rounds (in reverse)
        for round_num in range(self.rounds - 1, 0, -1):
            state = self._inv_shift_rows(state)
            state = self._inv_sub_bytes(state)
            state = self._add_round_key(state, self.round_keys[round_num])
            state = self._inv_mix_columns(state)
        
        # Final round (no InvMixColumns)
        state = self._inv_shift_rows(state)
        state = self._inv_sub_bytes(state)
        state = self._add_round_key(state, self.round_keys[0])
        
        return bytes(state)

# Example
aes_key = secrets.token_bytes(32)  # AES-256
aes = AES(aes_key)

plaintext_block = b"YELLOW SUBMARINE"
encrypted_block = aes.encrypt_block(plaintext_block)
decrypted_block = aes.decrypt_block(encrypted_block)

print(f"AES Plaintext: {plaintext_block}")
print(f"AES Encrypted: {encrypted_block.hex()}")
print(f"AES Decrypted: {decrypted_block}")
```

---

## 🔄 Block Cipher Modes of Operation

Block cipher modes determine how multiple blocks are encrypted.

### Electronic Codebook (ECB) Mode

```python
class ECB:
    """Electronic Codebook mode (insecure - for educational purposes)"""
    
    def __init__(self, cipher):
        self.cipher = cipher
        self.block_size = 16  # AES block size
    
    def _pad(self, data: bytes) -> bytes:
        """PKCS#7 padding"""
        padding_length = self.block_size - (len(data) % self.block_size)
        return data + bytes([padding_length] * padding_length)
    
    def _unpad(self, data: bytes) -> bytes:
        """Remove PKCS#7 padding"""
        padding_length = data[-1]
        return data[:-padding_length]
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """ECB encryption"""
        padded = self._pad(plaintext)
        ciphertext = b''
        
        for i in range(0, len(padded), self.block_size):
            block = padded[i:i + self.block_size]
            encrypted_block = self.cipher.encrypt_block(block)
            ciphertext += encrypted_block
        
        return ciphertext
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """ECB decryption"""
        plaintext = b''
        
        for i in range(0, len(ciphertext), self.block_size):
            block = ciphertext[i:i + self.block_size]
            decrypted_block = self.cipher.decrypt_block(block)
            plaintext += decrypted_block
        
        return self._unpad(plaintext)

# Example
aes_key = b"YELLOW SUBMARINE"
aes_cipher = AES(aes_key)
ecb = ECB(aes_cipher)

plaintext = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
ciphertext = ecb.encrypt(plaintext)
decrypted = ecb.decrypt(ciphertext)

print(f"ECB Original: {plaintext}")
print(f"ECB Encrypted: {ciphertext.hex()}")
print(f"ECB Decrypted: {decrypted}")
```

### Cipher Block Chaining (CBC) Mode

```python
class CBC:
    """Cipher Block Chaining mode"""
    
    def __init__(self, cipher, iv: bytes = None):
        self.cipher = cipher
        self.block_size = 16
        self.iv = iv if iv else secrets.token_bytes(self.block_size)
    
    def _pad(self, data: bytes) -> bytes:
        """PKCS#7 padding"""
        padding_length = self.block_size - (len(data) % self.block_size)
        return data + bytes([padding_length] * padding_length)
    
    def _unpad(self, data: bytes) -> bytes:
        """Remove PKCS#7 padding"""
        padding_length = data[-1]
        return data[:-padding_length]
    
    def _xor_blocks(self, block1: bytes, block2: bytes) -> bytes:
        """XOR two blocks"""
        return bytes(a ^ b for a, b in zip(block1, block2))
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """CBC encryption"""
        padded = self._pad(plaintext)
        ciphertext = self.iv  # Prepend IV
        previous_block = self.iv
        
        for i in range(0, len(padded), self.block_size):
            block = padded[i:i + self.block_size]
            # XOR with previous ciphertext block (or IV)
            xored = self._xor_blocks(block, previous_block)
            encrypted_block = self.cipher.encrypt_block(xored)
            ciphertext += encrypted_block
            previous_block = encrypted_block
        
        return ciphertext
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """CBC decryption"""
        # Extract IV
        iv = ciphertext[:self.block_size]
        ciphertext = ciphertext[self.block_size:]
        
        plaintext = b''
        previous_block = iv
        
        for i in range(0, len(ciphertext), self.block_size):
            block = ciphertext[i:i + self.block_size]
            decrypted_block = self.cipher.decrypt_block(block)
            # XOR with previous ciphertext block (or IV)
            plaintext_block = self._xor_blocks(decrypted_block, previous_block)
            plaintext += plaintext_block
            previous_block = block
        
        return self._unpad(plaintext)

# Example
aes_key = b"YELLOW SUBMARINE"
aes_cipher = AES(aes_key)
iv = secrets.token_bytes(16)
cbc = CBC(aes_cipher, iv)

plaintext = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
ciphertext = cbc.encrypt(plaintext)
decrypted = cbc.decrypt(ciphertext)

print(f"CBC Original: {plaintext}")
print(f"CBC Encrypted: {ciphertext.hex()}")
print(f"CBC Decrypted: {decrypted}")
```

### Counter (CTR) Mode

```python
class CTR:
    """Counter mode - turns block cipher into stream cipher"""
    
    def __init__(self, cipher, nonce: bytes = None):
        self.cipher = cipher
        self.block_size = 16
        self.nonce = nonce if nonce else secrets.token_bytes(8)
    
    def _increment_counter(self, counter: bytes) -> bytes:
        """Increment counter as big-endian integer"""
        counter_int = int.from_bytes(counter, 'big')
        counter_int = (counter_int + 1) % (2 ** (len(counter) * 8))
        return counter_int.to_bytes(len(counter), 'big')
    
    def _generate_keystream_block(self, counter: bytes) -> bytes:
        """Generate keystream block by encrypting nonce||counter"""
        # Combine nonce and counter
        input_block = self.nonce + counter
        # Pad to block size if necessary
        if len(input_block) < self.block_size:
            input_block += b'\x00' * (self.block_size - len(input_block))
        
        return self.cipher.encrypt_block(input_block)
    
    def _process(self, data: bytes) -> bytes:
        """Process data (same for encryption and decryption)"""
        result = bytearray()
        counter = b'\x00' * 8  # 8-byte counter
        
        for i in range(0, len(data), self.block_size):
            # Generate keystream block
            keystream_block = self._generate_keystream_block(counter)
            
            # XOR with data block
            data_block = data[i:i + self.block_size]
            for j in range(len(data_block)):
                result.append(data_block[j] ^ keystream_block[j])
            
            # Increment counter
            counter = self._increment_counter(counter)
        
        return bytes(result)
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """CTR encryption"""
        return self.nonce + self._process(plaintext)  # Prepend nonce
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """CTR decryption"""
        # Extract nonce
        nonce = ciphertext[:8]
        ciphertext = ciphertext[8:]
        
        # Create new CTR instance with extracted nonce
        ctr_decrypt = CTR(self.cipher, nonce)
        return ctr_decrypt._process(ciphertext)

# Example
aes_key = b"YELLOW SUBMARINE"
aes_cipher = AES(aes_key)
ctr = CTR(aes_cipher)

plaintext = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
ciphertext = ctr.encrypt(plaintext)
decrypted = ctr.decrypt(ciphertext)

print(f"CTR Original: {plaintext}")
print(f"CTR Encrypted: {ciphertext.hex()}")
print(f"CTR Decrypted: {decrypted}")
```

### Galois/Counter Mode (GCM)

```python
class GCM:
    """Galois/Counter Mode - provides both encryption and authentication"""
    
    def __init__(self, cipher):
        self.cipher = cipher
        self.block_size = 16
    
    def _gf_multiply(self, x: int, y: int) -> int:
        """Multiplication in GF(2^128)"""
        result = 0
        for i in range(128):
            if y & 1:
                result ^= x
            y >>= 1
            if x & (1 << 127):
                x = (x << 1) ^ 0x87  # Reduction polynomial
            else:
                x <<= 1
            x &= (1 << 128) - 1
        return result
    
    def _ghash(self, h: bytes, data: bytes) -> bytes:
        """GHASH function for authentication"""
        h_int = int.from_bytes(h, 'big')
        y = 0
        
        # Process data in 16-byte blocks
        for i in range(0, len(data), 16):
            block = data[i:i + 16]
            if len(block) < 16:
                block += b'\x00' * (16 - len(block))
            
            block_int = int.from_bytes(block, 'big')
            y ^= block_int
            y = self._gf_multiply(y, h_int)
        
        return y.to_bytes(16, 'big')
    
    def _increment_counter(self, counter: bytes) -> bytes:
        """Increment 32-bit counter in counter block"""
        counter_int = int.from_bytes(counter[-4:], 'big')
        counter_int = (counter_int + 1) % (2 ** 32)
        return counter[:-4] + counter_int.to_bytes(4, 'big')
    
    def encrypt_and_authenticate(self, plaintext: bytes, aad: bytes = b'', 
                                iv: bytes = None) -> tuple:
        """GCM encryption with authentication"""
        if iv is None:
            iv = secrets.token_bytes(12)
        
        # Generate hash key H by encrypting zero block
        h = self.cipher.encrypt_block(b'\x00' * 16)
        
        # Prepare initial counter block
        if len(iv) == 12:
            j0 = iv + b'\x00\x00\x00\x01'
        else:
            # For IV lengths != 96 bits, use GHASH
            iv_padded = iv + b'\x00' * (16 - len(iv) % 16) if len(iv) % 16 else iv
            iv_padded += (len(iv) * 8).to_bytes(8, 'big').rjust(16, b'\x00')
            j0 = self._ghash(h, iv_padded)
        
        # Encrypt plaintext using CTR mode
        ciphertext = bytearray()
        counter = j0
        
        for i in range(0, len(plaintext), 16):
            counter = self._increment_counter(counter)
            keystream_block = self.cipher.encrypt_block(counter)
            
            plaintext_block = plaintext[i:i + 16]
            for j in range(len(plaintext_block)):
                ciphertext.append(plaintext_block[j] ^ keystream_block[j])
        
        ciphertext = bytes(ciphertext)
        
        # Calculate authentication tag
        # Prepare data for GHASH: AAD || 0* || C || 0* || [len(A)]64 || [len(C)]64
        aad_padded = aad + b'\x00' * (16 - len(aad) % 16) if len(aad) % 16 else aad
        c_padded = ciphertext + b'\x00' * (16 - len(ciphertext) % 16) if len(ciphertext) % 16 else ciphertext
        
        lengths = (len(aad) * 8).to_bytes(8, 'big') + (len(ciphertext) * 8).to_bytes(8, 'big')
        
        auth_data = aad_padded + c_padded + lengths
        tag_pre = self._ghash(h, auth_data)
        
        # Encrypt with J0 to get final tag
        tag_keystream = self.cipher.encrypt_block(j0)
        tag = bytes(a ^ b for a, b in zip(tag_pre, tag_keystream))
        
        return ciphertext, tag, iv
    
    def decrypt_and_verify(self, ciphertext: bytes, tag: bytes, 
                          aad: bytes = b'', iv: bytes = None) -> bytes:
        """GCM decryption with authentication verification"""
        if iv is None:
            raise ValueError("IV required for decryption")
        
        # Generate hash key H
        h = self.cipher.encrypt_block(b'\x00' * 16)
        
        # Prepare initial counter block
        if len(iv) == 12:
            j0 = iv + b'\x00\x00\x00\x01'
        else:
            iv_padded = iv + b'\x00' * (16 - len(iv) % 16) if len(iv) % 16 else iv
            iv_padded += (len(iv) * 8).to_bytes(8, 'big').rjust(16, b'\x00')
            j0 = self._ghash(h, iv_padded)
        
        # Verify authentication tag
        aad_padded = aad + b'\x00' * (16 - len(aad) % 16) if len(aad) % 16 else aad
        c_padded = ciphertext + b'\x00' * (16 - len(ciphertext) % 16) if len(ciphertext) % 16 else ciphertext
        
        lengths = (len(aad) * 8).to_bytes(8, 'big') + (len(ciphertext) * 8).to_bytes(8, 'big')
        auth_data = aad_padded + c_padded + lengths
        tag_pre = self._ghash(h, auth_data)
        
        tag_keystream = self.cipher.encrypt_block(j0)
        expected_tag = bytes(a ^ b for a, b in zip(tag_pre, tag_keystream))
        
        if tag != expected_tag:
            raise ValueError("Authentication tag verification failed")
        
        # Decrypt ciphertext
        plaintext = bytearray()
        counter = j0
        
        for i in range(0, len(ciphertext), 16):
            counter = self._increment_counter(counter)
            keystream_block = self.cipher.encrypt_block(counter)
            
            ciphertext_block = ciphertext[i:i + 16]
            for j in range(len(ciphertext_block)):
                plaintext.append(ciphertext_block[j] ^ keystream_block[j])
        
        return bytes(plaintext)

# Example
aes_key = b"YELLOW SUBMARINE"
aes_cipher = AES(aes_key)
gcm = GCM(aes_cipher)

plaintext = b"THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
aad = b"Additional authenticated data"

ciphertext, tag, iv = gcm.encrypt_and_authenticate(plaintext, aad)
decrypted = gcm.decrypt_and_verify(ciphertext, tag, aad, iv)

print(f"GCM Original: {plaintext}")
print(f"GCM Encrypted: {ciphertext.hex()}")
print(f"GCM Tag: {tag.hex()}")
print(f"GCM Decrypted: {decrypted}")
```

---

*This is just the beginning of the comprehensive cryptography guide. The remaining sections (Authenticated Encryption, Key Derivation, Side-Channel Attacks, etc.) would continue with the same level of detail and practical implementations.*

*Next: [Asymmetric Cryptography →](04_asymmetric_cryptography.md)*
