#!/usr/bin/env python3
"""
Complete Cryptography Implementation Library
Educational implementations of major cryptographic algorithms
Author: Cryptography Mastery Guide
License: Educational Use Only
"""

import hashlib
import hmac
import secrets
import struct
import time
import json
import base64
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import math
from enum import Enum

# ===================================================================
# 1. MATHEMATICAL FOUNDATIONS
# ===================================================================

class MathUtils:
    """Mathematical utilities for cryptography"""
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest Common Divisor using Euclidean algorithm"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean Algorithm"""
        if a == 0:
            return b, 0, 1
        gcd_val, x1, y1 = MathUtils.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd_val, x, y
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> int:
        """Modular multiplicative inverse"""
        gcd_val, x, _ = MathUtils.extended_gcd(a, m)
        if gcd_val != 1:
            raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
        return (x % m + m) % m
    
    @staticmethod
    def fast_pow(base: int, exp: int, mod: int) -> int:
        """Fast modular exponentiation"""
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        return result
    
    @staticmethod
    def is_prime(n: int, k: int = 5) -> bool:
        """Miller-Rabin primality test"""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(k):
            a = secrets.randbelow(n - 3) + 2
            x = MathUtils.fast_pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = MathUtils.fast_pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    @staticmethod
    def generate_prime(bits: int) -> int:
        """Generate a prime number of specified bit length"""
        while True:
            # Generate odd number of required bit length
            candidate = secrets.randbits(bits)
            candidate |= (1 << bits - 1) | 1  # Set MSB and LSB
            
            if MathUtils.is_prime(candidate):
                return candidate

# ===================================================================
# 2. CLASSICAL CRYPTOGRAPHY
# ===================================================================

class ClassicalCiphers:
    """Implementation of classical cipher algorithms"""
    
    @staticmethod
    def caesar_cipher(text: str, shift: int, decrypt: bool = False) -> str:
        """Caesar cipher implementation"""
        if decrypt:
            shift = -shift
        
        result = ""
        for char in text:
            if char.isalpha():
                ascii_offset = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - ascii_offset + shift) % 26
                result += chr(shifted + ascii_offset)
            else:
                result += char
        return result
    
    @staticmethod
    def vigenere_cipher(text: str, key: str, decrypt: bool = False) -> str:
        """Vigenère cipher implementation"""
        result = ""
        key = key.upper()
        key_index = 0
        
        for char in text:
            if char.isalpha():
                ascii_offset = ord('A') if char.isupper() else ord('a')
                key_shift = ord(key[key_index % len(key)]) - ord('A')
                
                if decrypt:
                    key_shift = -key_shift
                
                shifted = (ord(char) - ascii_offset + key_shift) % 26
                result += chr(shifted + ascii_offset)
                key_index += 1
            else:
                result += char
        
        return result
    
    @staticmethod
    def playfair_cipher(text: str, key: str, decrypt: bool = False) -> str:
        """Playfair cipher implementation"""
        # Create 5x5 key matrix
        alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # J is omitted
        key = key.upper().replace('J', 'I')
        
        # Remove duplicates from key
        seen = set()
        key_unique = ""
        for char in key:
            if char not in seen and char in alphabet:
                key_unique += char
                seen.add(char)
        
        # Create matrix
        matrix_chars = key_unique + "".join(c for c in alphabet if c not in seen)
        matrix = [list(matrix_chars[i:i+5]) for i in range(0, 25, 5)]
        
        # Create position lookup
        char_pos = {}
        for i, row in enumerate(matrix):
            for j, char in enumerate(row):
                char_pos[char] = (i, j)
        
        # Prepare text
        text = text.upper().replace('J', 'I').replace(' ', '')
        
        # Create digrams
        digrams = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i] != text[i + 1]:
                digrams.append(text[i:i+2])
                i += 2
            else:
                digrams.append(text[i] + 'X')
                i += 1
        
        # Process digrams
        result = ""
        for digram in digrams:
            if len(digram) < 2:
                continue
                
            char1, char2 = digram[0], digram[1]
            
            if char1 not in char_pos or char2 not in char_pos:
                result += digram
                continue
            
            row1, col1 = char_pos[char1]
            row2, col2 = char_pos[char2]
            
            if row1 == row2:  # Same row
                if decrypt:
                    new_col1 = (col1 - 1) % 5
                    new_col2 = (col2 - 1) % 5
                else:
                    new_col1 = (col1 + 1) % 5
                    new_col2 = (col2 + 1) % 5
                result += matrix[row1][new_col1] + matrix[row2][new_col2]
            elif col1 == col2:  # Same column
                if decrypt:
                    new_row1 = (row1 - 1) % 5
                    new_row2 = (row2 - 1) % 5
                else:
                    new_row1 = (row1 + 1) % 5
                    new_row2 = (row2 + 1) % 5
                result += matrix[new_row1][col1] + matrix[new_row2][col2]
            else:  # Rectangle
                result += matrix[row1][col2] + matrix[row2][col1]
        
        return result

# ===================================================================
# 3. SYMMETRIC CRYPTOGRAPHY
# ===================================================================

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
    
    # Inverse S-box for InvSubBytes transformation
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
    RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
    
    def __init__(self, key: bytes):
        """Initialize AES with key"""
        self.key_size = len(key)
        if self.key_size not in [16, 24, 32]:
            raise ValueError("Key size must be 128, 192, or 256 bits")
        
        self.rounds = {16: 10, 24: 12, 32: 14}[self.key_size]
        self.round_keys = self._key_expansion(key)
    
    def _key_expansion(self, key: bytes) -> List[List[int]]:
        """AES key expansion algorithm"""
        # Convert key to words (4 bytes each)
        words = []
        for i in range(0, len(key), 4):
            word = list(key[i:i+4])
            words.append(word)
        
        # Number of words needed
        total_words = 4 * (self.rounds + 1)
        
        # Expand the key
        for i in range(len(words), total_words):
            temp = words[i-1][:]
            
            if i % (self.key_size // 4) == 0:
                # RotWord
                temp = temp[1:] + [temp[0]]
                # SubWord
                temp = [self.SBOX[b] for b in temp]
                # XOR with round constant
                temp[0] ^= self.RCON[(i // (self.key_size // 4)) - 1]
            elif self.key_size == 32 and i % 8 == 4:
                # Additional SubWord for AES-256
                temp = [self.SBOX[b] for b in temp]
            
            # XOR with word from key_size//4 positions back
            new_word = []
            for j in range(4):
                new_word.append(words[i - (self.key_size // 4)][j] ^ temp[j])
            
            words.append(new_word)
        
        # Group words into round keys
        round_keys = []
        for i in range(0, len(words), 4):
            round_key = []
            for j in range(4):
                round_key.extend(words[i+j])
            round_keys.append(round_key)
        
        return round_keys
    
    def _sub_bytes(self, state: List[List[int]]) -> List[List[int]]:
        """SubBytes transformation"""
        for i in range(4):
            for j in range(4):
                state[i][j] = self.SBOX[state[i][j]]
        return state
    
    def _inv_sub_bytes(self, state: List[List[int]]) -> List[List[int]]:
        """Inverse SubBytes transformation"""
        for i in range(4):
            for j in range(4):
                state[i][j] = self.INV_SBOX[state[i][j]]
        return state
    
    def _shift_rows(self, state: List[List[int]]) -> List[List[int]]:
        """ShiftRows transformation"""
        # Row 0: no shift
        # Row 1: shift left by 1
        state[1] = state[1][1:] + [state[1][0]]
        # Row 2: shift left by 2
        state[2] = state[2][2:] + state[2][:2]
        # Row 3: shift left by 3
        state[3] = state[3][3:] + state[3][:3]
        return state
    
    def _inv_shift_rows(self, state: List[List[int]]) -> List[List[int]]:
        """Inverse ShiftRows transformation"""
        # Row 0: no shift
        # Row 1: shift right by 1
        state[1] = [state[1][-1]] + state[1][:-1]
        # Row 2: shift right by 2
        state[2] = state[2][-2:] + state[2][:-2]
        # Row 3: shift right by 3
        state[3] = state[3][-3:] + state[3][:-3]
        return state
    
    def _gmul(self, a: int, b: int) -> int:
        """Galois field multiplication"""
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            carry = a & 0x80
            a <<= 1
            if carry:
                a ^= 0x1b  # Irreducible polynomial
            b >>= 1
        return p & 0xff
    
    def _mix_columns(self, state: List[List[int]]) -> List[List[int]]:
        """MixColumns transformation"""
        for j in range(4):
            col = [state[i][j] for i in range(4)]
            state[0][j] = self._gmul(0x02, col[0]) ^ self._gmul(0x03, col[1]) ^ col[2] ^ col[3]
            state[1][j] = col[0] ^ self._gmul(0x02, col[1]) ^ self._gmul(0x03, col[2]) ^ col[3]
            state[2][j] = col[0] ^ col[1] ^ self._gmul(0x02, col[2]) ^ self._gmul(0x03, col[3])
            state[3][j] = self._gmul(0x03, col[0]) ^ col[0] ^ col[1] ^ self._gmul(0x02, col[3])
        return state
    
    def _inv_mix_columns(self, state: List[List[int]]) -> List[List[int]]:
        """Inverse MixColumns transformation"""
        for j in range(4):
            col = [state[i][j] for i in range(4)]
            state[0][j] = self._gmul(0x0e, col[0]) ^ self._gmul(0x0b, col[1]) ^ self._gmul(0x0d, col[2]) ^ self._gmul(0x09, col[3])
            state[1][j] = self._gmul(0x09, col[0]) ^ self._gmul(0x0e, col[1]) ^ self._gmul(0x0b, col[2]) ^ self._gmul(0x0d, col[3])
            state[2][j] = self._gmul(0x0d, col[0]) ^ self._gmul(0x09, col[1]) ^ self._gmul(0x0e, col[2]) ^ self._gmul(0x0b, col[3])
            state[3][j] = self._gmul(0x0b, col[0]) ^ self._gmul(0x0d, col[1]) ^ self._gmul(0x09, col[2]) ^ self._gmul(0x0e, col[3])
        return state
    
    def _add_round_key(self, state: List[List[int]], round_key: List[int]) -> List[List[int]]:
        """AddRoundKey transformation"""
        for i in range(4):
            for j in range(4):
                state[i][j] ^= round_key[i * 4 + j]
        return state
    
    def _bytes_to_state(self, data: bytes) -> List[List[int]]:
        """Convert bytes to AES state matrix"""
        state = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                state[j][i] = data[i * 4 + j]
        return state
    
    def _state_to_bytes(self, state: List[List[int]]) -> bytes:
        """Convert AES state matrix to bytes"""
        data = bytearray(16)
        for i in range(4):
            for j in range(4):
                data[i * 4 + j] = state[j][i]
        return bytes(data)
    
    def encrypt_block(self, plaintext: bytes) -> bytes:
        """Encrypt a single 16-byte block"""
        if len(plaintext) != 16:
            raise ValueError("Block must be exactly 16 bytes")
        
        state = self._bytes_to_state(plaintext)
        
        # Initial round
        self._add_round_key(state, self.round_keys[0])
        
        # Main rounds
        for round_num in range(1, self.rounds):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state)
            self._add_round_key(state, self.round_keys[round_num])
        
        # Final round
        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, self.round_keys[self.rounds])
        
        return self._state_to_bytes(state)
    
    def decrypt_block(self, ciphertext: bytes) -> bytes:
        """Decrypt a single 16-byte block"""
        if len(ciphertext) != 16:
            raise ValueError("Block must be exactly 16 bytes")
        
        state = self._bytes_to_state(ciphertext)
        
        # Initial round
        self._add_round_key(state, self.round_keys[self.rounds])
        
        # Main rounds
        for round_num in range(self.rounds - 1, 0, -1):
            self._inv_shift_rows(state)
            self._inv_sub_bytes(state)
            self._add_round_key(state, self.round_keys[round_num])
            self._inv_mix_columns(state)
        
        # Final round
        self._inv_shift_rows(state)
        self._inv_sub_bytes(state)
        self._add_round_key(state, self.round_keys[0])
        
        return self._state_to_bytes(state)

class ChaCha20:
    """ChaCha20 stream cipher implementation"""
    
    def __init__(self, key: bytes, nonce: bytes, counter: int = 0):
        """Initialize ChaCha20"""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes")
        if len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes")
        
        self.key = key
        self.nonce = nonce
        self.counter = counter
    
    def _quarter_round(self, state: List[int], a: int, b: int, c: int, d: int):
        """ChaCha20 quarter round operation"""
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
        """Generate a ChaCha20 keystream block"""
        # Initialize state
        state = [0] * 16
        
        # Constants
        state[0] = 0x61707865
        state[1] = 0x3320646e
        state[2] = 0x79622d32
        state[3] = 0x6b206574
        
        # Key
        for i in range(8):
            state[4 + i] = struct.unpack('<I', self.key[i*4:(i+1)*4])[0]
        
        # Counter
        state[12] = counter
        
        # Nonce
        for i in range(3):
            state[13 + i] = struct.unpack('<I', self.nonce[i*4:(i+1)*4])[0]
        
        # Save initial state
        initial_state = state[:]
        
        # 20 rounds (10 double rounds)
        for _ in range(10):
            # Column rounds
            self._quarter_round(state, 0, 4, 8, 12)
            self._quarter_round(state, 1, 5, 9, 13)
            self._quarter_round(state, 2, 6, 10, 14)
            self._quarter_round(state, 3, 7, 11, 15)
            
            # Diagonal rounds
            self._quarter_round(state, 0, 5, 10, 15)
            self._quarter_round(state, 1, 6, 11, 12)
            self._quarter_round(state, 2, 7, 8, 13)
            self._quarter_round(state, 3, 4, 9, 14)
        
        # Add initial state
        for i in range(16):
            state[i] = (state[i] + initial_state[i]) & 0xffffffff
        
        # Convert to bytes
        keystream = b''
        for word in state:
            keystream += struct.pack('<I', word)
        
        return keystream
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt data with ChaCha20"""
        ciphertext = bytearray()
        counter = self.counter
        
        for i in range(0, len(plaintext), 64):
            keystream = self._chacha20_block(counter)
            chunk = plaintext[i:i+64]
            
            encrypted_chunk = bytes(a ^ b for a, b in zip(chunk, keystream))
            ciphertext.extend(encrypted_chunk)
            
            counter += 1
        
        return bytes(ciphertext)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt data with ChaCha20 (same as encrypt)"""
        return self.encrypt(ciphertext)

# ===================================================================
# 4. ASYMMETRIC CRYPTOGRAPHY
# ===================================================================

class RSA:
    """RSA public-key cryptosystem implementation"""
    
    def __init__(self, key_size: int = 2048):
        """Initialize RSA with key generation"""
        self.key_size = key_size
        self.public_key, self.private_key = self._generate_keypair()
    
    def _generate_keypair(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Generate RSA key pair"""
        # Generate two prime numbers
        p = MathUtils.generate_prime(self.key_size // 2)
        q = MathUtils.generate_prime(self.key_size // 2)
        
        # Compute n and phi(n)
        n = p * q
        phi_n = (p - 1) * (q - 1)
        
        # Choose e (commonly 65537)
        e = 65537
        if MathUtils.gcd(e, phi_n) != 1:
            e = 3
            while MathUtils.gcd(e, phi_n) != 1:
                e += 2
        
        # Compute d
        d = MathUtils.mod_inverse(e, phi_n)
        
        return (e, n), (d, n)
    
    def encrypt(self, message: bytes, public_key: Optional[Tuple[int, int]] = None) -> bytes:
        """Encrypt message with RSA"""
        if public_key is None:
            public_key = self.public_key
        
        e, n = public_key
        
        # Convert message to integer
        message_int = int.from_bytes(message, byteorder='big')
        
        if message_int >= n:
            raise ValueError("Message too large for key size")
        
        # Encrypt: c = m^e mod n
        ciphertext_int = MathUtils.fast_pow(message_int, e, n)
        
        # Convert back to bytes
        byte_length = (n.bit_length() + 7) // 8
        return ciphertext_int.to_bytes(byte_length, byteorder='big')
    
    def decrypt(self, ciphertext: bytes, private_key: Optional[Tuple[int, int]] = None) -> bytes:
        """Decrypt ciphertext with RSA"""
        if private_key is None:
            private_key = self.private_key
        
        d, n = private_key
        
        # Convert ciphertext to integer
        ciphertext_int = int.from_bytes(ciphertext, byteorder='big')
        
        # Decrypt: m = c^d mod n
        message_int = MathUtils.fast_pow(ciphertext_int, d, n)
        
        # Convert back to bytes
        byte_length = (message_int.bit_length() + 7) // 8
        return message_int.to_bytes(byte_length, byteorder='big')

class ECC:
    """Elliptic Curve Cryptography implementation"""
    
    def __init__(self, curve_name: str = "secp256k1"):
        """Initialize ECC with curve parameters"""
        self.curve_name = curve_name
        self.curve = self._get_curve_params(curve_name)
        self.private_key, self.public_key = self._generate_keypair()
    
    def _get_curve_params(self, curve_name: str) -> Dict[str, int]:
        """Get elliptic curve parameters"""
        if curve_name == "secp256k1":
            return {
                'p': 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
                'a': 0,
                'b': 7,
                'g_x': 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                'g_y': 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
                'n': 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            }
        else:
            raise ValueError(f"Unsupported curve: {curve_name}")
    
    def _point_add(self, px: int, py: int, qx: int, qy: int) -> Tuple[int, int]:
        """Add two points on elliptic curve"""
        if px is None:  # Point at infinity
            return qx, qy
        if qx is None:  # Point at infinity
            return px, py
        
        p = self.curve['p']
        
        if px == qx:
            if py == qy:
                # Point doubling
                s = (3 * px * px + self.curve['a']) * MathUtils.mod_inverse(2 * py, p) % p
            else:
                # Points are inverses
                return None, None  # Point at infinity
        else:
            # Point addition
            s = (qy - py) * MathUtils.mod_inverse(qx - px, p) % p
        
        rx = (s * s - px - qx) % p
        ry = (s * (px - rx) - py) % p
        
        return rx, ry
    
    def _point_multiply(self, k: int, px: int, py: int) -> Tuple[int, int]:
        """Multiply point by scalar using double-and-add"""
        if k == 0:
            return None, None  # Point at infinity
        if k == 1:
            return px, py
        
        result_x, result_y = None, None  # Point at infinity
        addend_x, addend_y = px, py
        
        while k:
            if k & 1:
                result_x, result_y = self._point_add(result_x, result_y, addend_x, addend_y)
            addend_x, addend_y = self._point_add(addend_x, addend_y, addend_x, addend_y)
            k >>= 1
        
        return result_x, result_y
    
    def _generate_keypair(self) -> Tuple[int, Tuple[int, int]]:
        """Generate ECC key pair"""
        # Private key is random integer
        private_key = secrets.randbelow(self.curve['n'])
        
        # Public key is private_key * G
        public_key = self._point_multiply(private_key, self.curve['g_x'], self.curve['g_y'])
        
        return private_key, public_key

# ===================================================================
# 5. HASH FUNCTIONS
# ===================================================================

class SHA256:
    """SHA-256 hash function implementation"""
    
    # SHA-256 constants
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    
    def __init__(self):
        """Initialize SHA-256"""
        self._h = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ]
        self._buffer = b''
        self._counter = 0
    
    def _right_rotate(self, value: int, amount: int) -> int:
        """Right rotate 32-bit value"""
        return ((value >> amount) | (value << (32 - amount))) & 0xffffffff
    
    def _process_chunk(self, chunk: bytes):
        """Process a 512-bit chunk"""
        # Break chunk into sixteen 32-bit words
        w = list(struct.unpack('>16I', chunk))
        
        # Extend to 64 words
        for i in range(16, 64):
            s0 = self._right_rotate(w[i-15], 7) ^ self._right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)
            s1 = self._right_rotate(w[i-2], 17) ^ self._right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)
        
        # Initialize working variables
        a, b, c, d, e, f, g, h = self._h
        
        # Main loop
        for i in range(64):
            s1 = self._right_rotate(e, 6) ^ self._right_rotate(e, 11) ^ self._right_rotate(e, 25)
            ch = (e & f) ^ (~e & g)
            temp1 = (h + s1 + ch + self.K[i] + w[i]) & 0xffffffff
            s0 = self._right_rotate(a, 2) ^ self._right_rotate(a, 13) ^ self._right_rotate(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (s0 + maj) & 0xffffffff
            
            h = g
            g = f
            f = e
            e = (d + temp1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xffffffff
        
        # Update hash values
        self._h[0] = (self._h[0] + a) & 0xffffffff
        self._h[1] = (self._h[1] + b) & 0xffffffff
        self._h[2] = (self._h[2] + c) & 0xffffffff
        self._h[3] = (self._h[3] + d) & 0xffffffff
        self._h[4] = (self._h[4] + e) & 0xffffffff
        self._h[5] = (self._h[5] + f) & 0xffffffff
        self._h[6] = (self._h[6] + g) & 0xffffffff
        self._h[7] = (self._h[7] + h) & 0xffffffff
    
    def update(self, data: bytes):
        """Update hash with new data"""
        self._buffer += data
        self._counter += len(data)
        
        # Process complete 512-bit chunks
        while len(self._buffer) >= 64:
            self._process_chunk(self._buffer[:64])
            self._buffer = self._buffer[64:]
    
    def digest(self) -> bytes:
        """Get final hash digest"""
        # Make a copy to avoid modifying state
        temp_buffer = self._buffer
        temp_counter = self._counter
        temp_h = self._h[:]
        
        # Padding
        msg_bit_length = temp_counter * 8
        temp_buffer += b'\x80'
        
        # Pad to 448 bits (56 bytes) mod 512
        while len(temp_buffer) % 64 != 56:
            temp_buffer += b'\x00'
        
        # Append original length as 64-bit big-endian integer
        temp_buffer += struct.pack('>Q', msg_bit_length)
        
        # Process final chunk(s)
        for i in range(0, len(temp_buffer), 64):
            chunk = temp_buffer[i:i+64]
            # Process chunk with temporary state
            w = list(struct.unpack('>16I', chunk))
            
            # Extend to 64 words
            for j in range(16, 64):
                s0 = self._right_rotate(w[j-15], 7) ^ self._right_rotate(w[j-15], 18) ^ (w[j-15] >> 3)
                s1 = self._right_rotate(w[j-2], 17) ^ self._right_rotate(w[j-2], 19) ^ (w[j-2] >> 10)
                w.append((w[j-16] + s0 + w[j-7] + s1) & 0xffffffff)
            
            # Initialize working variables
            a, b, c, d, e, f, g, h = temp_h
            
            # Main loop
            for j in range(64):
                s1 = self._right_rotate(e, 6) ^ self._right_rotate(e, 11) ^ self._right_rotate(e, 25)
                ch = (e & f) ^ (~e & g)
                temp1 = (h + s1 + ch + self.K[j] + w[j]) & 0xffffffff
                s0 = self._right_rotate(a, 2) ^ self._right_rotate(a, 13) ^ self._right_rotate(a, 22)
                maj = (a & b) ^ (a & c) ^ (b & c)
                temp2 = (s0 + maj) & 0xffffffff
                
                h = g
                g = f
                f = e
                e = (d + temp1) & 0xffffffff
                d = c
                c = b
                b = a
                a = (temp1 + temp2) & 0xffffffff
            
            # Update hash values
            temp_h[0] = (temp_h[0] + a) & 0xffffffff
            temp_h[1] = (temp_h[1] + b) & 0xffffffff
            temp_h[2] = (temp_h[2] + c) & 0xffffffff
            temp_h[3] = (temp_h[3] + d) & 0xffffffff
            temp_h[4] = (temp_h[4] + e) & 0xffffffff
            temp_h[5] = (temp_h[5] + f) & 0xffffffff
            temp_h[6] = (temp_h[6] + g) & 0xffffffff
            temp_h[7] = (temp_h[7] + h) & 0xffffffff
        
        # Return final hash
        return struct.pack('>8I', *temp_h)
    
    def hexdigest(self) -> str:
        """Get final hash digest as hex string"""
        return self.digest().hex()

# ===================================================================
# 6. CRYPTANALYSIS TOOLS
# ===================================================================

class FrequencyAnalysis:
    """Frequency analysis for breaking substitution ciphers"""
    
    ENGLISH_FREQUENCIES = {
        'A': 8.12, 'B': 1.49, 'C': 2.78, 'D': 4.25, 'E': 12.02, 'F': 2.23,
        'G': 2.02, 'H': 6.09, 'I': 6.97, 'J': 0.15, 'K': 0.77, 'L': 4.03,
        'M': 2.41, 'N': 6.75, 'O': 7.51, 'P': 1.93, 'Q': 0.10, 'R': 5.99,
        'S': 6.33, 'T': 9.06, 'U': 2.76, 'V': 0.98, 'W': 2.36, 'X': 0.15,
        'Y': 1.97, 'Z': 0.07
    }
    
    @staticmethod
    def analyze_frequencies(text: str) -> Dict[str, float]:
        """Analyze character frequencies in text"""
        text = text.upper()
        char_count = {}
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                char_count[char] = char_count.get(char, 0) + 1
                total_chars += 1
        
        frequencies = {}
        for char, count in char_count.items():
            frequencies[char] = (count / total_chars) * 100
        
        return frequencies
    
    @staticmethod
    def chi_squared_test(observed_freq: Dict[str, float]) -> float:
        """Calculate chi-squared statistic for frequency analysis"""
        chi_squared = 0
        
        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            observed = observed_freq.get(char, 0)
            expected = FrequencyAnalysis.ENGLISH_FREQUENCIES[char]
            chi_squared += ((observed - expected) ** 2) / expected
        
        return chi_squared
    
    @staticmethod
    def break_caesar_cipher(ciphertext: str) -> Tuple[str, int, float]:
        """Break Caesar cipher using frequency analysis"""
        best_shift = 0
        best_score = float('inf')
        best_plaintext = ""
        
        for shift in range(26):
            plaintext = ClassicalCiphers.caesar_cipher(ciphertext, shift, decrypt=True)
            frequencies = FrequencyAnalysis.analyze_frequencies(plaintext)
            score = FrequencyAnalysis.chi_squared_test(frequencies)
            
            if score < best_score:
                best_score = score
                best_shift = shift
                best_plaintext = plaintext
        
        return best_plaintext, best_shift, best_score

# ===================================================================
# 7. DEMONSTRATION FUNCTIONS
# ===================================================================

def demonstrate_complete_library():
    """Demonstrate all cryptographic implementations"""
    print("=== Complete Cryptography Library Demo ===\n")
    
    # 1. Mathematical Foundations
    print("1. Mathematical Foundations:")
    print(f"GCD(48, 18) = {MathUtils.gcd(48, 18)}")
    print(f"Modular inverse of 3 mod 11 = {MathUtils.mod_inverse(3, 11)}")
    print(f"3^7 mod 11 = {MathUtils.fast_pow(3, 7, 11)}")
    print(f"Is 97 prime? {MathUtils.is_prime(97)}")
    
    # 2. Classical Cryptography
    print("\n2. Classical Cryptography:")
    message = "HELLO WORLD"
    caesar_encrypted = ClassicalCiphers.caesar_cipher(message, 3)
    print(f"Caesar cipher (shift 3): '{message}' -> '{caesar_encrypted}'")
    
    vigenere_encrypted = ClassicalCiphers.vigenere_cipher(message, "KEY")
    print(f"Vigenère cipher (key 'KEY'): '{message}' -> '{vigenere_encrypted}'")
    
    # 3. AES Encryption
    print("\n3. AES Encryption:")
    key = secrets.token_bytes(32)  # 256-bit key
    aes = AES(key)
    plaintext = b"This is a secret message for AES encryption test!"
    
    # Pad to 16-byte boundary
    padded_plaintext = plaintext + b'\x00' * (16 - len(plaintext) % 16)
    
    encrypted_blocks = []
    for i in range(0, len(padded_plaintext), 16):
        block = padded_plaintext[i:i+16]
        encrypted_block = aes.encrypt_block(block)
        encrypted_blocks.append(encrypted_block)
    
    ciphertext = b''.join(encrypted_blocks)
    print(f"AES key: {key.hex()[:32]}...")
    print(f"Plaintext: {plaintext}")
    print(f"Ciphertext: {ciphertext.hex()[:64]}...")
    
    # 4. ChaCha20 Encryption
    print("\n4. ChaCha20 Encryption:")
    chacha_key = secrets.token_bytes(32)
    chacha_nonce = secrets.token_bytes(12)
    chacha = ChaCha20(chacha_key, chacha_nonce)
    
    chacha_plaintext = b"ChaCha20 is a fast stream cipher!"
    chacha_ciphertext = chacha.encrypt(chacha_plaintext)
    chacha_decrypted = chacha.decrypt(chacha_ciphertext)
    
    print(f"ChaCha20 key: {chacha_key.hex()[:32]}...")
    print(f"Plaintext: {chacha_plaintext}")
    print(f"Ciphertext: {chacha_ciphertext.hex()}")
    print(f"Decrypted: {chacha_decrypted}")
    
    # 5. RSA Encryption
    print("\n5. RSA Encryption:")
    rsa = RSA(key_size=1024)  # Small key for demo
    rsa_message = b"RSA public key encryption!"
    
    rsa_ciphertext = rsa.encrypt(rsa_message)
    rsa_decrypted = rsa.decrypt(rsa_ciphertext)
    
    print(f"RSA public key (e, n): ({rsa.public_key[0]}, {str(rsa.public_key[1])[:20]}...)")
    print(f"Plaintext: {rsa_message}")
    print(f"Ciphertext: {rsa_ciphertext.hex()[:40]}...")
    print(f"Decrypted: {rsa_decrypted}")
    
    # 6. SHA-256 Hashing
    print("\n6. SHA-256 Hashing:")
    sha = SHA256()
    hash_message = b"This message will be hashed with SHA-256"
    sha.update(hash_message)
    hash_digest = sha.hexdigest()
    
    print(f"Message: {hash_message}")
    print(f"SHA-256 hash: {hash_digest}")
    
    # 7. Frequency Analysis
    print("\n7. Frequency Analysis:")
    cipher_text = "WKLV LV D VHFUHW PHVVDJH"  # Caesar cipher with shift 3
    broken_text, shift, score = FrequencyAnalysis.break_caesar_cipher(cipher_text)
    
    print(f"Ciphertext: {cipher_text}")
    print(f"Broken plaintext: {broken_text}")
    print(f"Key (shift): {shift}")
    print(f"Chi-squared score: {score:.2f}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demonstrate_complete_library()
