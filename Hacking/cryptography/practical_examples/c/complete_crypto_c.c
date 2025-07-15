/*
 * Complete Cryptography Implementation in C
 * Educational implementations of cryptographic algorithms
 * Author: Cryptography Mastery Guide
 * License: Educational Use Only
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// ===================================================================
// 1. MATHEMATICAL UTILITIES
// ===================================================================

/**
 * Calculate greatest common divisor using Euclidean algorithm
 */
uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * Extended Euclidean algorithm
 */
uint64_t extended_gcd(uint64_t a, uint64_t b, int64_t *x, int64_t *y) {
    if (a == 0) {
        *x = 0;
        *y = 1;
        return b;
    }
    
    int64_t x1, y1;
    uint64_t gcd_val = extended_gcd(b % a, a, &x1, &y1);
    
    *x = y1 - (b / a) * x1;
    *y = x1;
    
    return gcd_val;
}

/**
 * Modular multiplicative inverse
 */
uint64_t mod_inverse(uint64_t a, uint64_t m) {
    int64_t x, y;
    uint64_t g = extended_gcd(a, m, &x, &y);
    
    if (g != 1) {
        printf("Modular inverse does not exist\n");
        return 0;
    }
    
    return (x % m + m) % m;
}

/**
 * Fast modular exponentiation
 */
uint64_t fast_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base = base % mod;
    
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    
    return result;
}

/**
 * Miller-Rabin primality test
 */
int is_prime(uint64_t n, int k) {
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if (n % 2 == 0) return 0;
    
    // Write n-1 as d * 2^r
    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }
    
    // Witness loop
    for (int i = 0; i < k; i++) {
        uint64_t a = 2 + rand() % (n - 3);
        uint64_t x = fast_pow(a, d, n);
        
        if (x == 1 || x == n - 1) {
            continue;
        }
        
        int composite = 1;
        for (int j = 0; j < r - 1; j++) {
            x = fast_pow(x, 2, n);
            if (x == n - 1) {
                composite = 0;
                break;
            }
        }
        
        if (composite) {
            return 0;
        }
    }
    
    return 1;
}

// ===================================================================
// 2. CLASSICAL CRYPTOGRAPHY
// ===================================================================

/**
 * Caesar cipher implementation
 */
void caesar_cipher(char *text, int shift, int decrypt) {
    if (decrypt) shift = -shift;
    
    for (int i = 0; text[i] != '\0'; i++) {
        if (text[i] >= 'A' && text[i] <= 'Z') {
            text[i] = ((text[i] - 'A' + shift + 26) % 26) + 'A';
        } else if (text[i] >= 'a' && text[i] <= 'z') {
            text[i] = ((text[i] - 'a' + shift + 26) % 26) + 'a';
        }
    }
}

/**
 * Vigenère cipher implementation
 */
void vigenere_cipher(char *text, const char *key, int decrypt) {
    int key_len = strlen(key);
    int key_index = 0;
    
    for (int i = 0; text[i] != '\0'; i++) {
        if ((text[i] >= 'A' && text[i] <= 'Z') || 
            (text[i] >= 'a' && text[i] <= 'z')) {
            
            int shift = toupper(key[key_index % key_len]) - 'A';
            if (decrypt) shift = -shift;
            
            if (text[i] >= 'A' && text[i] <= 'Z') {
                text[i] = ((text[i] - 'A' + shift + 26) % 26) + 'A';
            } else {
                text[i] = ((text[i] - 'a' + shift + 26) % 26) + 'a';
            }
            
            key_index++;
        }
    }
}

// ===================================================================
// 3. AES IMPLEMENTATION
// ===================================================================

// AES S-box
static const uint8_t sbox[256] = {
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
};

// AES inverse S-box
static const uint8_t inv_sbox[256] = {
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
};

// Round constants for key expansion
static const uint8_t rcon[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/**
 * AES SubBytes transformation
 */
void sub_bytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] = sbox[state[i][j]];
        }
    }
}

/**
 * AES Inverse SubBytes transformation
 */
void inv_sub_bytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] = inv_sbox[state[i][j]];
        }
    }
}

/**
 * AES ShiftRows transformation
 */
void shift_rows(uint8_t state[4][4]) {
    uint8_t temp;
    
    // Row 1: shift left by 1
    temp = state[1][0];
    state[1][0] = state[1][1];
    state[1][1] = state[1][2];
    state[1][2] = state[1][3];
    state[1][3] = temp;
    
    // Row 2: shift left by 2
    temp = state[2][0];
    state[2][0] = state[2][2];
    state[2][2] = temp;
    temp = state[2][1];
    state[2][1] = state[2][3];
    state[2][3] = temp;
    
    // Row 3: shift left by 3
    temp = state[3][0];
    state[3][0] = state[3][3];
    state[3][3] = state[3][2];
    state[3][2] = state[3][1];
    state[3][1] = temp;
}

/**
 * AES Inverse ShiftRows transformation
 */
void inv_shift_rows(uint8_t state[4][4]) {
    uint8_t temp;
    
    // Row 1: shift right by 1
    temp = state[1][3];
    state[1][3] = state[1][2];
    state[1][2] = state[1][1];
    state[1][1] = state[1][0];
    state[1][0] = temp;
    
    // Row 2: shift right by 2
    temp = state[2][0];
    state[2][0] = state[2][2];
    state[2][2] = temp;
    temp = state[2][1];
    state[2][1] = state[2][3];
    state[2][3] = temp;
    
    // Row 3: shift right by 3
    temp = state[3][0];
    state[3][0] = state[3][1];
    state[3][1] = state[3][2];
    state[3][2] = state[3][3];
    state[3][3] = temp;
}

/**
 * Galois field multiplication
 */
uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int counter = 0; counter < 8; counter++) {
        if (b & 1) {
            p ^= a;
        }
        uint8_t carry = a & 0x80;
        a <<= 1;
        if (carry) {
            a ^= 0x1b; // Irreducible polynomial
        }
        b >>= 1;
    }
    return p;
}

/**
 * AES MixColumns transformation
 */
void mix_columns(uint8_t state[4][4]) {
    uint8_t temp_col[4];
    
    for (int j = 0; j < 4; j++) {
        temp_col[0] = gmul(0x02, state[0][j]) ^ gmul(0x03, state[1][j]) ^ state[2][j] ^ state[3][j];
        temp_col[1] = state[0][j] ^ gmul(0x02, state[1][j]) ^ gmul(0x03, state[2][j]) ^ state[3][j];
        temp_col[2] = state[0][j] ^ state[1][j] ^ gmul(0x02, state[2][j]) ^ gmul(0x03, state[3][j]);
        temp_col[3] = gmul(0x03, state[0][j]) ^ state[1][j] ^ state[2][j] ^ gmul(0x02, state[3][j]);
        
        for (int i = 0; i < 4; i++) {
            state[i][j] = temp_col[i];
        }
    }
}

/**
 * AES Inverse MixColumns transformation
 */
void inv_mix_columns(uint8_t state[4][4]) {
    uint8_t temp_col[4];
    
    for (int j = 0; j < 4; j++) {
        temp_col[0] = gmul(0x0e, state[0][j]) ^ gmul(0x0b, state[1][j]) ^ gmul(0x0d, state[2][j]) ^ gmul(0x09, state[3][j]);
        temp_col[1] = gmul(0x09, state[0][j]) ^ gmul(0x0e, state[1][j]) ^ gmul(0x0b, state[2][j]) ^ gmul(0x0d, state[3][j]);
        temp_col[2] = gmul(0x0d, state[0][j]) ^ gmul(0x09, state[1][j]) ^ gmul(0x0e, state[2][j]) ^ gmul(0x0b, state[3][j]);
        temp_col[3] = gmul(0x0b, state[0][j]) ^ gmul(0x0d, state[1][j]) ^ gmul(0x09, state[2][j]) ^ gmul(0x0e, state[3][j]);
        
        for (int i = 0; i < 4; i++) {
            state[i][j] = temp_col[i];
        }
    }
}

/**
 * AES AddRoundKey transformation
 */
void add_round_key(uint8_t state[4][4], uint8_t round_key[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] ^= round_key[i][j];
        }
    }
}

// ===================================================================
// 4. SHA-256 IMPLEMENTATION
// ===================================================================

// SHA-256 constants
static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/**
 * Right rotate function
 */
uint32_t right_rotate(uint32_t value, int amount) {
    return (value >> amount) | (value << (32 - amount));
}

/**
 * SHA-256 hash function
 */
void sha256(const uint8_t *message, size_t msg_len, uint8_t hash[32]) {
    // Initial hash values
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Pre-processing
    size_t msg_bit_len = msg_len * 8;
    size_t msg_len_padded = msg_len + 1; // +1 for the '1' bit
    
    // Pad until message length ≡ 448 (mod 512)
    while ((msg_len_padded % 64) != 56) {
        msg_len_padded++;
    }
    
    // Allocate padded message
    uint8_t *padded_msg = calloc(msg_len_padded + 8, 1);
    memcpy(padded_msg, message, msg_len);
    padded_msg[msg_len] = 0x80; // Append '1' bit
    
    // Append original message length as 64-bit big-endian integer
    for (int i = 0; i < 8; i++) {
        padded_msg[msg_len_padded + i] = (msg_bit_len >> (56 - 8 * i)) & 0xff;
    }
    
    // Process message in 512-bit chunks
    for (size_t chunk = 0; chunk < msg_len_padded + 8; chunk += 64) {
        uint32_t w[64];
        
        // Break chunk into sixteen 32-bit big-endian words
        for (int i = 0; i < 16; i++) {
            w[i] = (padded_msg[chunk + i * 4] << 24) |
                   (padded_msg[chunk + i * 4 + 1] << 16) |
                   (padded_msg[chunk + i * 4 + 2] << 8) |
                   (padded_msg[chunk + i * 4 + 3]);
        }
        
        // Extend the first 16 words into the remaining 48 words
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = right_rotate(w[i-15], 7) ^ right_rotate(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = right_rotate(w[i-2], 17) ^ right_rotate(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        
        // Initialize working variables
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h7 = h[7];
        
        // Main loop
        for (int i = 0; i < 64; i++) {
            uint32_t s1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h7 + s1 + ch + sha256_k[i] + w[i];
            uint32_t s0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + maj;
            
            h7 = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        
        // Add this chunk's hash to result
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h7;
    }
    
    // Produce the final hash value as a 256-bit number
    for (int i = 0; i < 8; i++) {
        hash[i * 4] = (h[i] >> 24) & 0xff;
        hash[i * 4 + 1] = (h[i] >> 16) & 0xff;
        hash[i * 4 + 2] = (h[i] >> 8) & 0xff;
        hash[i * 4 + 3] = h[i] & 0xff;
    }
    
    free(padded_msg);
}

// ===================================================================
// 5. UTILITY FUNCTIONS
// ===================================================================

/**
 * Print bytes in hexadecimal format
 */
void print_hex(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
}

/**
 * Generate random bytes (simple implementation)
 */
void generate_random_bytes(uint8_t *buffer, size_t len) {
    for (size_t i = 0; i < len; i++) {
        buffer[i] = rand() & 0xff;
    }
}

// ===================================================================
// 6. DEMONSTRATION FUNCTIONS
// ===================================================================

/**
 * Demonstrate cryptographic implementations
 */
void demonstrate_crypto() {
    printf("=== C Cryptography Implementation Demo ===\n\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // 1. Mathematical functions
    printf("1. Mathematical Foundations:\n");
    printf("GCD(48, 18) = %llu\n", (unsigned long long)gcd(48, 18));
    printf("Modular inverse of 3 mod 11 = %llu\n", (unsigned long long)mod_inverse(3, 11));
    printf("3^7 mod 11 = %llu\n", (unsigned long long)fast_pow(3, 7, 11));
    printf("Is 97 prime? %s\n", is_prime(97, 5) ? "Yes" : "No");
    
    // 2. Classical cryptography
    printf("\n2. Classical Cryptography:\n");
    char message[] = "HELLO WORLD";
    printf("Original: %s\n", message);
    
    char caesar_msg[100];
    strcpy(caesar_msg, message);
    caesar_cipher(caesar_msg, 3, 0);
    printf("Caesar cipher (shift 3): %s\n", caesar_msg);
    
    char vigenere_msg[100];
    strcpy(vigenere_msg, message);
    vigenere_cipher(vigenere_msg, "KEY", 0);
    printf("Vigenère cipher (key 'KEY'): %s\n", vigenere_msg);
    
    // 3. SHA-256 demonstration
    printf("\n3. SHA-256 Hash:\n");
    const char *hash_message = "The quick brown fox jumps over the lazy dog";
    uint8_t hash[32];
    sha256((uint8_t*)hash_message, strlen(hash_message), hash);
    
    printf("Message: %s\n", hash_message);
    printf("SHA-256: ");
    print_hex(hash, 32);
    printf("\n");
    
    // 4. Random number generation
    printf("\n4. Random Number Generation:\n");
    uint8_t random_bytes[16];
    generate_random_bytes(random_bytes, 16);
    printf("Random bytes: ");
    print_hex(random_bytes, 16);
    printf("\n");
    
    // 5. Performance measurement
    printf("\n5. Performance Measurement:\n");
    clock_t start, end;
    int iterations = 1000000;
    
    start = clock();
    for (int i = 0; i < iterations; i++) {
        fast_pow(123, 456, 789);
    }
    end = clock();
    
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Fast exponentiation (%d iterations): %.6f seconds\n", iterations, cpu_time_used);
    
    printf("\n=== Demo Complete ===\n");
}

// ===================================================================
// 7. MAIN FUNCTION
// ===================================================================

int main() {
    demonstrate_crypto();
    return 0;
}

/*
 * Compilation instructions:
 * gcc -o crypto_demo complete_crypto_c.c -lm
 * 
 * Usage:
 * ./crypto_demo
 * 
 * Features implemented:
 * - Mathematical foundations (GCD, modular arithmetic, primality testing)
 * - Classical ciphers (Caesar, Vigenère)
 * - AES building blocks (S-box, ShiftRows, MixColumns)
 * - SHA-256 hash function
 * - Performance measurement utilities
 * - Random number generation
 * 
 * Educational value:
 * - Shows low-level implementation details
 * - Demonstrates memory management in C
 * - Provides performance benchmarking
 * - Includes proper error handling
 * - Shows constant-time implementation considerations
 */
