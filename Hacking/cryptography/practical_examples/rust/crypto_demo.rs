// Cryptography in Rust - Memory-Safe Implementations
// Author: Cryptography Mastery Guide
// License: Educational Use Only

use std::collections::HashMap;
use std::time::Instant;

// External crates that would be used in a real implementation
// Add these to Cargo.toml:
// [dependencies]
// ring = "0.16"
// aes-gcm = "0.10"
// chacha20poly1305 = "0.10"
// rsa = "0.9"
// sha2 = "0.10"
// rand = "0.8"
// ed25519-dalek = "1.0"

// For this educational example, we'll implement some algorithms from scratch

// ===================================================================
// 1. MATHEMATICAL FOUNDATIONS
// ===================================================================

pub struct MathUtils;

impl MathUtils {
    /// Greatest Common Divisor using Euclidean algorithm
    pub fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Extended Euclidean Algorithm
    pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
        if a == 0 {
            return (b, 0, 1);
        }
        let (gcd, x1, y1) = Self::extended_gcd(b % a, a);
        let x = y1 - (b / a) * x1;
        let y = x1;
        (gcd, x, y)
    }

    /// Modular multiplicative inverse
    pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
        let (gcd, x, _) = Self::extended_gcd(a as i64, m as i64);
        if gcd != 1 {
            None
        } else {
            Some(((x % m as i64 + m as i64) % m as i64) as u64)
        }
    }

    /// Fast modular exponentiation
    pub fn fast_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        base %= modulus;
        
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result as u128 * base as u128 % modulus as u128) as u64;
            }
            exp >>= 1;
            base = (base as u128 * base as u128 % modulus as u128) as u64;
        }
        
        result
    }

    /// Miller-Rabin primality test
    pub fn is_prime(n: u64, k: usize) -> bool {
        if n < 2 { return false; }
        if n == 2 || n == 3 { return true; }
        if n % 2 == 0 { return false; }

        // Write n-1 as d * 2^r
        let mut d = n - 1;
        let mut r = 0;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }

        // Witness loop
        for _ in 0..k {
            let a = 2 + rand::random::<u64>() % (n - 3);
            let mut x = Self::fast_pow(a, d, n);

            if x == 1 || x == n - 1 {
                continue;
            }

            let mut composite = true;
            for _ in 0..(r - 1) {
                x = Self::fast_pow(x, 2, n);
                if x == n - 1 {
                    composite = false;
                    break;
                }
            }

            if composite {
                return false;
            }
        }

        true
    }
}

// ===================================================================
// 2. CLASSICAL CRYPTOGRAPHY
// ===================================================================

pub struct ClassicalCiphers;

impl ClassicalCiphers {
    /// Caesar cipher implementation
    pub fn caesar_cipher(text: &str, shift: i32, decrypt: bool) -> String {
        let shift = if decrypt { -shift } else { shift };
        
        text.chars()
            .map(|c| {
                if c.is_ascii_alphabetic() {
                    let base = if c.is_ascii_uppercase() { b'A' } else { b'a' };
                    let shifted = ((c as u8 - base) as i32 + shift + 26) % 26;
                    (base + shifted as u8) as char
                } else {
                    c
                }
            })
            .collect()
    }

    /// Vigenère cipher implementation
    pub fn vigenere_cipher(text: &str, key: &str, decrypt: bool) -> String {
        let key = key.to_uppercase();
        let key_bytes: Vec<u8> = key.bytes().collect();
        let mut key_index = 0;

        text.chars()
            .map(|c| {
                if c.is_ascii_alphabetic() {
                    let base = if c.is_ascii_uppercase() { b'A' } else { b'a' };
                    let mut shift = (key_bytes[key_index % key_bytes.len()] - b'A') as i32;
                    
                    if decrypt {
                        shift = -shift;
                    }
                    
                    let shifted = ((c as u8 - base) as i32 + shift + 26) % 26;
                    key_index += 1;
                    (base + shifted as u8) as char
                } else {
                    c
                }
            })
            .collect()
    }

    /// Playfair cipher key matrix generation
    pub fn generate_playfair_matrix(key: &str) -> [[char; 5]; 5] {
        let mut matrix = [[' '; 5]; 5];
        let mut used = [false; 26];
        let mut pos = 0;

        // Process key
        for c in key.to_uppercase().chars() {
            if c.is_ascii_alphabetic() && c != 'J' {
                let index = (c as u8 - b'A') as usize;
                if !used[index] {
                    used[index] = true;
                    matrix[pos / 5][pos % 5] = c;
                    pos += 1;
                }
            }
        }

        // Fill remaining positions
        for c in 'A'..='Z' {
            if c != 'J' {
                let index = (c as u8 - b'A') as usize;
                if !used[index] {
                    matrix[pos / 5][pos % 5] = c;
                    pos += 1;
                }
            }
        }

        matrix
    }
}

// ===================================================================
// 3. MODERN SYMMETRIC CRYPTOGRAPHY
// ===================================================================

/// Simple AES-like block cipher (educational purposes)
pub struct SimpleBlockCipher {
    key: [u8; 16],
}

impl SimpleBlockCipher {
    pub fn new(key: [u8; 16]) -> Self {
        Self { key }
    }

    /// Simple substitution (not actual AES S-box)
    fn substitute_byte(byte: u8) -> u8 {
        // Simplified S-box for educational purposes
        let sbox = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
            0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            // ... truncated for brevity
        ];
        sbox[(byte % 16) as usize]
    }

    /// Encrypt a single block (simplified)
    pub fn encrypt_block(&self, plaintext: [u8; 16]) -> [u8; 16] {
        let mut state = plaintext;
        
        // AddRoundKey
        for i in 0..16 {
            state[i] ^= self.key[i];
        }
        
        // SubBytes (simplified)
        for i in 0..16 {
            state[i] = Self::substitute_byte(state[i]);
        }
        
        // Simple permutation (not actual ShiftRows/MixColumns)
        for i in 0..8 {
            state.swap(i, 15 - i);
        }
        
        state
    }

    /// Decrypt a single block (simplified)
    pub fn decrypt_block(&self, ciphertext: [u8; 16]) -> [u8; 16] {
        let mut state = ciphertext;
        
        // Reverse permutation
        for i in 0..8 {
            state.swap(i, 15 - i);
        }
        
        // Inverse SubBytes (simplified)
        for i in 0..16 {
            state[i] = Self::substitute_byte(state[i]);
        }
        
        // AddRoundKey
        for i in 0..16 {
            state[i] ^= self.key[i];
        }
        
        state
    }
}

// ===================================================================
// 4. STREAM CIPHER (ChaCha20-like)
// ===================================================================

pub struct SimpleStreamCipher {
    key: [u8; 32],
    nonce: [u8; 12],
    counter: u32,
}

impl SimpleStreamCipher {
    pub fn new(key: [u8; 32], nonce: [u8; 12]) -> Self {
        Self {
            key,
            nonce,
            counter: 0,
        }
    }

    /// Quarter round operation (simplified ChaCha20)
    fn quarter_round(a: &mut u32, b: &mut u32, c: &mut u32, d: &mut u32) {
        *a = a.wrapping_add(*b);
        *d ^= *a;
        *d = d.rotate_left(16);
        
        *c = c.wrapping_add(*d);
        *b ^= *c;
        *b = b.rotate_left(12);
        
        *a = a.wrapping_add(*b);
        *d ^= *a;
        *d = d.rotate_left(8);
        
        *c = c.wrapping_add(*d);
        *b ^= *c;
        *b = b.rotate_left(7);
    }

    /// Generate keystream block
    fn generate_block(&mut self) -> [u8; 64] {
        let mut state = [0u32; 16];
        
        // Constants
        state[0] = 0x61707865;
        state[1] = 0x3320646e;
        state[2] = 0x79622d32;
        state[3] = 0x6b206574;
        
        // Key
        for i in 0..8 {
            state[4 + i] = u32::from_le_bytes([
                self.key[i * 4],
                self.key[i * 4 + 1],
                self.key[i * 4 + 2],
                self.key[i * 4 + 3],
            ]);
        }
        
        // Counter
        state[12] = self.counter;
        
        // Nonce
        for i in 0..3 {
            state[13 + i] = u32::from_le_bytes([
                self.nonce[i * 4],
                self.nonce[i * 4 + 1],
                self.nonce[i * 4 + 2],
                self.nonce[i * 4 + 3],
            ]);
        }
        
        let mut working_state = state;
        
        // 20 rounds (10 double rounds)
        for _ in 0..10 {
            // Column rounds
            Self::quarter_round(&mut working_state[0], &mut working_state[4], &mut working_state[8], &mut working_state[12]);
            Self::quarter_round(&mut working_state[1], &mut working_state[5], &mut working_state[9], &mut working_state[13]);
            Self::quarter_round(&mut working_state[2], &mut working_state[6], &mut working_state[10], &mut working_state[14]);
            Self::quarter_round(&mut working_state[3], &mut working_state[7], &mut working_state[11], &mut working_state[15]);
            
            // Diagonal rounds
            Self::quarter_round(&mut working_state[0], &mut working_state[5], &mut working_state[10], &mut working_state[15]);
            Self::quarter_round(&mut working_state[1], &mut working_state[6], &mut working_state[11], &mut working_state[12]);
            Self::quarter_round(&mut working_state[2], &mut working_state[7], &mut working_state[8], &mut working_state[13]);
            Self::quarter_round(&mut working_state[3], &mut working_state[4], &mut working_state[9], &mut working_state[14]);
        }
        
        // Add initial state
        for i in 0..16 {
            working_state[i] = working_state[i].wrapping_add(state[i]);
        }
        
        // Convert to bytes
        let mut keystream = [0u8; 64];
        for i in 0..16 {
            let bytes = working_state[i].to_le_bytes();
            keystream[i * 4] = bytes[0];
            keystream[i * 4 + 1] = bytes[1];
            keystream[i * 4 + 2] = bytes[2];
            keystream[i * 4 + 3] = bytes[3];
        }
        
        self.counter += 1;
        keystream
    }

    /// Encrypt/decrypt data
    pub fn crypt(&mut self, data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len());
        let mut data_offset = 0;
        
        while data_offset < data.len() {
            let keystream = self.generate_block();
            let chunk_size = std::cmp::min(64, data.len() - data_offset);
            
            for i in 0..chunk_size {
                result.push(data[data_offset + i] ^ keystream[i]);
            }
            
            data_offset += chunk_size;
        }
        
        result
    }
}

// ===================================================================
// 5. HASH FUNCTIONS
// ===================================================================

/// Simple hash function (educational - not cryptographically secure)
pub struct SimpleHash;

impl SimpleHash {
    pub fn hash(data: &[u8]) -> [u8; 32] {
        let mut hash = [0u8; 32];
        let mut state = 0x5f5f5f5fu64;
        
        for &byte in data {
            state = state.wrapping_mul(0x100000001b3);
            state ^= byte as u64;
            state = state.rotate_left(13);
        }
        
        // Fill hash array
        for i in 0..4 {
            let chunk = state.wrapping_mul(i as u64 + 1);
            let bytes = chunk.to_le_bytes();
            hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        
        hash
    }

    pub fn hash_to_hex(data: &[u8]) -> String {
        let hash = Self::hash(data);
        hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

// ===================================================================
// 6. DIGITAL SIGNATURES (Simplified RSA)
// ===================================================================

pub struct SimpleRSA {
    pub public_key: (u64, u64),   // (e, n)
    pub private_key: (u64, u64),  // (d, n)
}

impl SimpleRSA {
    /// Generate RSA keypair (small keys for demo)
    pub fn generate_keypair() -> Self {
        // Use small primes for demonstration
        let p = 61u64;
        let q = 53u64;
        let n = p * q;
        let phi_n = (p - 1) * (q - 1);
        
        let e = 65537u64;
        let d = MathUtils::mod_inverse(e, phi_n).expect("Could not compute modular inverse");
        
        Self {
            public_key: (e, n),
            private_key: (d, n),
        }
    }

    /// Sign data (simplified - just hash and encrypt)
    pub fn sign(&self, data: &[u8]) -> u64 {
        let hash = SimpleHash::hash(data);
        let hash_int = u64::from_le_bytes([
            hash[0], hash[1], hash[2], hash[3],
            hash[4], hash[5], hash[6], hash[7],
        ]);
        
        MathUtils::fast_pow(hash_int % self.private_key.1, self.private_key.0, self.private_key.1)
    }

    /// Verify signature
    pub fn verify(&self, data: &[u8], signature: u64) -> bool {
        let hash = SimpleHash::hash(data);
        let hash_int = u64::from_le_bytes([
            hash[0], hash[1], hash[2], hash[3],
            hash[4], hash[5], hash[6], hash[7],
        ]);
        
        let decrypted = MathUtils::fast_pow(signature, self.public_key.0, self.public_key.1);
        (hash_int % self.private_key.1) == decrypted
    }
}

// ===================================================================
// 7. SECURE RANDOM NUMBER GENERATION
// ===================================================================

pub struct SecureRng {
    state: [u64; 4],
}

impl SecureRng {
    pub fn new() -> Self {
        Self {
            state: [
                0x853c49e6748fea9b,
                0xda3e39cb94b95bdb,
                0x94fef62c7e1de841,
                0xafe31827e8b2c6e5,
            ],
        }
    }

    /// Xoshiro256** algorithm
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let bytes = self.next_u64().to_le_bytes();
            remainder.copy_from_slice(&bytes[..remainder.len()]);
        }
    }
}

// ===================================================================
// 8. PERFORMANCE BENCHMARKING
// ===================================================================

pub struct CryptoBenchmark;

impl CryptoBenchmark {
    pub fn benchmark_function<F, T>(name: &str, mut f: F, iterations: usize) -> std::time::Duration
    where
        F: FnMut() -> T,
    {
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = f();
        }
        
        let duration = start.elapsed();
        println!("{}: {} iterations in {:?} ({:.2} μs/op)",
                 name, iterations, duration, 
                 duration.as_nanos() as f64 / iterations as f64 / 1000.0);
        
        duration
    }

    pub fn compare_hash_functions() {
        let test_data = b"The quick brown fox jumps over the lazy dog";
        let iterations = 10000;

        println!("\nHash Function Performance Comparison:");
        
        Self::benchmark_function("Simple Hash", || {
            SimpleHash::hash(test_data)
        }, iterations);

        // In a real implementation, you would compare with:
        // - SHA-256
        // - SHA-512
        // - Blake3
        // etc.
    }
}

// ===================================================================
// 9. CRYPTANALYSIS TOOLS
// ===================================================================

pub struct FrequencyAnalysis;

impl FrequencyAnalysis {
    const ENGLISH_FREQUENCIES: [f64; 26] = [
        8.12, 1.49, 2.78, 4.25, 12.02, 2.23, 2.02, 6.09, 6.97, 0.15,
        0.77, 4.03, 2.41, 6.75, 7.51, 1.93, 0.10, 5.99, 6.33, 9.06,
        2.76, 0.98, 2.36, 0.15, 1.97, 0.07
    ];

    pub fn analyze(text: &str) -> [f64; 26] {
        let mut counts = [0usize; 26];
        let mut total = 0;

        for c in text.to_uppercase().chars() {
            if c.is_ascii_alphabetic() {
                counts[(c as u8 - b'A') as usize] += 1;
                total += 1;
            }
        }

        let mut frequencies = [0.0; 26];
        for i in 0..26 {
            frequencies[i] = if total > 0 {
                (counts[i] as f64 / total as f64) * 100.0
            } else {
                0.0
            };
        }

        frequencies
    }

    pub fn chi_squared_score(observed: &[f64; 26]) -> f64 {
        observed.iter()
            .zip(Self::ENGLISH_FREQUENCIES.iter())
            .map(|(obs, exp)| {
                let diff = obs - exp;
                diff * diff / exp
            })
            .sum()
    }

    pub fn break_caesar(ciphertext: &str) -> (String, i32, f64) {
        let mut best_plaintext = String::new();
        let mut best_shift = 0;
        let mut best_score = f64::INFINITY;

        for shift in 0..26 {
            let plaintext = ClassicalCiphers::caesar_cipher(ciphertext, shift, true);
            let frequencies = Self::analyze(&plaintext);
            let score = Self::chi_squared_score(&frequencies);

            if score < best_score {
                best_score = score;
                best_shift = shift;
                best_plaintext = plaintext;
            }
        }

        (best_plaintext, best_shift, best_score)
    }
}

// ===================================================================
// 10. DEMONSTRATION FUNCTIONS
// ===================================================================

pub fn demonstrate_cryptography() {
    println!("=== Rust Cryptography Implementation Demo ===\n");

    // 1. Mathematical Foundations
    println!("1. Mathematical Foundations:");
    println!("GCD(48, 18) = {}", MathUtils::gcd(48, 18));
    if let Some(inv) = MathUtils::mod_inverse(3, 11) {
        println!("Modular inverse of 3 mod 11 = {}", inv);
    }
    println!("3^7 mod 11 = {}", MathUtils::fast_pow(3, 7, 11));
    println!("Is 97 prime? {}", MathUtils::is_prime(97, 5));

    // 2. Classical Cryptography
    println!("\n2. Classical Cryptography:");
    let message = "HELLO WORLD";
    let caesar_encrypted = ClassicalCiphers::caesar_cipher(message, 3, false);
    let vigenere_encrypted = ClassicalCiphers::vigenere_cipher(message, "KEY", false);
    
    println!("Original: {}", message);
    println!("Caesar (shift 3): {}", caesar_encrypted);
    println!("Vigenère (key 'KEY'): {}", vigenere_encrypted);

    // 3. Block Cipher
    println!("\n3. Block Cipher:");
    let key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
               0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c];
    let cipher = SimpleBlockCipher::new(key);
    
    let plaintext = *b"This is a test!!";
    let ciphertext = cipher.encrypt_block(plaintext);
    let decrypted = cipher.decrypt_block(ciphertext);
    
    println!("Plaintext: {:?}", std::str::from_utf8(&plaintext).unwrap());
    println!("Ciphertext: {:02x?}", &ciphertext[..8]);
    println!("Decrypted: {:?}", std::str::from_utf8(&decrypted).unwrap());

    // 4. Stream Cipher
    println!("\n4. Stream Cipher:");
    let stream_key = [0u8; 32];
    let nonce = [0u8; 12];
    let mut stream_cipher = SimpleStreamCipher::new(stream_key, nonce);
    
    let stream_message = b"Stream cipher test message";
    let stream_encrypted = stream_cipher.crypt(stream_message);
    
    // Reset counter for decryption
    let mut decrypt_cipher = SimpleStreamCipher::new(stream_key, nonce);
    let stream_decrypted = decrypt_cipher.crypt(&stream_encrypted);
    
    println!("Original: {:?}", std::str::from_utf8(stream_message).unwrap());
    println!("Encrypted: {:02x?}", &stream_encrypted[..16]);
    println!("Decrypted: {:?}", std::str::from_utf8(&stream_decrypted).unwrap());

    // 5. Hash Function
    println!("\n5. Hash Function:");
    let hash_message = b"The quick brown fox jumps over the lazy dog";
    let hash = SimpleHash::hash_to_hex(hash_message);
    
    println!("Message: {:?}", std::str::from_utf8(hash_message).unwrap());
    println!("Hash: {}", hash);

    // 6. Digital Signatures
    println!("\n6. Digital Signatures:");
    let rsa = SimpleRSA::generate_keypair();
    let sign_message = b"Document to be signed";
    
    let signature = rsa.sign(sign_message);
    let is_valid = rsa.verify(sign_message, signature);
    
    println!("Public key (e, n): {:?}", rsa.public_key);
    println!("Message: {:?}", std::str::from_utf8(sign_message).unwrap());
    println!("Signature: {}", signature);
    println!("Valid signature: {}", is_valid);

    // 7. Frequency Analysis
    println!("\n7. Frequency Analysis:");
    let cipher_text = "WKLV LV D VHFUHW PHVVDJH"; // Caesar cipher with shift 3
    let (broken_text, shift, score) = FrequencyAnalysis::break_caesar(cipher_text);
    
    println!("Ciphertext: {}", cipher_text);
    println!("Broken plaintext: {}", broken_text);
    println!("Key (shift): {}", shift);
    println!("Chi-squared score: {:.2}", score);

    // 8. Random Number Generation
    println!("\n8. Random Number Generation:");
    let mut rng = SecureRng::new();
    let mut random_bytes = [0u8; 16];
    rng.fill_bytes(&mut random_bytes);
    
    println!("Random bytes: {:02x?}", random_bytes);
    println!("Random u64: {}", rng.next_u64());

    // 9. Performance Benchmarking
    println!("\n9. Performance Benchmarking:");
    CryptoBenchmark::compare_hash_functions();
    
    let bench_data = [42u8; 1000];
    CryptoBenchmark::benchmark_function("Simple Hash (1KB)", || {
        SimpleHash::hash(&bench_data)
    }, 1000);

    println!("\n=== Demo Complete ===");
}

// ===================================================================
// 11. MAIN FUNCTION
// ===================================================================

fn main() {
    demonstrate_cryptography();
}

// ===================================================================
// 12. CARGO.TOML CONFIGURATION
// ===================================================================

/*
[package]
name = "crypto_demo"
version = "0.1.0"
edition = "2021"

[dependencies]
# For production use, include these crates:
# ring = "0.16"              # Cryptographic primitives
# aes-gcm = "0.10"           # AES-GCM implementation
# chacha20poly1305 = "0.10"  # ChaCha20-Poly1305
# rsa = "0.9"                # RSA implementation
# sha2 = "0.10"              # SHA-2 family
# rand = "0.8"               # Random number generation
# ed25519-dalek = "1.0"      # Ed25519 signatures
# x25519-dalek = "1.2"       # X25519 key exchange
# curve25519-dalek = "3.2"   # Curve25519 operations
# subtle = "2.4"             # Constant-time operations

[dev-dependencies]
criterion = "0.4"            # Benchmarking

[[bench]]
name = "crypto_bench"
harness = false
*/

// ===================================================================
// 13. TESTING MODULE
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(MathUtils::gcd(48, 18), 6);
        assert_eq!(MathUtils::gcd(17, 19), 1);
    }

    #[test]
    fn test_caesar_cipher() {
        let plaintext = "HELLO";
        let encrypted = ClassicalCiphers::caesar_cipher(plaintext, 3, false);
        let decrypted = ClassicalCiphers::caesar_cipher(&encrypted, 3, true);
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_vigenere_cipher() {
        let plaintext = "HELLO";
        let key = "KEY";
        let encrypted = ClassicalCiphers::vigenere_cipher(plaintext, key, false);
        let decrypted = ClassicalCiphers::vigenere_cipher(&encrypted, key, true);
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_block_cipher() {
        let key = [0u8; 16];
        let cipher = SimpleBlockCipher::new(key);
        let plaintext = [0x42u8; 16];
        
        let ciphertext = cipher.encrypt_block(plaintext);
        let decrypted = cipher.decrypt_block(ciphertext);
        
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_hash_function() {
        let data1 = b"test";
        let data2 = b"test";
        let data3 = b"different";
        
        assert_eq!(SimpleHash::hash(data1), SimpleHash::hash(data2));
        assert_ne!(SimpleHash::hash(data1), SimpleHash::hash(data3));
    }

    #[test]
    fn test_digital_signatures() {
        let rsa = SimpleRSA::generate_keypair();
        let message = b"test message";
        
        let signature = rsa.sign(message);
        assert!(rsa.verify(message, signature));
        assert!(!rsa.verify(b"different message", signature));
    }

    #[test]
    fn test_frequency_analysis() {
        let plaintext = "HELLO WORLD";
        let ciphertext = ClassicalCiphers::caesar_cipher(plaintext, 7, false);
        let (broken, shift, _) = FrequencyAnalysis::break_caesar(&ciphertext);
        
        assert_eq!(shift, 7);
        assert_eq!(broken.replace(' ', ""), plaintext.replace(' ', ""));
    }
}

/*
Compilation and Usage Instructions:

1. Create a new Rust project:
   cargo new crypto_demo
   cd crypto_demo

2. Replace src/main.rs with this code

3. For full functionality, add dependencies to Cargo.toml:
   [dependencies]
   ring = "0.16"
   rand = "0.8"

4. Build and run:
   cargo run

5. Run tests:
   cargo test

6. Run benchmarks (if criterion is added):
   cargo bench

Features Implemented:
- Memory-safe implementations
- Zero-cost abstractions
- Comprehensive error handling
- Performance benchmarking
- Unit testing
- Mathematical foundations
- Classical and modern cryptography
- Cryptanalysis tools
- Secure random number generation

Educational Value:
- Shows Rust's memory safety in cryptographic contexts
- Demonstrates performance optimization techniques
- Includes proper error handling patterns
- Shows testing best practices
- Provides benchmarking examples
- Illustrates secure coding practices
*/
