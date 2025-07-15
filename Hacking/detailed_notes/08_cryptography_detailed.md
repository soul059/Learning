# Detailed Notes: Cryptography Fundamentals

## Mathematical Foundations of Cryptography

### Number Theory Basics

#### Prime Numbers and Factorization
**Prime Number Properties:**
- Only divisible by 1 and itself
- Infinite number of primes (Euclid's theorem)
- Difficulty of factoring large composite numbers is basis for RSA

**Prime Number Generation:**
```python
import random

def is_prime(n, k=5):
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
        a = random.randrange(2, n - 1)
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

def generate_prime(bits):
    """Generate a prime number with specified bit length"""
    while True:
        num = random.getrandbits(bits)
        num |= (1 << bits - 1) | 1  # Set MSB and LSB
        if is_prime(num):
            return num
```

#### Modular Arithmetic
**Basic Operations:**
```python
# Modular addition
def mod_add(a, b, m):
    return (a + b) % m

# Modular multiplication
def mod_mul(a, b, m):
    return (a * b) % m

# Modular exponentiation (fast)
def mod_pow(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

# Extended Euclidean Algorithm
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

# Modular multiplicative inverse
def mod_inverse(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        return None  # Inverse doesn't exist
    return (x % m + m) % m
```

### Symmetric Cryptography - Deep Analysis

#### Advanced Encryption Standard (AES)

**AES Algorithm Structure:**
```python
# AES S-Box (Substitution Box)
S_BOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    # ... (complete S-box would have 256 values)
]

def sub_bytes(state):
    """Apply S-box substitution to each byte"""
    for i in range(4):
        for j in range(4):
            state[i][j] = S_BOX[state[i][j]]
    return state

def shift_rows(state):
    """Shift rows in the state matrix"""
    # Row 0: no shift
    # Row 1: shift left by 1
    state[1] = state[1][1:] + state[1][:1]
    # Row 2: shift left by 2
    state[2] = state[2][2:] + state[2][:2]
    # Row 3: shift left by 3
    state[3] = state[3][3:] + state[3][:3]
    return state

def mix_columns(state):
    """Mix columns using Galois Field arithmetic"""
    # Simplified version - actual implementation uses GF(2^8)
    for c in range(4):
        a = [state[r][c] for r in range(4)]
        state[0][c] = galois_mul(0x02, a[0]) ^ galois_mul(0x03, a[1]) ^ a[2] ^ a[3]
        state[1][c] = a[0] ^ galois_mul(0x02, a[1]) ^ galois_mul(0x03, a[2]) ^ a[3]
        state[2][c] = a[0] ^ a[1] ^ galois_mul(0x02, a[2]) ^ galois_mul(0x03, a[3])
        state[3][c] = galois_mul(0x03, a[0]) ^ a[1] ^ a[2] ^ galois_mul(0x02, a[3])
    return state

# Complete AES implementation with proper key scheduling
class AES:
    def __init__(self, key):
        self.key = key
        self.round_keys = self.key_expansion(key)
    
    def encrypt_block(self, plaintext):
        state = self.bytes_to_matrix(plaintext)
        state = self.add_round_key(state, self.round_keys[0])
        
        for round in range(1, 10):
            state = sub_bytes(state)
            state = shift_rows(state)
            state = mix_columns(state)
            state = self.add_round_key(state, self.round_keys[round])
        
        # Final round (no MixColumns)
        state = sub_bytes(state)
        state = shift_rows(state)
        state = self.add_round_key(state, self.round_keys[10])
        
        return self.matrix_to_bytes(state)
```

#### Block Cipher Modes of Operation

**1. Electronic Codebook (ECB) Mode:**
```python
def ecb_encrypt(plaintext, key, cipher_func):
    """ECB mode - NOT RECOMMENDED for most use cases"""
    block_size = 16  # AES block size
    ciphertext = b''
    
    for i in range(0, len(plaintext), block_size):
        block = plaintext[i:i+block_size]
        if len(block) < block_size:
            block = pkcs7_pad(block, block_size)
        ciphertext += cipher_func(block, key)
    
    return ciphertext

# ECB weakness: identical plaintexts produce identical ciphertexts
```

**2. Cipher Block Chaining (CBC) Mode:**
```python
import os

def cbc_encrypt(plaintext, key, cipher_func):
    """CBC mode - secure when used with random IV"""
    block_size = 16
    iv = os.urandom(block_size)  # Random initialization vector
    ciphertext = iv  # Prepend IV to ciphertext
    
    previous_block = iv
    for i in range(0, len(plaintext), block_size):
        block = plaintext[i:i+block_size]
        if len(block) < block_size:
            block = pkcs7_pad(block, block_size)
        
        # XOR with previous ciphertext block
        xored_block = xor_bytes(block, previous_block)
        encrypted_block = cipher_func(xored_block, key)
        ciphertext += encrypted_block
        previous_block = encrypted_block
    
    return ciphertext

def cbc_decrypt(ciphertext, key, decipher_func):
    """CBC decryption"""
    block_size = 16
    iv = ciphertext[:block_size]
    ciphertext = ciphertext[block_size:]
    
    plaintext = b''
    previous_block = iv
    
    for i in range(0, len(ciphertext), block_size):
        block = ciphertext[i:i+block_size]
        decrypted_block = decipher_func(block, key)
        plaintext_block = xor_bytes(decrypted_block, previous_block)
        plaintext += plaintext_block
        previous_block = block
    
    return pkcs7_unpad(plaintext)
```

**3. Galois/Counter Mode (GCM):**
```python
def gcm_encrypt(plaintext, key, aad=b''):
    """GCM mode - provides both confidentiality and authenticity"""
    block_size = 16
    iv = os.urandom(12)  # 96-bit IV for GCM
    
    # Initialize counter
    counter = iv + b'\x00\x00\x00\x01'
    
    # Encrypt plaintext using CTR mode
    ciphertext = b''
    for i in range(0, len(plaintext), block_size):
        block = plaintext[i:i+block_size]
        keystream = aes_encrypt(counter, key)
        
        if len(block) < block_size:
            keystream = keystream[:len(block)]
        
        ciphertext += xor_bytes(block, keystream)
        counter = increment_counter(counter)
    
    # Calculate authentication tag using GHASH
    auth_tag = ghash(aad + ciphertext, key, iv)
    
    return iv + ciphertext + auth_tag
```

#### Stream Ciphers

**ChaCha20 Implementation:**
```python
def chacha20_quarter_round(a, b, c, d, state):
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

def chacha20_block(key, counter, nonce):
    """Generate a ChaCha20 keystream block"""
    # ChaCha20 constants
    constants = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
    
    # Initialize state
    state = constants[:]
    state.extend(struct.unpack('<8I', key))
    state.append(counter)
    state.extend(struct.unpack('<3I', nonce))
    
    working_state = state[:]
    
    # 20 rounds (10 double rounds)
    for _ in range(10):
        # Column rounds
        chacha20_quarter_round(0, 4, 8, 12, working_state)
        chacha20_quarter_round(1, 5, 9, 13, working_state)
        chacha20_quarter_round(2, 6, 10, 14, working_state)
        chacha20_quarter_round(3, 7, 11, 15, working_state)
        
        # Diagonal rounds
        chacha20_quarter_round(0, 5, 10, 15, working_state)
        chacha20_quarter_round(1, 6, 11, 12, working_state)
        chacha20_quarter_round(2, 7, 8, 13, working_state)
        chacha20_quarter_round(3, 4, 9, 14, working_state)
    
    # Add original state
    for i in range(16):
        working_state[i] = (working_state[i] + state[i]) & 0xffffffff
    
    return struct.pack('<16I', *working_state)
```

### Asymmetric Cryptography - Comprehensive Analysis

#### RSA Algorithm Implementation

**RSA Key Generation:**
```python
import random

def generate_rsa_keypair(bits):
    """Generate RSA public/private key pair"""
    # Step 1: Generate two large prime numbers
    p = generate_prime(bits // 2)
    q = generate_prime(bits // 2)
    
    # Step 2: Compute n = p * q
    n = p * q
    
    # Step 3: Compute Euler's totient function
    phi_n = (p - 1) * (q - 1)
    
    # Step 4: Choose public exponent e
    e = 65537  # Common choice (2^16 + 1)
    while gcd(e, phi_n) != 1:
        e += 2
    
    # Step 5: Compute private exponent d
    d = mod_inverse(e, phi_n)
    
    # Public key: (n, e)
    # Private key: (n, d)
    return {'public': (n, e), 'private': (n, d), 'p': p, 'q': q}

def rsa_encrypt(message, public_key):
    """RSA encryption"""
    n, e = public_key
    # Convert message to integer (proper padding should be used)
    m = int.from_bytes(message, 'big')
    if m >= n:
        raise ValueError("Message too large for key size")
    
    # c = m^e mod n
    ciphertext = pow(m, e, n)
    return ciphertext.to_bytes((n.bit_length() + 7) // 8, 'big')

def rsa_decrypt(ciphertext, private_key):
    """RSA decryption"""
    n, d = private_key
    c = int.from_bytes(ciphertext, 'big')
    
    # m = c^d mod n
    message = pow(c, d, n)
    return message.to_bytes((message.bit_length() + 7) // 8, 'big')
```

**RSA Padding Schemes:**

**PKCS#1 v1.5 Padding:**
```python
def pkcs1_v15_pad(message, key_length):
    """PKCS#1 v1.5 padding for encryption"""
    if len(message) > key_length - 11:
        raise ValueError("Message too long")
    
    ps_length = key_length - len(message) - 3
    ps = bytes([random.randint(1, 255) for _ in range(ps_length)])
    
    return b'\x00\x02' + ps + b'\x00' + message

def pkcs1_v15_unpad(padded_message):
    """Remove PKCS#1 v1.5 padding"""
    if padded_message[0] != 0 or padded_message[1] != 2:
        raise ValueError("Invalid padding")
    
    separator_index = padded_message.index(0, 2)
    return padded_message[separator_index + 1:]
```

**Optimal Asymmetric Encryption Padding (OAEP):**
```python
import hashlib

def mgf1(seed, length, hash_func=hashlib.sha256):
    """Mask Generation Function based on hash function"""
    if length >= (1 << 32) * hash_func().digest_size:
        raise ValueError("MGF1 mask too long")
    
    T = b''
    counter = 0
    while len(T) < length:
        C = counter.to_bytes(4, 'big')
        T += hash_func(seed + C).digest()
        counter += 1
    
    return T[:length]

def oaep_pad(message, key_length, label=b'', hash_func=hashlib.sha256):
    """OAEP padding"""
    hlen = hash_func().digest_size
    if len(message) > key_length - 2 * hlen - 2:
        raise ValueError("Message too long")
    
    # Step 1: Length checking done above
    # Step 2: EME-OAEP encoding
    lhash = hash_func(label).digest()
    ps_length = key_length - len(message) - 2 * hlen - 2
    ps = b'\x00' * ps_length
    db = lhash + ps + b'\x01' + message
    
    seed = os.urandom(hlen)
    db_mask = mgf1(seed, key_length - hlen - 1, hash_func)
    masked_db = xor_bytes(db, db_mask)
    
    seed_mask = mgf1(masked_db, hlen, hash_func)
    masked_seed = xor_bytes(seed, seed_mask)
    
    return b'\x00' + masked_seed + masked_db
```

#### Elliptic Curve Cryptography (ECC)

**Elliptic Curve Point Operations:**
```python
class EllipticCurve:
    def __init__(self, a, b, p):
        """Elliptic curve y^2 = x^3 + ax + b (mod p)"""
        self.a = a
        self.b = b
        self.p = p
    
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
                s = (3 * x1 * x1 + self.a) * mod_inverse(2 * y1, self.p) % self.p
            else:
                # P + (-P) = O (point at infinity)
                return None
        else:
            # Point addition
            s = (y2 - y1) * mod_inverse(x2 - x1, self.p) % self.p
        
        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_mult(self, k, P):
        """Scalar multiplication k*P using double-and-add"""
        if k == 0:
            return None  # Point at infinity
        if k == 1:
            return P
        
        result = None
        addend = P
        
        while k:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_add(addend, addend)
            k >>= 1
        
        return result

# NIST P-256 curve parameters
p256 = EllipticCurve(
    a=0xffffffff00000001000000000000000000000000fffffffffffffffffffffffc,
    b=0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b,
    p=0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
)

# Base point G
G = (
    0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296,
    0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5
)

# Order of the base point
n = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
```

**ECDSA Digital Signatures:**
```python
def ecdsa_sign(message, private_key, curve, base_point, order):
    """Generate ECDSA signature"""
    z = int.from_bytes(hashlib.sha256(message).digest(), 'big')
    
    while True:
        k = random.randint(1, order - 1)
        x, y = curve.scalar_mult(k, base_point)
        r = x % order
        
        if r == 0:
            continue
        
        k_inv = mod_inverse(k, order)
        s = (k_inv * (z + r * private_key)) % order
        
        if s == 0:
            continue
        
        return (r, s)

def ecdsa_verify(message, signature, public_key, curve, base_point, order):
    """Verify ECDSA signature"""
    r, s = signature
    
    if not (1 <= r < order and 1 <= s < order):
        return False
    
    z = int.from_bytes(hashlib.sha256(message).digest(), 'big')
    
    s_inv = mod_inverse(s, order)
    u1 = (z * s_inv) % order
    u2 = (r * s_inv) % order
    
    point1 = curve.scalar_mult(u1, base_point)
    point2 = curve.scalar_mult(u2, public_key)
    x, y = curve.point_add(point1, point2)
    
    return (x % order) == r
```

#### Diffie-Hellman Key Exchange

**Classic Diffie-Hellman:**
```python
def diffie_hellman_keygen(p, g):
    """Generate Diffie-Hellman key pair"""
    private_key = random.randint(2, p - 2)
    public_key = pow(g, private_key, p)
    return private_key, public_key

def diffie_hellman_shared_secret(other_public_key, my_private_key, p):
    """Compute shared secret"""
    return pow(other_public_key, my_private_key, p)

# Example usage
p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
g = 2

alice_private, alice_public = diffie_hellman_keygen(p, g)
bob_private, bob_public = diffie_hellman_keygen(p, g)

# Both parties compute the same shared secret
alice_shared = diffie_hellman_shared_secret(bob_public, alice_private, p)
bob_shared = diffie_hellman_shared_secret(alice_public, bob_private, p)

assert alice_shared == bob_shared
```

**Elliptic Curve Diffie-Hellman (ECDH):**
```python
def ecdh_keygen(curve, base_point, order):
    """Generate ECDH key pair"""
    private_key = random.randint(1, order - 1)
    public_key = curve.scalar_mult(private_key, base_point)
    return private_key, public_key

def ecdh_shared_secret(other_public_key, my_private_key, curve):
    """Compute ECDH shared secret"""
    shared_point = curve.scalar_mult(my_private_key, other_public_key)
    # Use x-coordinate as shared secret
    return shared_point[0] if shared_point else 0

# Example with P-256
alice_private, alice_public = ecdh_keygen(p256, G, n)
bob_private, bob_public = ecdh_keygen(p256, G, n)

alice_shared = ecdh_shared_secret(bob_public, alice_private, p256)
bob_shared = ecdh_shared_secret(alice_public, bob_private, p256)

assert alice_shared == bob_shared
```

### Hash Functions and Message Authentication

#### SHA-256 Implementation

**SHA-256 Algorithm:**
```python
import struct

def sha256_constants():
    """Generate SHA-256 constants"""
    # First 32 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131]
    
    # Square roots of first 8 primes (initial hash values)
    h = []
    for p in primes[:8]:
        frac = (p ** 0.5) % 1
        h.append(int(frac * (2**32)))
    
    # Cube roots of first 64 primes (round constants)
    k = []
    for p in primes[:64]:
        frac = (p ** (1/3)) % 1
        k.append(int(frac * (2**32)))
    
    return h, k

def sha256_pad(message):
    """Pad message according to SHA-256 specification"""
    msg_len = len(message)
    message += b'\x80'
    
    # Pad to 512 bits - 64 bits (for length)
    while len(message) % 64 != 56:
        message += b'\x00'
    
    # Append original length as 64-bit big-endian integer
    message += struct.pack('>Q', msg_len * 8)
    return message

def sha256_compress(chunk, h):
    """SHA-256 compression function"""
    # Constants
    _, k = sha256_constants()
    
    # Create message schedule
    w = list(struct.unpack('>16I', chunk))
    for i in range(16, 64):
        s0 = right_rotate(w[i-15], 7) ^ right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = right_rotate(w[i-2], 17) ^ right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)
    
    # Initialize working variables
    a, b, c, d, e, f, g, h_val = h
    
    # Main loop
    for i in range(64):
        s1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)
        ch = (e & f) ^ (~e & g)
        temp1 = (h_val + s1 + ch + k[i] + w[i]) & 0xffffffff
        
        s0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (s0 + maj) & 0xffffffff
        
        h_val = g
        g = f
        f = e
        e = (d + temp1) & 0xffffffff
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xffffffff
    
    # Add compressed chunk to current hash
    return [(h[i] + [a, b, c, d, e, f, g, h_val][i]) & 0xffffffff for i in range(8)]

def sha256(message):
    """Complete SHA-256 hash function"""
    h, _ = sha256_constants()
    padded_msg = sha256_pad(message)
    
    # Process message in 512-bit chunks
    for i in range(0, len(padded_msg), 64):
        chunk = padded_msg[i:i+64]
        h = sha256_compress(chunk, h)
    
    # Produce final hash value
    return struct.pack('>8I', *h)

def right_rotate(value, amount):
    """Right rotate a 32-bit integer"""
    return ((value >> amount) | (value << (32 - amount))) & 0xffffffff
```

#### HMAC Implementation

**Hash-based Message Authentication Code:**
```python
def hmac(key, message, hash_func, block_size=64):
    """HMAC implementation"""
    if len(key) > block_size:
        key = hash_func(key).digest()
    
    if len(key) < block_size:
        key = key + b'\x00' * (block_size - len(key))
    
    opad = bytes([0x5C] * block_size)
    ipad = bytes([0x36] * block_size)
    
    o_key_pad = xor_bytes(key, opad)
    i_key_pad = xor_bytes(key, ipad)
    
    inner_hash = hash_func(i_key_pad + message).digest()
    return hash_func(o_key_pad + inner_hash).digest()

# Example usage
import hashlib
key = b'secret_key'
message = b'Hello, World!'
mac = hmac(key, message, hashlib.sha256)
```

### Password Hashing and Key Derivation

#### PBKDF2 Implementation

**Password-Based Key Derivation Function 2:**
```python
def pbkdf2(password, salt, iterations, dk_length, hash_func=hashlib.sha256):
    """PBKDF2 key derivation function"""
    def prf(data):
        return hmac(password, data, hash_func)
    
    h_length = hash_func().digest_size
    blocks_needed = (dk_length + h_length - 1) // h_length
    
    derived_key = b''
    for i in range(1, blocks_needed + 1):
        u = prf(salt + struct.pack('>I', i))
        result = u
        
        for _ in range(iterations - 1):
            u = prf(u)
            result = xor_bytes(result, u)
        
        derived_key += result
    
    return derived_key[:dk_length]

# Secure password hashing
def hash_password(password, salt=None, iterations=100000):
    """Hash password using PBKDF2"""
    if salt is None:
        salt = os.urandom(32)
    
    key = pbkdf2(password.encode(), salt, iterations, 32)
    return salt + key

def verify_password(password, stored_hash):
    """Verify password against stored hash"""
    salt = stored_hash[:32]
    stored_key = stored_hash[32:]
    
    key = pbkdf2(password.encode(), salt, 100000, 32)
    return key == stored_key
```

#### Argon2 (Modern Password Hashing)

**Argon2 Key Concepts:**
```python
# Simplified Argon2 implementation (conceptual)
def argon2(password, salt, t_cost=3, m_cost=12, p_cost=1):
    """
    Argon2 password hashing
    t_cost: time cost (iterations)
    m_cost: memory cost (2^m_cost KB)
    p_cost: parallelism degree
    """
    # Memory allocation
    memory_size = 2 ** m_cost
    memory_blocks = [b'\x00' * 1024] * memory_size
    
    # Initial hash
    initial_hash = hashlib.blake2b(
        password + salt + 
        struct.pack('<I', p_cost) +
        struct.pack('<I', 32) +  # tag length
        struct.pack('<I', memory_size) +
        struct.pack('<I', t_cost) +
        struct.pack('<I', 2) +   # Argon2id
        struct.pack('<I', 0x13)  # version
    ).digest()
    
    # Fill memory blocks (simplified)
    for i in range(memory_size):
        if i == 0:
            memory_blocks[i] = hashlib.blake2b(initial_hash + struct.pack('<I', 0)).digest()
        else:
            memory_blocks[i] = hashlib.blake2b(memory_blocks[i-1] + struct.pack('<I', i)).digest()
    
    # Perform iterations
    for t in range(t_cost):
        for i in range(memory_size):
            # Simplified mixing function
            prev_block = memory_blocks[(i - 1) % memory_size]
            ref_block = memory_blocks[int.from_bytes(prev_block[:4], 'little') % memory_size]
            memory_blocks[i] = hashlib.blake2b(prev_block + ref_block).digest()[:1024]
    
    # Final hash
    return hashlib.blake2b(b''.join(memory_blocks)).digest()[:32]
```

### Cryptographic Attacks and Vulnerabilities

#### Side-Channel Attacks

**Timing Attack on RSA:**
```python
import time

def vulnerable_rsa_decrypt(ciphertext, private_key):
    """Vulnerable RSA decryption (timing attack)"""
    n, d = private_key
    c = int.from_bytes(ciphertext, 'big')
    
    # Timing depends on the value of d
    result = pow(c, d, n)  # Time varies based on bit pattern of d
    return result

def constant_time_rsa_decrypt(ciphertext, private_key):
    """Constant-time RSA decryption using CRT"""
    n, d, p, q = private_key  # Include p and q for CRT
    
    # Chinese Remainder Theorem speedup with blinding
    c = int.from_bytes(ciphertext, 'big')
    
    # Generate random blinding factor
    r = random.randint(2, n - 1)
    r_inv = mod_inverse(r, n)
    
    # Blind the ciphertext
    blinded_c = (c * pow(r, 65537, n)) % n  # e = 65537
    
    # CRT decryption
    dp = d % (p - 1)
    dq = d % (q - 1)
    qinv = mod_inverse(q, p)
    
    m1 = pow(blinded_c, dp, p)
    m2 = pow(blinded_c, dq, q)
    h = (qinv * (m1 - m2)) % p
    m = m2 + h * q
    
    # Unblind the result
    return (m * r_inv) % n
```

**Power Analysis Attack Mitigation:**
```python
def secure_scalar_multiplication(k, P, curve):
    """Scalar multiplication resistant to power analysis"""
    # Montgomery ladder for constant-time execution
    if k == 0:
        return None
    
    R0 = None  # Point at infinity
    R1 = P
    
    bit_length = k.bit_length()
    for i in range(bit_length - 1, -1, -1):
        if (k >> i) & 1:
            R0 = curve.point_add(R0, R1)
            R1 = curve.point_add(R1, R1)
        else:
            R1 = curve.point_add(R0, R1)
            R0 = curve.point_add(R0, R0)
    
    return R0
```

#### Cryptanalytic Attacks

**Frequency Analysis (Classical Ciphers):**
```python
def frequency_analysis(ciphertext):
    """Analyze letter frequency for substitution ciphers"""
    # English letter frequencies
    english_freq = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97, 'N': 6.95,
        'S': 6.28, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
        'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
        'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
        'Q': 0.10, 'Z': 0.07
    }
    
    # Count letter frequencies in ciphertext
    cipher_freq = {}
    total_letters = 0
    
    for char in ciphertext.upper():
        if char.isalpha():
            cipher_freq[char] = cipher_freq.get(char, 0) + 1
            total_letters += 1
    
    # Convert to percentages
    for char in cipher_freq:
        cipher_freq[char] = (cipher_freq[char] / total_letters) * 100
    
    # Sort by frequency
    sorted_cipher = sorted(cipher_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_english = sorted(english_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Suggest substitution mapping
    mapping = {}
    for i, (cipher_char, _) in enumerate(sorted_cipher):
        if i < len(sorted_english):
            english_char, _ = sorted_english[i]
            mapping[cipher_char] = english_char
    
    return mapping, cipher_freq
```

**Birthday Attack on Hash Functions:**
```python
def birthday_attack_simulation(hash_bits=32):
    """Simulate birthday attack on hash function"""
    import math
    
    # Birthday paradox: ~sqrt(2^n) attempts needed
    expected_attempts = int(math.sqrt(2 ** hash_bits))
    
    seen_hashes = {}
    attempts = 0
    
    while True:
        # Generate random message
        message = random.randbits(64).to_bytes(8, 'big')
        hash_value = hashlib.sha256(message).digest()[:hash_bits//8]
        
        attempts += 1
        
        if hash_value in seen_hashes:
            print(f"Collision found after {attempts} attempts")
            print(f"Message 1: {seen_hashes[hash_value]}")
            print(f"Message 2: {message}")
            print(f"Hash: {hash_value.hex()}")
            break
        
        seen_hashes[hash_value] = message
        
        if attempts > expected_attempts * 10:  # Safety limit
            print("No collision found within reasonable attempts")
            break
    
    return attempts
```

### Public Key Infrastructure (PKI)

#### Certificate Authority Implementation

**X.509 Certificate Generation:**
```python
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

def create_ca_certificate():
    """Create a Certificate Authority certificate"""
    # Generate CA private key
    ca_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096
    )
    
    # Create CA certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Example CA"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Example Root CA"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        ca_private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=3650)  # 10 years
    ).add_extension(
        x509.SubjectKeyIdentifier.from_public_key(ca_private_key.public_key()),
        critical=False,
    ).add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_private_key.public_key()),
        critical=False,
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None),
        critical=True,
    ).add_extension(
        x509.KeyUsage(
            key_cert_sign=True,
            crl_sign=True,
            digital_signature=False,
            key_encipherment=False,
            key_agreement=False,
            data_encipherment=False,
            content_commitment=False,
            encipher_only=False,
            decipher_only=False
        ),
        critical=True,
    ).sign(ca_private_key, hashes.SHA256())
    
    return cert, ca_private_key

def create_server_certificate(ca_cert, ca_private_key, server_name):
    """Create a server certificate signed by CA"""
    # Generate server private key
    server_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    # Create server certificate
    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Example Corp"),
        x509.NameAttribute(NameOID.COMMON_NAME, server_name),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        ca_cert.issuer
    ).public_key(
        server_private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)  # 1 year
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(server_name),
        ]),
        critical=False,
    ).add_extension(
        x509.BasicConstraints(ca=False, path_length=None),
        critical=True,
    ).add_extension(
        x509.KeyUsage(
            key_cert_sign=False,
            crl_sign=False,
            digital_signature=True,
            key_encipherment=True,
            key_agreement=False,
            data_encipherment=False,
            content_commitment=False,
            encipher_only=False,
            decipher_only=False
        ),
        critical=True,
    ).add_extension(
        x509.ExtendedKeyUsage([
            x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
        ]),
        critical=True,
    ).sign(ca_private_key, hashes.SHA256())
    
    return cert, server_private_key
```

#### Certificate Validation

**Certificate Chain Validation:**
```python
def validate_certificate_chain(cert_chain, trusted_ca_certs):
    """Validate certificate chain"""
    if not cert_chain:
        return False, "Empty certificate chain"
    
    # Start with the end-entity certificate
    current_cert = cert_chain[0]
    
    # Check each certificate in the chain
    for i, cert in enumerate(cert_chain):
        # Check certificate validity period
        now = datetime.datetime.utcnow()
        if now < cert.not_valid_before or now > cert.not_valid_after:
            return False, f"Certificate {i} is expired or not yet valid"
        
        # For non-root certificates, verify signature
        if i < len(cert_chain) - 1:
            issuer_cert = cert_chain[i + 1]
            try:
                issuer_cert.public_key().verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    cert.signature_algorithm_oid._name
                )
            except Exception as e:
                return False, f"Certificate {i} signature verification failed: {e}"
        else:
            # Root certificate - check against trusted CAs
            if cert not in trusted_ca_certs:
                return False, "Root certificate not in trusted CA list"
    
    return True, "Certificate chain is valid"

def check_certificate_revocation(cert, crl_url=None):
    """Check if certificate is revoked"""
    # In practice, this would check CRL or OCSP
    # Simplified implementation
    if crl_url:
        # Download and parse CRL
        # Check if certificate serial number is in CRL
        pass
    
    # For demonstration, assume certificate is not revoked
    return False  # Not revoked
```
