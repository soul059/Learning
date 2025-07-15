# 🏛️ Classical Cryptography: Ancient to Pre-Computer Era

## 📖 Table of Contents
1. [Ancient Cryptographic Methods](#ancient-cryptographic-methods)
2. [Substitution Ciphers](#substitution-ciphers)
3. [Transposition Ciphers](#transposition-ciphers)
4. [Polyalphabetic Ciphers](#polyalphabetic-ciphers)
5. [Mechanical Cipher Machines](#mechanical-cipher-machines)
6. [Cryptanalysis Techniques](#cryptanalysis-techniques)
7. [Historical Impact](#historical-impact)

---

## 🏺 Ancient Cryptographic Methods

### Steganography: The Art of Hiding

#### Invisible Inks
- **Ancient Methods**: Lemon juice, milk, onion juice
- **Modern Detection**: UV light, heat application
- **Security**: Relies on obscurity, not cryptographic strength

```python
def invisible_ink_simulation(message, method="lemon"):
    """Simulate invisible ink encoding"""
    methods = {
        "lemon": lambda x: f"[INVISIBLE: {x}]",
        "milk": lambda x: f"[HEAT_REVEAL: {x}]",
        "uv": lambda x: f"[UV_VISIBLE: {x}]"
    }
    
    return methods.get(method, methods["lemon"])(message)

# Example
hidden_message = invisible_ink_simulation("ATTACK AT DAWN", "lemon")
```

#### Null Ciphers
Hide messages in seemingly innocent text by using specific letters, words, or positions.

```python
def null_cipher_encode(message, cover_text):
    """Hide message using first letters of words"""
    words = cover_text.split()
    encoded = []
    
    for i, char in enumerate(message.upper()):
        if i < len(words):
            # Ensure word starts with required character
            if words[i][0].upper() != char:
                # Find or create word starting with char
                word_map = {
                    'A': 'ATTACK', 'B': 'BRING', 'C': 'COME', 'D': 'DOWN',
                    'E': 'EVERY', 'F': 'FIGHT', 'G': 'GO', 'H': 'HELP',
                    'I': 'IN', 'J': 'JUMP', 'K': 'KEEP', 'L': 'LOOK',
                    'M': 'MOVE', 'N': 'NOW', 'O': 'OVER', 'P': 'PUSH',
                    'Q': 'QUICK', 'R': 'RUN', 'S': 'STOP', 'T': 'TAKE',
                    'U': 'UP', 'V': 'VERY', 'W': 'WHEN', 'X': 'X-RAY',
                    'Y': 'YES', 'Z': 'ZERO'
                }
                encoded.append(word_map.get(char, words[i]))
            else:
                encoded.append(words[i])
    
    return ' '.join(encoded)

def null_cipher_decode(encoded_text):
    """Extract hidden message from first letters"""
    words = encoded_text.split()
    return ''.join([word[0].upper() for word in words])

# Example
cover = "Bring every soldier to defend our position immediately"
message = "HELP"
encoded = null_cipher_encode(message, cover)
decoded = null_cipher_decode(encoded)
```

### Transposition Methods

#### Spartan Scytale (7th Century BCE)
The first known military cryptographic device.

```python
class Scytale:
    def __init__(self, diameter):
        self.diameter = diameter
    
    def encrypt(self, plaintext):
        """Encrypt using scytale method"""
        # Remove spaces and pad if necessary
        text = plaintext.replace(' ', '').upper()
        
        # Calculate required padding
        remainder = len(text) % self.diameter
        if remainder:
            text += 'X' * (self.diameter - remainder)
        
        # Arrange in columns
        rows = len(text) // self.diameter
        matrix = []
        
        for i in range(rows):
            row = text[i * self.diameter:(i + 1) * self.diameter]
            matrix.append(list(row))
        
        # Read column-wise
        ciphertext = ''
        for col in range(self.diameter):
            for row in range(rows):
                ciphertext += matrix[row][col]
        
        return ciphertext
    
    def decrypt(self, ciphertext):
        """Decrypt scytale cipher"""
        rows = len(ciphertext) // self.diameter
        matrix = [['' for _ in range(self.diameter)] for _ in range(rows)]
        
        # Fill matrix column-wise
        index = 0
        for col in range(self.diameter):
            for row in range(rows):
                matrix[row][col] = ciphertext[index]
                index += 1
        
        # Read row-wise
        plaintext = ''
        for row in range(rows):
            plaintext += ''.join(matrix[row])
        
        return plaintext.rstrip('X')

# Example
scytale = Scytale(4)
encrypted = scytale.encrypt("ATTACK AT DAWN")
print(f"Encrypted: {encrypted}")
decrypted = scytale.decrypt(encrypted)
print(f"Decrypted: {decrypted}")
```

---

## 🔤 Substitution Ciphers

### Monoalphabetic Substitution

#### Caesar Cipher (Enhanced Implementation)
```python
class CaesarCipher:
    def __init__(self):
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def encrypt(self, plaintext, key):
        """Enhanced Caesar cipher with error handling"""
        if not isinstance(key, int) or key < 0:
            raise ValueError("Key must be a non-negative integer")
        
        key = key % 26  # Normalize key
        result = []
        
        for char in plaintext.upper():
            if char in self.alphabet:
                old_index = self.alphabet.index(char)
                new_index = (old_index + key) % 26
                result.append(self.alphabet[new_index])
            else:
                result.append(char)  # Keep non-alphabetic characters
        
        return ''.join(result)
    
    def decrypt(self, ciphertext, key):
        """Decrypt by using negative key"""
        return self.encrypt(ciphertext, -key)
    
    def brute_force_attack(self, ciphertext):
        """Try all possible keys"""
        results = {}
        for key in range(26):
            decrypted = self.decrypt(ciphertext, key)
            results[key] = decrypted
        return results

# Example with frequency analysis
def frequency_analysis_attack(ciphertext):
    """Attack Caesar cipher using frequency analysis"""
    # English letter frequencies (%)
    english_freq = [8.12, 1.49, 2.78, 4.25, 12.02, 2.23, 2.02, 6.09, 6.97, 0.15,
                   0.77, 4.03, 2.41, 6.75, 7.68, 1.93, 0.10, 5.99, 6.33, 9.10,
                   2.76, 0.98, 2.36, 0.15, 1.97, 0.07]
    
    caesar = CaesarCipher()
    best_key = 0
    best_score = float('inf')
    
    for key in range(26):
        decrypted = caesar.decrypt(ciphertext, key)
        score = calculate_chi_squared_score(decrypted, english_freq)
        
        if score < best_score:
            best_score = score
            best_key = key
    
    return best_key, caesar.decrypt(ciphertext, best_key)

def calculate_chi_squared_score(text, expected_freq):
    """Calculate chi-squared statistic for goodness of fit"""
    text = ''.join([c for c in text.upper() if c.isalpha()])
    if not text:
        return float('inf')
    
    observed_freq = [0] * 26
    for char in text:
        observed_freq[ord(char) - ord('A')] += 1
    
    # Convert to percentages
    total = sum(observed_freq)
    observed_freq = [count / total * 100 for count in observed_freq]
    
    # Calculate chi-squared
    chi_squared = 0
    for i in range(26):
        expected = expected_freq[i]
        observed = observed_freq[i]
        if expected > 0:
            chi_squared += (observed - expected) ** 2 / expected
    
    return chi_squared
```

#### Atbash Cipher
Ancient Hebrew cipher that reverses the alphabet.

```python
class AtbashCipher:
    def __init__(self):
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.reversed_alphabet = self.alphabet[::-1]
    
    def encrypt(self, plaintext):
        """Atbash encryption (A->Z, B->Y, etc.)"""
        result = []
        for char in plaintext.upper():
            if char in self.alphabet:
                index = self.alphabet.index(char)
                result.append(self.reversed_alphabet[index])
            else:
                result.append(char)
        return ''.join(result)
    
    def decrypt(self, ciphertext):
        """Atbash is self-inverse"""
        return self.encrypt(ciphertext)

# Example
atbash = AtbashCipher()
encrypted = atbash.encrypt("HELLO WORLD")  # "SVOOL DLIOW"
decrypted = atbash.decrypt(encrypted)      # "HELLO WORLD"
```

#### Affine Cipher
Combines multiplicative and additive shifts.

```python
class AffineCipher:
    def __init__(self):
        self.alphabet_size = 26
    
    def gcd(self, a, b):
        """Greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def mod_inverse(self, a, m):
        """Modular multiplicative inverse"""
        def extended_gcd(a, b):
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
    
    def encrypt(self, plaintext, a, b):
        """Affine encryption: E(x) = (ax + b) mod 26"""
        if self.gcd(a, self.alphabet_size) != 1:
            raise ValueError(f"'a' must be coprime to {self.alphabet_size}")
        
        result = []
        for char in plaintext.upper():
            if char.isalpha():
                x = ord(char) - ord('A')
                encrypted_x = (a * x + b) % self.alphabet_size
                result.append(chr(encrypted_x + ord('A')))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def decrypt(self, ciphertext, a, b):
        """Affine decryption: D(y) = a^(-1)(y - b) mod 26"""
        a_inv = self.mod_inverse(a, self.alphabet_size)
        
        result = []
        for char in ciphertext.upper():
            if char.isalpha():
                y = ord(char) - ord('A')
                decrypted_y = (a_inv * (y - b)) % self.alphabet_size
                result.append(chr(decrypted_y + ord('A')))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def brute_force_attack(self, ciphertext):
        """Try all valid combinations of a and b"""
        results = []
        
        for a in range(1, 26):
            if self.gcd(a, 26) == 1:  # a must be coprime to 26
                for b in range(26):
                    try:
                        decrypted = self.decrypt(ciphertext, a, b)
                        results.append(((a, b), decrypted))
                    except ValueError:
                        continue
        
        return results

# Example
affine = AffineCipher()
encrypted = affine.encrypt("HELLO", 5, 8)  # a=5, b=8
decrypted = affine.decrypt(encrypted, 5, 8)
```

### Polygraphic Substitution

#### Playfair Cipher
Encrypts pairs of letters using a 5×5 key square.

```python
class PlayfairCipher:
    def __init__(self, key):
        self.key = self.generate_key_square(key)
    
    def generate_key_square(self, key):
        """Generate 5x5 key square"""
        # Remove duplicates and J (combine with I)
        key = key.upper().replace('J', 'I')
        seen = set()
        key_chars = []
        
        for char in key:
            if char.isalpha() and char not in seen:
                seen.add(char)
                key_chars.append(char)
        
        # Add remaining letters
        alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'  # No J
        for char in alphabet:
            if char not in seen:
                key_chars.append(char)
        
        # Create 5x5 grid
        square = []
        for i in range(5):
            row = key_chars[i*5:(i+1)*5]
            square.append(row)
        
        return square
    
    def find_position(self, char):
        """Find position of character in key square"""
        for i, row in enumerate(self.key):
            for j, c in enumerate(row):
                if c == char:
                    return i, j
        return None
    
    def prepare_text(self, text):
        """Prepare text for Playfair encryption"""
        text = text.upper().replace('J', 'I')
        text = ''.join([c for c in text if c.isalpha()])
        
        # Insert X between duplicate letters in pairs
        prepared = []
        i = 0
        while i < len(text):
            if i == len(text) - 1:
                prepared.append(text[i])
                prepared.append('X')  # Pad single letter
                break
            elif text[i] == text[i + 1]:
                prepared.append(text[i])
                prepared.append('X')
                i += 1
            else:
                prepared.append(text[i])
                prepared.append(text[i + 1])
                i += 2
        
        return ''.join(prepared)
    
    def encrypt_pair(self, char1, char2):
        """Encrypt a pair of characters"""
        row1, col1 = self.find_position(char1)
        row2, col2 = self.find_position(char2)
        
        if row1 == row2:  # Same row
            new_col1 = (col1 + 1) % 5
            new_col2 = (col2 + 1) % 5
            return self.key[row1][new_col1] + self.key[row2][new_col2]
        elif col1 == col2:  # Same column
            new_row1 = (row1 + 1) % 5
            new_row2 = (row2 + 1) % 5
            return self.key[new_row1][col1] + self.key[new_row2][col2]
        else:  # Rectangle
            return self.key[row1][col2] + self.key[row2][col1]
    
    def decrypt_pair(self, char1, char2):
        """Decrypt a pair of characters"""
        row1, col1 = self.find_position(char1)
        row2, col2 = self.find_position(char2)
        
        if row1 == row2:  # Same row
            new_col1 = (col1 - 1) % 5
            new_col2 = (col2 - 1) % 5
            return self.key[row1][new_col1] + self.key[row2][new_col2]
        elif col1 == col2:  # Same column
            new_row1 = (row1 - 1) % 5
            new_row2 = (row2 - 1) % 5
            return self.key[new_row1][col1] + self.key[new_row2][col2]
        else:  # Rectangle
            return self.key[row1][col2] + self.key[row2][col1]
    
    def encrypt(self, plaintext):
        """Encrypt entire message"""
        prepared = self.prepare_text(plaintext)
        ciphertext = ''
        
        for i in range(0, len(prepared), 2):
            pair = self.encrypt_pair(prepared[i], prepared[i + 1])
            ciphertext += pair
        
        return ciphertext
    
    def decrypt(self, ciphertext):
        """Decrypt entire message"""
        plaintext = ''
        
        for i in range(0, len(ciphertext), 2):
            pair = self.decrypt_pair(ciphertext[i], ciphertext[i + 1])
            plaintext += pair
        
        return plaintext

# Example
playfair = PlayfairCipher("PLAYFAIREXAMPLE")
encrypted = playfair.encrypt("HIDE THE GOLD IN THE TREE STUMP")
decrypted = playfair.decrypt(encrypted)
```

#### Four-Square Cipher
Uses four 5×5 squares for enhanced security.

```python
class FourSquareCipher:
    def __init__(self, key1, key2):
        self.alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'  # No J
        self.plain_square1 = self.create_plain_square()
        self.plain_square2 = self.create_plain_square()
        self.cipher_square1 = self.create_key_square(key1)
        self.cipher_square2 = self.create_key_square(key2)
    
    def create_plain_square(self):
        """Create standard alphabet square"""
        square = []
        for i in range(5):
            row = list(self.alphabet[i*5:(i+1)*5])
            square.append(row)
        return square
    
    def create_key_square(self, key):
        """Create keyed alphabet square"""
        key = key.upper().replace('J', 'I')
        seen = set()
        key_chars = []
        
        for char in key:
            if char.isalpha() and char not in seen:
                seen.add(char)
                key_chars.append(char)
        
        for char in self.alphabet:
            if char not in seen:
                key_chars.append(char)
        
        square = []
        for i in range(5):
            row = key_chars[i*5:(i+1)*5]
            square.append(row)
        
        return square
    
    def find_position(self, char, square):
        """Find position of character in square"""
        for i, row in enumerate(square):
            for j, c in enumerate(row):
                if c == char:
                    return i, j
        return None
    
    def encrypt(self, plaintext):
        """Encrypt using four-square cipher"""
        plaintext = plaintext.upper().replace('J', 'I')
        plaintext = ''.join([c for c in plaintext if c.isalpha()])
        
        if len(plaintext) % 2 == 1:
            plaintext += 'X'
        
        ciphertext = ''
        for i in range(0, len(plaintext), 2):
            char1, char2 = plaintext[i], plaintext[i + 1]
            
            # Find positions in plain squares
            row1, col1 = self.find_position(char1, self.plain_square1)
            row2, col2 = self.find_position(char2, self.plain_square2)
            
            # Get corresponding characters from cipher squares
            cipher_char1 = self.cipher_square1[row1][col2]
            cipher_char2 = self.cipher_square2[row2][col1]
            
            ciphertext += cipher_char1 + cipher_char2
        
        return ciphertext
    
    def decrypt(self, ciphertext):
        """Decrypt using four-square cipher"""
        plaintext = ''
        
        for i in range(0, len(ciphertext), 2):
            char1, char2 = ciphertext[i], ciphertext[i + 1]
            
            # Find positions in cipher squares
            row1, col1 = self.find_position(char1, self.cipher_square1)
            row2, col2 = self.find_position(char2, self.cipher_square2)
            
            # Get corresponding characters from plain squares
            plain_char1 = self.plain_square1[row1][col2]
            plain_char2 = self.plain_square2[row2][col1]
            
            plaintext += plain_char1 + plain_char2
        
        return plaintext

# Example
four_square = FourSquareCipher("EXAMPLE", "KEYWORD")
encrypted = four_square.encrypt("ATTACK AT DAWN")
decrypted = four_square.decrypt(encrypted)
```

---

## 🔄 Transposition Ciphers

### Columnar Transposition
Rearranges letters according to a keyword-based column order.

```python
class ColumnarTransposition:
    def __init__(self, key):
        self.key = key.upper()
        self.key_order = self.get_key_order()
    
    def get_key_order(self):
        """Determine column order based on alphabetical sorting of key"""
        sorted_key = sorted(enumerate(self.key), key=lambda x: x[1])
        return [i for i, _ in sorted_key]
    
    def encrypt(self, plaintext):
        """Encrypt using columnar transposition"""
        # Remove spaces and convert to uppercase
        text = ''.join(plaintext.split()).upper()
        
        # Calculate number of rows needed
        cols = len(self.key)
        rows = (len(text) + cols - 1) // cols  # Ceiling division
        
        # Pad text if necessary
        text += 'X' * (rows * cols - len(text))
        
        # Create matrix
        matrix = []
        for r in range(rows):
            row = text[r * cols:(r + 1) * cols]
            matrix.append(list(row))
        
        # Read columns in key order
        ciphertext = ''
        for col_index in self.key_order:
            for row in range(rows):
                ciphertext += matrix[row][col_index]
        
        return ciphertext
    
    def decrypt(self, ciphertext):
        """Decrypt columnar transposition"""
        cols = len(self.key)
        rows = len(ciphertext) // cols
        
        # Create empty matrix
        matrix = [['' for _ in range(cols)] for _ in range(rows)]
        
        # Fill matrix column by column in key order
        index = 0
        for col_index in self.key_order:
            for row in range(rows):
                matrix[row][col_index] = ciphertext[index]
                index += 1
        
        # Read matrix row by row
        plaintext = ''
        for row in range(rows):
            for col in range(cols):
                plaintext += matrix[row][col]
        
        return plaintext.rstrip('X')

# Example
transposition = ColumnarTransposition("ZEBRAS")
encrypted = transposition.encrypt("WE ARE DISCOVERED FLEE AT ONCE")
decrypted = transposition.decrypt(encrypted)
```

### Rail Fence Cipher
Text is written in a zigzag pattern across multiple rails.

```python
class RailFenceCipher:
    def __init__(self, rails):
        self.rails = rails
    
    def encrypt(self, plaintext):
        """Encrypt using rail fence cipher"""
        if self.rails == 1:
            return plaintext
        
        # Create rails
        fence = [[] for _ in range(self.rails)]
        rail = 0
        direction = 1
        
        for char in plaintext:
            fence[rail].append(char)
            rail += direction
            
            # Change direction at boundaries
            if rail == self.rails - 1 or rail == 0:
                direction = -direction
        
        # Concatenate all rails
        return ''.join([''.join(rail) for rail in fence])
    
    def decrypt(self, ciphertext):
        """Decrypt rail fence cipher"""
        if self.rails == 1:
            return ciphertext
        
        # Calculate lengths of each rail
        rail_lengths = [0] * self.rails
        rail = 0
        direction = 1
        
        for _ in ciphertext:
            rail_lengths[rail] += 1
            rail += direction
            
            if rail == self.rails - 1 or rail == 0:
                direction = -direction
        
        # Fill rails with characters
        fence = []
        index = 0
        for length in rail_lengths:
            fence.append(list(ciphertext[index:index + length]))
            index += length
        
        # Read in zigzag pattern
        result = []
        rail = 0
        direction = 1
        rail_indices = [0] * self.rails
        
        for _ in ciphertext:
            result.append(fence[rail][rail_indices[rail]])
            rail_indices[rail] += 1
            rail += direction
            
            if rail == self.rails - 1 or rail == 0:
                direction = -direction
        
        return ''.join(result)

# Example
rail_fence = RailFenceCipher(3)
encrypted = rail_fence.encrypt("WE ARE DISCOVERED FLEE AT ONCE")
decrypted = rail_fence.decrypt(encrypted)
```

---

## 🔄 Polyalphabetic Ciphers

### Vigenère Cipher (Enhanced)
The most famous polyalphabetic cipher.

```python
class VigenereCipher:
    def __init__(self):
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def encrypt(self, plaintext, key):
        """Encrypt using Vigenère cipher"""
        key = key.upper()
        plaintext = plaintext.upper()
        result = []
        key_index = 0
        
        for char in plaintext:
            if char in self.alphabet:
                # Get shift value from key
                key_char = key[key_index % len(key)]
                shift = ord(key_char) - ord('A')
                
                # Apply shift
                char_index = ord(char) - ord('A')
                new_index = (char_index + shift) % 26
                result.append(chr(new_index + ord('A')))
                
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def decrypt(self, ciphertext, key):
        """Decrypt Vigenère cipher"""
        key = key.upper()
        ciphertext = ciphertext.upper()
        result = []
        key_index = 0
        
        for char in ciphertext:
            if char in self.alphabet:
                # Get shift value from key
                key_char = key[key_index % len(key)]
                shift = ord(key_char) - ord('A')
                
                # Apply reverse shift
                char_index = ord(char) - ord('A')
                new_index = (char_index - shift) % 26
                result.append(chr(new_index + ord('A')))
                
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def kasiski_examination(self, ciphertext):
        """Find likely key length using Kasiski examination"""
        ciphertext = ''.join([c for c in ciphertext.upper() if c.isalpha()])
        coincidences = {}
        
        # Find repeated sequences of length 3 or more
        for seq_length in range(3, min(20, len(ciphertext) // 4)):
            for i in range(len(ciphertext) - seq_length):
                sequence = ciphertext[i:i + seq_length]
                
                # Look for repetitions
                for j in range(i + seq_length, len(ciphertext) - seq_length + 1):
                    if ciphertext[j:j + seq_length] == sequence:
                        distance = j - i
                        
                        # Find factors of distance
                        for factor in range(2, distance + 1):
                            if distance % factor == 0:
                                coincidences[factor] = coincidences.get(factor, 0) + 1
        
        # Return most likely key lengths
        if coincidences:
            sorted_coincidences = sorted(coincidences.items(), 
                                       key=lambda x: x[1], reverse=True)
            return sorted_coincidences[:5]  # Top 5 candidates
        else:
            return [(1, 0)]  # Default to key length 1
    
    def index_of_coincidence(self, text):
        """Calculate index of coincidence"""
        text = ''.join([c for c in text.upper() if c.isalpha()])
        n = len(text)
        
        if n <= 1:
            return 0
        
        # Count frequencies
        frequencies = {}
        for char in text:
            frequencies[char] = frequencies.get(char, 0) + 1
        
        # Calculate IC
        ic = sum(f * (f - 1) for f in frequencies.values()) / (n * (n - 1))
        return ic
    
    def cryptanalysis(self, ciphertext):
        """Attempt to break Vigenère cipher"""
        # Step 1: Determine key length
        key_candidates = self.kasiski_examination(ciphertext)
        
        results = []
        
        for key_length, _ in key_candidates[:3]:  # Try top 3 candidates
            if key_length == 1:
                continue
            
            # Step 2: Split into groups
            groups = [''] * key_length
            clean_text = ''.join([c for c in ciphertext.upper() if c.isalpha()])
            
            for i, char in enumerate(clean_text):
                groups[i % key_length] += char
            
            # Step 3: Analyze each group as monoalphabetic
            key = ''
            for group in groups:
                best_shift = self.find_best_shift(group)
                key += chr(best_shift + ord('A'))
            
            # Step 4: Test the key
            decrypted = self.decrypt(ciphertext, key)
            ic = self.index_of_coincidence(decrypted)
            
            results.append((key, decrypted, ic, key_length))
        
        # Return result with highest IC (closest to English ~0.067)
        if results:
            best_result = max(results, key=lambda x: x[2])
            return best_result[0], best_result[1]
        else:
            return None, None
    
    def find_best_shift(self, text):
        """Find best Caesar shift for given text"""
        english_freq = [8.12, 1.49, 2.78, 4.25, 12.02, 2.23, 2.02, 6.09, 6.97, 0.15,
                       0.77, 4.03, 2.41, 6.75, 7.68, 1.93, 0.10, 5.99, 6.33, 9.10,
                       2.76, 0.98, 2.36, 0.15, 1.97, 0.07]
        
        best_shift = 0
        best_score = float('inf')
        
        for shift in range(26):
            # Decrypt with this shift
            decrypted = ''
            for char in text:
                new_char = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                decrypted += new_char
            
            # Calculate chi-squared
            score = self.chi_squared_score(decrypted, english_freq)
            if score < best_score:
                best_score = score
                best_shift = shift
        
        return best_shift
    
    def chi_squared_score(self, text, expected_freq):
        """Calculate chi-squared statistic"""
        if not text:
            return float('inf')
        
        observed_freq = [0] * 26
        for char in text:
            observed_freq[ord(char) - ord('A')] += 1
        
        # Convert to percentages
        total = sum(observed_freq)
        if total == 0:
            return float('inf')
        
        observed_freq = [count / total * 100 for count in observed_freq]
        
        # Calculate chi-squared
        chi_squared = 0
        for i in range(26):
            expected = expected_freq[i]
            observed = observed_freq[i]
            if expected > 0:
                chi_squared += (observed - expected) ** 2 / expected
        
        return chi_squared

# Example usage
vigenere = VigenereCipher()
key = "LEMON"
plaintext = "ATTACK AT DAWN"
encrypted = vigenere.encrypt(plaintext, key)
decrypted = vigenere.decrypt(encrypted, key)

# Cryptanalysis
recovered_key, recovered_text = vigenere.cryptanalysis(encrypted)
```

### Beaufort Cipher
Variant of Vigenère using different operation.

```python
class BeaufortCipher:
    def __init__(self):
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def encrypt(self, plaintext, key):
        """Beaufort encryption: C = K - P (mod 26)"""
        key = key.upper()
        plaintext = plaintext.upper()
        result = []
        key_index = 0
        
        for char in plaintext:
            if char in self.alphabet:
                key_char = key[key_index % len(key)]
                key_val = ord(key_char) - ord('A')
                char_val = ord(char) - ord('A')
                
                # Beaufort operation: key - plaintext
                cipher_val = (key_val - char_val) % 26
                result.append(chr(cipher_val + ord('A')))
                
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def decrypt(self, ciphertext, key):
        """Beaufort decryption: P = K - C (mod 26)"""
        # Beaufort cipher is reciprocal
        return self.encrypt(ciphertext, key)

# Example
beaufort = BeaufortCipher()
encrypted = beaufort.encrypt("ATTACK AT DAWN", "FORTIFICATION")
decrypted = beaufort.decrypt(encrypted, "FORTIFICATION")
```

---

## ⚙️ Mechanical Cipher Machines

### Enigma Machine Simulation
Simplified simulation of the famous WWII cipher machine.

```python
class EnigmaRotor:
    def __init__(self, wiring, notch, position=0):
        self.wiring = wiring
        self.reverse_wiring = self.create_reverse_wiring()
        self.notch = notch
        self.position = position
    
    def create_reverse_wiring(self):
        """Create reverse wiring for return path"""
        reverse = [''] * 26
        for i, char in enumerate(self.wiring):
            reverse[ord(char) - ord('A')] = chr(i + ord('A'))
        return ''.join(reverse)
    
    def encode_forward(self, char):
        """Encode character going through rotor forward"""
        # Adjust for rotor position
        input_pos = (ord(char) - ord('A') + self.position) % 26
        output_char = self.wiring[input_pos]
        # Adjust output for position
        output_pos = (ord(output_char) - ord('A') - self.position) % 26
        return chr(output_pos + ord('A'))
    
    def encode_backward(self, char):
        """Encode character on return path"""
        # Adjust for rotor position
        input_pos = (ord(char) - ord('A') + self.position) % 26
        output_char = self.reverse_wiring[input_pos]
        # Adjust output for position
        output_pos = (ord(output_char) - ord('A') - self.position) % 26
        return chr(output_pos + ord('A'))
    
    def step(self):
        """Advance rotor by one position"""
        self.position = (self.position + 1) % 26
        return self.position == self.notch
    
    def at_notch(self):
        """Check if rotor is at notch position"""
        return self.position == self.notch

class EnigmaReflector:
    def __init__(self, wiring):
        self.wiring = wiring
    
    def reflect(self, char):
        """Reflect character"""
        index = ord(char) - ord('A')
        return self.wiring[index]

class EnigmaPlugboard:
    def __init__(self, pairs=None):
        self.swaps = {}
        if pairs:
            for pair in pairs:
                if len(pair) == 2:
                    self.swaps[pair[0]] = pair[1]
                    self.swaps[pair[1]] = pair[0]
    
    def swap(self, char):
        """Swap character if it's in plugboard"""
        return self.swaps.get(char, char)

class EnigmaMachine:
    def __init__(self, rotors, reflector, plugboard=None, rotor_positions=None):
        self.rotors = rotors
        self.reflector = reflector
        self.plugboard = plugboard or EnigmaPlugboard()
        
        if rotor_positions:
            for i, pos in enumerate(rotor_positions):
                if i < len(self.rotors):
                    self.rotors[i].position = pos
    
    def step_rotors(self):
        """Step rotors according to Enigma rules"""
        # Check for double stepping
        if len(self.rotors) >= 2 and self.rotors[1].at_notch():
            self.rotors[1].step()
            if len(self.rotors) >= 3:
                self.rotors[2].step()
        
        # Step middle rotor if it's at notch
        if len(self.rotors) >= 2 and self.rotors[0].at_notch():
            self.rotors[1].step()
        
        # Always step first rotor
        self.rotors[0].step()
    
    def encode_char(self, char):
        """Encode a single character"""
        if not char.isalpha():
            return char
        
        char = char.upper()
        
        # Step rotors before encoding
        self.step_rotors()
        
        # Through plugboard
        char = self.plugboard.swap(char)
        
        # Through rotors (forward)
        for rotor in self.rotors:
            char = rotor.encode_forward(char)
        
        # Through reflector
        char = self.reflector.reflect(char)
        
        # Through rotors (backward)
        for rotor in reversed(self.rotors):
            char = rotor.encode_backward(char)
        
        # Through plugboard again
        char = self.plugboard.swap(char)
        
        return char
    
    def encode_message(self, message):
        """Encode entire message"""
        result = []
        for char in message:
            result.append(self.encode_char(char))
        return ''.join(result)

# Example Enigma setup
def create_enigma_example():
    # Historical rotor wirings (simplified)
    rotor1 = EnigmaRotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", 16)  # Rotor I
    rotor2 = EnigmaRotor("AJDKSIRUXBLHWTMCQGZNPYFVOE", 4)   # Rotor II
    rotor3 = EnigmaRotor("BDFHJLCPRTXVZNYEIWGAKMUSQO", 21)  # Rotor III
    
    # Reflector B
    reflector = EnigmaReflector("YRUHQSLDPXNGOKMIEBFZCWVJAT")
    
    # Plugboard pairs
    plugboard = EnigmaPlugboard(["AR", "GK", "OX"])
    
    return EnigmaMachine([rotor1, rotor2, rotor3], reflector, plugboard)

# Usage
enigma = create_enigma_example()
encrypted = enigma.encode_message("HELLO WORLD")

# Reset machine to same state for decryption
enigma = create_enigma_example()
decrypted = enigma.encode_message(encrypted)
```

---

## 🔍 Cryptanalysis Techniques

### Frequency Analysis
Statistical analysis of letter, bigram, and trigram frequencies.

```python
class FrequencyAnalyzer:
    def __init__(self):
        # English letter frequencies (%)
        self.english_freq = {
            'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97,
            'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
            'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
            'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
            'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
        }
        
        # Common English bigrams
        self.english_bigrams = ['TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ED', 'ND', 'ON', 'EN']
        
        # Common English trigrams
        self.english_trigrams = ['THE', 'AND', 'ING', 'HER', 'HAT', 'HIS', 'THA', 'ERE', 'FOR', 'ENT']
    
    def letter_frequency(self, text):
        """Calculate letter frequencies in text"""
        text = ''.join([c.upper() for c in text if c.isalpha()])
        total = len(text)
        
        if total == 0:
            return {}
        
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        # Convert to percentages
        for char in freq:
            freq[char] = (freq[char] / total) * 100
        
        return freq
    
    def bigram_frequency(self, text):
        """Calculate bigram frequencies"""
        text = ''.join([c.upper() for c in text if c.isalpha()])
        bigrams = {}
        
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        
        return sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
    
    def trigram_frequency(self, text):
        """Calculate trigram frequencies"""
        text = ''.join([c.upper() for c in text if c.isalpha()])
        trigrams = {}
        
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            trigrams[trigram] = trigrams.get(trigram, 0) + 1
        
        return sorted(trigrams.items(), key=lambda x: x[1], reverse=True)
    
    def chi_squared_test(self, observed_freq):
        """Chi-squared test against English frequencies"""
        chi_squared = 0
        
        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            observed = observed_freq.get(char, 0)
            expected = self.english_freq.get(char, 0)
            
            if expected > 0:
                chi_squared += (observed - expected) ** 2 / expected
        
        return chi_squared
    
    def index_of_coincidence(self, text):
        """Calculate index of coincidence"""
        text = ''.join([c.upper() for c in text if c.isalpha()])
        n = len(text)
        
        if n <= 1:
            return 0
        
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        ic = sum(f * (f - 1) for f in freq.values()) / (n * (n - 1))
        return ic
    
    def language_detection(self, text):
        """Simple language detection based on IC"""
        ic = self.index_of_coincidence(text)
        
        if 0.06 <= ic <= 0.075:
            return "English"
        elif 0.074 <= ic <= 0.082:
            return "German"
        elif 0.072 <= ic <= 0.080:
            return "French"
        elif 0.045 <= ic <= 0.055:
            return "Random/Encrypted"
        else:
            return "Unknown"

# Example usage
analyzer = FrequencyAnalyzer()
text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"

letter_freq = analyzer.letter_frequency(text)
bigram_freq = analyzer.bigram_frequency(text)
ic = analyzer.index_of_coincidence(text)
language = analyzer.language_detection(text)
```

### Kasiski Examination
Method for determining the key length of polyalphabetic ciphers.

```python
def kasiski_examination(ciphertext, min_length=3, max_length=20):
    """Perform Kasiski examination to find key length"""
    ciphertext = ''.join([c.upper() for c in ciphertext if c.isalpha()])
    
    # Find repeated sequences
    sequences = {}
    
    for length in range(min_length, min(max_length, len(ciphertext) // 3)):
        for i in range(len(ciphertext) - length + 1):
            sequence = ciphertext[i:i + length]
            
            if sequence in sequences:
                sequences[sequence].append(i)
            else:
                sequences[sequence] = [i]
    
    # Find distances between repetitions
    distances = []
    
    for sequence, positions in sequences.items():
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                distance = positions[i + 1] - positions[i]
                distances.append(distance)
    
    # Find factors of distances
    factor_counts = {}
    
    for distance in distances:
        factors = []
        for i in range(2, distance + 1):
            if distance % i == 0:
                factors.append(i)
        
        for factor in factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    # Sort by frequency
    likely_key_lengths = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
    
    return likely_key_lengths[:10]  # Return top 10 candidates

# Example
ciphertext = "CHREEVOAHMAERATBIAXXWTNXBEEOPHBSBQMQEQERBWRVXUOAKXAOSXXWEAHBWGJMMQMNKGRFVGXWTRZXWIAKLXFPSKAUTEMNDCMGTSXMXBTUIADNGMGPSRELXNJELXVRVPRTULHDNQWTWDTYGBPHXTFALJHASVBFXNGLLCHRZBWELEKMSJIKNBHWRJGNMGJSGLXFEYPHAGNRBIEQJTAMRVLCRREMNDGLXRRIMGNSNRWCHRQHAEYEVTAQEBBIPEEWEVKAKOEWADREMXMTBHHCHRTKDNVRZCHRCLQOHPWQAIIWXNRMGWOIIFKEE"
key_lengths = kasiski_examination(ciphertext)
print("Likely key lengths:", key_lengths)
```

---

## 📜 Historical Impact

### Military Applications

#### World War I
- **Zimmermann Telegram**: German diplomatic cable intercepted and decrypted by British
- **ADFGVX Cipher**: German field cipher broken by French cryptanalyst Georges Painvin
- **Trench Codes**: Simple field ciphers for tactical communications

#### World War II
- **Enigma**: German machine cipher, broken by Polish and British cryptanalysts
- **Purple**: Japanese diplomatic cipher, broken by US cryptanalysts
- **Ultra**: British codebreaking program, crucial for Allied victory

```python
def zimmermann_telegram_analysis():
    """Analysis of the famous Zimmermann Telegram"""
    # Original encrypted telegram (partial)
    encrypted = "FFNHE TFRUB NNYHL FNZRH RKTNE YMUZS OTZPO ZHWLU JRBTS ZVRZH"
    
    # Historical context
    context = {
        "date": "January 16, 1917",
        "sender": "Arthur Zimmermann (German Foreign Secretary)",
        "recipient": "German Ambassador to Mexico",
        "cipher": "German diplomatic code 0075",
        "intercepted_by": "British Room 40",
        "impact": "Helped bring US into WWI"
    }
    
    # Decrypted content (historical)
    decrypted = """We intend to begin on the first of February unrestricted submarine warfare. 
    We shall endeavor in spite of this to keep the United States of America neutral. 
    In the event of this not succeeding, we make Mexico a proposal of alliance on the 
    following basis: make war together, make peace together, generous financial support 
    and an understanding on our part that Mexico is to reconquer the lost territory 
    in Texas, New Mexico, and Arizona."""
    
    return context, decrypted

# Historical analysis
context, decrypted = zimmermann_telegram_analysis()
```

### Diplomatic Communications

#### Renaissance Ciphers
- **Great Cipher**: Louis XIV's diplomatic correspondence
- **Papal Ciphers**: Vatican's encrypted communications
- **Venetian Ciphers**: Commercial and diplomatic secrets

#### Modern Era
- **Cold War Cryptography**: Extensive use of one-time pads
- **Digital Diplomacy**: Modern encrypted communications
- **Whistleblower Protection**: Anonymous communication systems

---

## 🎯 Exercises and Challenges

### Beginner Challenges

1. **Caesar Cipher Variants**
   - Implement Caesar cipher with custom alphabet
   - Create a "number shift" cipher for digits
   - Develop automatic key detection

2. **Substitution Cipher Creator**
   - Generate random substitution keys
   - Implement keyword-based substitutions
   - Create a cipher that preserves word lengths

3. **Transposition Exercises**
   - Implement block transposition
   - Create irregular columnar transposition
   - Develop route cipher (spiral, zigzag)

### Intermediate Challenges

1. **Playfair Variants**
   - Implement Two-Square cipher
   - Create 6×6 Playfair for alphanumeric text
   - Develop automatic Playfair key recovery

2. **Vigenère Extensions**
   - Implement Gronsfeld cipher (numeric key)
   - Create running key cipher
   - Develop progressive key Vigenère

3. **Mechanical Cipher Simulation**
   - Build simplified Enigma with different rotors
   - Implement Hagelin M-209 simulation
   - Create custom rotor machine

### Advanced Challenges

1. **Historical Cipher Breaking**
   - Break actual historical ciphertexts
   - Implement Friedman's method for Vigenère
   - Develop machine learning approach to cipher identification

2. **Cipher Design and Analysis**
   - Design a novel classical cipher
   - Analyze its security properties
   - Develop automated cryptanalysis

3. **Modern Classical Applications**
   - Implement classical ciphers for Unicode text
   - Create steganographic classical ciphers
   - Develop quantum-resistant classical methods

---

## 🔗 Connections to Modern Cryptography

### Lessons Learned
1. **Security through Obscurity**: Classical ciphers show why this fails
2. **Key Management**: Historical challenges still relevant today
3. **Frequency Analysis**: Foundation for modern statistical attacks
4. **Perfect Secrecy**: One-time pad principles in modern systems

### Modern Applications
1. **CTF Competitions**: Classical ciphers in capture-the-flag events
2. **Educational Tools**: Teaching cryptographic principles
3. **Puzzle Games**: Entertainment and brain training
4. **Historical Research**: Decrypting historical documents

---

*Next: [Symmetric Cryptography →](03_symmetric_cryptography.md)*
