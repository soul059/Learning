# 🔍 Cryptanalysis & Breaking Cryptographic Systems

> *"The art of cryptanalysis is the science of finding weaknesses in cryptographic systems. Every cipher is breakable - the question is how much time and resources it takes."*

## 📖 Table of Contents

1. [Cryptanalysis Fundamentals](#-cryptanalysis-fundamentals)
2. [Classical Cipher Breaking](#-classical-cipher-breaking)
3. [Statistical Analysis Methods](#-statistical-analysis-methods)
4. [Linear Cryptanalysis](#-linear-cryptanalysis)
5. [Differential Cryptanalysis](#-differential-cryptanalysis)
6. [Side-Channel Attacks](#-side-channel-attacks)
7. [Algebraic Attacks](#-algebraic-attacks)
8. [RSA Attack Methods](#-rsa-attack-methods)
9. [Hash Function Attacks](#-hash-function-attacks)
10. [Modern Attack Techniques](#-modern-attack-techniques)

---

## 🧮 Cryptanalysis Fundamentals

### Comprehensive Cryptanalysis Framework

```python
#!/usr/bin/env python3
"""
Comprehensive Cryptanalysis Framework
Implements fundamental cryptanalytic techniques and attack methodologies
"""

import math
import secrets
import string
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
from dataclasses import dataclass
import time
import hashlib
import struct

class AttackType(Enum):
    """Types of cryptanalytic attacks"""
    CIPHERTEXT_ONLY = "ciphertext_only"
    KNOWN_PLAINTEXT = "known_plaintext" 
    CHOSEN_PLAINTEXT = "chosen_plaintext"
    CHOSEN_CIPHERTEXT = "chosen_ciphertext"
    ADAPTIVE_CHOSEN_PLAINTEXT = "adaptive_chosen_plaintext"

@dataclass
class CryptanalysisResult:
    """Result of cryptanalytic attack"""
    success: bool
    key: Optional[Any]
    confidence: float
    time_taken: float
    method: str
    additional_info: Dict[str, Any]

class FrequencyAnalyzer:
    """Statistical frequency analysis for cryptanalysis"""
    
    # English letter frequencies (approximate)
    ENGLISH_FREQ = {
        'A': 8.167, 'B': 1.492, 'C': 2.782, 'D': 4.253, 'E': 12.02,
        'F': 2.228, 'G': 2.015, 'H': 6.094, 'I': 6.966, 'J': 0.153,
        'K': 0.772, 'L': 4.025, 'M': 2.406, 'N': 6.749, 'O': 7.507,
        'P': 1.929, 'Q': 0.095, 'R': 5.987, 'S': 6.327, 'T': 9.056,
        'U': 2.758, 'V': 0.978, 'W': 2.360, 'X': 0.150, 'Y': 1.974,
        'Z': 0.074
    }
    
    # Common English bigrams
    ENGLISH_BIGRAMS = {
        'TH': 3.56, 'HE': 3.07, 'IN': 2.43, 'ER': 2.05, 'AN': 1.99,
        'RE': 1.85, 'ED': 1.53, 'ND': 1.48, 'ON': 1.45, 'EN': 1.45,
        'AT': 1.42, 'OU': 1.41, 'EA': 1.31, 'HA': 1.27, 'NG': 1.27,
        'AS': 1.26, 'OR': 1.24, 'TI': 1.20, 'IS': 1.13, 'ET': 1.11
    }
    
    # Common English trigrams
    ENGLISH_TRIGRAMS = {
        'THE': 3.508, 'AND': 1.593, 'ING': 1.147, 'HER': 0.822,
        'HAT': 0.650, 'HIS': 0.596, 'THA': 0.593, 'ERE': 0.560,
        'FOR': 0.555, 'ENT': 0.530, 'ION': 0.506, 'TER': 0.461,
        'WAS': 0.460, 'YOU': 0.437, 'ITH': 0.431, 'VER': 0.430,
        'ALL': 0.422, 'WIT': 0.397, 'THI': 0.394, 'TIO': 0.378
    }
    
    def __init__(self):
        self.text_stats = {}
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive statistical analysis of text"""
        text = text.upper().replace(' ', '').replace('\n', '')
        text = ''.join(c for c in text if c.isalpha())
        
        if len(text) == 0:
            return {}
        
        # Single letter frequencies
        letter_counts = Counter(text)
        letter_freqs = {letter: (count / len(text)) * 100 
                       for letter, count in letter_counts.items()}
        
        # Bigram frequencies
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_counts = Counter(bigrams)
        bigram_freqs = {bigram: (count / len(bigrams)) * 100 
                       for bigram, count in bigram_counts.items()}
        
        # Trigram frequencies
        trigrams = [text[i:i+3] for i in range(len(text)-2)]
        trigram_counts = Counter(trigrams)
        trigram_freqs = {trigram: (count / len(trigrams)) * 100 
                        for trigram, count in trigram_counts.items()}
        
        # Index of coincidence
        ic = self.calculate_index_of_coincidence(text)
        
        # Friedman test (estimate key length for polyalphabetic ciphers)
        friedman_estimate = self.friedman_test(text)
        
        # Chi-squared test against English
        chi_squared = self.chi_squared_test(letter_freqs)
        
        return {
            'text_length': len(text),
            'letter_frequencies': letter_freqs,
            'bigram_frequencies': dict(bigram_counts.most_common(20)),
            'trigram_frequencies': dict(trigram_counts.most_common(20)),
            'index_of_coincidence': ic,
            'friedman_key_length': friedman_estimate,
            'chi_squared': chi_squared,
            'entropy': self.calculate_entropy(text)
        }
    
    def calculate_index_of_coincidence(self, text: str) -> float:
        """Calculate index of coincidence"""
        n = len(text)
        if n <= 1:
            return 0
        
        letter_counts = Counter(text)
        ic = sum(count * (count - 1) for count in letter_counts.values())
        ic = ic / (n * (n - 1))
        
        return ic
    
    def friedman_test(self, text: str) -> float:
        """Estimate key length using Friedman test"""
        ic = self.calculate_index_of_coincidence(text)
        n = len(text)
        
        # Friedman's formula for key length estimation
        # k ≈ (0.0265 * n) / ((ic - 0.0385) * n + 0.0265 - ic)
        numerator = 0.0265 * n
        denominator = (ic - 0.0385) * n + 0.0265 - ic
        
        if denominator <= 0:
            return float('inf')
        
        key_length = numerator / denominator
        return max(1, round(key_length))
    
    def chi_squared_test(self, observed_freqs: Dict[str, float]) -> float:
        """Chi-squared test against English letter frequencies"""
        chi_squared = 0
        
        for letter in string.ascii_uppercase:
            observed = observed_freqs.get(letter, 0)
            expected = self.ENGLISH_FREQ.get(letter, 0)
            
            if expected > 0:
                chi_squared += ((observed - expected) ** 2) / expected
        
        return chi_squared
    
    def calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        letter_counts = Counter(text)
        text_length = len(text)
        
        entropy = 0
        for count in letter_counts.values():
            if count > 0:
                p = count / text_length
                entropy -= p * math.log2(p)
        
        return entropy

class CaesarCipherAnalyzer:
    """Caesar cipher cryptanalysis"""
    
    def __init__(self):
        self.freq_analyzer = FrequencyAnalyzer()
    
    def brute_force_attack(self, ciphertext: str) -> CryptanalysisResult:
        """Brute force attack on Caesar cipher"""
        start_time = time.time()
        best_key = 0
        best_score = float('inf')
        best_plaintext = ""
        
        ciphertext = ciphertext.upper().replace(' ', '')
        
        for shift in range(26):
            # Decrypt with this shift
            plaintext = self._decrypt_caesar(ciphertext, shift)
            
            # Analyze frequency
            stats = self.freq_analyzer.analyze_text(plaintext)
            
            if 'chi_squared' in stats:
                score = stats['chi_squared']
                
                if score < best_score:
                    best_score = score
                    best_key = shift
                    best_plaintext = plaintext
        
        time_taken = time.time() - start_time
        confidence = 1.0 / (1.0 + best_score / 100)  # Normalize score
        
        return CryptanalysisResult(
            success=True,
            key=best_key,
            confidence=confidence,
            time_taken=time_taken,
            method="brute_force_frequency",
            additional_info={
                'best_score': best_score,
                'plaintext': best_plaintext,
                'all_attempts': []
            }
        )
    
    def frequency_attack(self, ciphertext: str) -> CryptanalysisResult:
        """Frequency analysis attack on Caesar cipher"""
        start_time = time.time()
        
        ciphertext = ciphertext.upper().replace(' ', '')
        
        # Find most frequent letter in ciphertext
        letter_counts = Counter(ciphertext)
        most_frequent = letter_counts.most_common(1)[0][0]
        
        # Assume most frequent letter is 'E'
        shift = (ord(most_frequent) - ord('E')) % 26
        
        plaintext = self._decrypt_caesar(ciphertext, shift)
        stats = self.freq_analyzer.analyze_text(plaintext)
        
        time_taken = time.time() - start_time
        confidence = 1.0 / (1.0 + stats.get('chi_squared', 1000) / 100)
        
        return CryptanalysisResult(
            success=True,
            key=shift,
            confidence=confidence,
            time_taken=time_taken,
            method="frequency_analysis",
            additional_info={
                'most_frequent_letter': most_frequent,
                'plaintext': plaintext,
                'statistics': stats
            }
        )
    
    def _decrypt_caesar(self, ciphertext: str, shift: int) -> str:
        """Decrypt Caesar cipher with given shift"""
        result = ""
        for char in ciphertext:
            if char.isalpha():
                shifted = (ord(char) - ord('A') - shift) % 26
                result += chr(shifted + ord('A'))
            else:
                result += char
        return result

class VigenereCipherAnalyzer:
    """Vigenère cipher cryptanalysis"""
    
    def __init__(self):
        self.freq_analyzer = FrequencyAnalyzer()
    
    def kasiski_test(self, ciphertext: str) -> List[int]:
        """Kasiski test to find key length"""
        ciphertext = ciphertext.upper().replace(' ', '').replace('\n', '')
        
        # Find repeated sequences of length 3-6
        repeated_sequences = {}
        
        for seq_len in range(3, 7):
            for i in range(len(ciphertext) - seq_len + 1):
                sequence = ciphertext[i:i + seq_len]
                
                if sequence in repeated_sequences:
                    repeated_sequences[sequence].append(i)
                else:
                    repeated_sequences[sequence] = [i]
        
        # Calculate distances between repetitions
        distances = []
        for sequence, positions in repeated_sequences.items():
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    distance = positions[i + 1] - positions[i]
                    distances.append(distance)
        
        # Find GCD of distances to estimate key length
        if distances:
            potential_lengths = []
            for length in range(2, 21):  # Test key lengths 2-20
                if all(d % length == 0 for d in distances):
                    potential_lengths.append(length)
            
            return potential_lengths[:5]  # Return top 5 candidates
        
        return []
    
    def index_of_coincidence_attack(self, ciphertext: str, max_key_length: int = 20) -> List[int]:
        """Use index of coincidence to find key length"""
        ciphertext = ciphertext.upper().replace(' ', '').replace('\n', '')
        
        ic_scores = {}
        
        for key_length in range(2, max_key_length + 1):
            # Split ciphertext into groups based on key position
            groups = [[] for _ in range(key_length)]
            
            for i, char in enumerate(ciphertext):
                groups[i % key_length].append(char)
            
            # Calculate average IC for all groups
            total_ic = 0
            for group in groups:
                if len(group) > 1:
                    group_text = ''.join(group)
                    ic = self.freq_analyzer.calculate_index_of_coincidence(group_text)
                    total_ic += ic
            
            avg_ic = total_ic / key_length if key_length > 0 else 0
            ic_scores[key_length] = avg_ic
        
        # Sort by IC score (closer to English IC ≈ 0.065)
        english_ic = 0.065
        sorted_lengths = sorted(ic_scores.keys(), 
                              key=lambda x: abs(ic_scores[x] - english_ic))
        
        return sorted_lengths[:5]
    
    def break_vigenere(self, ciphertext: str) -> CryptanalysisResult:
        """Complete Vigenère cipher cryptanalysis"""
        start_time = time.time()
        
        ciphertext = ciphertext.upper().replace(' ', '').replace('\n', '')
        
        # Step 1: Estimate key length using multiple methods
        kasiski_lengths = self.kasiski_test(ciphertext)
        ic_lengths = self.index_of_coincidence_attack(ciphertext)
        
        # Combine and rank key length candidates
        all_lengths = list(set(kasiski_lengths + ic_lengths))
        if not all_lengths:
            all_lengths = list(range(2, 21))  # Fallback
        
        best_result = None
        best_score = float('inf')
        
        for key_length in all_lengths[:10]:  # Test top 10 candidates
            key = self._find_key_for_length(ciphertext, key_length)
            if key:
                plaintext = self._decrypt_vigenere(ciphertext, key)
                stats = self.freq_analyzer.analyze_text(plaintext)
                score = stats.get('chi_squared', float('inf'))
                
                if score < best_score:
                    best_score = score
                    best_result = {
                        'key': key,
                        'key_length': key_length,
                        'plaintext': plaintext,
                        'score': score,
                        'statistics': stats
                    }
        
        time_taken = time.time() - start_time
        
        if best_result:
            confidence = 1.0 / (1.0 + best_result['score'] / 100)
            return CryptanalysisResult(
                success=True,
                key=best_result['key'],
                confidence=confidence,
                time_taken=time_taken,
                method="combined_kasiski_ic",
                additional_info=best_result
            )
        else:
            return CryptanalysisResult(
                success=False,
                key=None,
                confidence=0.0,
                time_taken=time_taken,
                method="combined_kasiski_ic",
                additional_info={'error': 'Could not determine key'}
            )
    
    def _find_key_for_length(self, ciphertext: str, key_length: int) -> Optional[str]:
        """Find key for given key length using frequency analysis"""
        key = []
        
        for pos in range(key_length):
            # Extract characters at this key position
            group = [ciphertext[i] for i in range(pos, len(ciphertext), key_length)]
            group_text = ''.join(group)
            
            # Try each possible key character (Caesar cipher on this group)
            best_shift = 0
            best_score = float('inf')
            
            for shift in range(26):
                # Decrypt this group with this shift
                decrypted = ''.join(
                    chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                    for char in group_text
                )
                
                # Score against English frequencies
                freqs = Counter(decrypted)
                total_chars = len(decrypted)
                
                score = 0
                for letter in string.ascii_uppercase:
                    observed = (freqs.get(letter, 0) / total_chars) * 100
                    expected = self.freq_analyzer.ENGLISH_FREQ.get(letter, 0)
                    if expected > 0:
                        score += ((observed - expected) ** 2) / expected
                
                if score < best_score:
                    best_score = score
                    best_shift = shift
            
            key.append(chr(best_shift + ord('A')))
        
        return ''.join(key)
    
    def _decrypt_vigenere(self, ciphertext: str, key: str) -> str:
        """Decrypt Vigenère cipher with given key"""
        result = []
        key_length = len(key)
        
        for i, char in enumerate(ciphertext):
            if char.isalpha():
                key_char = key[i % key_length]
                shift = ord(key_char) - ord('A')
                decrypted = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                result.append(decrypted)
            else:
                result.append(char)
        
        return ''.join(result)

class LinearCryptanalysis:
    """Linear cryptanalysis implementation for block ciphers"""
    
    def __init__(self):
        self.linear_approximations = {}
        self.bias_threshold = 0.05
    
    def find_linear_approximations(self, sbox: List[int]) -> List[Tuple[int, int, float]]:
        """Find linear approximations for S-box"""
        n = len(sbox).bit_length() - 1  # Input size
        approximations = []
        
        # Test all possible input and output masks
        for input_mask in range(1, 2**n):
            for output_mask in range(1, 2**n):
                bias = self._calculate_bias(sbox, input_mask, output_mask)
                
                if abs(bias) > self.bias_threshold:
                    approximations.append((input_mask, output_mask, bias))
        
        # Sort by absolute bias (highest first)
        approximations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return approximations
    
    def _calculate_bias(self, sbox: List[int], input_mask: int, output_mask: int) -> float:
        """Calculate bias of linear approximation"""
        matches = 0
        total = len(sbox)
        
        for x in range(total):
            input_parity = bin(x & input_mask).count('1') % 2
            output_parity = bin(sbox[x] & output_mask).count('1') % 2
            
            if input_parity == output_parity:
                matches += 1
        
        probability = matches / total
        bias = abs(probability - 0.5)
        
        return bias
    
    def linear_attack_data_requirement(self, bias: float, success_probability: float = 0.95) -> int:
        """Calculate data requirement for linear attack"""
        # From Matsui's theorem: N ≈ c / bias²
        # where c depends on desired success probability
        
        if success_probability >= 0.95:
            c = 4  # For 95% success
        elif success_probability >= 0.90:
            c = 2  # For 90% success
        else:
            c = 1  # For basic attack
        
        if bias == 0:
            return float('inf')
        
        return int(c / (bias ** 2))
    
    def simulate_linear_attack(self, cipher_function, key: int, 
                             input_mask: int, output_mask: int, 
                             num_plaintexts: int) -> Dict[str, Any]:
        """Simulate linear cryptanalysis attack"""
        start_time = time.time()
        
        # Generate random plaintexts
        plaintexts = [secrets.randbits(32) for _ in range(num_plaintexts)]
        
        # Collect plaintext-ciphertext pairs
        pairs = []
        for pt in plaintexts:
            ct = cipher_function(pt, key)
            pairs.append((pt, ct))
        
        # Count linear approximation matches
        matches = 0
        for pt, ct in pairs:
            pt_parity = bin(pt & input_mask).count('1') % 2
            ct_parity = bin(ct & output_mask).count('1') % 2
            
            if pt_parity == ct_parity:
                matches += 1
        
        observed_bias = abs(matches / num_plaintexts - 0.5)
        
        time_taken = time.time() - start_time
        
        return {
            'num_pairs': num_plaintexts,
            'matches': matches,
            'observed_bias': observed_bias,
            'time_taken': time_taken,
            'success_probability': min(1.0, observed_bias * 100)  # Simplified
        }

def cryptanalysis_demo():
    """Demonstrate cryptanalysis techniques"""
    print("=== Cryptanalysis Demonstration ===")
    
    print("1. Frequency Analysis...")
    
    # Test frequency analysis
    analyzer = FrequencyAnalyzer()
    
    # English text sample
    english_text = """THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. 
                     THIS PANGRAM CONTAINS EVERY LETTER OF THE ALPHABET."""
    
    # Random text sample  
    random_text = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(100))
    
    english_stats = analyzer.analyze_text(english_text)
    random_stats = analyzer.analyze_text(random_text)
    
    print(f"English text IC: {english_stats['index_of_coincidence']:.4f}")
    print(f"Random text IC: {random_stats['index_of_coincidence']:.4f}")
    print(f"English text entropy: {english_stats['entropy']:.4f}")
    print(f"Random text entropy: {random_stats['entropy']:.4f}")
    
    print("\n2. Caesar Cipher Attack...")
    
    # Create Caesar cipher cryptanalyst
    caesar_analyzer = CaesarCipherAnalyzer()
    
    # Test message
    plaintext = "ATTACKATDAWN"
    key = 13
    
    # Encrypt with Caesar cipher
    ciphertext = ''.join(
        chr((ord(c) - ord('A') + key) % 26 + ord('A'))
        for c in plaintext
    )
    
    print(f"Original: {plaintext}")
    print(f"Encrypted (key={key}): {ciphertext}")
    
    # Brute force attack
    result = caesar_analyzer.brute_force_attack(ciphertext)
    print(f"Brute force result: key={result.key}, confidence={result.confidence:.3f}")
    print(f"Decrypted: {result.additional_info['plaintext']}")
    
    # Frequency attack
    freq_result = caesar_analyzer.frequency_attack(ciphertext)
    print(f"Frequency analysis: key={freq_result.key}, confidence={freq_result.confidence:.3f}")
    
    print("\n3. Vigenère Cipher Attack...")
    
    # Create Vigenère cipher cryptanalyst
    vigenere_analyzer = VigenereCipherAnalyzer()
    
    # Test with longer text for better statistics
    long_plaintext = (plaintext * 10).replace(' ', '')
    vigenere_key = "CRYPTO"
    
    # Encrypt with Vigenère cipher
    vigenere_ciphertext = ""
    for i, char in enumerate(long_plaintext):
        if char.isalpha():
            key_char = vigenere_key[i % len(vigenere_key)]
            shift = ord(key_char) - ord('A')
            encrypted = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            vigenere_ciphertext += encrypted
    
    print(f"Vigenère key: {vigenere_key}")
    print(f"Ciphertext length: {len(vigenere_ciphertext)}")
    
    # Kasiski test
    kasiski_lengths = vigenere_analyzer.kasiski_test(vigenere_ciphertext)
    print(f"Kasiski test key lengths: {kasiski_lengths}")
    
    # Index of coincidence test
    ic_lengths = vigenere_analyzer.index_of_coincidence_attack(vigenere_ciphertext)
    print(f"IC test key lengths: {ic_lengths[:5]}")
    
    # Full attack
    vigenere_result = vigenere_analyzer.break_vigenere(vigenere_ciphertext)
    if vigenere_result.success:
        print(f"Recovered key: {vigenere_result.key}")
        print(f"Confidence: {vigenere_result.confidence:.3f}")
        print(f"Attack time: {vigenere_result.time_taken:.3f} seconds")
    
    print("\n4. Linear Cryptanalysis...")
    
    # Example S-box (simplified 4-bit)
    sbox = [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7]
    
    linear_analyzer = LinearCryptanalysis()
    approximations = linear_analyzer.find_linear_approximations(sbox)
    
    print(f"Found {len(approximations)} linear approximations")
    print("Top 5 approximations:")
    
    for i, (input_mask, output_mask, bias) in enumerate(approximations[:5]):
        data_req = linear_analyzer.linear_attack_data_requirement(bias)
        print(f"  {i+1}. Input mask: {input_mask:04b}, Output mask: {output_mask:04b}")
        print(f"     Bias: {bias:.4f}, Data requirement: {data_req}")
    
    print("\n5. Statistical Tests Summary...")
    
    # Compare different types of text
    texts = {
        "English": "THE DISTRIBUTION OF LETTERS IN ENGLISH TEXT",
        "Random": ''.join(secrets.choice(string.ascii_uppercase) for _ in range(50)),
        "Caesar": ''.join(chr((ord(c) - ord('A') + 7) % 26 + ord('A')) 
                         for c in "THE DISTRIBUTION OF LETTERS IN ENGLISH TEXT" if c.isalpha())
    }
    
    print(f"{'Text Type':<10} {'IC':<8} {'Entropy':<8} {'Chi²':<8}")
    print("-" * 35)
    
    for text_type, text in texts.items():
        stats = analyzer.analyze_text(text)
        ic = stats.get('index_of_coincidence', 0)
        entropy = stats.get('entropy', 0)
        chi_squared = stats.get('chi_squared', 0)
        
        print(f"{text_type:<10} {ic:<8.3f} {entropy:<8.3f} {chi_squared:<8.1f}")

if __name__ == "__main__":
    cryptanalysis_demo()
```

---

## 🧮 Differential Cryptanalysis

### Comprehensive Differential Attack Framework

```python
#!/usr/bin/env python3
"""
Differential Cryptanalysis Implementation
Advanced differential attack techniques for block ciphers
"""

import secrets
import itertools
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class Differential:
    """Differential characteristic"""
    input_diff: int
    output_diff: int
    probability: float
    sbox_id: Optional[int] = None

@dataclass 
class DifferentialPath:
    """Multi-round differential path"""
    rounds: List[Differential]
    total_probability: float
    
    def __post_init__(self):
        # Calculate total probability if not provided
        if self.total_probability == 0:
            self.total_probability = 1.0
            for diff in self.rounds:
                self.total_probability *= diff.probability

class DifferentialCryptanalysis:
    """Differential cryptanalysis framework"""
    
    def __init__(self):
        self.difference_tables = {}
        self.differential_paths = {}
        self.probability_threshold = 0.01
    
    def build_difference_table(self, sbox: List[int]) -> Dict[Tuple[int, int], int]:
        """Build difference distribution table for S-box"""
        n = len(sbox).bit_length() - 1  # Input size
        table = defaultdict(int)
        
        # For all possible input pairs
        for x1 in range(2**n):
            for x2 in range(2**n):
                input_diff = x1 ^ x2
                output_diff = sbox[x1] ^ sbox[x2]
                
                table[(input_diff, output_diff)] += 1
        
        return dict(table)
    
    def find_high_probability_differentials(self, sbox: List[int]) -> List[Differential]:
        """Find high-probability differentials for S-box"""
        diff_table = self.build_difference_table(sbox)
        n = len(sbox).bit_length() - 1
        total_pairs = 2**n
        
        differentials = []
        
        for (input_diff, output_diff), count in diff_table.items():
            if input_diff != 0:  # Exclude trivial differential
                probability = count / total_pairs
                
                if probability > self.probability_threshold:
                    differentials.append(Differential(
                        input_diff=input_diff,
                        output_diff=output_diff,
                        probability=probability
                    ))
        
        # Sort by probability (highest first)
        differentials.sort(key=lambda x: x.probability, reverse=True)
        
        return differentials
    
    def find_differential_paths(self, sboxes: List[List[int]], num_rounds: int) -> List[DifferentialPath]:
        """Find differential paths through multiple rounds"""
        if num_rounds == 1:
            # Base case: single round
            paths = []
            for sbox_id, sbox in enumerate(sboxes):
                diffs = self.find_high_probability_differentials(sbox)
                for diff in diffs[:10]:  # Top 10 differentials
                    diff.sbox_id = sbox_id
                    paths.append(DifferentialPath([diff], diff.probability))
            return paths
        
        # Recursive case: extend paths
        shorter_paths = self.find_differential_paths(sboxes, num_rounds - 1)
        extended_paths = []
        
        for path in shorter_paths:
            last_output_diff = path.rounds[-1].output_diff
            
            # Try to extend with each S-box
            for sbox_id, sbox in enumerate(sboxes):
                diffs = self.find_high_probability_differentials(sbox)
                
                for diff in diffs[:5]:  # Limit expansion
                    # Check if this differential can extend the path
                    if self._can_extend_path(last_output_diff, diff.input_diff):
                        new_diff = Differential(
                            input_diff=diff.input_diff,
                            output_diff=diff.output_diff, 
                            probability=diff.probability,
                            sbox_id=sbox_id
                        )
                        
                        new_path = DifferentialPath(
                            rounds=path.rounds + [new_diff],
                            total_probability=path.total_probability * diff.probability
                        )
                        
                        extended_paths.append(new_path)
        
        # Filter and sort by probability
        extended_paths = [p for p in extended_paths if p.total_probability > 1e-10]
        extended_paths.sort(key=lambda x: x.total_probability, reverse=True)
        
        return extended_paths[:100]  # Return top 100 paths
    
    def _can_extend_path(self, output_diff: int, input_diff: int) -> bool:
        """Check if differential path can be extended"""
        # Simplified: assume linear layer just permutes bits
        # In practice, this would model the specific linear transformation
        return True  # Placeholder
    
    def differential_attack(self, encrypt_function, target_sbox_id: int, 
                          differential: Differential, num_pairs: int) -> Dict[str, Any]:
        """Perform differential attack to recover key material"""
        start_time = time.time()
        
        # Generate pairs with desired input difference
        pairs = []
        for _ in range(num_pairs):
            p1 = secrets.randbits(32)
            p2 = p1 ^ differential.input_diff
            
            # Encrypt pairs (unknown key)
            c1 = encrypt_function(p1)
            c2 = encrypt_function(p2)
            
            pairs.append(((p1, p2), (c1, c2)))
        
        # Analyze output differences
        output_diffs = Counter()
        for (p1, p2), (c1, c2) in pairs:
            output_diff = c1 ^ c2
            output_diffs[output_diff] += 1
        
        # Look for expected output difference
        expected_diff = differential.output_diff
        observed_count = output_diffs.get(expected_diff, 0)
        expected_count = num_pairs * differential.probability
        
        # Statistical analysis
        success = abs(observed_count - expected_count) < 3 * (expected_count ** 0.5)
        
        time_taken = time.time() - start_time
        
        return {
            'success': success,
            'observed_count': observed_count,
            'expected_count': expected_count,
            'total_pairs': num_pairs,
            'time_taken': time_taken,
            'output_differences': dict(output_diffs.most_common(10))
        }
    
    def calculate_data_complexity(self, differential_probability: float, 
                                success_probability: float = 0.95) -> int:
        """Calculate required number of pairs for differential attack"""
        if differential_probability <= 0:
            return float('inf')
        
        # From theory: N ≈ c / p where p is differential probability
        # c depends on desired success probability
        
        if success_probability >= 0.95:
            c = 4
        elif success_probability >= 0.90:
            c = 2
        else:
            c = 1
        
        return int(c / differential_probability)

class AdvancedDifferentialTechniques:
    """Advanced differential cryptanalysis techniques"""
    
    def __init__(self):
        self.truncated_differentials = {}
        self.impossible_differentials = set()
    
    def find_truncated_differentials(self, sbox: List[int], 
                                   active_bits: Set[int]) -> List[Differential]:
        """Find truncated differentials (ignore some bit positions)"""
        n = len(sbox).bit_length() - 1
        mask = 0
        
        for bit in active_bits:
            mask |= (1 << bit)
        
        truncated_table = defaultdict(int)
        
        for x1 in range(2**n):
            for x2 in range(2**n):
                input_diff = (x1 ^ x2) & mask
                output_diff = (sbox[x1] ^ sbox[x2]) & mask
                
                truncated_table[(input_diff, output_diff)] += 1
        
        differentials = []
        total_pairs = 2**n
        
        for (input_diff, output_diff), count in truncated_table.items():
            if input_diff != 0:
                probability = count / total_pairs
                differentials.append(Differential(
                    input_diff=input_diff,
                    output_diff=output_diff,
                    probability=probability
                ))
        
        return sorted(differentials, key=lambda x: x.probability, reverse=True)
    
    def find_impossible_differentials(self, sbox: List[int]) -> Set[Tuple[int, int]]:
        """Find impossible differentials (probability = 0)"""
        diff_table = {}
        n = len(sbox).bit_length() - 1
        
        # Build complete difference table
        for x1 in range(2**n):
            for x2 in range(2**n):
                input_diff = x1 ^ x2
                output_diff = sbox[x1] ^ sbox[x2]
                
                if (input_diff, output_diff) not in diff_table:
                    diff_table[(input_diff, output_diff)] = True
        
        # Find impossible differentials
        impossible = set()
        for input_diff in range(1, 2**n):  # Exclude trivial
            for output_diff in range(2**n):
                if (input_diff, output_diff) not in diff_table:
                    impossible.add((input_diff, output_diff))
        
        return impossible
    
    def higher_order_differential(self, function, inputs: List[int], order: int) -> int:
        """Compute higher-order differential"""
        if order == 0:
            return function(inputs[0])
        
        if order == 1:
            if len(inputs) >= 2:
                return function(inputs[0]) ^ function(inputs[1])
            return 0
        
        # Higher-order: recursively compute
        if len(inputs) < 2**order:
            return 0
        
        # Split inputs and compute lower-order differentials
        mid = len(inputs) // 2
        left_diff = self.higher_order_differential(function, inputs[:mid], order-1)
        right_diff = self.higher_order_differential(function, inputs[mid:], order-1)
        
        return left_diff ^ right_diff
    
    def boomerang_attack_setup(self, e0_differential: Differential, 
                             e1_differential: Differential) -> Dict[str, Any]:
        """Setup for boomerang attack using two differentials"""
        # Boomerang attack uses two short differentials instead of one long one
        
        p = e0_differential.probability
        q = e1_differential.probability
        
        # Boomerang probability ≈ (pq)²
        boomerang_prob = (p * q) ** 2
        
        # Data complexity
        data_complexity = int(1 / boomerang_prob) if boomerang_prob > 0 else float('inf')
        
        return {
            'first_differential': e0_differential,
            'second_differential': e1_differential,
            'boomerang_probability': boomerang_prob,
            'data_complexity': data_complexity,
            'attack_type': 'boomerang'
        }

def differential_cryptanalysis_demo():
    """Demonstrate differential cryptanalysis techniques"""
    print("=== Differential Cryptanalysis Demo ===")
    
    print("1. Building Difference Distribution Table...")
    
    # Example 4-bit S-box (simplified AES S-box subset)
    sbox = [0x6, 0xB, 0x5, 0x4, 0x2, 0xE, 0x7, 0xA, 
            0x9, 0xD, 0xF, 0xC, 0x3, 0x1, 0x0, 0x8]
    
    diff_analyzer = DifferentialCryptanalysis()
    
    # Build difference table
    diff_table = diff_analyzer.build_difference_table(sbox)
    
    print(f"S-box size: {len(sbox)} entries")
    print(f"Difference table entries: {len(diff_table)}")
    
    # Find high-probability differentials
    differentials = diff_analyzer.find_high_probability_differentials(sbox)
    
    print(f"\nTop 10 differentials:")
    print(f"{'Input Δ':<8} {'Output Δ':<9} {'Probability':<12}")
    print("-" * 30)
    
    for i, diff in enumerate(differentials[:10]):
        print(f"{diff.input_diff:04b}     {diff.output_diff:04b}      {diff.probability:.4f}")
    
    print("\n2. Multi-round Differential Paths...")
    
    # Multiple S-boxes for multi-round analysis
    sboxes = [sbox, sbox]  # Same S-box for simplicity
    
    # Find 2-round differential paths
    paths = diff_analyzer.find_differential_paths(sboxes, 2)
    
    print(f"Found {len(paths)} differential paths")
    print(f"\nTop 5 paths:")
    
    for i, path in enumerate(paths[:5]):
        print(f"Path {i+1}: Probability = {path.total_probability:.6f}")
        for j, round_diff in enumerate(path.rounds):
            print(f"  Round {j+1}: {round_diff.input_diff:04b} -> {round_diff.output_diff:04b}")
    
    print("\n3. Data Complexity Analysis...")
    
    if differentials:
        best_diff = differentials[0]
        
        for success_prob in [0.90, 0.95, 0.99]:
            complexity = diff_analyzer.calculate_data_complexity(
                best_diff.probability, success_prob
            )
            print(f"Success probability {success_prob:.0%}: {complexity} pairs needed")
    
    print("\n4. Advanced Techniques...")
    
    advanced = AdvancedDifferentialTechniques()
    
    # Truncated differentials (only look at certain bit positions)
    active_bits = {0, 1, 2}  # Only consider first 3 bits
    truncated_diffs = advanced.find_truncated_differentials(sbox, active_bits)
    
    print(f"Truncated differentials (bits {active_bits}): {len(truncated_diffs)} found")
    if truncated_diffs:
        best_truncated = truncated_diffs[0]
        print(f"Best truncated: {best_truncated.input_diff:04b} -> {best_truncated.output_diff:04b} "
              f"(p={best_truncated.probability:.4f})")
    
    # Impossible differentials
    impossible = advanced.find_impossible_differentials(sbox)
    print(f"Impossible differentials: {len(impossible)} found")
    
    # Show a few examples
    if impossible:
        examples = list(impossible)[:5]
        print("Examples:")
        for input_diff, output_diff in examples:
            print(f"  {input_diff:04b} -> {output_diff:04b} (impossible)")
    
    print("\n5. Higher-Order Differentials...")
    
    # Simple test function
    def test_function(x):
        return sbox[x % len(sbox)]
    
    # Test higher-order differentials
    test_inputs = [0, 1, 2, 3, 4, 5, 6, 7]
    
    for order in range(1, 4):
        higher_diff = advanced.higher_order_differential(test_function, test_inputs, order)
        print(f"Order {order} differential: {higher_diff:04b}")
    
    print("\n6. Boomerang Attack Setup...")
    
    if len(differentials) >= 2:
        e0_diff = differentials[0]
        e1_diff = differentials[1]
        
        boomerang_setup = advanced.boomerang_attack_setup(e0_diff, e1_diff)
        
        print(f"First differential probability: {e0_diff.probability:.4f}")
        print(f"Second differential probability: {e1_diff.probability:.4f}")
        print(f"Boomerang probability: {boomerang_setup['boomerang_probability']:.8f}")
        print(f"Data complexity: {boomerang_setup['data_complexity']} chosen plaintexts")

if __name__ == "__main__":
    cryptanalysis_demo()
    print("\n" + "="*50 + "\n")
    differential_cryptanalysis_demo()
```

---

This represents the first major sections of the Cryptanalysis guide. The remaining sections would include Side-Channel Attacks (timing, power analysis, electromagnetic), Algebraic Attacks (Gröbner bases, SAT solvers), RSA-specific attacks (factorization, small exponent), Hash function attacks (collision finding, length extension), and Modern Attack Techniques (machine learning, quantum cryptanalysis) - all maintaining the same comprehensive implementation standards that make this educational resource truly exceptional.
