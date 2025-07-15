#!/usr/bin/env python3
"""
Advanced Cryptographic Tools and Utilities
Specialized tools for cryptographic analysis and implementation
"""

import hashlib
import hmac
import secrets
import struct
import time
import json
import base64
import itertools
from typing import List, Tuple, Optional, Dict, Any, Union, Iterator
from dataclasses import dataclass, field
import math
import statistics
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
# ADVANCED CRYPTANALYSIS TOOLS
# ===================================================================

class CryptoAnalyzer:
    """Advanced cryptanalysis toolkit"""
    
    @staticmethod
    def entropy_analysis(data: bytes) -> Dict[str, float]:
        """Calculate entropy and randomness metrics"""
        if not data:
            return {'entropy': 0, 'max_entropy': 0, 'randomness_ratio': 0}
        
        # Byte frequency analysis
        byte_counts = Counter(data)
        total_bytes = len(data)
        
        # Calculate Shannon entropy
        entropy = 0
        for count in byte_counts.values():
            probability = count / total_bytes
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        max_entropy = 8.0  # Maximum entropy for byte data
        randomness_ratio = entropy / max_entropy
        
        return {
            'entropy': entropy,
            'max_entropy': max_entropy,
            'randomness_ratio': randomness_ratio,
            'unique_bytes': len(byte_counts),
            'most_common_byte': byte_counts.most_common(1)[0] if byte_counts else None
        }
    
    @staticmethod
    def hamming_distance(data1: bytes, data2: bytes) -> int:
        """Calculate Hamming distance between two byte strings"""
        if len(data1) != len(data2):
            raise ValueError("Data lengths must be equal")
        
        distance = 0
        for b1, b2 in zip(data1, data2):
            # XOR and count set bits
            xor_result = b1 ^ b2
            distance += bin(xor_result).count('1')
        
        return distance
    
    @staticmethod
    def index_of_coincidence(text: str) -> float:
        """Calculate Index of Coincidence for text"""
        text = text.upper()
        letter_counts = Counter(c for c in text if c.isalpha())
        n = sum(letter_counts.values())
        
        if n <= 1:
            return 0
        
        ic = sum(count * (count - 1) for count in letter_counts.values())
        ic = ic / (n * (n - 1))
        
        return ic
    
    @staticmethod
    def autocorrelation(data: List[float], max_lag: int = 50) -> List[float]:
        """Calculate autocorrelation for data analysis"""
        n = len(data)
        if n == 0:
            return []
        
        # Normalize data
        mean_val = statistics.mean(data)
        normalized = [x - mean_val for x in data]
        
        autocorr = []
        for lag in range(min(max_lag, n)):
            if n - lag <= 0:
                autocorr.append(0)
                continue
            
            sum_prod = sum(normalized[i] * normalized[i + lag] 
                          for i in range(n - lag))
            
            # Normalize by variance and sample size
            variance = sum(x * x for x in normalized) / n
            if variance == 0:
                autocorr.append(0)
            else:
                autocorr.append(sum_prod / ((n - lag) * variance))
        
        return autocorr

class VigenereAnalyzer:
    """Specialized Vigenère cipher analysis"""
    
    @staticmethod
    def estimate_key_length(ciphertext: str, max_length: int = 20) -> List[Tuple[int, float]]:
        """Estimate Vigenère key length using Index of Coincidence"""
        ciphertext = ''.join(c.upper() for c in ciphertext if c.isalpha())
        results = []
        
        for key_length in range(1, min(max_length + 1, len(ciphertext) // 2)):
            # Split ciphertext into groups based on key length
            groups = [''] * key_length
            for i, char in enumerate(ciphertext):
                groups[i % key_length] += char
            
            # Calculate average IC for all groups
            total_ic = sum(CryptoAnalyzer.index_of_coincidence(group) 
                          for group in groups if group)
            avg_ic = total_ic / key_length if key_length > 0 else 0
            
            results.append((key_length, avg_ic))
        
        # Sort by IC value (higher is better for key length estimation)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    @staticmethod
    def break_vigenere(ciphertext: str, key_length: int) -> Tuple[str, str]:
        """Break Vigenère cipher with known key length"""
        from complete_crypto_library import FrequencyAnalysis, ClassicalCiphers
        
        ciphertext = ''.join(c.upper() for c in ciphertext if c.isalpha())
        key = []
        
        # For each position in the key
        for pos in range(key_length):
            # Extract characters at this position
            substring = ''
            for i in range(pos, len(ciphertext), key_length):
                substring += ciphertext[i]
            
            # Find best Caesar shift for this substring
            best_shift = 0
            best_score = float('inf')
            
            for shift in range(26):
                decrypted = ClassicalCiphers.caesar_cipher(substring, shift, decrypt=True)
                frequencies = FrequencyAnalysis.analyze_frequencies(decrypted)
                score = FrequencyAnalysis.chi_squared_test(frequencies)
                
                if score < best_score:
                    best_score = score
                    best_shift = shift
            
            key.append(chr(ord('A') + best_shift))
        
        key_str = ''.join(key)
        plaintext = ClassicalCiphers.vigenere_cipher(ciphertext, key_str, decrypt=True)
        
        return plaintext, key_str

class BlockCipherAnalyzer:
    """Analysis tools for block ciphers"""
    
    @staticmethod
    def detect_ecb_mode(ciphertext: bytes, block_size: int = 16) -> Dict[str, Any]:
        """Detect ECB mode by looking for repeated blocks"""
        if len(ciphertext) % block_size != 0:
            return {'is_ecb': False, 'confidence': 0, 'repeated_blocks': 0}
        
        blocks = []
        for i in range(0, len(ciphertext), block_size):
            blocks.append(ciphertext[i:i + block_size])
        
        unique_blocks = set(blocks)
        repeated_blocks = len(blocks) - len(unique_blocks)
        
        # Calculate confidence based on repetition rate
        repetition_rate = repeated_blocks / len(blocks) if blocks else 0
        confidence = min(repetition_rate * 10, 1.0)  # Scale to 0-1
        
        return {
            'is_ecb': repeated_blocks > 0,
            'confidence': confidence,
            'repeated_blocks': repeated_blocks,
            'total_blocks': len(blocks),
            'unique_blocks': len(unique_blocks),
            'repetition_rate': repetition_rate
        }
    
    @staticmethod
    def padding_oracle_simulation(oracle_function, ciphertext: bytes, 
                                block_size: int = 16) -> bytes:
        """Simulate padding oracle attack (educational purposes)"""
        # This is a simplified simulation for educational understanding
        # Real padding oracle attacks are more complex
        
        if len(ciphertext) % block_size != 0:
            raise ValueError("Ciphertext length must be multiple of block size")
        
        blocks = []
        for i in range(0, len(ciphertext), block_size):
            blocks.append(ciphertext[i:i + block_size])
        
        decrypted_blocks = []
        
        # Attack each block (except the first, which is IV)
        for block_index in range(1, len(blocks)):
            target_block = blocks[block_index]
            previous_block = blocks[block_index - 1]
            
            # This would contain the actual padding oracle attack logic
            # For simulation, we'll return a placeholder
            decrypted_block = b'SIMULATED_DECRYPT_' + bytes([block_index])
            decrypted_blocks.append(decrypted_block[:block_size])
        
        return b''.join(decrypted_blocks)

class SideChannelAnalyzer:
    """Side-channel attack analysis tools"""
    
    @staticmethod
    def timing_attack_simulation(operation_function, inputs: List[Any], 
                               samples: int = 100) -> Dict[str, Any]:
        """Simulate timing attack analysis"""
        timing_data = defaultdict(list)
        
        for input_val in inputs:
            for _ in range(samples):
                start_time = time.perf_counter()
                operation_function(input_val)
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000000  # microseconds
                timing_data[str(input_val)].append(execution_time)
        
        # Statistical analysis
        analysis = {}
        for input_val, times in timing_data.items():
            analysis[input_val] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times),
                'samples': len(times)
            }
        
        # Check for timing differences
        means = [data['mean'] for data in analysis.values()]
        timing_variance = statistics.variance(means) if len(means) > 1 else 0
        
        return {
            'analysis': analysis,
            'timing_variance': timing_variance,
            'vulnerable': timing_variance > 1.0,  # Threshold for concern
            'recommendation': 'Use constant-time implementations' if timing_variance > 1.0 else 'Timing appears constant'
        }
    
    @staticmethod
    def power_analysis_simulation(key_guess: bytes, known_plaintext: bytes) -> Dict[str, float]:
        """Simulate power analysis correlation"""
        # Simulate Hamming weight model for power consumption
        def hamming_weight(value: int) -> int:
            return bin(value).count('1')
        
        # Simulate intermediate values during encryption
        correlations = {}
        
        for key_byte_pos in range(min(len(key_guess), 16)):  # AES has 16 bytes
            key_byte = key_guess[key_byte_pos]
            
            # Simulate S-box output for first round
            if key_byte_pos < len(known_plaintext):
                sbox_input = known_plaintext[key_byte_pos] ^ key_byte
                
                # Simplified AES S-box (first few values)
                sbox = [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5]
                sbox_output = sbox[sbox_input % len(sbox)]
                
                # Calculate hypothetical power consumption
                power_consumption = hamming_weight(sbox_output)
                
                # Simulate correlation with "measured" power traces
                correlation = abs(power_consumption - 4) / 4  # Normalize to 0-1
                correlations[f'byte_{key_byte_pos}'] = correlation
        
        return correlations

# ===================================================================
# PROTOCOL ANALYSIS TOOLS
# ===================================================================

class ProtocolAnalyzer:
    """Protocol security analysis tools"""
    
    @staticmethod
    def tls_handshake_analyzer(cipher_suites: List[str]) -> Dict[str, Any]:
        """Analyze TLS cipher suites for security"""
        analysis = {
            'secure_suites': [],
            'weak_suites': [],
            'deprecated_suites': [],
            'warnings': []
        }
        
        # Define security classifications
        weak_algorithms = ['RC4', 'DES', '3DES', 'MD5', 'SHA1']
        deprecated_algorithms = ['SSL', 'TLS1.0', 'TLS1.1']
        secure_algorithms = ['AES', 'ChaCha20', 'SHA256', 'SHA384', 'ECDHE', 'DHE']
        
        for suite in cipher_suites:
            suite_upper = suite.upper()
            
            is_weak = any(weak in suite_upper for weak in weak_algorithms)
            is_deprecated = any(dep in suite_upper for dep in deprecated_algorithms)
            is_secure = any(sec in suite_upper for sec in secure_algorithms)
            
            if is_weak:
                analysis['weak_suites'].append(suite)
                analysis['warnings'].append(f"Weak cipher suite: {suite}")
            elif is_deprecated:
                analysis['deprecated_suites'].append(suite)
                analysis['warnings'].append(f"Deprecated cipher suite: {suite}")
            elif is_secure:
                analysis['secure_suites'].append(suite)
            else:
                analysis['warnings'].append(f"Unknown cipher suite: {suite}")
        
        # Overall security assessment
        total_suites = len(cipher_suites)
        secure_ratio = len(analysis['secure_suites']) / total_suites if total_suites > 0 else 0
        
        if secure_ratio > 0.8:
            security_level = "Good"
        elif secure_ratio > 0.5:
            security_level = "Moderate"
        else:
            security_level = "Poor"
        
        analysis['security_level'] = security_level
        analysis['secure_ratio'] = secure_ratio
        
        return analysis
    
    @staticmethod
    def certificate_analyzer(cert_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze certificate security properties"""
        analysis = {
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'security_score': 100
        }
        
        # Check key size
        key_size = cert_info.get('key_size', 0)
        if key_size < 2048:
            analysis['issues'].append(f"Weak key size: {key_size} bits")
            analysis['security_score'] -= 30
        elif key_size < 3072:
            analysis['warnings'].append(f"Key size {key_size} bits is adequate but consider upgrading")
            analysis['security_score'] -= 10
        
        # Check signature algorithm
        sig_algorithm = cert_info.get('signature_algorithm', '').lower()
        if 'sha1' in sig_algorithm:
            analysis['issues'].append("Weak signature algorithm: SHA-1")
            analysis['security_score'] -= 40
        elif 'md5' in sig_algorithm:
            analysis['issues'].append("Critically weak signature algorithm: MD5")
            analysis['security_score'] -= 60
        
        # Check expiration
        days_until_expiry = cert_info.get('days_until_expiry', 0)
        if days_until_expiry <= 0:
            analysis['issues'].append("Certificate has expired")
            analysis['security_score'] -= 50
        elif days_until_expiry <= 30:
            analysis['warnings'].append(f"Certificate expires in {days_until_expiry} days")
            analysis['security_score'] -= 20
        
        # Generate recommendations
        if analysis['issues']:
            analysis['recommendations'].append("Replace certificate immediately")
        if key_size < 3072:
            analysis['recommendations'].append("Upgrade to at least 3072-bit keys")
        if 'sha' not in sig_algorithm or 'sha1' in sig_algorithm:
            analysis['recommendations'].append("Use SHA-256 or stronger signature algorithm")
        
        # Overall assessment
        if analysis['security_score'] >= 80:
            analysis['assessment'] = "Secure"
        elif analysis['security_score'] >= 60:
            analysis['assessment'] = "Acceptable"
        else:
            analysis['assessment'] = "Insecure"
        
        return analysis

# ===================================================================
# PERFORMANCE BENCHMARKING TOOLS
# ===================================================================

class CryptoBenchmark:
    """Cryptographic performance benchmarking"""
    
    @staticmethod
    def benchmark_function(func, *args, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark a cryptographic function"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # milliseconds
        
        return {
            'mean_time_ms': statistics.mean(times),
            'median_time_ms': statistics.median(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0,
            'iterations': iterations
        }
    
    @staticmethod
    def compare_algorithms(algorithms: Dict[str, callable], 
                          test_data: Any, iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Compare performance of multiple algorithms"""
        results = {}
        
        for name, algorithm in algorithms.items():
            try:
                results[name] = CryptoBenchmark.benchmark_function(
                    algorithm, test_data, iterations=iterations
                )
                results[name]['status'] = 'success'
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results

# ===================================================================
# VISUALIZATION TOOLS
# ===================================================================

class CryptoVisualizer:
    """Cryptographic visualization tools"""
    
    @staticmethod
    def plot_frequency_analysis(text: str, title: str = "Character Frequency Analysis"):
        """Plot character frequency analysis"""
        from complete_crypto_library import FrequencyAnalysis
        
        frequencies = FrequencyAnalysis.analyze_frequencies(text)
        expected_freq = FrequencyAnalysis.ENGLISH_FREQUENCIES
        
        chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        observed = [frequencies.get(char, 0) for char in chars]
        expected = [expected_freq[char] for char in chars]
        
        x = range(len(chars))
        width = 0.35
        
        plt.figure(figsize=(15, 6))
        plt.bar([i - width/2 for i in x], observed, width, label='Observed', alpha=0.7)
        plt.bar([i + width/2 for i in x], expected, width, label='Expected English', alpha=0.7)
        
        plt.xlabel('Characters')
        plt.ylabel('Frequency (%)')
        plt.title(title)
        plt.xticks(x, chars)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_entropy_over_time(data: bytes, window_size: int = 256):
        """Plot entropy changes over data"""
        entropies = []
        positions = []
        
        for i in range(0, len(data) - window_size, window_size // 4):
            window = data[i:i + window_size]
            entropy_info = CryptoAnalyzer.entropy_analysis(window)
            entropies.append(entropy_info['entropy'])
            positions.append(i)
        
        plt.figure(figsize=(12, 6))
        plt.plot(positions, entropies, 'b-', linewidth=2)
        plt.axhline(y=8.0, color='r', linestyle='--', label='Maximum Entropy')
        plt.xlabel('Position in Data')
        plt.ylabel('Entropy (bits)')
        plt.title('Entropy Analysis Over Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_timing_analysis(timing_data: Dict[str, Dict[str, float]]):
        """Plot timing analysis results"""
        inputs = list(timing_data.keys())
        means = [data['mean'] for data in timing_data.values()]
        stdevs = [data['stdev'] for data in timing_data.values()]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(range(len(inputs)), means, yerr=stdevs, 
                    fmt='o-', capsize=5, capthick=2)
        plt.xlabel('Input Values')
        plt.ylabel('Execution Time (μs)')
        plt.title('Timing Analysis - Potential Side Channel')
        plt.xticks(range(len(inputs)), inputs, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ===================================================================
# DEMONSTRATION FUNCTIONS
# ===================================================================

def demonstrate_advanced_tools():
    """Demonstrate advanced cryptographic tools"""
    print("=== Advanced Cryptographic Tools Demo ===\n")
    
    # 1. Entropy Analysis
    print("1. Entropy Analysis:")
    random_data = secrets.token_bytes(1000)
    pattern_data = b'AAAA' * 250  # Low entropy data
    
    random_entropy = CryptoAnalyzer.entropy_analysis(random_data)
    pattern_entropy = CryptoAnalyzer.entropy_analysis(pattern_data)
    
    print(f"Random data entropy: {random_entropy['entropy']:.2f} bits")
    print(f"Pattern data entropy: {pattern_entropy['entropy']:.2f} bits")
    print(f"Random data randomness ratio: {random_entropy['randomness_ratio']:.2f}")
    print(f"Pattern data randomness ratio: {pattern_entropy['randomness_ratio']:.2f}")
    
    # 2. Vigenère Analysis
    print("\n2. Vigenère Cipher Analysis:")
    # Sample Vigenère ciphertext
    vigenere_cipher = "LXFOPVEFRNHR"  # "HELLO WORLD" encrypted with key "KEY"
    
    key_lengths = VigenereAnalyzer.estimate_key_length(vigenere_cipher, max_length=10)
    print("Estimated key lengths (length, IC score):")
    for length, ic in key_lengths[:3]:
        print(f"  Length {length}: IC = {ic:.4f}")
    
    if key_lengths:
        best_length = key_lengths[0][0]
        plaintext, key = VigenereAnalyzer.break_vigenere(vigenere_cipher, best_length)
        print(f"Broken cipher with key length {best_length}:")
        print(f"  Key: {key}")
        print(f"  Plaintext: {plaintext}")
    
    # 3. Block Cipher Analysis
    print("\n3. Block Cipher Analysis:")
    # Simulate ECB mode detection
    repeated_block = b'\x41' * 16  # Repeated block
    ecb_like_data = repeated_block * 3 + secrets.token_bytes(16)
    cbc_like_data = secrets.token_bytes(64)  # Random-looking data
    
    ecb_analysis = BlockCipherAnalyzer.detect_ecb_mode(ecb_like_data)
    cbc_analysis = BlockCipherAnalyzer.detect_ecb_mode(cbc_like_data)
    
    print(f"ECB-like data: {ecb_analysis['is_ecb']}, confidence: {ecb_analysis['confidence']:.2f}")
    print(f"CBC-like data: {cbc_analysis['is_ecb']}, confidence: {cbc_analysis['confidence']:.2f}")
    
    # 4. Protocol Analysis
    print("\n4. Protocol Analysis:")
    test_cipher_suites = [
        "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
        "TLS_RSA_WITH_RC4_128_SHA",  # Weak
        "TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA",  # Deprecated
        "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256"
    ]
    
    tls_analysis = ProtocolAnalyzer.tls_handshake_analyzer(test_cipher_suites)
    print(f"TLS Security Level: {tls_analysis['security_level']}")
    print(f"Secure suites: {len(tls_analysis['secure_suites'])}")
    print(f"Weak suites: {len(tls_analysis['weak_suites'])}")
    if tls_analysis['warnings']:
        print("Warnings:")
        for warning in tls_analysis['warnings'][:3]:
            print(f"  - {warning}")
    
    # 5. Certificate Analysis
    print("\n5. Certificate Analysis:")
    test_cert = {
        'key_size': 2048,
        'signature_algorithm': 'sha256WithRSAEncryption',
        'days_until_expiry': 90
    }
    
    cert_analysis = ProtocolAnalyzer.certificate_analyzer(test_cert)
    print(f"Certificate assessment: {cert_analysis['assessment']}")
    print(f"Security score: {cert_analysis['security_score']}/100")
    if cert_analysis['recommendations']:
        print("Recommendations:")
        for rec in cert_analysis['recommendations']:
            print(f"  - {rec}")
    
    # 6. Performance Benchmarking
    print("\n6. Performance Benchmarking:")
    
    # Benchmark hash functions
    test_data = b"The quick brown fox jumps over the lazy dog" * 100
    
    hash_algorithms = {
        'SHA-256 (hashlib)': lambda data: hashlib.sha256(data).digest(),
        'SHA-512 (hashlib)': lambda data: hashlib.sha512(data).digest(),
        'MD5 (hashlib)': lambda data: hashlib.md5(data).digest(),
    }
    
    benchmark_results = CryptoBenchmark.compare_algorithms(
        hash_algorithms, test_data, iterations=100
    )
    
    print("Hash function performance (100 iterations):")
    for name, result in benchmark_results.items():
        if result['status'] == 'success':
            print(f"  {name}: {result['mean_time_ms']:.3f} ms (avg)")
    
    # 7. Side-Channel Analysis
    print("\n7. Side-Channel Analysis Simulation:")
    
    # Simulate vulnerable comparison function
    def vulnerable_compare(guess: str) -> bool:
        secret = "SECRET123"
        for i, (g, s) in enumerate(zip(guess, secret)):
            if g != s:
                return False
            # Simulate some processing time per character
            time.sleep(0.0001)  # 0.1ms per character
        return len(guess) == len(secret)
    
    test_inputs = ["S", "SE", "SEC", "SECR", "WRONG"]
    timing_results = SideChannelAnalyzer.timing_attack_simulation(
        vulnerable_compare, test_inputs, samples=10
    )
    
    print(f"Timing analysis vulnerability: {timing_results['vulnerable']}")
    print(f"Timing variance: {timing_results['timing_variance']:.2f}")
    print(f"Recommendation: {timing_results['recommendation']}")
    
    print("\n=== Advanced Tools Demo Complete ===")

if __name__ == "__main__":
    demonstrate_advanced_tools()
