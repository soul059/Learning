/**
 * Web Cryptography API Implementations
 * Modern browser-based cryptographic operations
 * Author: Cryptography Mastery Guide
 * License: Educational Use Only
 */

// ===================================================================
// 1. WEB CRYPTO API UTILITIES
// ===================================================================

class WebCryptoUtils {
    /**
     * Generate cryptographically secure random bytes
     */
    static generateRandomBytes(length) {
        const array = new Uint8Array(length);
        crypto.getRandomValues(array);
        return array;
    }

    /**
     * Convert ArrayBuffer to hex string
     */
    static arrayBufferToHex(buffer) {
        const byteArray = new Uint8Array(buffer);
        const hexCodes = [...byteArray].map(value => {
            const hexCode = value.toString(16);
            return hexCode.padStart(2, '0');
        });
        return hexCodes.join('');
    }

    /**
     * Convert hex string to ArrayBuffer
     */
    static hexToArrayBuffer(hex) {
        const bytes = new Uint8Array(hex.length / 2);
        for (let i = 0; i < hex.length; i += 2) {
            bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
        }
        return bytes.buffer;
    }

    /**
     * Convert string to ArrayBuffer
     */
    static stringToArrayBuffer(str) {
        const encoder = new TextEncoder();
        return encoder.encode(str);
    }

    /**
     * Convert ArrayBuffer to string
     */
    static arrayBufferToString(buffer) {
        const decoder = new TextDecoder();
        return decoder.decode(buffer);
    }

    /**
     * Encode ArrayBuffer to Base64
     */
    static arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        bytes.forEach(byte => binary += String.fromCharCode(byte));
        return btoa(binary);
    }

    /**
     * Decode Base64 to ArrayBuffer
     */
    static base64ToArrayBuffer(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes.buffer;
    }
}

// ===================================================================
// 2. SYMMETRIC ENCRYPTION
// ===================================================================

class SymmetricCrypto {
    /**
     * Generate AES key
     */
    static async generateAESKey(length = 256) {
        return await crypto.subtle.generateKey(
            {
                name: "AES-GCM",
                length: length
            },
            true, // extractable
            ["encrypt", "decrypt"]
        );
    }

    /**
     * AES-GCM encryption
     */
    static async encryptAESGCM(data, key, additionalData = null) {
        const iv = WebCryptoUtils.generateRandomBytes(12); // 96-bit IV for GCM
        
        const encrypted = await crypto.subtle.encrypt(
            {
                name: "AES-GCM",
                iv: iv,
                additionalData: additionalData ? WebCryptoUtils.stringToArrayBuffer(additionalData) : undefined
            },
            key,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );

        return {
            ciphertext: encrypted,
            iv: iv,
            additionalData: additionalData
        };
    }

    /**
     * AES-GCM decryption
     */
    static async decryptAESGCM(encryptedData, key) {
        const decrypted = await crypto.subtle.decrypt(
            {
                name: "AES-GCM",
                iv: encryptedData.iv,
                additionalData: encryptedData.additionalData ? 
                    WebCryptoUtils.stringToArrayBuffer(encryptedData.additionalData) : undefined
            },
            key,
            encryptedData.ciphertext
        );

        return WebCryptoUtils.arrayBufferToString(decrypted);
    }

    /**
     * ChaCha20-Poly1305 encryption (if supported)
     */
    static async encryptChaCha20Poly1305(data, key) {
        try {
            const iv = WebCryptoUtils.generateRandomBytes(12);
            
            const encrypted = await crypto.subtle.encrypt(
                {
                    name: "ChaCha20-Poly1305",
                    iv: iv
                },
                key,
                typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
            );

            return {
                ciphertext: encrypted,
                iv: iv
            };
        } catch (error) {
            throw new Error('ChaCha20-Poly1305 not supported in this browser');
        }
    }

    /**
     * Key derivation using PBKDF2
     */
    static async deriveKeyFromPassword(password, salt, iterations = 100000, keyLength = 256) {
        const keyMaterial = await crypto.subtle.importKey(
            "raw",
            WebCryptoUtils.stringToArrayBuffer(password),
            { name: "PBKDF2" },
            false,
            ["deriveKey"]
        );

        return await crypto.subtle.deriveKey(
            {
                name: "PBKDF2",
                salt: salt,
                iterations: iterations,
                hash: "SHA-256"
            },
            keyMaterial,
            {
                name: "AES-GCM",
                length: keyLength
            },
            true,
            ["encrypt", "decrypt"]
        );
    }
}

// ===================================================================
// 3. ASYMMETRIC ENCRYPTION
// ===================================================================

class AsymmetricCrypto {
    /**
     * Generate RSA key pair
     */
    static async generateRSAKeyPair(modulusLength = 2048) {
        return await crypto.subtle.generateKey(
            {
                name: "RSA-OAEP",
                modulusLength: modulusLength,
                publicExponent: new Uint8Array([1, 0, 1]), // 65537
                hash: "SHA-256"
            },
            true,
            ["encrypt", "decrypt"]
        );
    }

    /**
     * RSA-OAEP encryption
     */
    static async encryptRSA(data, publicKey) {
        return await crypto.subtle.encrypt(
            {
                name: "RSA-OAEP"
            },
            publicKey,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
    }

    /**
     * RSA-OAEP decryption
     */
    static async decryptRSA(encryptedData, privateKey) {
        const decrypted = await crypto.subtle.decrypt(
            {
                name: "RSA-OAEP"
            },
            privateKey,
            encryptedData
        );

        return WebCryptoUtils.arrayBufferToString(decrypted);
    }

    /**
     * Generate ECDH key pair
     */
    static async generateECDHKeyPair(namedCurve = "P-256") {
        return await crypto.subtle.generateKey(
            {
                name: "ECDH",
                namedCurve: namedCurve
            },
            true,
            ["deriveKey"]
        );
    }

    /**
     * ECDH key derivation
     */
    static async deriveSharedSecret(privateKey, publicKey) {
        return await crypto.subtle.deriveKey(
            {
                name: "ECDH",
                public: publicKey
            },
            privateKey,
            {
                name: "AES-GCM",
                length: 256
            },
            true,
            ["encrypt", "decrypt"]
        );
    }

    /**
     * Export public key to JWK format
     */
    static async exportPublicKey(key) {
        return await crypto.subtle.exportKey("jwk", key);
    }

    /**
     * Import public key from JWK format
     */
    static async importPublicKey(jwk, algorithm) {
        return await crypto.subtle.importKey(
            "jwk",
            jwk,
            algorithm,
            true,
            algorithm.name === "RSA-OAEP" ? ["encrypt"] : ["verify"]
        );
    }
}

// ===================================================================
// 4. DIGITAL SIGNATURES
// ===================================================================

class DigitalSignatures {
    /**
     * Generate RSA-PSS signing key pair
     */
    static async generateRSASigningKeyPair(modulusLength = 2048) {
        return await crypto.subtle.generateKey(
            {
                name: "RSA-PSS",
                modulusLength: modulusLength,
                publicExponent: new Uint8Array([1, 0, 1]),
                hash: "SHA-256"
            },
            true,
            ["sign", "verify"]
        );
    }

    /**
     * Generate ECDSA signing key pair
     */
    static async generateECDSAKeyPair(namedCurve = "P-256") {
        return await crypto.subtle.generateKey(
            {
                name: "ECDSA",
                namedCurve: namedCurve
            },
            true,
            ["sign", "verify"]
        );
    }

    /**
     * Generate Ed25519 signing key pair (if supported)
     */
    static async generateEd25519KeyPair() {
        try {
            return await crypto.subtle.generateKey(
                {
                    name: "Ed25519"
                },
                true,
                ["sign", "verify"]
            );
        } catch (error) {
            throw new Error('Ed25519 not supported in this browser');
        }
    }

    /**
     * RSA-PSS signature
     */
    static async signRSAPSS(data, privateKey) {
        return await crypto.subtle.sign(
            {
                name: "RSA-PSS",
                saltLength: 32
            },
            privateKey,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
    }

    /**
     * RSA-PSS signature verification
     */
    static async verifyRSAPSS(signature, data, publicKey) {
        return await crypto.subtle.verify(
            {
                name: "RSA-PSS",
                saltLength: 32
            },
            publicKey,
            signature,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
    }

    /**
     * ECDSA signature
     */
    static async signECDSA(data, privateKey) {
        return await crypto.subtle.sign(
            {
                name: "ECDSA",
                hash: "SHA-256"
            },
            privateKey,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
    }

    /**
     * ECDSA signature verification
     */
    static async verifyECDSA(signature, data, publicKey) {
        return await crypto.subtle.verify(
            {
                name: "ECDSA",
                hash: "SHA-256"
            },
            publicKey,
            signature,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
    }

    /**
     * HMAC signature
     */
    static async signHMAC(data, key) {
        return await crypto.subtle.sign(
            "HMAC",
            key,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
    }

    /**
     * HMAC signature verification
     */
    static async verifyHMAC(signature, data, key) {
        return await crypto.subtle.verify(
            "HMAC",
            key,
            signature,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
    }

    /**
     * Generate HMAC key
     */
    static async generateHMACKey(length = 256) {
        return await crypto.subtle.generateKey(
            {
                name: "HMAC",
                hash: "SHA-256",
                length: length
            },
            true,
            ["sign", "verify"]
        );
    }
}

// ===================================================================
// 5. HASH FUNCTIONS
// ===================================================================

class HashFunctions {
    /**
     * SHA-256 hash
     */
    static async sha256(data) {
        const hashBuffer = await crypto.subtle.digest(
            "SHA-256",
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
        return hashBuffer;
    }

    /**
     * SHA-384 hash
     */
    static async sha384(data) {
        const hashBuffer = await crypto.subtle.digest(
            "SHA-384",
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
        return hashBuffer;
    }

    /**
     * SHA-512 hash
     */
    static async sha512(data) {
        const hashBuffer = await crypto.subtle.digest(
            "SHA-512",
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
        return hashBuffer;
    }

    /**
     * Calculate hash with hex output
     */
    static async hashToHex(data, algorithm = "SHA-256") {
        const hashBuffer = await crypto.subtle.digest(
            algorithm,
            typeof data === 'string' ? WebCryptoUtils.stringToArrayBuffer(data) : data
        );
        return WebCryptoUtils.arrayBufferToHex(hashBuffer);
    }
}

// ===================================================================
// 6. SECURE MESSAGING IMPLEMENTATION
// ===================================================================

class SecureMessaging {
    constructor() {
        this.keyPair = null;
        this.contacts = new Map();
    }

    /**
     * Initialize user's key pair
     */
    async initialize() {
        this.keyPair = await AsymmetricCrypto.generateECDHKeyPair();
        return await AsymmetricCrypto.exportPublicKey(this.keyPair.publicKey);
    }

    /**
     * Add contact's public key
     */
    async addContact(contactId, publicKeyJWK) {
        const publicKey = await crypto.subtle.importKey(
            "jwk",
            publicKeyJWK,
            {
                name: "ECDH",
                namedCurve: "P-256"
            },
            true,
            []
        );
        this.contacts.set(contactId, publicKey);
    }

    /**
     * Send encrypted message
     */
    async sendMessage(contactId, message) {
        if (!this.contacts.has(contactId)) {
            throw new Error('Contact not found');
        }

        const contactPublicKey = this.contacts.get(contactId);
        
        // Derive shared secret
        const sharedKey = await AsymmetricCrypto.deriveSharedSecret(
            this.keyPair.privateKey,
            contactPublicKey
        );

        // Encrypt message
        const encrypted = await SymmetricCrypto.encryptAESGCM(message, sharedKey);

        return {
            to: contactId,
            iv: WebCryptoUtils.arrayBufferToBase64(encrypted.iv),
            ciphertext: WebCryptoUtils.arrayBufferToBase64(encrypted.ciphertext),
            timestamp: Date.now()
        };
    }

    /**
     * Receive encrypted message
     */
    async receiveMessage(encryptedMessage, senderPublicKeyJWK) {
        // Import sender's public key
        const senderPublicKey = await crypto.subtle.importKey(
            "jwk",
            senderPublicKeyJWK,
            {
                name: "ECDH",
                namedCurve: "P-256"
            },
            true,
            []
        );

        // Derive shared secret
        const sharedKey = await AsymmetricCrypto.deriveSharedSecret(
            this.keyPair.privateKey,
            senderPublicKey
        );

        // Decrypt message
        const decryptedData = {
            ciphertext: WebCryptoUtils.base64ToArrayBuffer(encryptedMessage.ciphertext),
            iv: WebCryptoUtils.base64ToArrayBuffer(encryptedMessage.iv)
        };

        return await SymmetricCrypto.decryptAESGCM(decryptedData, sharedKey);
    }
}

// ===================================================================
// 7. PERFORMANCE BENCHMARKING
// ===================================================================

class CryptoBenchmark {
    /**
     * Benchmark a cryptographic operation
     */
    static async benchmarkOperation(operation, iterations = 100) {
        const times = [];
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await operation();
            const end = performance.now();
            times.push(end - start);
        }

        return {
            mean: times.reduce((a, b) => a + b) / times.length,
            min: Math.min(...times),
            max: Math.max(...times),
            median: times.sort((a, b) => a - b)[Math.floor(times.length / 2)],
            iterations: iterations
        };
    }

    /**
     * Compare multiple algorithms
     */
    static async compareAlgorithms(algorithms, testData, iterations = 50) {
        const results = {};
        
        for (const [name, operation] of Object.entries(algorithms)) {
            try {
                results[name] = await this.benchmarkOperation(
                    () => operation(testData),
                    iterations
                );
                results[name].status = 'success';
            } catch (error) {
                results[name] = {
                    status: 'error',
                    error: error.message
                };
            }
        }

        return results;
    }
}

// ===================================================================
// 8. DEMONSTRATION FUNCTIONS
// ===================================================================

async function demonstrateWebCrypto() {
    console.log('=== Web Cryptography API Demo ===\n');

    try {
        // 1. Hash Functions
        console.log('1. Hash Functions:');
        const message = "The quick brown fox jumps over the lazy dog";
        const sha256Hash = await HashFunctions.hashToHex(message, "SHA-256");
        const sha512Hash = await HashFunctions.hashToHex(message, "SHA-512");
        
        console.log(`Message: ${message}`);
        console.log(`SHA-256: ${sha256Hash}`);
        console.log(`SHA-512: ${sha512Hash}`);

        // 2. Symmetric Encryption
        console.log('\n2. Symmetric Encryption:');
        const aesKey = await SymmetricCrypto.generateAESKey(256);
        const plaintext = "This is a secret message!";
        
        const encrypted = await SymmetricCrypto.encryptAESGCM(plaintext, aesKey);
        const decrypted = await SymmetricCrypto.decryptAESGCM(encrypted, aesKey);
        
        console.log(`Plaintext: ${plaintext}`);
        console.log(`Encrypted: ${WebCryptoUtils.arrayBufferToHex(encrypted.ciphertext).substring(0, 32)}...`);
        console.log(`Decrypted: ${decrypted}`);
        console.log(`IV: ${WebCryptoUtils.arrayBufferToHex(encrypted.iv)}`);

        // 3. Password-based Key Derivation
        console.log('\n3. Password-based Key Derivation:');
        const password = "my_secure_password";
        const salt = WebCryptoUtils.generateRandomBytes(32);
        const derivedKey = await SymmetricCrypto.deriveKeyFromPassword(password, salt, 100000);
        
        console.log(`Password: ${password}`);
        console.log(`Salt: ${WebCryptoUtils.arrayBufferToHex(salt)}`);
        console.log('Derived key generated successfully');

        // 4. Asymmetric Encryption
        console.log('\n4. Asymmetric Encryption:');
        const rsaKeyPair = await AsymmetricCrypto.generateRSAKeyPair(2048);
        const rsaMessage = "RSA encrypted message";
        
        const rsaEncrypted = await AsymmetricCrypto.encryptRSA(rsaMessage, rsaKeyPair.publicKey);
        const rsaDecrypted = await AsymmetricCrypto.decryptRSA(rsaEncrypted, rsaKeyPair.privateKey);
        
        console.log(`RSA Message: ${rsaMessage}`);
        console.log(`RSA Encrypted: ${WebCryptoUtils.arrayBufferToHex(rsaEncrypted).substring(0, 32)}...`);
        console.log(`RSA Decrypted: ${rsaDecrypted}`);

        // 5. Digital Signatures
        console.log('\n5. Digital Signatures:');
        const ecdsaKeyPair = await DigitalSignatures.generateECDSAKeyPair();
        const signMessage = "Document to be signed";
        
        const signature = await DigitalSignatures.signECDSA(signMessage, ecdsaKeyPair.privateKey);
        const isValid = await DigitalSignatures.verifyECDSA(signature, signMessage, ecdsaKeyPair.publicKey);
        
        console.log(`Signed message: ${signMessage}`);
        console.log(`Signature: ${WebCryptoUtils.arrayBufferToHex(signature).substring(0, 32)}...`);
        console.log(`Signature valid: ${isValid}`);

        // 6. HMAC
        console.log('\n6. HMAC Authentication:');
        const hmacKey = await DigitalSignatures.generateHMACKey();
        const hmacMessage = "Message to authenticate";
        
        const hmacSignature = await DigitalSignatures.signHMAC(hmacMessage, hmacKey);
        const hmacValid = await DigitalSignatures.verifyHMAC(hmacSignature, hmacMessage, hmacKey);
        
        console.log(`HMAC message: ${hmacMessage}`);
        console.log(`HMAC: ${WebCryptoUtils.arrayBufferToHex(hmacSignature).substring(0, 32)}...`);
        console.log(`HMAC valid: ${hmacValid}`);

        // 7. Secure Messaging
        console.log('\n7. Secure Messaging:');
        const alice = new SecureMessaging();
        const bob = new SecureMessaging();
        
        const alicePublicKey = await alice.initialize();
        const bobPublicKey = await bob.initialize();
        
        await alice.addContact('bob', bobPublicKey);
        await bob.addContact('alice', alicePublicKey);
        
        const secretMessage = "Hello Bob, this is a secret message!";
        const encryptedMsg = await alice.sendMessage('bob', secretMessage);
        const decryptedMsg = await bob.receiveMessage(encryptedMsg, alicePublicKey);
        
        console.log(`Original message: ${secretMessage}`);
        console.log(`Encrypted message: ${encryptedMsg.ciphertext.substring(0, 32)}...`);
        console.log(`Decrypted message: ${decryptedMsg}`);

        // 8. Performance Benchmarking
        console.log('\n8. Performance Benchmarking:');
        const testData = "Performance test data";
        
        const hashAlgorithms = {
            'SHA-256': (data) => HashFunctions.sha256(data),
            'SHA-384': (data) => HashFunctions.sha384(data),
            'SHA-512': (data) => HashFunctions.sha512(data)
        };
        
        const hashBenchmarks = await CryptoBenchmark.compareAlgorithms(hashAlgorithms, testData, 20);
        
        console.log('Hash function performance (20 iterations):');
        for (const [algorithm, result] of Object.entries(hashBenchmarks)) {
            if (result.status === 'success') {
                console.log(`  ${algorithm}: ${result.mean.toFixed(3)} ms (avg)`);
            }
        }

        // 9. Key Export/Import
        console.log('\n9. Key Export/Import:');
        const exportedRSAPublicKey = await AsymmetricCrypto.exportPublicKey(rsaKeyPair.publicKey);
        console.log('RSA Public Key (JWK format):');
        console.log(JSON.stringify(exportedRSAPublicKey, null, 2));

        console.log('\n=== Demo Complete ===');

    } catch (error) {
        console.error('Error during demonstration:', error);
    }
}

// ===================================================================
// 9. UTILITY FOR BROWSER COMPATIBILITY
// ===================================================================

function checkWebCryptoSupport() {
    const features = {
        'Web Crypto API': typeof crypto !== 'undefined' && typeof crypto.subtle !== 'undefined',
        'TextEncoder': typeof TextEncoder !== 'undefined',
        'TextDecoder': typeof TextDecoder !== 'undefined',
        'Performance API': typeof performance !== 'undefined',
        'Async/Await': true // If this code runs, async/await is supported
    };

    console.log('Browser Crypto Support:');
    for (const [feature, supported] of Object.entries(features)) {
        console.log(`  ${feature}: ${supported ? '✓' : '✗'}`);
    }

    return features['Web Crypto API'];
}

// ===================================================================
// 10. HTML PAGE INTEGRATION
// ===================================================================

const htmlTemplate = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Cryptography Demo</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        .crypto-demo {
            background-color: #2d2d30;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .crypto-output {
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 4px;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
            border-left: 4px solid #007acc;
        }
        button {
            background-color: #007acc;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background-color: #005a9e;
        }
        .error {
            color: #f14c4c;
        }
        .success {
            color: #73c991;
        }
    </style>
</head>
<body>
    <h1>🔐 Web Cryptography API Demo</h1>
    
    <div class="crypto-demo">
        <h2>Browser Compatibility Check</h2>
        <button onclick="runCompatibilityCheck()">Check Support</button>
        <div id="compatibility-output" class="crypto-output"></div>
    </div>

    <div class="crypto-demo">
        <h2>Full Cryptography Demo</h2>
        <button onclick="runFullDemo()">Run Demo</button>
        <div id="demo-output" class="crypto-output"></div>
    </div>

    <div class="crypto-demo">
        <h2>Interactive Encryption</h2>
        <input type="text" id="message-input" placeholder="Enter message to encrypt" style="width: 300px; padding: 8px;">
        <button onclick="encryptMessage()">Encrypt</button>
        <button onclick="decryptMessage()">Decrypt</button>
        <div id="encryption-output" class="crypto-output"></div>
    </div>

    <script>
        // Include all the cryptography classes here
        ${document.querySelector('script').textContent}
        
        let globalKey = null;
        let globalEncrypted = null;

        function runCompatibilityCheck() {
            const output = document.getElementById('compatibility-output');
            output.innerHTML = '';
            
            const supported = checkWebCryptoSupport();
            if (supported) {
                output.innerHTML += '<span class="success">✓ Web Crypto API is supported!</span>\\n';
            } else {
                output.innerHTML += '<span class="error">✗ Web Crypto API is not supported in this browser.</span>\\n';
            }
        }

        async function runFullDemo() {
            const output = document.getElementById('demo-output');
            output.innerHTML = 'Running cryptography demonstration...\\n\\n';
            
            // Redirect console.log to output div
            const originalLog = console.log;
            console.log = (...args) => {
                output.innerHTML += args.join(' ') + '\\n';
                originalLog(...args);
            };

            try {
                await demonstrateWebCrypto();
                output.innerHTML += '\\n<span class="success">Demo completed successfully!</span>';
            } catch (error) {
                output.innerHTML += '\\n<span class="error">Error: ' + error.message + '</span>';
            } finally {
                console.log = originalLog;
            }
        }

        async function encryptMessage() {
            const messageInput = document.getElementById('message-input');
            const output = document.getElementById('encryption-output');
            
            const message = messageInput.value;
            if (!message) {
                output.innerHTML = '<span class="error">Please enter a message</span>';
                return;
            }

            try {
                if (!globalKey) {
                    globalKey = await SymmetricCrypto.generateAESKey(256);
                    output.innerHTML = 'Generated new AES key\\n';
                }

                globalEncrypted = await SymmetricCrypto.encryptAESGCM(message, globalKey);
                
                output.innerHTML += 'Message encrypted successfully:\\n';
                output.innerHTML += 'Original: ' + message + '\\n';
                output.innerHTML += 'Encrypted: ' + WebCryptoUtils.arrayBufferToHex(globalEncrypted.ciphertext).substring(0, 64) + '...\\n';
                output.innerHTML += 'IV: ' + WebCryptoUtils.arrayBufferToHex(globalEncrypted.iv) + '\\n';
                
            } catch (error) {
                output.innerHTML = '<span class="error">Encryption error: ' + error.message + '</span>';
            }
        }

        async function decryptMessage() {
            const output = document.getElementById('encryption-output');
            
            if (!globalEncrypted || !globalKey) {
                output.innerHTML = '<span class="error">No encrypted message available. Encrypt a message first.</span>';
                return;
            }

            try {
                const decrypted = await SymmetricCrypto.decryptAESGCM(globalEncrypted, globalKey);
                output.innerHTML += 'Message decrypted successfully:\\n';
                output.innerHTML += 'Decrypted: ' + decrypted + '\\n';
                
            } catch (error) {
                output.innerHTML = '<span class="error">Decryption error: ' + error.message + '</span>';
            }
        }

        // Initialize compatibility check on page load
        window.onload = function() {
            runCompatibilityCheck();
        };
    </script>
</body>
</html>
`;

// Export for use in HTML file
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        WebCryptoUtils,
        SymmetricCrypto,
        AsymmetricCrypto,
        DigitalSignatures,
        HashFunctions,
        SecureMessaging,
        CryptoBenchmark,
        demonstrateWebCrypto,
        checkWebCryptoSupport,
        htmlTemplate
    };
}

console.log('Web Cryptography API utilities loaded. Run demonstrateWebCrypto() to see the demo.');
