# Final Exam Study Guide - Key Points

## Chaotic Maps (Q1-15)

**Q1: What are chaotic maps?**
• Mathematical functions with
  chaotic behavior
• Highly sensitive to initial conditions
• Show unpredictability over time

**Q2: Primary characteristic of chaotic system?**
• Sensitive dependence on initial conditions
• Small input changes → vastly different outcomes

**Q3: Can chaotic maps be deterministic?**
• Yes, follow deterministic equations
• Outcomes appear random due to extreme sensitivity

**Q4: Well-known example of chaotic map?**
• Logistic Map: x_{n+1} = r x_n (1 - x_n)
• Exhibits chaos for certain r values

**Q5: Why useful in cryptography?**
• Provide strong randomness and unpredictability
• Enable secure encryption methods

**Q6: Role of Lyapunov exponent?**
• Measures rate of trajectory divergence
• Indicates chaos level in system

**Q7: Tent Map vs Logistic Map?**
• Tent Map: piecewise linear function
• Logistic Map: quadratic form
• Both exhibit chaotic behavior under specific conditions

**Q8: What is bifurcation diagram?**
• Shows system behavior changes as parameter varies
• Reveals transitions from stable states to chaotic regimes

**Q9: Can generate pseudo-random numbers?**
• Yes, trajectories appear unpredictable
• Serve as pseudo-random number generators

**Q10: Main engineering applications?**
• Cryptography, signal processing
• Secure communications, nonlinear control systems

**Q11: Topological mixing property?**
• Initial point sets eventually spread throughout entire space
• Ensures unpredictability over time

**Q12: Use in image encryption?**
• Randomly shuffle pixel positions
• Alter intensity values
• Make encrypted images difficult to decipher

**Q13: Relationship with fractals?**
• Chaotic systems generate fractals
• Exhibit self-similarity and complexity at different scales

**Q14: Can chaotic maps be reversed?**
• Some have inverse functions
• Accurate reversal extremely difficult due to sensitivity

**Q15: Truly unpredictable or controllable?**
• Appear random but deterministic
• Can be controlled/synchronized using adaptive techniques

## Digital Signatures (Q16-30)

**Q16: What is digital signature?**
• Cryptographic mechanism for verification
• Ensures authenticity and integrity of digital messages/documents

**Q17: Why important?**
• Provide authentication, integrity, non-repudiation
• Ensure data security and verifiability

**Q18: Common algorithms?**
• RSA, DSA, ECDSA

**Q19: Difference from handwritten signature?**
• Generated using cryptographic techniques
• Much more secure than handwritten

**Q20: Role of private key?**
• Used to generate digital signature
• Must be kept secret to prevent unauthorized access

**Q21: Verification process?**
• Recipient uses sender's public key
• Checks signature validity and data integrity

**Q22: Hash function's role?**
• Creates fixed-size data digest
• Ensures integrity, makes signing secure

**Q23: Can prevent tampering?**
• Yes, any message alteration causes verification failure

**Q24: Certificate Authority (CA)?**
• Trusted entity issuing/verifying digital certificates
• Ensures digital signature authenticity

**Q25: Digital vs electronic signature?**
• Digital: cryptography-based security
• Electronic: any digital mark or consent

**Q26: PKI relation?**
• Manages digital certificates and encryption keys
• Ensures secure authentication and verification

**Q27: Can be forged?**
• Extremely difficult due to strong cryptographic protections

**Q28: Associated risks?**
• Private key exposure
• Weak encryption algorithms
• Fake Certificate Authorities

**Q29: Quantum computer threat?**
• Could break current encryption algorithms
• Lead to insecure digital signatures

**Q30: Post-quantum algorithms?**
• Designed to resist quantum computer attacks
• Ensure future security

## AES Encryption (Q31-45)

**Q31: What is AES?**
• Symmetric encryption algorithm
• Converts plaintext to ciphertext using secret key

**Q32: Supported key lengths?**
• 128, 192, and 256 bits

**Q33: NIST adoption year?**
• 2001

**Q34: AES developers?**
• Vincent Rijmen and Joan Daemen
• Originally called Rijndael

**Q35: Difference from DES?**
• More complex substitution-permutation network
• Larger key sizes, much more secure

**Q36: Why considered secure?**
• Multiple transformation rounds
• Strong key expansion
• Non-linear substitution operations
• Resistant to attacks

**Q37: Four main transformation steps?**
• SubBytes, ShiftRows, MixColumns, AddRoundKey

**Q38: S-Box contribution to security?**
• Provides non-linearity
• Protects against differential and linear cryptanalysis

**Q39: AES-CBC mode?**
• Cipher Block Chaining
• Combines each block with previous ciphertext block
• Makes identical plaintext blocks encrypt differently

**Q40: AES-256 vs AES-128 security?**
• AES-256 more secure due to longer key size
• Makes brute-force attacks significantly harder

**Q41: Round key generation?**
• Key schedule algorithm with rotations, substitutions, XOR operations
• Generates subkeys for each round

**Q42: MixColumns importance?**
• Improves diffusion
• Changes in one byte spread across multiple bytes

**Q43: Known attacks against AES?**
• Resists brute force, differential cryptanalysis, side-channel attacks
• No successful full-scale attack has broken AES

**Q44: Hardware optimization?**
• Dedicated circuits and parallel processing
• Reduces computational overhead, increases speed

**Q45: Quantum computing threat?**
• Grover's algorithm could halve brute-force time
• AES-256 remains quantum-resistant for now

## Symmetric vs Asymmetric Encryption (Q46-61)

**Q46: Symmetric encryption?**
• Same key for encryption and decryption

**Q47: Asymmetric encryption?**
• Key pair: public key (encryption), private key (decryption)

**Q48: Primary difference?**
• Symmetric: faster
• Asymmetric: more secure but slower

**Q49: Secure messaging apps use?**
• Symmetric encryption (speed and efficiency)

**Q50: Digital signatures use?**
• Asymmetric encryption (verification without exposing private key)

**Q51: Key sharing in symmetric?**
• Secret key must be securely exchanged between parties

**Q52: Asymmetric solves key exchange how?**
• Public key for encryption eliminates secret key sharing need

**Q53: Popular symmetric algorithms?**
• AES, DES, Blowfish

**Q54: Popular asymmetric algorithms?**
• RSA, ECC, DSA

**Q55: HTTPS encryption type?**
• Both: asymmetric for key exchange, symmetric for data transfer

**Q56: Why asymmetric slower?**
• Complex mathematical calculations, computationally intensive

**Q57: MixColumns crucial in AES why?**
• Enhances diffusion, one byte changes affect multiple bytes

**Q58: Diffie-Hellman key exchange?**
• Method for two parties to securely generate shared symmetric key
• No prior key exchange needed

**Q59: ECC preferred over RSA why?**
• Same security as RSA with smaller key sizes
• More efficient

**Q60: Can asymmetric replace symmetric entirely?**
• No, asymmetric computationally expensive
• Often combined with symmetric for efficiency
