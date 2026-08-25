"""
Password hashing utilities using direct bcrypt.
Never import plain-text passwords outside this module.
"""

import bcrypt


def hash_password(plain: str) -> str:
    """Hash a plain-text password using bcrypt. Call during registration."""
    pwd_bytes = plain.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored hash. Call during login."""
    pwd_bytes = plain.encode('utf-8')
    hash_bytes = hashed.encode('utf-8')
    return bcrypt.checkpw(pwd_bytes, hash_bytes)
