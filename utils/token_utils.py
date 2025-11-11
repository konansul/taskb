import jwt

def decode_access_token(access_token: str) -> dict:
    """
    Decodes a JWT access token without verifying the signature.
    Returns all available claims: name, surname, email, FIN, etc.
    """
    try:
        # decode without signature verification
        decoded = jwt.decode(access_token, options={"verify_signature": False})
        return decoded or {}
    except jwt.DecodeError:
        print("❌ Invalid token format.")
        return {}
    except Exception as e:
        print(f"⚠️ Error while decoding token: {e}")
        return {}