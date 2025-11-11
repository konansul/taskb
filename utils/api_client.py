import requests

BASE_URL = "https://4sim-website-gateway-dev.cloudraft.net/api/v1/auth"


# Register a new user
def register_user(first_name, last_name, fin, email, password):
    url = f"{BASE_URL}/register"
    payload = {
        "firstName": first_name.strip(),
        "lastName": last_name.strip(),
        "fin": fin.strip(),
        "email": email.strip(),
        "password": password.strip()
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    print("➡️ Sending to server:", payload)

    try:
        response = requests.post(url, json=payload, headers=headers)
        print("⬅️ Server response:", response.status_code, response.text)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "status_code": getattr(e.response, "status_code", None),
            "response": e.response.text if e.response else None
        }


# User login
def login_user(fin, password):
    url = f"{BASE_URL}/login"
    payload = {
        "fin": fin,
        "password": password
    }
    return send_request(url, payload)


# Logout
def logout_user(refresh_token):
    url = f"{BASE_URL}/logout"
    payload = {
        "refreshToken": refresh_token
    }
    return send_request(url, payload)


# Password reset (via email and OTP)
def reset_password(email, otp, new_password):
    url = f"{BASE_URL}/reset-password"
    payload = {
        "email": email,
        "otp": otp,
        "newPassword": new_password
    }
    return send_request(url, payload)


# Forgot password
def forgot_password(email):
    url = f"{BASE_URL}/forgot-password"
    payload = {"email": email}
    return send_request(url, payload)


# Helper function for sending POST requests ---
def send_request(url, payload):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "status_code": getattr(e.response, "status_code", None),
            "response": e.response.text if e.response else None
        }