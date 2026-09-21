def mask_secret(secret: str, visible: int = 4) -> str:
    # Напишите реализацию.
    return "*" * (len(secret) - visible) + secret[-visible:]


print(mask_secret("test", 2))
