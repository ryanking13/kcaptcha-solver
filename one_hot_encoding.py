import numpy as np
import settings


def encode(text):
    vector = np.zeros(settings.CHAR_SET_LEN * settings.CAPTCHA_LENGTH, dtype=float)
    for i, c in enumerate(text):
        idx = i * settings.CHAR_SET_LEN + int(c)
        vector[idx] = 1.0
    return vector


def decode(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        digit = c % settings.CHAR_SET_LEN
        text.append(str(digit))
    return "".join(text)