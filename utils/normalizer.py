import re


def persian_normalizer(text):
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "آ",
        "ٱ": "ا",
        "ء": "",
        "ؤ": "و",
        "ى": "ی",
        "ي": "ی",
        "ة": "ه",
        "ۀ": "ه",
        "ۆ": "و",
        "ڵ": "ل",
        "ێ": "ی",
        "ە": "ه",
        "ڤ": "و",
        "ٔ": "",
        "\u200d": "\u200c",
        "\u200e": "\u200c",
        "\u200f": "\u200c",
        "\ufeff": "\u200c",
    }

    for src, dest in replacements.items():
        text = text.replace(src, dest)

    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    for pd, ed in zip(persian_digits, english_digits):
        text = text.replace(pd, ed)

    allowed_pattern = re.compile(
        r"[^"
        r"ئآابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
        r"a-zA-Z"
        r"0-9۰-۹"
        r'.,:!?\'"()\[\]{}\-–—'
        r"٪،؛«»؟%&+=/"
        r"\s\u200c"
        r"]"
    )
    text = allowed_pattern.sub("", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[\u200c]{2,}", "\u200c", text)
    text = text.strip()

    return text


def persian_normalizer_no_punc(text):
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "آ",
        "ٱ": "ا",
        "ء": "",
        "ؤ": "و",
        "ى": "ی",
        "ي": "ی",
        "ة": "ه",
        "ۀ": "ه",
        "ۆ": "و",
        "ڵ": "ل",
        "ێ": "ی",
        "ە": "ه",
        "ڤ": "و",
        "ٔ": "",
        "\u200d": "\u200c",
        "\u200e": "\u200c",
        "\u200f": "\u200c",
        "\ufeff": "\u200c",
    }

    for src, dest in replacements.items():
        text = text.replace(src, dest)

    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    english_digits = "0123456789"
    for pd, ed in zip(persian_digits, english_digits):
        text = text.replace(pd, ed)

    allowed_pattern = re.compile(
        r"[^" r"ئآابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی" r"a-zA-Z" r"0-9" r"\s\u200c" r"]"
    )
    text = allowed_pattern.sub("", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[\u200c]{2,}", "\u200c", text)
    text = text.strip()

    return text
