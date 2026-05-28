from typing import Callable, Final, Optional

import numpy as np

MORSE_LETTER_INTRASPACE: Final[str] = " "
MORSE_LETTER_INTERSPACE: Final[str] = "   "
MORSE_WORD_INTERSPACE: Final[str] = "       "
MORSE_CODES: Final[dict[str, str]] = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
    " ": MORSE_WORD_INTERSPACE,
}


def morse_encode(message: str, unit_size: int) -> np.ndarray:
    """Encodes the string according to the morse code chart."""

    def interspace(
        items: list[str], space_item: str, predicate: Callable[[str, str], bool]
    ) -> list[str]:
        result = list[str]()

        prev: Optional[str] = None
        for item in items:
            if prev is not None and predicate(prev, item):
                result.append(space_item)
            result.append(item)
            prev = item

        return result

    codes = [MORSE_CODES[letter] for letter in message]

    # Add letter interspaces
    codes = interspace(
        codes,
        MORSE_LETTER_INTERSPACE,
        lambda lhs, rhs: MORSE_WORD_INTERSPACE not in (lhs, rhs),
    )

    # Add letter intraspaces
    chars = [char for code in codes for char in code]
    chars = interspace(
        chars,
        MORSE_LETTER_INTRASPACE,
        lambda lhs, rhs: MORSE_LETTER_INTRASPACE not in (lhs, rhs),
    )

    # Convert to numpy
    dot = np.ones(unit_size)
    dash = np.ones(3 * unit_size)
    unit_count = chars.count(" ") + chars.count(".") + 3 * chars.count("-")
    signal = np.zeros(unit_count * unit_size)
    index = 0
    for char in chars:
        if char == " ":
            index += unit_size
        elif char == ".":
            signal[index : index + len(dot)] += dot
            index += len(dot)
        elif char == "-":
            signal[index : index + len(dash)] += dash
            index += len(dash)
        else:
            raise Exception(f"Invalid char: `{char}`.")

    return signal
