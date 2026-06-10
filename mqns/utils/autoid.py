class AutoIncrementIdentifier:
    """
    Auto-increment identifier generator.

    Each identifier has a 4-char prefix (must end with underscore) followed by 28-digit hexadecimal number.
    """

    def __init__(self, prefix: str):
        assert len(prefix) == 4
        assert prefix[-1] == "_"
        self._prefix = prefix
        self._n = 0

    def __call__(self) -> str:
        n = self._n
        self._n += 1
        return f"{self._prefix}{n:028x}"
