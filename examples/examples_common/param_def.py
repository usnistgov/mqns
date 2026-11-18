import json
import sys
import tomllib

import tomli_w
from pydantic import BaseModel


def load_param_def[T: BaseModel](arg: str, sample: T) -> T | None:
    """
    Load and validate parameter definitions.

    Args:
        arg: Command line argument.
        sample: Pydantic model and sample data.

    Returns:
        Loaded parameters, or None if schema/sample is written to stdout.

    Acceptable values for ``arg``:

    * **+schema**: write JSON schema to stdout.
    * **+json**: write JSON sample to stdout.
    * **+toml**: write TOML sample to stdout.
    * **-json**: read JSON parameter from stdin.
    * **-toml**: read TOML parameter from stdin.
    * Filename that contains either JSON or TOML.
    """
    match arg:
        case "+schema":
            json.dump(sample.model_json_schema(), sys.stdout)
            return None
        case "+json":
            json.dump(sample.model_dump(mode="json"), sys.stdout)
            return None
        case "+toml":
            tomli_w.dump(sample.model_dump(mode="json"), sys.stdout.buffer)
            return None
        case "-json":
            d = json.load(sys.stdin)
        case "-toml":
            d = tomllib.load(sys.stdin.buffer)
        case _:
            try:
                with open(arg, "rb") as f:
                    d = tomllib.load(f)
            except tomllib.TOMLDecodeError:
                with open(arg) as f:
                    d = json.load(f)
    return type(sample).model_validate(d)
