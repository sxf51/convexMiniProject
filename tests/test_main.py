import io
from contextlib import redirect_stdout

import main


def test_main_prints_greeting():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main.main()

    assert buffer.getvalue().strip() == "Hello from convexminiproject!"
