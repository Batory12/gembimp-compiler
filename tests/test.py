from src.main import main
import pytest
from pathlib import Path
import subprocess
programs = Path(__file__).parent.joinpath("programs").iterdir()
@pytest.mark.parametrize("file", programs, ids=lambda p: p.name)
def test_compilation(file: Path):
    name = file.name
    input_file = Path(__file__).parent.joinpath("inputs", name)
    expected_file = Path(__file__).parent.joinpath("outputs", name)
    temp_name = "temp.vm"
    main(debug=False, input_file=str(file), output_file=temp_name)
    result = subprocess.run(
        ["bin/maszyna-wirtualna-cln", temp_name],
        stdin=open(input_file),
        capture_output=True,
        timeout=5
    )
    expected_output = expected_file.read_text()
    assert "\n".join(filter(lambda s: ">" in s, result.stdout.decode().split("\n"))) == expected_output


