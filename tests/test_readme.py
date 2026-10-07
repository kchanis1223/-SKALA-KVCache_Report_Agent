import re
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_readme_follows_the_presentation_template_without_external_setup():
    readme = README.read_text(encoding="utf-8")

    assert re.findall(r"^## (.+)$", readme, flags=re.MULTILINE) == [
        "Subject",
        "Overview",
        "Selected Technologies",
        "Features",
        "Tech Stack",
        "Agents",
        "State Schema",
        "Architecture",
        "Design Decisions",
        "Directory Structure",
        "Usage",
        "Contributors",
    ]
    assert "make run" in readme
    for image in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        assert (README.parent / image).is_file(), image
    assert "API 키" not in readme
    assert "TAVILY_API_KEY" not in readme
