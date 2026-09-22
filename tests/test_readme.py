import re
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_readme_follows_the_presentation_template_without_external_setup():
    readme = README.read_text(encoding="utf-8")

    assert re.findall(r"^## (.+)$", readme, flags=re.MULTILINE) == [
        "Subject",
        "Overview",
        "Selected Technologies",
        "Key Features",
        "Agents",
        "Agent Model Strategy",
        "RAG",
        "Architecture",
        "State Design",
        "Differentiators",
        "Report Highlights",
        "Lessons Learned",
        "Directory Structure",
        "Usage",
        "Contributors",
    ]
    assert "make run" in readme
    assert "API 키" not in readme
    assert "TAVILY_API_KEY" not in readme
