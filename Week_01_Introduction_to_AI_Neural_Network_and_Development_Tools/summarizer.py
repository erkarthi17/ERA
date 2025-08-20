import re
from html.parser import HTMLParser


class EmExtractor(HTMLParser):
    """Custom HTML parser to extract <em> text without bs4."""
    def __init__(self):
        super().__init__()
        self.inside_em = False
        self.em_texts = []
        self.all_text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "em":
            self.inside_em = True

    def handle_endtag(self, tag):
        if tag.lower() == "em":
            self.inside_em = False

    def handle_data(self, data):
        if self.inside_em:
            self.em_texts.append(data.strip())
        self.all_text.append(data.strip())

    def get_em_text(self):
        return " ".join([t for t in self.em_texts if t])

    def get_plain_text(self):
        return " ".join([t for t in self.all_text if t])


def extract_em_text(html: str) -> str:
    parser = EmExtractor()
    parser.feed(html or "")
    em_text = parser.get_em_text()
    return em_text if em_text else parser.get_plain_text()


def summarize_text(text: str) -> str:
    """Simple summarizer with no ML/dependencies."""
    if not text:
        return ""

    # Prefer emphasized text
    emphasized = extract_em_text(text)
    if emphasized:
        return emphasized

    # Otherwise, take the first 2–3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:3])[:600]