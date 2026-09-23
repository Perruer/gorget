from pathlib import Path

from gorget.exception import GorgetValidationError
from gorget.util import calculate_risk_score, get_logger, lazy_load_dep

from .base import Scanner

LOGGER = get_logger()
_lexicon = "vader_lexicon"

# VADER lexicon (MIT, C.J. Hutto) ships with the package, so no NLTK download is needed.
_BUNDLED_LEXICONS = {
    "vader_lexicon": Path(__file__).resolve().parent.parent / "data" / "vader_lexicon.txt",
}
_NLTK_LEXICON_PATHS = {
    "vader_lexicon": "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt",
}


def _read_lexicon(lexicon: str) -> str:
    """Lexicon text from the package, a file path or an installed NLTK resource."""
    if lexicon in _BUNDLED_LEXICONS:
        return _BUNDLED_LEXICONS[lexicon].read_text(encoding="utf-8")

    if Path(lexicon).is_file():
        return Path(lexicon).read_text(encoding="utf-8")

    nltk = lazy_load_dep("nltk")
    resource = _NLTK_LEXICON_PATHS.get(lexicon, lexicon)
    try:
        return nltk.data.load(resource, format="text")
    except LookupError as exc:
        raise GorgetValidationError(
            f"Lexicon {lexicon} is not installed. Download it once with "
            f"`python -m nltk.downloader {lexicon}` or pass a path to a lexicon file."
        ) from exc


def _analyzer(lexicon_text: str):
    vader = lazy_load_dep("nltk.sentiment.vader", "nltk")

    class _Analyzer(vader.SentimentIntensityAnalyzer):
        # NLTK only reads lexicons through nltk.data, which refuses paths outside
        # nltk_data, so the text is handed over directly.
        def __init__(self, text: str) -> None:
            self.lexicon_file = text
            self.lexicon = self.make_lex_dict()
            self.constants = vader.VaderConstants()

    return _Analyzer(lexicon_text)


class Sentiment(Scanner):
    """
    A sentiment scanner based on the NLTK's SentimentIntensityAnalyzer. It is used to detect if a prompt
    has a sentiment score lower than the threshold, indicating a negative sentiment.
    """

    def __init__(self, *, threshold: float = -0.3, lexicon: str = _lexicon) -> None:
        """
        Initializes Sentiment with a threshold and a chosen lexicon.

        Parameters:
           threshold (float): Threshold for the sentiment score (from -1 to 1). Default is 0.3.
           lexicon (str): Lexicon for the SentimentIntensityAnalyzer. Default is 'vader_lexicon'.

        Raises:
           None.
        """

        self._sentiment_analyzer = _analyzer(_read_lexicon(lexicon))
        self._threshold = threshold

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if not prompt:
            return prompt, True, -1.0

        sentiment_score = self._sentiment_analyzer.polarity_scores(prompt)
        sentiment_score_compound = sentiment_score["compound"]
        if sentiment_score_compound > self._threshold:
            LOGGER.debug(
                "Sentiment score is below the threshold",
                sentiment_score=sentiment_score_compound,
                threshold=self._threshold,
            )

            return prompt, True, 0.0

        LOGGER.warning(
            "Sentiment score is above the threshold",
            sentiment_score=sentiment_score_compound,
            threshold=self._threshold,
        )

        return (
            prompt,
            False,
            calculate_risk_score(abs(sentiment_score_compound), self._threshold),
        )
