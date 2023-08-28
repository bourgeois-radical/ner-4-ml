# built-in
import string
import re
from typing import List

# installed
# import spacy
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


class DefaultPreprocess:
    """Preprocessing routines for English and German."""

    @staticmethod
    def tokenize_english(text: str) -> List[str]:
        """Transforms English text to tokens.

        Parameters
        ----------
        text : str
            The whole text.

        Returns
        -------
        tokens : List[str]
        """
        tokens = nltk.tokenize.word_tokenize(text, language='english')
        return tokens

    def remove_newline(tokens: List[str]) -> List[str]:
        """Removes these '\n' guys.

        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens.

        Returns
        -------
        tokens : List[str]
        """

        tokens = [token.replace("\n", "") for token in tokens]

        return tokens

    @staticmethod
    def remove_english_stop_words(tokens: List[str]) -> List[str]:
        """
        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens.

        Returns
        -------
        tokens : List[str]
        """
        english_stop_words = stopwords.words('english')
        tokens = [token for token in tokens if token not in english_stop_words]
        return tokens

    @staticmethod
    def to_lowercase(tokens: List[str]) -> List[str]:
        """
        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens.

        Returns
        -------
        tokens : List[str]
        """
        for idx, token in enumerate(tokens):
            a_lower_token = token.casefold()
            tokens[idx] = a_lower_token

        return tokens

    @staticmethod
    def remove_punctuation_tokens(tokens: List[str]) -> List[str]:
        """Removes tokens which are punctuation symbols themselves.

        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens.

        Returns
        -------
        tokens : List[str]
        """
        regular_punct = list(string.punctuation)  # getting the list of regular punctuation symbols

        # TODO: get all lowercase letters from some library
        #   use regexp [a-z]

        regular_punct += extra_punct
        tokens = [token for token in tokens if token not in regular_punct]

        return tokens

    @staticmethod
    def remove_number_tokens(tokens: List[str]) -> List[str]:
        """Removes tokens which are numbers themselves.

        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens.

        Returns
        -------
        tokens : List[str]
        """
        tokens = [token for token in tokens if not token.isdigit()]
        return tokens

    @staticmethod
    def remove_empty_tokens(tokens: List[str]) -> List[str]:
        """
        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens.

        Returns
        -------
        tokens : List[str]
        """
        tokens = [token for token in tokens if not (token.isspace() and len(token) == 0)]
        return tokens

    @staticmethod
    def remove_punctuation_from_words(tokens: List[str]) -> List[str]:
        """Removes punctuation inside words. Tokens remain.

        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens.

        Returns
        -------
        tokens : List[str]
        """
        for idx, token in enumerate(tokens):
            a_token_with_no_punct = token.translate(str.maketrans('', '', string.punctuation))
            tokens[idx] = a_token_with_no_punct

        return tokens

    @staticmethod
    def remove_numbers_from_words(tokens: List[str]) -> List[str]:
        """Removes numbers inside words. Tokens remain.

        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens

        Returns
        -------
        tokens : List[str]
        """
        for idx, token in enumerate(tokens):
            a_token_with_no_numbers = re.sub(r'\d+', '', token).strip()
            tokens[idx] = a_token_with_no_numbers

        return tokens

    @staticmethod
    def eng_lemmatization(tokens: List[str]) -> List[str]:
        """Lemmatization of English texts with NLTK

        Parameters
        ----------
        tokens : List[str]
            The whole text as tokens

        Returns
        -------
        tokens : List[str]
        """
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    @staticmethod
    def __call__(text: str, remove_stop_words=False) -> List[str]:
        """Allows to call the whole preprocessing routine straight away.

        In order to do so, use an instance of DefaultPreprocess as a function.

        Parameters
        ----------
        text : str
            The whole text.
        language : str
            'english' or 'german'
        remove_stop_words : bool
            If True all stop words will be removed.

        Returns
        -------
        tokens : List[str]
        """

        tokens = DefaultPreprocess.tokenize_english(text)
        tokens = DefaultPreprocess.to_lowercase(tokens)
        tokens = DefaultPreprocess.remove_newline(tokens)

        if remove_stop_words:
            tokens = DefaultPreprocess.remove_english_stop_words(tokens)

        tokens = DefaultPreprocess.remove_punctuation_tokens(tokens)
        tokens = DefaultPreprocess.remove_number_tokens(tokens)
        tokens = DefaultPreprocess.remove_empty_tokens(tokens)
        tokens = DefaultPreprocess.remove_punctuation_from_words(tokens)
        tokens = DefaultPreprocess.remove_numbers_from_words(tokens)
        tokens = DefaultPreprocess.eng_lemmatization(tokens)

        return tokens
