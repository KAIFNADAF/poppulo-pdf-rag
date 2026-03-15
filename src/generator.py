import re

import requests
from groq import Groq

from src.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)


class Generator:
    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        timeout: int = 120,
    ) -> None:
        self.provider = provider.lower()
        self.timeout = timeout

        if self.provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, prompt: str) -> str:
        """
        Generating an answer using the configured provider.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Cannot generate an answer from an empty prompt.")

        if self.provider == "groq":
            raw_answer = self._generate_with_groq(prompt)
            return self._clean_answer(raw_answer)

        if self.provider == "ollama":
            raw_answer = self._generate_with_ollama(prompt)
            return self._clean_answer(raw_answer)

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _generate_with_groq(self, prompt: str) -> str:
        """
        Generating a response through the Groq API.
        """
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )

            if not getattr(response, "choices", None):
                raise RuntimeError("Groq response contained no choices.")

            message = response.choices[0].message
            content = getattr(message, "content", None)

            if content is None:
                raise RuntimeError("Groq response contained no message content.")

            content = content.strip()

            if not content:
                raise RuntimeError("The language model returned an empty response.")

            return content

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Groq generation failed: {exc}") from exc

    def _generate_with_ollama(self, prompt: str) -> str:
        """
        Generating a response through the Ollama API.
        """
        url = f"{OLLAMA_BASE_URL}/api/generate"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama generation timed out after {self.timeout} seconds."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to get response from Ollama at {url}: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

        if "response" not in data:
            raise RuntimeError("Ollama response missing 'response' field.")

        content = data["response"]

        if content is None:
            raise RuntimeError("Ollama response field 'response' was null.")

        content = content.strip()

        if not content:
            raise RuntimeError("The language model returned an empty response.")

        return content

    def _clean_answer(self, answer: str) -> str:
        """
        Cleaning common model-output artifacts while keeping the actual
        answer content intact.
        """
        cleaned = answer.strip()

        # Removing common leading answer labels
        prefix_patterns = [
            r"^\s*final answer\s*:\s*",
            r"^\s*answer\s*:\s*",
            r"^\s*plain answer\s*:\s*",
            r"^\s*response\s*:\s*",
        ]
        for pattern in prefix_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Removing leading source or document metadata lines
        metadata_patterns = [
            r"^\s*source\s*\d+\s*\|.*?(?:\n|$)",
            r"^\s*document\s*:.*?(?:\n|$)",
            r"^\s*pages?\s*:.*?(?:\n|$)",
            r"^\s*\[passage\s*\d+\].*?(?:\n|$)",
            r"^\s*\[source\s*\d+\].*?(?:\n|$)",
        ]
        for pattern in metadata_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Removing echoed question blocks when the model repeats them
        cleaned = re.sub(
            r"^\s*question\s*:.*?(?:\n|$)",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # Removing common introductory boilerplate
        intro_patterns = [
            r"^\s*based on the (?:retrieved )?context,\s*",
            r"^\s*according to the (?:retrieved )?text,\s*",
            r"^\s*according to the context,\s*",
            r"^\s*the document states that\s*",
            r"^\s*the text states that\s*",
        ]
        for pattern in intro_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Removing surrounding quotes when the whole answer is wrapped in them
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()

        # Collapsing excessive blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # Keeping only the first meaningful block when the model starts adding extra sections
        split_patterns = [
            r"\n\s*supporting evidence\s*:?",
            r"\n\s*evidence\s*:?",
            r"\n\s*citations?\s*:?",
            r"\n\s*sources?\s*:?",
        ]
        for pattern in split_patterns:
            parts = re.split(pattern, cleaned, flags=re.IGNORECASE)
            if parts:
                cleaned = parts[0].strip()

        return cleaned.strip()

    def healthcheck(self) -> bool:
        """
        Checking whether the configured provider is reachable.
        """
        if self.provider == "groq":
            return True

        if self.provider == "ollama":
            try:
                response = requests.get(
                    f"{OLLAMA_BASE_URL}/api/tags",
                    timeout=10,
                )
                return response.status_code == 200
            except requests.RequestException:
                return False

        return False