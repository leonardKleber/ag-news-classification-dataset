import random
import math
from collections import defaultdict, Counter
from typing import List, Tuple
from task_two import preprocess_dataset


class NGramLanguageModel:
    def __init__(self, n: int):
        assert n >= 2, "This implementation supports n >= 2"
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.V = 0  # vocabulary size

    def train(self, texts: List[str]):
        """
        Train n-gram model from a list of preprocessed text strings.
        """
        for text in texts:
            tokens = ["<s>"] * (self.n - 1) + text.split() + ["</s>"]
            self.vocab.update(tokens)

            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                target = tokens[i + self.n - 1]

                self.ngram_counts[context][target] += 1
                self.context_counts[context] += 1

        self.V = len(self.vocab)

    def probability(self, context: Tuple[str], word: str) -> float:
        """
        Compute smoothed probability P(word | context)
        using Laplace smoothing.
        """
        numerator = self.ngram_counts[context][word] + 1
        denominator = self.context_counts[context] + self.V
        return numerator / denominator

    def generate(self, max_length: int = 20) -> str:
        """
        Generate a text sequence from the language model.
        """
        context = ["<s>"] * (self.n - 1)
        generated = []

        for _ in range(max_length):
            context_tuple = tuple(context)
            candidates = list(self.vocab)

            probs = [
                self.probability(context_tuple, word)
                for word in candidates
            ]

            next_word = random.choices(candidates, weights=probs, k=1)[0]

            if next_word == "</s>":
                break

            generated.append(next_word)
            context = context[1:] + [next_word]

        return " ".join(generated)

    def perplexity(self, texts: List[str]) -> float:
        """
        Compute perplexity of the model on a list of texts.
        """
        log_prob_sum = 0.0
        token_count = 0

        for text in texts:
            tokens = ["<s>"] * (self.n - 1) + text.split() + ["</s>"]

            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]

                prob = self.probability(context, word)
                log_prob_sum += math.log(prob)
                token_count += 1

        return math.exp(-log_prob_sum / token_count)


if __name__ == "__main__":
    df = preprocess_dataset("train.csv")
    # Prepare training data
    train_texts = df["clean_text"].tolist()

    # Bigram model
    bigram_model = NGramLanguageModel(n=2)
    bigram_model.train(train_texts)

    # Trigram model
    trigram_model = NGramLanguageModel(n=3)
    trigram_model.train(train_texts)

    # Generate sample text
    print("Bigram generation:")
    print(bigram_model.generate())

    print("\nTrigram generation:")
    print(trigram_model.generate())

    # Compute perplexity on training data
    bigram_ppl = bigram_model.perplexity(train_texts)
    trigram_ppl = trigram_model.perplexity(train_texts)

    print(f"Bigram perplexity: {bigram_ppl:.2f}")
    print(f"Trigram perplexity: {trigram_ppl:.2f}")
