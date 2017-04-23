# BigramsDict - class designed to store bigrams that occurred in text, maintain they count and
# compute conditional probability of word occurrence based on words in its surroundings


class BigramsDict:

    def __init__(self):
        self._bigrams = {} # key - bigram, value - occurrences count
        self._word_counts = {} # dict for occurrences of single words
        self._dummy_const = 0.0000001 # const, that goal is to prevent multiplication by zero when counting
        # probability but in the same time lowering it value to make sequences when both previous and next word
        # matches score higher

    def insert(self, bigram):
        # inserting bigram BigramsDict object
        # TODO: maybe use Counter instead of handling that counts manually?
        if bigram is None:
            raise Exception("Empty bigram passed")
        bigram = tuple(bigram[i].lower() for i in range(0, len(bigram)))
        if bigram in self._bigrams:
            self._bigrams[bigram] += 1
        else:
            self._bigrams[bigram] = 0

        # "" - means end of sentence, it is added in earlier processing
        # TODO: replace it with some constant to make it more readable?
        word = bigram[0]
        if word in self._word_counts:
            self._word_counts[word] += 1
        else:
            self._word_counts[word] = 1
        if bigram[1] == "":
            self._word_counts[""] += 1

    def get_prob(self, sequence):
        # returns probability of passed sequence occurrence
        # sequence = tuple of three - prev_word, word, next_word
        # if both prev and next hasn't occurred earlier (in training) in word context then the returned prob is 0
        prev_context, next_context = (sequence[0], sequence[1]), (sequence[1], sequence[2])
        if prev_context in self._bigrams and next_context in self._bigrams:
            prev_word_prob = self._bigrams[prev_context] / self._word_counts[sequence[1]]
            next_word_prob = self._bigrams[next_context] / self._word_counts[sequence[1]]
            return prev_word_prob * next_word_prob
        elif prev_context in self._bigrams and next_context not in self._bigrams:
            prev_word_prob = self._bigrams[prev_context] / self._word_counts[sequence[1]]
            next_word_prob = self._dummy_const / self._word_counts[sequence[1]]
            return prev_word_prob * next_word_prob
        elif prev_context not in self._bigrams and next_context in self._bigrams:
            prev_word_prob = self._dummy_const / self._word_counts[sequence[1]]
            next_word_prob = self._bigrams[next_context] / self._word_counts[sequence[1]]
            return prev_word_prob * next_word_prob
        else:
            return 0.0

    def get_count(self, word):
        # returns count of total occurrences of word in train data
        # TODO: there shouldn't be possibility for use get_count with word that hasn't occured earlier
        # but it occurred once - investigate it
        return self._word_counts[word] if word in self._word_counts else 0
