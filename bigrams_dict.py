
class BigramsDict:

    def __init__(self):
        self._bigrams = {} #klucz - bigram, wartosc - ilosc wystapien
        self._word_counts = {}
        self._dummy_const = 0.0000001 #stala, w przypadku gdy jesden bigram nie istnieje,
        #zapewnia ze w takim przypadku prawdopodobienstwo nie bedzie zrem ale obnizy tez wartosc na tyle by byla mniejsza
        #od kazdego prawdopodobienstwa w przypadku dwoch bigramow

    def insert(self, bigram):
        if bigram is None:
            raise Exception("Empty bigram passed")
        bigram = tuple(bigram[i].lower() for i in range(0, len(bigram)))
        if bigram in self._bigrams:
            self._bigrams[bigram] += 1
        else:
            self._bigrams[bigram] = 0

        #dol bedzie dzialac bo w ramach treningu podajemy wszystkie bigramy
        #nie zawsze musi byc poprzednie/nastepne - problem pustego
        #prev_counts - ile razy przed podanym slowem cos bylo
        #next_counts - ile razy po podanym slowie cos bylo
        #ale to bedzie to samo bo dodajemy puste
        #wiec moze byc zwykly word count

        word = bigram[0]
        if word in self._word_counts:
            self._word_counts[word] += 1
        else:
            self._word_counts[word] = 1
        if bigram[1] == "":
            self._word_counts[""] += 1

    def get_prob(self, sequence): #sequence = tuple of three - prev_word, word, next_word
        #zwracac iloczyn czy 'punkty' na podstawie sumy prawdopodobienstwa poprzedniego i nastepnego?
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
        #to w ogole nie powinno zajsc! zwykly ret powinien starczyc ale chwilowo praca nad czym innym wiec do zbadania potem
        #return self._word_counts[word]
        return self._word_counts[word] if word in self._word_counts else 0
