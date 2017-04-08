import os
import json
import re
from collections import defaultdict#slowniki zliczajace nie sa defaultductem - po dopiero przy wersji 1.02 napotkalem
# go w dokumentacji do potencjalnej poprawki potem
from bigrams_dict import BigramsDict
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import xml.dom.minidom as minidom
#ponizsza byla wykorzystana od poczatku, to wyzej do upiekszenia pliku wynikowego
from xml.etree.ElementTree import Element, SubElement, tostring
import matplotlib.pyplot as plt
import collections
from nltk_tests import get_sents_from_nltk_corpuses

#Mozliwe ze zbyt uproszczone, trzeba bedzie pamietac o polskich znakach, w wersji angielskiej natomiast uwzglednic skroty
#z apostrofami he's, won't itd.
#dodano - chwilowo zezwala na apostrof - chociaz to moze falszowac prawdopodobienstwo dla np. won't i will not
#ale za duzo jest przypadkow 's przy przymiotnikach
# w sumie paramtery mozna by z ctora SP przerzucic do correct()

def get_words(text):
    return re.findall('[A-Za-z\']+', text)


alphabet = set('abcdefghijklmnopqrstuvwxyz\'')

class SpellingCorrector:

    '''
    ctor, tains corrector basing on files in train_data
    '''
    def __init__(self, create_report_file=True, report_filename='report', return_results=False):
        self._create_report_file = create_report_file
        self._report_filename = report_filename
        self._return_results = return_results
        self._shortcuts = None
        self._load_shortcuts()
        self._train_sentences = None
        self._words_set = None #possible words, filled when creting bigrams
        self._load_train_data()
        self._bigrams_dict = BigramsDict()
        self._generate_bigrams_and_words_set()
        self._test_text = None
        self._test_sentences = None


    '''
    Main function of the class, corrects all files in directory test_data
    Params:
    output_mode, possibilities are 'file', 'var', respectively saves output to file, or returns variable with results
    (otherwise return none). If mode is 'file', filepath param should be provided too
    filepath, specifies output destination in case of 'file' return mode
    '''
    #z zalozenia dzialamy wtedy gdy slowo nie wystepuje w slowniku
    def correct(self, input_filename, output_mode='', filepath='test_output/results.txt'):
        self._load_test_data(input_filename)
        corrected_words = []
        corrected_sentences = [] #nf
        for sentence in self._test_sentences: #ale interpunkcje to bym chcial zachowac
            #na razie bez uwzgledniania interpunkcji
            # Update: chwilowo tracimy info o duzych literach, interpunkcja za to ogarnieta
            orig = sentence.lower()
            sentence = get_words(sentence.lower())
            for i in range(0, len(sentence)):
                if sentence[i] not in self._words_set:
                    next_word = "" if i + 1 >= len(sentence) else sentence[i + 1]
                    prev_word = "" if i - 1 < 0 else sentence[i - 1]
                    sequence = (prev_word, sentence[i], next_word)
                    corrected_word = self._correct_word(sequence) #tu operacja na kopii, poprawic potem
                    print("Corrected {:s} to {:s}".format(sentence[i], corrected_word)) #chwilowo taki debug
                    corrected_words.append((sentence[i], corrected_word))
                    orig = orig.replace(sentence[i], corrected_word)

            orig = orig[0].upper() + orig[1:] #handling upper letter at sentence beginning
            corrected_sentences.append(orig)

        #dbg
        #print(self._find_test_sentneces_after_areas())
        #input()
        corrected_sentences = self._add_after_areas(corrected_sentences)
        if self._create_report_file:
            self._generate_xml_report(corrected_words)
        if output_mode == 'file':
            self._save_corrected_text_to_file(corrected_sentences, filepath)
        if self._return_results:
            return corrected_words
        else:
            return None

    def _load_train_data(self):
        train_data = ""
        for file in os.listdir("train_data"):
            path = os.path.join("train_data", file)
            train_data += open(path, "r").read() + "\n"
        self._train_sentences = self._split_text_to_sentences(train_data)
        #print(len(self._train_sentences))
        #input()
        self._train_sentences += get_sents_from_nltk_corpuses()
        #print(len(self._train_sentences))
        #input()
        self._train_sentences = [self._expand_shortcuts(sentence) for sentence in self._train_sentences]

        #!!!
        #print(len(self._train_sentences))
        #raise ValueError

    #opposite to train data it loads only one file - still from test_data folder but specified by param
    def _load_test_data(self, filename):
        test_data = ""
        path = os.path.join("test_data", filename)
        test_data += open(path, "r").read()
        self._test_text = test_data
        self._test_sentences = self._split_text_to_sentences(test_data)
        self._test_sentences = [self._expand_shortcuts(sentence) for sentence in self._test_sentences]
        #print(self._test_sentences)
        #input() ok

    def _split_text_to_sentences(self, text):
        # nltk, code for spliting text into sentences
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])  #do przypomnienia - trzeba rozwinac recznie
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentences = sentence_splitter.tokenize(text)
        #print(type(sentences))
        #input()
        #print(type(sentences[0]))
        #input()
        #print(sentences[898])
        return sentences

    def _generate_bigrams_and_words_set(self): #ale tu juz przecinki i reszta wroci
        self._words_set = set()
        for sentence in self._train_sentences:
            sentence_words = get_words(sentence.lower())
            #nadal watpliwosci odnosnie slow krancowych - chwilow brak slowa przed czy po reprezentowany przez slowo puste
            words_count = len(sentence_words)
            for i in range(0, words_count):
                self._words_set.add(sentence_words[i])
                bigram = None
                if i == 0:
                    bigram = ("", sentence_words[i])
                elif i == words_count - 1:
                    bigram = (sentence_words[i], "")
                else:
                    bigram = (sentence_words[i-1], sentence_words[i])
                self._bigrams_dict.insert(bigram)

    def _expand_shortcuts(self, sentence): #jeszcze przecinki itd
        for key, value in self._shortcuts.items():
            if key in sentence:
                sentence.replace(key, value)
        return sentence

    def _load_shortcuts(self):
        path = os.path.join("other", "shortcuts.json")
        json_object = json.load(open(path, 'r'))
        self._shortcuts = {}
        for pair in json_object["shortcuts"]:
            self._shortcuts[list(pair.keys())[0]] = list(pair.values())[0]

    def _generate_candidates(self, faulty_word):
        s = [(faulty_word[:i], faulty_word[i:]) for i in range(len(faulty_word) + 1)]
        deletes = [a + b[1:] for a, b in s if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in s for c in alphabet if b]
        inserts = [a + c + b for a, b in s for c in alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _known(self, words):
        return set(word for word in words if word in self._words_set)

    def _correct_word(self, sequence):
        candidates = self._generate_candidates(sequence[1])
        print("Wykryte jako błąd słowo:", sequence[1])
        candidates = [(sequence[0], candidate, sequence[2]) for candidate in candidates if candidate in self._words_set]
        #dodatek: przefiltrowanie kandydatow na tych w ktorych sa slowa znane
        print("Przefiltrowani kandydaci: ", candidates)
        probabilities = {key: self._bigrams_dict.get_prob(key) for key in candidates}
        #tu nizej jest blad + jeszcze pomysl na filtracja bezsensownych slow przed badaniem z kontekstem

        best_prob = max(probabilities.values()) if len(candidates) > 0 else 0.0
        print("Uwaga best prob: ", best_prob)
        if best_prob != 0.0:
            winner = max(probabilities, key=probabilities.get)[1]
            print("Prawdopodobienstwo wybranego slowa {} wynosi {}".format(winner, probabilities[
                (sequence[0], winner, sequence[2])]))
            return max(probabilities, key=probabilities.get)[1]
        #at least one of bigrams wee not found and prob is 0 - zmiana w bigram dictie, w takim przypadku zwroci na podstawie
        #jednego bigramu, 0 tylko gdy oba nie wystepuja
        #jesli nadal nic to sprawdzamy czy ktorykolwiek z kandydatow jest w slowniku i zwracamy go
        candidates = [sequence[1] for sequence in candidates]
        known_words_from_candidates = self._known(candidates)

        if len(known_words_from_candidates) > 0:
            word_counts = {word: self._bigrams_dict.get_count(word) for word in known_words_from_candidates}
            choosen_word = max(word_counts, key=word_counts.get)
            print("Prawdopodobienstwo 0, z posrod znanych wybrano najczesciej wystepujace: {}".format(choosen_word))
            return choosen_word
        print("Wszystko zawiodlo, zwrocono niepoprawione: {}".format(sequence[1]))
        return sequence[1]

    def _generate_xml_report(self, corrected_words):
        report = Element('report')
        for corrected_from, corrected_to in corrected_words:
            error = SubElement(report, "spelling_error")
            corrected_from_elem = SubElement(error, "corrected_from")
            corrected_from_elem.text = corrected_from
            corrected_to_elem = SubElement(error, "corrected_to")
            corrected_to_elem.text = corrected_to
        xml_string = tostring(report)

        path = os.path.join("test_output", self._report_filename + ".xml")
        xml = minidom.parseString(xml_string)
        pretty_xml_as_string = xml.toprettyxml()
        open(path, 'w').write(pretty_xml_as_string)

    def _find_test_sentneces_after_areas(self):
        after_areas = []
        for i in range(0, len(self._test_sentences)-1):
            #print(self._test_sentences[i] + r'(\n|.)*?' + self._test_sentences[i+1]) #dbg
            #print(self._test_text)
            #input()
            area = re.findall(r'(?<={}).*?(?={})'.format(re.escape(self._test_sentences[i]), re.escape(self._test_sentences[i+1])),
                            self._test_text, re.MULTILINE | re.DOTALL)[0]
            #area = re.match(self._test_sentences[i] + r'(\n|.)*?' + self._test_sentences[i+1], self._test_text, re.M)\
            #    .group(1)
            after_areas.append(area)
        # case of last sentence in text
        area = re.findall(r'(?<={}).*$'.format(re.escape(self._test_sentences[-1])), self._test_text,
                          re.MULTILINE | re.DOTALL)[0]
        after_areas.append(area)
        #print(len(after_areas))
        return after_areas

    def _add_after_areas(self, correction_results):
        after_areas = self._find_test_sentneces_after_areas()
        for i in range(len(correction_results)):
            correction_results[i] += after_areas[i]
        return correction_results

    def _save_corrected_text_to_file(self, correction_results, filepath):
       # correction_results = self._add_after_areas(correction_results)
        correction_results = sentences_list_to_text(correction_results)
        with open(filepath, 'w') as f:
            f.write(correction_results)


def sentences_list_to_text(sentences):
    text = ''
    for sentence in sentences:
        text += sentence
    return text


#error stats - dict
def _visualize_errors_stats(error_stats):
    fig, ax = plt.subplots(figsize=(12, 12))
    error_stats = sorted(error_stats.items(), key=lambda pair: pair[0])
    labels = [pair[0] for pair in error_stats]
    values = [pair[1] for pair in error_stats]
    total_count = sum(values)
    sizes = [value/total_count * 100 for value in values]

    ax.pie(sizes, labels=labels, autopct='%1.1f%%',  colors=[(1, 0, 0), (1, 0.33, 0), (0, 0.47, 0), (0.56, 0.72, 0)])

    ax.axis('equal')
    #ax.set_title('Correction Results')
    #ax.legend(loc='upper right')
    plt.show()

#program_output - lista par, kazdy blad w test secie chwilowo jest unikalny
def benchmark_func(program_output):
    spelling_errors = [["tha", "that"], ["garten", "garden"], ["presss", "press"], ["tains", "trains"], ["oit", "out"],
                       ["realy", "really"], ["bougt", "bought"], ["waering", "wearing"], ["opnion", "opinion"],
                       ["vboltage", "voltage"], ["yout", "youth"], ["cadtle", "castle"], ["lettter", "letter"],
                       ["clse", "close"], ["dihes", "dishes"], ["witc", "witch"], ["b", "by"], ["looj", "look"],
                       ["runnfng", "running"], ["jacke", "jacket"], ["inded", "indeed"], ["mostr", "most"],
                       ["shal", "shall"], ["danerous", "dangerous"], ["haits", "hates"], ["clen", "clean"],
                       ["botle", "bottle"], ["cearfully", "carefully"], ["intresting", "interesting"],
                       ["classiccal", "classical"], ["caln", "calm"], ["favouritte", "favourite"], ["wan", "want"],
                       ["quech", "quench"], ["laerning", "learning"], ["talll", "tall"], ["nwspaper", "newspaper"],
                       ["wful", "awful"], ["whu", "who"], ["ootside", "outside"], ["somebady", "somebody"],
                       ["agly", "ugly"], ["hadf", "had"], ["bldae", "blade"], ["elephants", "elephantss"],
                       ["mony", "money"], ["everibody", "everybody"], ["muhc", "much"], ["lennd", "lend"],
                       ["ttendance", "attendance"]]

    spelling_errors_dict = dict(spelling_errors)
    errors_stats = {"Corrected good": 0, "Corrected bad:": 0, "Corrected good when shouldn't": 0,
                    "Corrected bad when shouldn't": 0}

    for corrected_from, corrected_to in program_output:
        if corrected_from in spelling_errors_dict:
            if corrected_to == spelling_errors_dict[corrected_from]:
                errors_stats["Corrected good"] += 1
            else:
                errors_stats["Corrected bad:"] += 1
        else:
            if corrected_to == corrected_from:
                errors_stats["Corrected good when shouldn't"] += 1
            else:
                errors_stats["Corrected bad when shouldn't"] += 1

    _visualize_errors_stats(errors_stats)


    def benchmark_func2(program_output):
        pass



        #J.D.
    # def _edits1(self, faulty_word):
    #     s = [(faulty_word[:i], faulty_word[i:]) for i in range(len(faulty_word) + 1)]
    #     deletes = [a + b[1:] for a, b in s if b]
    #     transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b) > 1]
    #     replaces = [a + c + b[1:] for a, b in s for c in alphabet if b]
    #     inserts = [a + c + b for a, b in s for c in alphabet]
    #     # print(set((deletes + transposes + replaces + inserts)))
    #     return set(deletes + transposes + replaces + inserts)
    #
    # def _known_edits2(self, faulty_word):
    #     return set(e2 for e1 in self._edits1(faulty_word) for e2 in self._edits1(e1) if e2 in self._words_set)
    #
    # def _known(self, words):
    #     return set(w for w in words if w in self._words_set)
    #
    # def _generate_candidates(self, faulty_word):
    #     return self._known([faulty_word]) or self._known(self._edits1(faulty_word)) or self._known_edits2(faulty_word) or [faulty_word]