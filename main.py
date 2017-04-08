from spelling_corrector import SpellingCorrector
from spelling_corrector import benchmark_func

test_data_filename = "test2.txt"


def main():
    sp = SpellingCorrector(return_results=True)
    print("i'm" in sp._words_set)
    print(("", "i'm") in sp._bigrams_dict._bigrams)
    print("i'm" in sp._bigrams_dict._word_counts)
    results = sp.correct(test_data_filename)
    results = sp.correct(test_data_filename, output_mode='file')
    benchmark_func(results)

main()

#wersja 1.0 alfa
#jestesmy na etapie dzialajacego - ale tylko gdy sa oba bigramy, teraz idziemy dalej - jest schemat dla reszty
#poczyszczone troche z komentarzy
#pojawiaja sie (ofc ;/) bledy:
#- sa (wlasciwie jest, bo jeden) przypadek prawdopodobienstwa powyzej 1 (2.0, that) - POPRAWIONO
#- garden jest w secie-slowniku a mimo nie zostal wziety pod uwage przy poprawianiu garten, zostalo one jak jest
#-licznosc slow w slowniku ~40k, jak na ta liczbe wydaje mi sie ze nie wykrywa zbyt popularnych slow, poprawil np. scary
#-albo niepoprawiono runnfng na running - to raczej przesada, cos musi byc nie tak tu

#wersja 1.01 alfa
#niektore faktycznie nie wystepuja w tekstach - np. scary (mozna by dorzucic jakas porzadniejsza baze tekstow)
#ale brakuje tez w word_secie takich jak np. i'm - pytanie czy nltk czegos tu nie dziala w trakcie
#zreszta dobrym przykladem ze cos sie nie zgadza jest nowowygenerowany blad (ten na ostatnim poprawianym slowie) - ZALATANE CHWILOWO ALE ZNALEZC PRZYCZYNE
#dodano generacje xmla, na razie tylko jakie slowo wykryto jako blad i na co poprawiono

#wersja 1.02 alfa
#dodac funkcje-benchmark - DONE
#naprawic bledy z poprzednich, chwilowa latke, garden, i'm - blad z i'm FOUND AND DONE
#zwiekszyc ilosc testow UPDATE 50/50
#odpalic w koncu i sprawdzic ; p
#wyczyscic kod - troszke przeczysczony
#dosc duzo jest przypadkow nr 3 - czyli przy prob 0 tylko na podstawie liczności