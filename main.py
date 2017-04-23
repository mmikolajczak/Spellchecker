from spelling_corrector import SpellingCorrector
from spelling_corrector import benchmark_func

test_data_filename = "test2.txt"


def main():
    sp = SpellingCorrector(return_results=True)
    # results = sp.correct(test_data_filename)
    results = sp.correct(test_data_filename, output_mode='file')
    benchmark_func(results)

main()
