import nltk
from nltk.corpus import brown, gutenberg, reuters, twitter_samples


def filter_out_interpunction(sentences):
    interpuntcion = ['.', ';', ',', '\'', ':', '"', '!', '?', '-', '[', ']', '(', ')']
    sents_wo_interpunction = []
    for sentence in sentences:
        filtered = [word for word in sentence if word not in interpuntcion]
        sents_wo_interpunction.append(filtered)
    return sents_wo_interpunction


def sentences_lists_to_strings(sentences):
    result_list = []
    for sentence in sentences:
        tmp_str = ''
        for word in sentence:
            tmp_str += word + ' '
            # print(result_string)
            # input()
        result_list.append(tmp_str)
    return result_list


def sentences_to_single_string(sentences):
    result_string = ''
    for sentence in sentences:
        for word in sentence:
            result_string += word + ' '
        #print(result_string)
        #input()
    return result_string


def get_sents_from_nltk_corpuses():
    brown_sents = filter_out_interpunction(brown.sents())
    gutenberg_sents = filter_out_interpunction(gutenberg.sents())
    reuters_sents = filter_out_interpunction(reuters.sents())
    sentences = brown_sents + gutenberg_sents + reuters_sents
    sentences = sentences_lists_to_strings(sentences)
    return sentences


def get_sents_string_from_nltk_corpuses():
    sentences = get_sents_from_nltk_corpuses()
    return sentences_to_single_string(sentences)


#print(get_sents_from_nltk_corpuses()[500])
# sentences = brown.sents()
# print(len(sentences))
# sentences = gutenberg.sents()
# print(sentences[0:10])
# sentences = filter_out_interpunction(sentences)
# print(sentences[0:10])
# print(len(sentences))
# sentences = reuters.sents()
# print(len(sentences))



'''
def _load_train_data(self):
    train_data = ""
    for file in os.listdir("train_data"):
        path = os.path.join("train_data", file)
        train_data += open(path, "r").read() + "\n"
    self._train_sentences = self._split_text_to_sentences(train_data)
    self._train_sentences = [self._expand_shortcuts(sentence) for sentence in self._train_sentences]
'''