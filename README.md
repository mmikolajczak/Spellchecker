# Spellchecker

Spellchecker finds and correct spelling errors in text.
The method of correction is based on conditional probability and Bayes Theorem.
It analyses rather little context of error word - bigrams.

Training data is generated from some literature classic from Project Gutenberg and 
(added recently) corporas from nltk.

Program has some options that allows to better analyse what and how was corrected like 
creating xml report file that contains pairs (error_word, corrected_word) or visualizing
statistics of correction in hraph form.

This is a initial readme, more details will be added along with newer versions of project.
