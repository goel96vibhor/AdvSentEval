from __future__ import absolute_import
import nltk
from nltk.corpus import wordnet


synonyms = dict()
antonyms = dict()


# synonyms = []
# antonyms = []
# for syn in wordnet.synsets("good"):
#     for l in syn.lemmas():
#         synonyms.append(l.name())
#         if l.antonyms():
#             antonyms.append(l.antonyms()[0].name())
#
# print(set(synonyms))
# print(set(antonyms))



def get_word_synonym(word):
    if word in synonyms.keys():
        # print "returning already present"
        return synonyms[word]
    else :
        word_syns = []
        word_ants = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                word_syns.append(l.name())
                if l.antonyms():
                    word_ants.append(l.antonyms()[0].name())
        synonyms[word] = set(word_syns)
        antonyms[word] = set(word_ants)
        return synonyms[word]


# word = "good"
# word_syns, word_ants = get_word_synonym(word)
#
# print synonyms[word], antonyms[word]

