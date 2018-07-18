import nltk
from nltk import pos_tag
import nltk.tokenize 
from nltk.tokenize import word_tokenize
import numpy
nltk.download('maxent_ne_chunker')
nltk.download('words')

f = open(r'C:/Users/prana/Downloads/positive.csv', encoding="utf8")
data = f.readlines()

for line in data:
    tokens = nltk.word_tokenize(line)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    print(entities)
f.close()