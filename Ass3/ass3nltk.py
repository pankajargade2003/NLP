import nltk 
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 
nltk.download('maxent_ne_chunker') 
nltk.download('words') 

sentence = "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. The company is based in Cupertino, California. In 2007, Apple introduced the iPhone, which revolutionized the smartphone industry. Apple is currently led by CEO Tim Cook, who succeeded Steve Jobs in 2011. Today, Apple is one of the most valuable companies in the world, with a market capitalization of over 2 trillion dollars."

for sent in nltk.sent_tokenize(sentence): 
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))): 
        if hasattr(chunk, 'label'): 
            print(chunk.label(), ' '.join(c[0] for c in chunk))
