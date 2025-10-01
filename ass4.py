from nltk.util import ngrams  

# Unigram model
n = 1
sentence = ("My best friend and I study computer science together at college. "
            "We often spend long hours in the library preparing for exams. "
            "During lunch breaks, we enjoy talking about new technologies. "
            "Sometimes we also help each other with coding assignments. "
            "Our friendship has grown stronger through teamwork and learning.")
unigrams = ngrams(sentence.split(), n)
print(f"\n***********   UNIGRAM    ************************")
for item in unigrams:
    print(item)

# Bigram model
n = 2
sentence = ("The professor explained the assignment clearly during the lecture. "
            "Many students asked questions to understand the topic better. "
            "After class, we gathered in the study room to discuss ideas. "
            "Working in groups helped us learn from each other. "
            "The professor encouraged collaboration and creativity.")
bigrams = ngrams(sentence.split(), n)
print(f"\n***********   BIGRAM    ************************")
for item in bigrams:
    print(item)

# Trigram model
n = 3
sentence = ("Our college library is full of useful resources for research. "
            "I often borrow books on history, literature, and science. "
            "The digital section provides access to online journals and articles. "
            "Quiet study areas make it easier to focus on assignments. "
            "Spending time in the library has improved my learning habits.")
trigrams = ngrams(sentence.split(), n)
print(f"\n***********   TRIGRAM    ************************")
for item in trigrams:
    print(item)

# Four-gram model
n = 4
sentence = ("My friend enjoys playing football after college classes end. "
            "Sometimes we all gather to watch matches on the big screen. "
            "Sports bring us closer and teach us discipline and teamwork. "
            "Even during busy weeks, we find time for recreation. "
            "Balancing studies and sports makes life more enjoyable.")
four = ngrams(sentence.split(), n)
print(f"\n***********   FOUR    ************************")
for item in four:
    print(item)

# Five-gram model
n = 5
sentence = ("The group of friends planned a trip during the summer vacation. "
            "They saved money and organized everything in advance. "
            "The journey created unforgettable memories for everyone. "
            "They took pictures, shared stories, and enjoyed nature together. "
            "The trip strengthened their bond and brought them happiness.")
five = ngrams(sentence.split(), n)
print(f"\n***********   FIVE    ************************")
for item in five:
    print(item)
