import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# New input text (4–5 lines)
text = (
    "Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University. The company is headquartered in Mountain View, California. In 2015, Google was reorganized under a new parent company called Alphabet Inc. Sundar Pichai became the CEO of Google in 2015 and later the CEO of Alphabet in 2019. "
)

# Process text
doc = nlp(text)

# Print entity details
for ent in doc.ents:
    print(
        f"""
{ent.text = }
{ent.start_char = }
{ent.end_char = }
{ent.label_ = }
spacy.explain('{ent.label_}') = {spacy.explain(ent.label_)}"""
    )
