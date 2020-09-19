import pandas as pd  # import data
import numpy as np  # mathematical functions
from string import punctuation  # removing punctuations from text
import re  # regex for garbage collection
from nltk import PorterStemmer  # converting words in base form

# reading training data
df = pd.read_csv("train.csv")

# cleaning data
drop_col = ["location", "id"]
df.drop(drop_col, axis=1, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# collecting hashtags, links and mentions
hashtags = []
links = []
mentions = []

# patterns
hash_pattern = r"#[a-z0-9_]+"
# http://[a-z0-9.]/+ |
link_pattern = r"http://[a-z0-9.]+/[a-z0-9.]+"
mention_pattern = r"@[a-z0-9]+"

for t in df["text"]:
    hashtags.extend(re.findall(hash_pattern, t, re.I))
    links.extend(re.findall(link_pattern, t, re.I))
    mentions.extend(re.findall(mention_pattern, t, re.I))

# print(hashtags[:50])
# print(links[:50])
# print(mentions[:50])

# writting hashtags, lings and mentions to text file.
with open("specials.txt", "w") as f:
    f.write("All hashtags\n")
    [f.write(h + "\t") for h in hashtags]
    f.write("\n")
    f.write("All links\n")
    [f.write(l + "\t") for l in links]
    f.write("\n")
    f.write("All mentions\n")
    [f.write(m + "\t") for m in mentions]

# removing hastags, links and mentions form text

# for t in df["text"][:15]:
#     print(re.findall(garbage_pattern, t, re.I))


def clean(text):
    garbage = []
    # return re.sub(garbage_pattern, " ", text)
    garbage.extend(re.findall(hash_pattern, text, re.I))
    garbage.extend(re.findall(link_pattern, text, re.I))
    garbage.extend(re.findall(mention_pattern, text, re.I))
    text_list = text.split()
    text = " ".join([i for i in text_list if i not in garbage])
    return text


df["clean_text"] = df["text"].map(clean)

# removing punctuations
df["no_punc"] = df["clean_text"].map(
    lambda x: "".join([w.strip(punctuation) for w in x]))

# combining all text into one string
corpus = "\n".join([t for t in df[["no_punc", "keyword"]]])

# stemming text: converting word to their base form
stemmer = PorterStemmer()
df["stemmed_text"] = df["no_punc"].map(lambda x: stemmer.stem(x))

# save clean data
df.to_csv("clean_train.csv", index=False)


