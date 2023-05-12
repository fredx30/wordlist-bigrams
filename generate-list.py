import nltk
from nltk.corpus import stopwords

# Install/update
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download('reuters')

# Filter out stop words
stop_words = set(stopwords.words('english'))

# Load the corpus & Filter out stop words
from nltk.corpus import reuters, brown, gutenberg
words = []
for corpusView in [reuters]:
    words += [word.lower() for word in corpusView.words() if word.isalpha() and word.lower() not in stop_words]
freq_dist = nltk.FreqDist(words)


# Count the frequency of each word and bigram in the corpus
bigrams = list(nltk.bigrams(words))
freq_dist_bigrams = nltk.FreqDist(bigrams)


# Sort the words and bigrams by frequency
sorted_words = sorted(words, key=lambda word: freq_dist[word], reverse=True)
sorted_bigrams = sorted(bigrams, key=lambda bigram: freq_dist_bigrams[bigram], reverse=True)

# Dedupe the bigrams while maintaining frequency counts
deduped_bigrams = {}
for bigram, freq in freq_dist_bigrams.items():
    sorted_bigram = tuple(sorted(bigram))
    if sorted_bigram in deduped_bigrams:
        deduped_bigrams[sorted_bigram] += freq
    else:
        deduped_bigrams[sorted_bigram] = freq

# Sort the deduped bigrams by frequency
sorted_deduped_bigrams = sorted(deduped_bigrams.items(), key=lambda x: x[1], reverse=True)

# Write list to file
with open('bigrams-reuters-g-2.txt', 'w') as f:
    for bigram, freq in sorted_deduped_bigrams:
        if freq > 2:
            f.write(f"{'-'.join(bigram)}\n")