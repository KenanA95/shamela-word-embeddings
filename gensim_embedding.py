from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


def read_corpus(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            yield simple_preprocess(line)


if __name__ == "__main__":
    for century in list(range(100, 200, 100)):
        filename = "data/compiled/corpus_" + str(century) + "AH_lemmatized_clean"
        out_file = "data/gensim/corpus_" + str(century) + "AH_lemmatized.model"
        print("Starting on corpus: " + filename)
        sentences = list(read_corpus(filename))
        model = Word2Vec(sentences, size=300, window=8, min_count=10, workers=8)
        print("Saving the model: ", out_file)
        model.save(out_file)
        # model = Word2Vec.load(out_file)
        print("Top 3 closest to محمد ", model.wv.most_similar("محمد", topn=5))
        print("Top 3 closest to شتى ", model.wv.most_similar("شتى", topn=5))
