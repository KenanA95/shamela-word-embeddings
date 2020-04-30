from collections import Counter


def read_stop_words(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        words = set()
        [words.add(line.strip()) for line in f]
    return words


def get_word_frequencies(corpus_file):
    freq = Counter()

    # Avoid loading the whole corpus into memory
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            freq.update(line.split())

    return dict(freq)


def remove_stop_words(input_file, stop_words, output_file):
    out = open(output_file, 'w', encoding='utf-8')
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split()
            filtered_sentence = [w.lower().strip() for w in words if not w in stop_words]
            out.write(" ".join(filtered_sentence) + "\n")
    out.close()



