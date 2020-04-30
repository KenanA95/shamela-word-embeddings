import csv
import os
import re
from collections import Counter, OrderedDict

ARABIC_REGEX = r'[\u0621-\u064A]+'


def get_start_of_book_index(file_contents):
    matchers = ['#####FILENAME', '#####ARABICA']
    return [file_contents.index(s) for s in file_contents if any(xs in s for xs in matchers)][1]


def get_arabic_lines_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        file_contents = file.readlines()
        book_text = file_contents[get_start_of_book_index(file_contents) + 1:]
        for line in book_text:
            arabic_text = " ".join(re.findall(ARABIC_REGEX, line))
            yield arabic_text


def walk_data_directory(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext[-4:] == 'ara1':
                yield os.path.join(subdir, file)


def build_century_corpus(data_dir, out_file):
    with open(out_file, 'w', encoding='utf-8') as out:
        for f in walk_data_directory(data_dir):
            print("Writing text from file: " + f)
            for line in get_arabic_lines_from_file(f):
                out.write(line + "\n")


def generate_corpus_summary(corpus_filename, out_file, min_count=10):
    freq = Counter()

    # Avoid loading the whole corpus into memory
    with open(corpus_filename, 'r', encoding='utf-8') as f:
        for line in f:
            freq.update(line.split())

    with open(out_file, 'w', encoding='utf-8') as corpus_summary:
        fieldnames = ['word', 'frequency']
        writer = csv.DictWriter(corpus_summary, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in OrderedDict(freq.most_common()).items():
            if v >= min_count:
                writer.writerow({"word": k, "frequency": v})
        print(out_file + " contains: + " + str(len(freq)) + " unique tokens")
