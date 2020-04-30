import os
import re
from project.database import *
import pymongo

# Could probably use pyparser but this works and it was quick to write...

header_arabic_fields = {
    "AuthorNAME": "name",
    ".BookTITLE	": "title",
    "BookTITLEalt": "alt_title",
    "BookSUBJ": "subject"
}

header_number_fields = {
    "011.AuthorDIED": "died",
    "BookVOLS": "volumes"
}

ARABIC_REGEX = r'[\u0621-\u064A]+'


def split_header_and_text(filename):
    start, stop, index = 0, 0, 0
    file = open(filename, "r", encoding="utf8")
    with open(filename, "r", encoding="utf8") as f:
        line = f.readline()
        while line:
            if "BEGofRECORD" in line:
                start = index + 1
            if "#####FILENAME" or "#####ARABICA" in line:
                end = index - 1
                full_contexts = file.readlines()
                return full_contexts[start:end], " ".join(full_contexts[end + 1:])
            index += 1
            line = f.readline()


def parse_doc_data(filename):
    header, text = split_header_and_text(filename)
    metadata = {}

    for key, val in header_arabic_fields.items():
        line = [i for i in header if key in i][0]
        result = " ".join(re.findall(ARABIC_REGEX, line))
        metadata[val] = result

    for key, val in header_number_fields.items():
        line = [i for i in header if key in i][0]
        line = line.replace("\n", '')
        result = line.split("::")[1]
        metadata[val] = result

    return {
        "title": metadata['title'],
        "alt_title": metadata['alt_title'],
        "subject": metadata['subject'],
        "volumes": metadata['volumes'],
        "author_name": metadata['name'],
        "author_died": metadata['died'],
        "text": " ".join(re.findall(ARABIC_REGEX, text))
    }


def walk_data_directory(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext[-4:] == 'ara1':
                yield os.path.join(subdir, file)


if __name__ == "__main__":
    collection_name = '400AH'
    connection_string = "mongodb+srv://" + mongo_username + ":" + mongo_password + \
                        "@cluster0-ann00.gcp.mongodb.net/test?retryWrites=true&w=majority"
    client = MongoClient(connection_string)
    db = client.ArabicTexts
    collection = db[collection_name]

    for index, file_name in enumerate(walk_data_directory('data/' + collection_name)):
        print("Index: " + str(index) + " Parsing: " + file_name)
        book = parse_doc_data(file_name)
        save(collection_name, book)

    collection.create_index([('text', pymongo.TEXT)])
    result = get_word_frequency(collection_name, 'كتاب')
    print(result)
