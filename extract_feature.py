from sklearn.feature_extraction.text import CountVectorizer
from ntpath import join
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from gensim.parsing.preprocessing import remove_stopwords
import string


def extractandfeature(name,compression):

    lines=name.split(".")

    doc_test = []
    for i in range(len(lines)):
        doc_test.append(lines[i].split('.'))

    final_doc = []
    for i in range(len(doc_test)):
        for j in range(len(doc_test[i])):
            final_doc.append(doc_test[i][j])

    without_stopwords = []
    for i in final_doc:
        filtered_sentence = remove_stopwords(i)
        without_stopwords.append(filtered_sentence)

    without_SandP = []
    for i in without_stopwords:
        filtered_sentence = i.translate(
            str.maketrans('', '', string.punctuation))
        without_SandP.append(filtered_sentence)

    vectorizer = CountVectorizer()

    bag_of_words = vectorizer.fit_transform(without_SandP)
    bag_of_words.todense()
    svd = TruncatedSVD(n_components=1)
    lsa = svd.fit_transform(bag_of_words)
    topic_encoded_df = pd.DataFrame(lsa, columns=["topic1"])
    topic_encoded_df["without_stopwords"] = without_stopwords
    # topic_encoded_df[["without_stopwords", "topic1"]]
    dictionary = vectorizer.get_feature_names()
    # print(dictionary)
    encoding_matrix = pd.DataFrame(svd.components_, index=[
                                   "topic1"], columns=dictionary).T
    encoding_matrix['abs_topic1'] = np.abs(encoding_matrix)
    encoding_matrix.sort_values('abs_topic1', ascending=False)
    final_matrix = encoding_matrix.sort_values('abs_topic1', ascending=False)
    # sentence1 = final_matrix[final_matrix["abs_topic1"] >= 0.2]
    sentence1 = final_matrix.head(4)
    index_list = list(sentence1.index.values)

    words_after_compression = []
    if compression <= 0.33:
        words_after_compression.append(index_list[0])
    elif compression <= 0.66 and compression > 0.33:
        words_after_compression.append(index_list[0])
        words_after_compression.append(index_list[1])
    else:
        words_after_compression.extend(index_list)


    final_conclusion = []
    for i in range(len(final_doc)):
        for j in range(len(words_after_compression)):
            if words_after_compression[j] in final_doc[i]:
                final_conclusion.append(final_doc[i])

    list_final = list(set(final_conclusion))

    summ_final = '.'.join(list_final) + "."
    significant_final = ' '.join(index_list)

    res = {
        "summary": summ_final,
        "significant_words": significant_final
    }

    return res
