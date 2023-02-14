from sklearn.feature_extraction.text import CountVectorizer
from ntpath import join
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from gensim.parsing.preprocessing import remove_stopwords
import string

# importing
from transformers import pipeline
# loading summarizer pipeline
summarizer = pipeline('summarization')

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


article = """
Thank you. Thank you. 
Thank you to Vice President Pence. 
He's a good guy. We've done a great job together.
And Merry Christmas, Michigan. Thank you, Michigan. 
What a victory we had in Michigan. What a victory was that. One of the greats.
Was that the greatest evening? 
But I'm thrilled to be here with thousands of hardworking patriots as we celebrate the miracle of Christmas, the greatness of America and the glory of God. Thank you very much. And did you notice that everybody is saying Merry Christmas again? Did you notice? Saying Merry Christmas. I remember when I first started this beautiful trip, this beautiful journey, I just said to the First Lady, "You are so lucky. I took you on this fantastic journey. It's so much fun. They want to impeach you. They want to do worse than that." By the way, by the way, by the way, it doesn't really feel like we're being impeached. The country is doing better than ever before. We did nothing wrong. We did nothing wrong. And we have tremendous support in the Republican Party like we've never had before. Nobody's ever had this kind of support. But this sacred season, our country is thriving and it's thriving truly like it has never, it has never happened before to the extent what's happening now. And by the way, your state, because of us, not because of local government, but because of us, because of the job that we've done. Because I understand she's not fixing those potholes. That's what the word is. It was all about roads and they want to raise those gasoline taxes on you. We don't want to do that. 
"""
print(summarizer(article,max_length=200,min_length=100,do_sample=False)[0]['summary_text'])