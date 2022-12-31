from extract_feature import extractandfeature

out_final=extractandfeature('sample.txt')

significant_words= out_final[0]
extract_summ = out_final[1]

print(significant_words)
print(extract_summ)