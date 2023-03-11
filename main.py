from extract_feature import extractandfeature

sample_str = """Two student groups clashed at Ratna Rajya Laxmi Campus in the Capital on Saturday following a dispute over nomination filing for the Free Student Union elections.
Student leaders of the All Nepal National Free Students Union(ANNFSU), a student wing of the CPN-UML, have accused the students associated with the CPN(Unified Socialist) of attacking them with khukuri.
Lokesh Khadka, unit chairman of the ANNFSU, said students Raj Neupane, Anil Bohara and Vivek Gaire were injured in the attack. Khadka claimed they had retaliated after the student wing of the Unified Socialist attempted to attack the election officer and disrupt nomination filing.
“Gangsters from all over Kathmandu were called to attack us and disrupt the election process, ” accused Khadka.
The student wing of the Unified Socialist, however, refuted the allegations.
“The campus chief and the election officer admitted fake students. We have demanded an all-party meeting for that, ” said Sudesh Parajuli of the student union.
Police have been stationed outside the campus after the clash. Kathmandu Police said they have detained those involved in the attack.
The students have gathered outside the campus, while talks between the campus chief, the election committee and student representatives are underway.
Sanjiv Dhakal, unit president of the Nepal Student Union, said that the clash ensued following a dispute over fake students.
Meanwhile, the campus administration has extended the deadline for nomination filing till Sunday following the clash.
Nominations are being filed at campuses across the country on Saturday for the Free Student Union election slated for March 19.
"""



out_final = extractandfeature(sample_str, 0.2)

significant_words = out_final['significant_words']
extract_summ = out_final['summary']
abs_summ = out_final['abs_summ']

print(significant_words)
print(extract_summ)
print(abs_summ)
