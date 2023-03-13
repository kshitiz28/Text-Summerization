from extract_feature import extractandfeature

sample_str =  """In Sanskrit, Maha means great or big, and Shivaratri means a night dedicated to Shiva. According to the Rudra Samhita, the wedding of Shiva and Parvati took place in the Himalayas. On that day, Parvati transformed herself into Chandraghanta with golden skin and ten arms, and they got married in their beautiful divine forms at Triyuginarayan village in Rudraprayag, India. So, their marriage is celebrated as Mahashivaratri every year.

 Mahashivaratri is an important festival that is widely celebrated in both Nepal and India, though with different perspectives and processes. Some fast for the entire day of Mahashivaratri and perform vedic or tantric worship of Shiva, while others practise meditative yoga. People also perform the Rudra Abhishek in the day of Mahashivaratri, a special type of puja to please Lord Shiva and seek his blessings. The rituals are carried out throughout the day or in different muhurtas(ancient measurement units for time). Though the daytime of the Mahashivaratri rituals differ, at night people generally stay awake doing bhajan, kirtans, meditation, sadhana, upasana, etc.

 Mahashivaratri is a magnificent occasion for the followers of Lord Shiva to praise him and seek his blessings. In fact, for devotees of Lord Shiva, nothing is more important than fasting on this day, ” said Narayan Bhatt, a priest at Pashupatinath Temple. A common ritual a lot of people follow is to take a bath in a river early in the morning or in warm water with sesame seeds at their homes in order to clean themselves. “Devotees can fast for 24 hours in the day of Mahashivaratri without eating or drinking, but they can also fast by drinking water and eating sattvic food(unprocessed food with yogic qualities to increase energy), ” added the priest.

"""

out_final = extractandfeature(sample_str, 0.5)

significant_words = out_final['significant_words']
extract_summ = out_final['summary']
abs_summ = out_final['abs_summ']

print(significant_words)
print(extract_summ)
print(abs_summ)
