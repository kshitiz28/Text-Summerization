from extract_feature import extractandfeature

sample_str = """The World Soil Day is being observed today globally as well as in Nepal with the objective of raising public awareness on the significance of healthy soil and for the sustainable management of the soil fertility.The United Nations General Assembly had in December 2013 declared December 5, 2014 as the World Soil Day and it was formally marked throughout the world since then. Nepal has been observing the Day since 2015. The theme of the World Soil Day this year is 'Soils: where food begins.' Soil is at the heart of all agricultural activities, food security, nutrition security and climate conservation.
The World Soil Day programme reiterates the importance of soil for mankind and the crucial need for its conservation and proper management while at the same time increasing its fertility, the Department of Agriculture said.Director General of the Department, Dr Rewati Raman Poudel said that the debate about food and nutritional security, sustainable agriculture development, conservation of bio-diversity and organic agriculture will have no meaning without the conservation, promotion and proper management of soil.
The soil fertility is deteriorating throughout the world including in Nepal in the recent years with the declining physical, chemical and biological features of the soil. Therefore, this problem of declining soil fertility has been taken as the common global problem.
Poudel said the World Soil Day is being marked with the main goal of raising extensive public awareness to tackle this growing problem of loss in soil fertility.
The World Soil Day is being celebrated at the national level today in Nepal amidst various programmes under the aegis of the Department, Central Agricultural Laboratory, the National Soil Science Research Centre (NARC), Food and Nutrition Security Improvement Project, Rural Enterprises and Economic Development Project, United Nations, Food and Agriculture Organization and the Nepalese Society of Soil Science.
The UN has said that over the last 70 years, the level of vitamins and nutrients in food has drastically decreased, and it is estimated that 2 billion people worldwide suffer from lack of micronutrients, known as hidden hunger because it is difficult to detect.
Soil degradation induces some soils to be nutrient depleted losing their capacity to support crops, while others have such a high nutrient concentration that represent a toxic environment to plants and animals, pollutes the environment and cause climate change.
World Soil Day 2022 and its campaign "Soils: Where food begins" aims to raise awareness of the importance of maintaining healthy ecosystems and human well-being by addressing the growing challenges in soil management, increasing soil awareness and encouraging societies to improve soil health.
"""

out_final=extractandfeature(sample_str,0.8)

significant_words = out_final['significant_words']
extract_summ = out_final['summary']

print(significant_words)
print(extract_summ)