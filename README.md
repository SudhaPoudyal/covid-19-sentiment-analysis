###	Problem Statement and Background:

Since the start of year 2020, people all around the world are coping or struggling with a new disease called Coronavirus COVID-19. The total cases are more than 195 million, total death from this disease is more than 4 million and recovered cases are more than 177 million, US being the highest and India the second highest in terms of total cases. If we compare by total deaths, US again has the highest deaths followed up by Brazil. Similarly, if we compare by total number of recovered cases, India has the highest followed up by USA [1].  On 8 December 2020, the United Kingdom became the first country to roll out COVID-19 vaccine to its citizens which was then followed by other countries all over the world then onwards [2]. 
The vaccination is made free by their government to their citizens; however, people are hesitant towards the vaccination. Such risk perception of the people can be seen in all developed and undeveloped countries. The purpose of this study is to analyse the sentiment of people towards the vaccination. How are people responding, positively or negatively or neutral towards the different vaccines produced by different countries?
As a Data Scientist, I am interested in finding the best model while analysing the given data. With that in mind, this study will also focus on finding the best machine learning algorithms in terms of accuracy. In this study, three algorithms viz Bernoulli Naïve Bayes algorithm, Logistic Regression and Support Vector Machine (SVM) are taken into consideration.

Research questions:
1.	Finding the sentiments of people on the vaccines
2.	Which model works better for analysing social media data?

### Resources: 
The main source of data is from social media i.e., tweets from twitter. It will be impossible to collect data from the people who have strong perception towards vaccination, however, do not have a social media presence. 
The datasets were collected from Kaggle [3]. “The tweets were queried using vaccine-related keywords, the specific vaccine being referred to in the tweet is not explicitly included as a column in the dataset” [4]. The dataset is in the .csv format so it was quite easy to import in python to pre-process and exploratory data analysis. The tweets were collected from 12 December 2020 to 31 July 2021 with altogether 150,990 tweets within this period of almost 7.5 months. December 12, 2020 was the day of approval of the Pfizer-BioNTech vaccine [4]. Altogether there are 150,990 rows and 16 columns, however; during the analysis phase, some of the columns were dropped out and new calculated columns were generated. The main data for our purpose are mainly tweets, users_followers, users_friends and hashtags. There is mixed type of data numerical, strings, both categorial and discrete.

### Business Model:
The project is related with the application of technology in the health sector for the overall wellbeing of the society. With the help of this project, we can analyse some insights and draw conclusions mainly on the perception of the people towards covid-19 vaccinations and the remedial measures that can be considered to improve the positivity for vaccination. We can also find the major players who are actively supporting or who are strongly against the vaccination. The network of those actors in the community can be analysed and they can be used to influence other people in the network connection who are hesitant or not willing to have vaccines. Within the variety of vaccines, there might be preference of one vaccine over other brand of vaccines, with their own benefits and side effects. In general, the application of this study mainly includes in social media monitoring, customer feedback, brand monitoring and reputation management, product analysis [5].
Data availability from the open source Kaggle has clean data to some extent which was imported into Jupyter notebook with the help of different libraries. Data wrangling and pre-processing were carried out before exploratory data analysis. Within that data, with the help of Jupyter notebook, further calculations of sentiments were done based on the tweets collected. In this study, the polarity of the sentiments along with the intensity of those sentiments were calculated using Valence Aware Dictionalry and sEntiment Reasoner (VADER), a Python lexicon and rule-based sentiment analysis tool. “VADER is designed to determine sentiments of social media posts based on individual words, emoticons and sentences” [6]. Then an overall picture of how people perceive towards those vaccines were figured out. A model was developed in Python using Bernoulli Naïve Bayes algorithm, Logistic Regression and Support Vector Machine (SVM) to compare the accuracy of each model for this analysis.

![image](https://user-images.githubusercontent.com/75915138/131266823-51366166-5739-4220-b91d-e2d83e7999fd.png)
Figure 1: Workflow diagram for this study

Data acquisition was one of the challenges in the selected project. Due to insufficient resources and limited time and expertise, open-source data was extracted, and all the analysis are relied on the same data. Covid-19 has changed the world dramatically and uncertainties have increased day by day. The result obtained from this study might not give the real picture specially on the sentiment of people and the perception towards vaccination as data were from 6 months back.  Besides that, sentiment analysis of tweets is all about contextual mining of text which basically identifies and extract subjective information from the online conversations using machine learning algorithms. With time and advancement in technology in deep learning, although the ability to analyse text has improved considerably [7] however; there are always shortcoming, and nothing is absolute. 
The result of this study shows the distribution pattern of sentiments so medical practitioners or policy makers, or decision makers can work out to increase the positivity and decrease the negativity towards the perception of vaccination. Awareness campaign of long-term effect of covid-19 if infected versus the impact of vaccination in the long run can be delivered using social media. The users with higher number of followers and friends can influence other users within their network which is one of the strengths of using social media for disseminating information to the public. 

### Data Analysis:
The data collected from Kaggle was pre-processed to remove unwanted characters such as punctuations, URLs, numbers, stop words along with tokenisation and stemming. The text mining was carried out for each tweets using VADER lexicon and rule-based sentiment analysis tool which gives polarity and intensity of each word based on the text specified in that tweet. Sentiment intensity analyser was created to categorise the dataset. If the compound score >=0.05, it is positive, if the score is >-0.05 and <0.05, it is categorised as neutral and for the compound score of <=-0.05, it is negative sentiment [8]. From the Figure 2, the total number of neutral sentiments are higher of all, followed up by positive sentiments.

 ![image](https://user-images.githubusercontent.com/75915138/131266801-b629c58a-9d21-4bac-a398-e80adab4f3ee.png)

Figure 2: Total count of sentiments

Data visualisation is one of the most important steps in Machine Learning projects as it gives an approximate picture about the dataset. Cloud of words is very popular visualisation techniques, where most frequent words appear in larger size and vice versa. This visualisation can be generated using a package in Python [9]. 

 ![image](https://user-images.githubusercontent.com/75915138/131266785-92009fb4-55c5-4b04-9ecf-d4eb8572cbc6.png)

Figure 3: Cloud of words for positive sentiments
An example is presented for positive tweets (Figure 3) where PfizerBioNTech, Moderna, SputnikV, are some of the names of vaccines that are listed and sorted by frequency. PfizerBioNTech vaccine was appeared with highest number in the tweets. Similarly, word cloud was plotted for each negative and neutral sentiments in the notebook.

 ![image](https://user-images.githubusercontent.com/75915138/131266828-bc7d3a8f-7231-4e93-86e2-1707eb146702.png)

Figure 4: Distribution of sentiment over time

From the Figure 4 and Figure 5, we can see how the sentiment were distributed with respect to time. There were ups and downs in the sentiments during the first three months i.e., December 2020, January 2021, and February 2021 then it gradually slowed down.

 ![image](https://user-images.githubusercontent.com/75915138/131266829-660e3c60-86b8-4f51-9a1a-18173d0bdaab.png)

Figure 5: Distribution of sentiment over time (Positive, Negative and Neutral)

To extract features from clean tweets, Term Frequency-Inverse Document Frequency (TF-IDF) was used which is a statistical measure to evaluate the importance of a word to a document in a collection or corpus. The more a word appears in the document, the more importance it is but it is offset by the frequency of the word in the collection [10]. There is a package in scikit-learn known as TfidfVectorizer which was imported during analysing process in Jupyter notebook.
The dataset was then split into train and test subset in the ratio of 80:20 so that we can train and test the model. The next process is application of machine learning models to our dataset. Three different supervised machine learning algorithms namely Bernoulli Naïve Bayes, Logistic Regression and SVC model were tried in the dataset to find the model with best performance. The logic behind selecting these models was to find the performance of those simple to complex models [11]. Each of the model was evaluated against accuracy and F1 scores obtained from the confusion matrix for three different values: positive, negative, and neutral. From the confusion matrix, true positive, true negative, false positive and false negative values can be obtained, for each of those models.

### Result Interpretation:
The overall sentiments of the tweets were more neutral and positive than that of negative. The positive extreme during the initial stage of vaccine rollout can be attributed because of discovery of a shield against the coronavirus. The negative extreme at the same time can be a result of rumours on the credibility of the vaccines against this new virus [3]. In this case, if we see PfizerBioNTech, this term appeared in both positive and negative sentiments in the cloud of words. However, with time the perception of people might have changed due to severity of virus or effectiveness of vaccines which is very clearly represented by the visualisations. After almost first three months, the sentiments have become normalised and hopefully it would accelerate in the coming months.

 ![image](https://user-images.githubusercontent.com/75915138/131266834-775c73bd-f531-4c5c-8129-1fa9f78240bf.png)

Figure 6: Box plot of sentiments in terms of intensity

 ![image](https://user-images.githubusercontent.com/75915138/131266838-b7829f32-5b3f-4794-b2ad-7e81709926a0.png)

Figure 7: Distribution of positive sentiment wrt Time

For this specific analysis of twitter data of altogether 150,990 tweets, from last 7.5 months, Support Vector Machine stood out with highest performance in terms of accuracy and F1 score which can be seen from  Figure 8 and Table 1.

![image](https://user-images.githubusercontent.com/75915138/131266842-74920c09-023d-4cb2-b113-0655871e1b78.png)

Figure 8: Confusion matrix of SVM model

Table 1: Comparison of three different models with 80:20 train – test dataset and 70:30 dataset

![image](https://user-images.githubusercontent.com/75915138/131267085-7c9be905-4ad7-436c-a5c6-84308c08e03b.png)

Based on the table above, SVM model has the highest accuracy and the highest F1 score as compared with other two models. There are lots of topics within this study to go in detail for future exploration. Model with higher train data results better than the one with lower train data. Accuracy and F1 Score for all three models behave either equal or higher with 80:20 train -test dataset than with 70:30 dataset.


### References

[1] 	"Coronavirus: Worldometers," [Online]. Available: https://www.worldometers.info/coronavirus/. [Accessed 27 07 2021].
[2] 	"News: Aljazeera," [Online]. Available: https://www.aljazeera.com/news/2020/12/24/vaccine-rollout-which-countries-have-started. [Accessed 27 07 2021].
[3] 	Kaggle, "Sentiment Analysis of Covid19 Vaccination tweets," 30 05 2021. [Online]. Available: https://www.kaggle.com/raghav2002sharma/sentiment-analysis-of-covid19-vaccination-tweets. [Accessed 09 08 2021].
[4] 	S. Dua, "Sentiment Analysis of COVID-19 Vaccine Tweets," towards data science, 23 03 2021. [Online]. Available: https://towardsdatascience.com/sentiment-analysis-of-covid-19-vaccine-tweets-dc6f41a5e1af. [Accessed 08 08 2021].
[5] 	MonkeyLearn, "8 Applications of Sentiment Analysis," [Online]. Available: https://monkeylearn.com/blog/sentiment-analysis-applications/. [Accessed 13 08 2021].
[6] 	R. D. S. M. A. P. S. S. Samira Yousefinaghani, "An analysis of COVID-19 vaccine sentiments and opinions on Twitter," International Journal of Infectious Diseases, vol. 108, no. DOI:https://doi.org/10.1016/j.ijid.2021.05.059, pp. 256-262, July 01, 2021. 
[7] 	towards data science, "Sentiment Analysis: Concept, Analysis and Applications," 08 01 2018. [Online]. Available: https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17. [Accessed 13 08 2021].
[8] 	C. Hutto, "VADER-Sentiment-Analysis," 18 11 2014. [Online]. Available: https://github.com/cjhutto/vaderSentiment#about-the-scoring. [Accessed 06 08 2021].
[9] 	D. Das, "Social Media Sentiment Analysis using Machine Learning : Part — I," towards data science, 07 09 2019. [Online]. Available: https://towardsdatascience.com/social-media-sentiment-analysis-49b395771197. [Accessed 12 08 2021].
[10] 	D. Das, "Social Media Sentiment Analysis using Machine Learning : Part — II," towards data science, 23 09 2019. [Online]. Available: https://towardsdatascience.com/social-media-sentiment-analysis-part-ii-bcacca5aaa39. [Accessed 12 08 2021].
[11] 	Gunjan28, "Twitter Sentiment Analysis- A NLP Use-Case for Beginners," Analytics Vidhya, 11 06 2021. [Online]. Available: https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/. [Accessed 13 08 2021].
[12] 	C. H. a. E. Gilbert, "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text," Eighth International Conference on Weblogs and Social Media (ICWSM-14), Ann Arbor, June 2014.



