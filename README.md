Multi-Class Sentiment Analysis of US Tariffs Comments
This project performs multi-class sentiment analysis on US tariffs-related comments. The comments are classified into Positive, Neutral, and Negative categories using TF-IDF vectorization and an XGBoost classifier.
________________________________________
Features
•	Compute sentiment polarity using TextBlob if not provided.

•	Create sentiment labels: Positive, Neutral, Negative.

•	Handle class imbalance with RandomOverSampler.

•	Train XGBoost model for multi-class classification.

•	Evaluate model performance:
o	Classification report (precision, recall, f1-score)

o	Confusion matrix

o	Accuracy visualization

o	Top 20 important words/features

•	Visualize data:
o	Sentiment distribution before and after balancing

o	Word clouds for Positive, Neutral, and Negative comments

•	Save outputs:
o	us_tariffs_final_predictions.csv

o	positive_comments.csv

o	negative_comments.csv

•	Display top 5 Positive and Negative comments directly in the console.
________________________________________
Example Outputs
Sentiment Distribution Before Balancing
Sentiment Before
 
Sentiment Before
Sentiment Distribution After Balancing
Sentiment After
 
Sentiment After
Confusion Matrix
Confusion Matrix
 
Confusion Matrix
Word Clouds
 
•	Positive: Positive Word Cloud

•	Neutral: Neutral Word Cloud

•	Negative: Negative Word Cloud
Feature Importance
Top Words
 
Top Words
________________________________________
Top Comments (Example)
Top 5 Positive Comments: 1. I don't really understand this comment. Trump's and putin's goal is to destabilize America. This is exactly what Agent Orange wants. Putin is the one who is winning here.
2. As far as I'm aware they don't own any American roads. What you might be thinking of are stories about Chinese companies forming a public-private partnership with other companies to build American infrastructure (including roads and bridges). In those kinds of deals, the Chinese company oversees construction while the actual work is mostly done by American labor. The argument for doing it this way is that it was cheaper, and the argument against it is that it hurts local competitors and means that the more skilled jobs like engineering miss out on a chance for local training and experience cos the Chinese firms tended to use their own staff for that sort of thing.
3. To my recollection this affects Michigan significantly. Granted it was only by 1.5%, but Michigan voting for Trump 2 of the last 3 elections is a bit ridiculous.

Republicans largely opposed the bailouts for the big 3. Could have cared less about the lead pipes in Flint or the bankruptcy problems in Detroit.
The high  Muslim population there thought voting for the guy who was saying he would help Isreal finish the job in Gaza was a good idea.Now this. I feel for the people who knew better and voted accordingly, but empathy for others has waned.    
4. Trump really is misunderstanding the Chinese. They don't give a shit, and its an actual dictatorship who doesn't care about it's people.
Adding the fact, China will not lose face and it's now a USA vs China nationalistic issue.
5. Can someone way more educated than me, tell me why the conservative sub is seeing this as a win? They seem to think it's a game of chicken and that China can't win since they export to the US more so than the other way around.

Top 5 Negative Comments:
1. Canada can out last the US/Trump in this silly unnecessary contest. In the US, if the stock market continues to fall and people see their 401K and personal wealth evaporate, they are going to get angry real fast. 
To Canada they are taking this potential austerity as a matter of patriotic pride. Trump continues to insult them so their anger burns. They are willing to endure more hardships for longer periods of time. They will work to establish other markets and the bad will generated will remain for years.None of this was necessary and Trump is driving the car off the cliff.
2. It feels like the US is playing with fire because China 100% has the stomach to just inject capital investments into their economy while the US is cutting food to food banks. The Republicans have no stomach to invest in America.
3. The funniest part of this? China clearly has the upper hand here. Communists/socialists are over the fucking moon with this because once the pain starts to hit everyone, they will have LOADS of propaganda and talking points to utilize on how the capitalist-dominated, American-dominated world trade system is flawed and must be destroyed
4. 35. But yeah.
1990, exchange rate is 2:1. China 500B GDP, US 7T.
2025 exchange rate 6:1, going to 8:1. Gdp 17T, US 27T.
So while our economy 3.5X, theirs is 35x, but the currency is -3x, going to 4x.
Fucking insane advantage. You think people would buy their products if they were even 3x the cost? Or our products would not sell like hotcakes at 1/3rd the cost?
And that will go for everyone. India, Malaysia,  etc... their main competitor just gave themselves a 25% pricing advantage in the world market.
5. Yeah how they didn’t go about this slowly and selectively until they snowballed (IF I should say) some “wins” and keep it a controlled burn. At least then we’d boil slowly I guess .. better then this crap
PS D:\Tariff>
________________________________________
Requirements
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn xgboost imbalanced-learn textblob
________________________________________
How to Run
1.	Place the dataset us_tariffs_comments_sentiment.csv in the project folder.

2.	Run the script:
python sentiment_analysis_final.py
3.	The script will generate:
o	Console outputs: classification report, top comments, feature importance

o	CSV files:
	us_tariffs_final_predictions.csv

	positive_comments.csv

	negative_comments.csv

________________________________________
Folder Structure
Sentiment-Analysis-Tariffs/
│
├─ sentiment_analysis_final.py
├─ us_tariffs_comments_sentiment.csv
├─ README.md
├─ positive_comments.csv
├─ negative_comments.csv
├─ us_tariffs_final_predictions.csv
└─ images/           
________________________________________
Author
Elpsha

