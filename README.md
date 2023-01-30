## Yelp Recommendation System


Description of the method: My approach consists of the following components combined linearly:

1. I used the similar method in hwk3 to determine item-based CF prediction.
2. XGBregressor: I use the same procedure as in hwk3 to generate XGBregressor prediction, but I make some adjustments for better results.
3. Take from business the average rate for each business (column name: bus mean). 
4. Take from the business the review count for each company (column name: bus review count).
5. Take the user.json value from average stars to get the user's average rating (column name: user mean).
6. Take from user.json->useful the number of usefuls received by the user (column name: user useful).
7. The weighted rate of a business (column name: useful bus rate): A business review will be given more weight if more users have given it the rating of "useful." 
To save execution time, I keep track of the coefficients and intersections for each part and apply them to data.


Error Distribution:

#### >= 0 and < 1: 41286,
#### >= 1 and < 2: 15244,
#### >= 2 and < 3: 5260,
#### >= 3 and < 4: 795,
#### >= 4: 1



#### RMSE: 0.9796709211544142

#### Time: 660 sec
