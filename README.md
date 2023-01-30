## Yelp Recommendation System


Description of the method: My approach consists of the following components combined linearly:

1. I used a similar method to hwk3 to determine the item-based Collaborative Filtering prediction.
2. XGBregressor: I used the same procedure as in hwk3 to generate XGBregressor prediction, but I made some adjustments for better results.
3. Took the average rate for each business from business (column name: bus mean). 
4. Took the review count for each business from the business (column name: bus review count).
5. From user.json, took average stars value to get the user's average rating (column name: user mean).
6. From user.json-> the number of "useful"s sent by the user (column name: user useful).
7. The weighted rate of a business (column name: useful bus rate)-> A business review will be given more weight if more users have rated it "useful".
To reduce complexity and execution time, I kept track of the coefficients and intersections for each part and applied them to the data.


Error Distribution:

#### >= 0 and < 1: 41286,
#### >= 1 and < 2: 15244,
#### >= 2 and < 3: 5260,
#### >= 3 and < 4: 795,
#### >= 4: 1



#### RMSE: 
0.9796709211544142

#### Time: 
660 sec
