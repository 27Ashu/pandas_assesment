#1. Import pandas under the name pd.
import pandas as pd
import numpy as np

#2. Print the version of pandas that has been imported(
print(pd.__version__)

#3. Print out all the version information of the libraries that are required by the pandas library
print(pd.show_versions())

#Consider the following Python dictionary data and Python list labels
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

#4.Create a DataFrame df from this dictionary data which has the index labels.
df = pd.DataFrame(data,index = labels)
print(df)

#5.Display a summary of the basic information about this DataFrame and its data.
print(df.info())

#6.Return the first 3 rows of the DataFrame
print(df.iloc[:3])

#7.select just the 'animal' and 'age' columns from the DataFrame
print(df[['animal','age']])

#8.Select the data in rows [3, 4, 8] and in columns ['animal', 'age'] using iloc
print(df.loc[df.index[[3,4,8]],['animal', 'age']])

#9.Select only the rows where the number of visits is greater than 3
print(df[df['visits'] > 3])

#10.Select the rows where the age is missing, i.e. is NaN
print(df[df['age'].isnull()])

#11.Select the rows where the animal is a cat and the age is less than 3
print(df[(df['animal'] == 'cat') & (df['age'] < 3)])

#12.Select the rows the age is between 2 and 4 (inclusive)
print(df[df['age'].between(2,4)])

#13.Change the age in row 'f' to 1.5
print(df.loc['f','age'])#this is before changing
df.loc['f','age'] = 1.5
print(df.loc['f','age'])#this is after changing

#14.Calculate the sum of all visits (the total number of visits).
vis_sum = df['visits'].sum()
print(vis_sum)

#15.Calculate the mean age for each different animal in df.
print(df.groupby('animal')['age'].mean())

#16.Append a new row 'k' to df with your choice of values for each column.
df.loc['k'] = ['cow', 5, 4,'no']
print(df)
#Then delete that row to return the original DataFrame
df = df.drop('k')
print(df)

#17.Count the number of each type of animal in df
print(df['animal'].value_counts())

#18.Sort df first by the values in the 'age' in decending order, then by the value in the 'visit' column in ascending orde
print(df.sort_values(by=['age'],ascending=[False]))
print(df.sort_values(by=['visits'],ascending=[True]))

#19.The 'priority' column contains the values 'yes' and 'no'.Replace this column with a column of boolean values:'yes' should be True and 'no' should be False.
print(df['priority'].map({'yes': True, 'no': False}))

#20.In the 'animal' column, change the 'snake' entries to 'python'.
print(df['animal'].replace('snake','python'))

#21.For each animal type and each number of visits, find the mean age. In other words, each row is an animal, each column is a number of visits and the values are the mean ages (hint: use a pivot table).
print(df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean'))

#22.You have a DataFrame df with a column 'A' of integers. For example:
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
#How do you filter out rows which contain the same integer as the row immediately above?
print(df.loc[df['A'].shift()!=df['A']])

#23.Given a DataFrame of numeric values, say
df = pd.DataFrame(np.random.random(size=(5, 3))) # a 5x3 frame of float values
#how do you subtract the row mean from each element in the row?
print(df.sub(df.mean(axis=1), axis=0))

#24.Suppose you have DataFrame with 10 columns of real numbers, for example:
df = pd.DataFrame(np.random.random(size=(5,10)), columns=list('abcdefghij'))
print(df)
#Which column of numbers has the smallest sum? (Find that column's label.)
print(df.sum().idxmin())

#25.How do you count how many unique rows a DataFrame has (i.e. ignore all rows that are duplicates)?
print(len(df.drop_duplicates(keep=False)))

#26.You have a DataFrame that consists of 10 columns of floating--point numbers.
# Suppose that exactly 5 entries in each row are NaN values. For each row of the DataFrame,
#find the column which contains the third NaN value.
#(You should return a Series of column labels.)
print((df.isnull().cumsum(axis=1) == 3).idxmax(axis=1))

#27.A DataFrame has a column of groups 'grps' and and column of numbers 'vals'. For example:

df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'),
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
#For each group,find the sum of the three greatest values.
great_val = df.groupby('grps')['vals'].nlargest(3).sum(level=0)
print(great_val)

#28.A DataFrame has two integer columns 'A' and 'B'.
# The values in 'A' are between 1 and 100 (inclusive).
#For each group of 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...)
#calculate the sum of the corresponding values in column 'B'
df = pd.DataFrame(np.random.RandomState(8765).randint(1, 101, size=(100, 2)), columns = ["A", "B"])
sum_B = df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()
print(sum_B)

#29. Consider a DataFrame df where there is an integer column 'X':
df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
#For each value, count the difference back to the previous zero (or the start of the Series, whichever is closer).
#These values should therefore be [1, 2, 0, 1, 2, 3, 4, 0, 1, 2]. Make this a new column 'Y'

df['Y'] = df.groupby((df['X'] == 0).cumsum()).cumcount()
# We're off by one before we reach the first zero.
first_zero_idx = (df['X'] == 0).idxmax()
df['Y'].iloc[0:first_zero_idx] += 1
print(df['Y'])

#Here's an alternative approach based on a cookbook recipe:
x = (df['X'] != 0).cumsum()
y = x != x.shift()
df['Y'] = y.groupby((y != y.shift()).cumsum()).cumsum()
print(df[['X','Y']])

#30. Consider a DataFrame containing rows and columns of purely numerical data.
#Create a list of the row-column index locations of the 3 largest values.
lar_3_list = df.unstack().sort_values()[-3:].index.tolist()
print(lar_3_list)

#31.Given a DataFrame with a column of group IDs, 'grps', and a column of corresponding integer values
#'vals', replace any negative values in 'vals' with the group mean.
df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'),
                   'vals': [12,345,-3,1,45,-14,4,-52,54,23,235,21,57,3,87]})
def replace(group):
    mask = group<0
    group[mask] = group[~mask].mean()
    return group

print(df.groupby(['grps'])['vals'].transform(replace))

#32.Implement a rolling mean over groups with window size 3,
# which ignores NaN value. For example consider the following DataFrame:
df = pd.DataFrame({'group': list('aabbabbbabab'),
                       'value': [1, 2, 3, np.nan, 2, 3,
                                 np.nan, 1, 7, 3, np.nan, 8]})
print(df)

g1 = df.groupby(['group'])['value']
g2 = df.fillna(0).groupby(['group'])['value']

s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count() # compute means

print(s.reset_index(level=0, drop=True).sort_index())

#Series and DateTimeIndex
#excercuse for creating and manipulating Series with datatime and time

#33. Create a DatetimeIndex that contains each business day of 2015
# and use it to index a Series of random numbers. Let's call this Series s.
date_timeindx = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B')
s = pd.Series(np.random.rand(len(date_timeindx)), index=date_timeindx)
print(s)

#34. Find the sum of the values in s for every Wednesday.
print(s[s.index.weekday == 2].sum())

#35.For each calendar month in s, find the mean of values
print(s.resample('M').mean())

#36.For each group of four consecutive calendar months in s, find the date on which the highest value occurred
print(s.groupby(pd.Grouper(freq = '4M')).idxmax())

#37.Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016
pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')

#Cleaning Data
#Making a DataFrame easier to work with
#Difficulty: easy/medium
"""It happens all the time: someone gives you data containing malformed strings, Python, lists and missing data. How do you tidy it up so you can get on with the analysis?
Take this monstrosity as the DataFrame to use in the following puzzles:
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
(It's some flight data I made up; it's not meant to be accurate in any way.)"""
#38. Some values in the the FlightNumber column are missing.
#These numbers are meant to increase by 10 with each row so 10055 and 10075 need to be put in place.
# Fill in these missing numbers and make the column an integer column (instead of a float column).

df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
                               '12. Air France', '"Swiss Air"']})
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)
print(df)
#39.The From_To column would be better as two separate columns! Split each string on the underscore delimiter _
# to give a new temporary DataFrame with the correct values.
# Assign the correct column names to this temporary DataFrame.
temp_df = df.From_To.str.split('_', expand=True)
temp_df.columns = ['From', 'To']
print(temp_df)

#40. Notice how the capitalisation of the city names is all mixed up in this temporary DataFrame.
# Standardise the strings so that only the first letter is uppercase (e.g. "londON" should become "London".)
temp_df['From'] = temp_df['From'].str.capitalize()
temp_df['To'] = temp_df['To'].str.capitalize()
print(temp_df)

#41. Delete the From_To column from df and attach the temporary DataFrame from the previous questions.
df = df.drop('From_To', axis=1)
df = df.join(temp_df)
print(df)

#42. In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names.
#Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'
print(df['Airline'])#before removing the extra punctuation
"""
                 KLM(!)
1      <Air France> (12)
2    (British Airways. )
3         12. Air France
4            "Swiss Air" """
df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()
print(df['Airline'])
"""After removing the punctuations
0                KLM
1         Air France
2    British Airways
3         Air France
4          Swiss Air"""

#43 In the RecentDelays column, the values have been entered into the DataFrame as a list.
# We would like each first value in its own column, each second value in its own column, and so on.
# If there isn't an Nth value, the value should be NaN.
# Expand the Series of lists into a DataFrame named delays, rename the columns delay_1, delay_2, etc.
# and replace the unwanted RecentDelays column in df with delays.

delays = df['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]
df = df.drop('RecentDelays', axis=1).join(delays)
print(df)

"""Using MultiIndexes
Go beyond flat DataFrames with additional index levels"""

#44.Given the lists letters = ['A', 'B', 'C'] and numbers = list(range(10)),
# construct a MultiIndex object from the product of the two lists.
# Use it to index a Series of random numbers. Call this Series s.
letters = ['A', 'B', 'C']
numbers = list(range(10))

mul_idx = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=mul_idx)
print(s)

#45.Check the index of s is lexicographically sorted
#(this is a necessary proprty for indexing to work correctly with a MultiIndex).
print(s.index.is_lexsorted())

#46.Select the labels 1, 3 and 6 from the second level of the MultiIndexed Series.
print(s.loc[:, [1, 3, 6]])

#47.Slice the Series s; slice up to label 'B' for the first level and from label 5 onwards for the second level
s.loc[pd.IndexSlice[:'B', 5:]]
print(s)

#48.Sum the values in s for each label in the first level (you should have Series giving you a total for labels A, B and C).
s.sum(level=0)

#49.Suppose that sum() (and other methods) did not accept a level keyword argument.
# How else could you perform the equivalent of s.sum(level=1)?
s.unstack().sum(axis=0)

#50.Exchange the levels of the MultiIndex so we have an index of the form (letters, numbers).
# Is this new Series properly lexsorted? If not, sort it.
new_s = s.swaplevel(0, 1)
# check
new_s.index.is_lexsorted()
# sort
new_s = new_s.sort_index()
print(new_s)

#Minesweeper
#Generate the numbers for safe squares in a Minesweeper grid
""" If you've ever used an older version of Windows, there's a good chance you've played with Minesweeper. 
If you're not familiar with the game, imagine a grid of squares: some of these squares conceal a mine. 
If you click on a mine, you lose instantly. If you click on a safe square, 
you reveal a number telling you how many mines are found in the squares that are immediately adjacent. 
The aim of the game is to uncover all squares in the grid that do not contain a mine.

In this section, we'll make a DataFrame that contains the necessary data for a game of Minesweeper: 
coordinates of the squares, whether the square contains a mine and the number of mines found on adjacent squares."""
"""
51. Let's suppose we're playing Minesweeper on a 5 by 4 grid, i.e.
X = 5
Y = 4
To begin, generate a DataFrame df with two columns, 'x' and 'y' containing every coordinate for this grid. 
That is,the DataFrame should start:
   x  y
0  0  0
1  0  1
2  0  2 """
X = 5
Y = 4
p = pd.core.reshape.util.cartesian_product([np.arange(X), np.arange(Y)])
df = pd.DataFrame(np.asarray(p).T, columns=['x', 'y'])
print(df)

#52.For this DataFrame df, create a new column of zeros (safe) and ones (mine).
#The probability of a mine occuring at each location should be 0.4.
df['mine'] = np.random.binomial(1, 0.4, X*Y)
print(df)

"""
#53. Now create a new column for this DataFrame called 'adjacent'. 
This column should contain the number of mines found on adjacent squares in the grid.
(E.g. for the first row, which is the entry for the coordinate (0, 0), 
count how many mines are found on the coordinates (0, 1), (1, 0) and (1, 1).)"""
df['adjacent'] = \
    df.merge(df + [ 1,  1, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 1, -1, 0], on=['x', 'y'], how='left')\
      .merge(df + [-1,  1, 0], on=['x', 'y'], how='left')\
      .merge(df + [-1, -1, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 1,  0, 0], on=['x', 'y'], how='left')\
      .merge(df + [-1,  0, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 0,  1, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 0, -1, 0], on=['x', 'y'], how='left')\
       .iloc[:, 3:]\
        .sum(axis=1)


#54.For rows of the DataFrame that contain a mine, set the value in the 'adjacent' column to NaN.
df.loc[df['mine'] == 1, 'adjacent'] = np.nan

#55.Finally, convert the DataFrame to grid of the adjacent mine counts: columns are the x coordinate, rows are the y coordinate.
df.drop('mine', axis=1).set_index(['y', 'x']).unstack()

"""56. Pandas is highly integrated with the plotting library matplotlib, and makes plotting DataFrames
 very user-friendly! Plotting in a notebook environment usually makes use of the following boilerplate:

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
matplotlib is the plotting library which pandas' plotting functionality is built upon, 
and it is usually aliased to plt.

%matplotlib inline tells the notebook to show plots inline, instead of creating them in a separate window.

plt.style.use('ggplot') is a style theme that most people find agreeable, based upon the styling of R's ggplot package.
For starters, make a scatter plot of this random data, but use black X's instead of the default markers.
df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})

Consult the documentation if you get stuck!
"""
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})
df.plot.scatter("xs", "ys", color = "black", marker = "x")

#57. Columns in your DataFrame can also be used to modify colors and sizes.
# Bill has been keeping track of his performance at work over time, as well as how good he was feeling that day,
# and whether he had a cup of coffee in the morning. Make a plot which incorporates all four features of this DataFrame.
#(Hint: If you're having trouble seeing the plot, try multiplying the Series which you choose to represent size by 10 or more)
#The chart doesn't have to be pretty: this isn't a course in data viz!

df = pd.DataFrame({"productivity":[5,2,3,1,4,5,6,7,8,3,4,8,9],
                   "hours_in"    :[1,9,6,5,3,9,2,9,1,7,4,2,2],
                   "happiness"   :[2,1,3,2,3,1,2,3,1,2,2,1,3],
                   "caffienated" :[0,0,1,1,0,0,0,0,1,1,0,1,0]})
df.plot.scatter("hours_in", "productivity", s = df.happiness * 30, c = df.caffienated)
print(df)

#58. What if we want to plot multiple things? Pandas allows you to pass in a matplotlib Axis object for plots, and plots will also return an Axis object.
#Make a bar plot of monthly revenue with a line plot of monthly advertising spending (numbers in millions)

df = pd.DataFrame({"revenue":[57,68,63,71,72,90,80,62,59,51,47,52],
                   "advertising":[2.1,1.9,2.7,3.0,3.6,3.2,2.7,2.4,1.8,1.6,1.3,1.9],
                   "month":range(12)
                  })
ax = df.plot.bar("month", "revenue", color = "green")
df.plot.line("month", "advertising", secondary_y = True, ax = ax)
print(ax.set_xlim((-1,12)))

"""
Now we're finally ready to create a candlestick chart, which is a very common tool used to analyze stock price data. A candlestick chart shows the opening, closing, highest, and lowest price for a stock during a time window. The color of the "candle" (the thick part of the bar) is green if the stock closed above its opening price, or red if below.

Candlestick Example

This was initially designed to be a pandas plotting challenge, but it just so happens that this type of plot is just not feasible using pandas' methods. If you are unfamiliar with matplotlib, we have provided a function that will plot the chart for you so long as you can use pandas to get the data into the correct format.

Your first step should be to get the data in the correct format using pandas' time-series grouping function. We would like each candle to represent an hour's worth of data. You can write your own aggregation function which returns the open/high/low/close, but pandas has a built-in which also does this.

The below cell contains helper functions. Call day_stock_data() to generate a DataFrame containing the prices a hypothetical stock sold for, and the time the sale occurred. Call plot_candlestick(df) on your properly aggregated and formatted stock data to print the candlestick chart.
"""
import numpy as np
def float_to_time(x):
    return str(int(x)) + ":" + str(int(x%1 * 60)).zfill(2) + ":" + str(int(x*60 % 1 * 60)).zfill(2)

def day_stock_data():
    #NYSE is open from 9:30 to 4:00
    time = 9.5
    price = 100
    results = [(float_to_time(time), price)]
    while time < 16:
        elapsed = np.random.exponential(.001)
        time += elapsed
        if time > 16:
            break
        price_diff = np.random.uniform(.999, 1.001)
        price *= price_diff
        results.append((float_to_time(time), price))


    df = pd.DataFrame(results, columns = ['time','price'])
    df.time = pd.to_datetime(df.time)
    return df

#Don't read me unless you get stuck!
def plot_candlestick(agg):
    """
    agg is a DataFrame which has a DatetimeIndex and five columns: ["open","high","low","close","color"]
    """
    fig, ax = plt.subplots()
    for time in agg.index:
        ax.plot([time.hour] * 2, agg.loc[time, ["high","low"]].values, color = "black")
        ax.plot([time.hour] * 2, agg.loc[time, ["open","close"]].values, color = agg.loc[time, "color"], linewidth = 10)

    ax.set_xlim((8,16))
    ax.set_ylabel("Price")
    ax.set_xlabel("Hour")
    ax.set_title("OHLC of Stock Value During Trading Day")
    plt.show()
#59. Generate a day's worth of random stock data, and aggregate / reformat it so that it has hourly summaries of the opening, highest, lowest, and closing prices
df = day_stock_data()
df.head()

df.set_index("time", inplace = True)
agg = df.resample("H").ohlc()
agg.columns = agg.columns.droplevel()
agg["color"] = (agg.close > agg.open).map({True:"green",False:"red"})
print(agg.head())

#60. Now that you have your properly-formatted data, try to plot it yourself as a candlestick chart. Use the plot_candlestick(df) function above, or
#matplotlib's plot documentation if you get stuck.
plot1 = plot_candlestick(agg)
print(plot1)