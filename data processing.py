import pandas as pd

weatherList = []
for year in range(1999,2010):
    weatherData = pd.read_csv(r"C:\Users\jfan\OneDrive\Documents\Harvey Mudd College\neural network\final project\data set\w" + str(year)+ ".csv", sep='\t', encoding='iso-8859-1')
    weatherList.append(weatherData.iloc[:,5])

dateData = pd.read_csv(r"C:\Users\jfan\OneDrive\Documents\Harvey Mudd College\neural network\final project\data set\w1999.csv", sep='\t', encoding='iso-8859-1')
month = dateData.loc[:,"Month"]
day = dateData.loc[:,"Day"]

weatherFrame = pd.DataFrame(
    {
        "month": month,
        "day": day,
        "1999": weatherList[0],
        "2000": weatherList[1],
        "2001": weatherList[2],
        "2002": weatherList[3],
        "2003": weatherList[4],
        "2004": weatherList[5],
        "2005": weatherList[6],
        "2006": weatherList[7],
        "2007": weatherList[8],
        "2008": weatherList[9],
        "2009": weatherList[10],
    }
)
print weatherFrame