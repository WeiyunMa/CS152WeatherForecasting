import pandas as pd

weather_variables = {"Max Temp": 5, 
                     "Min Temp": 7,
                     "Mean Temp": 9,
                     "Heat Deg Days": 11,
                     "Cool Deg Days": 13,
                     "Total Rain": 15,
                     "Total Snow": 17,
                     "Total Precip": 19,
                     "Snow on Grnd": 21,
                     "Dir of Max Gust": 23,
                     "Spd of Max Gust": 25}
weather_list = []
for year in range(1999,2010):
    weather_data = pd.read_csv("data/w" + str(year)+ ".csv", sep='\t', encoding='iso-8859-1')
    weather_list.append(weather_data.iloc[:,weather_variables["Max Temp"]])

date_data = pd.read_csv("data/w1999.csv", sep='\t', encoding='iso-8859-1')
month = date_data.loc[:,"Month"]
day = date_data.loc[:,"Day"]

weather_frame = pd.DataFrame(
    {
        "month": month,
        "day": day,
        "1999": weather_list[0],
        "2000": weather_list[1],
        "2001": weather_list[2],
        "2002": weather_list[3],
        "2003": weather_list[4],
        "2004": weather_list[5],
        "2005": weather_list[6],
        "2006": weather_list[7],
        "2007": weather_list[8],
        "2008": weather_list[9],
        "2009": weather_list[10],
    }
)

weather_frame.to_csv("data/mean_temp.csv", sep='\t')
