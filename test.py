import meteostat as ms
from meteostat import Point
from datetime import datetime, timedelta, timezone

lat, lon = 10.7769, 106.7009
end = datetime.now(timezone.utc).replace(tzinfo=None)
start = end - timedelta(hours=42)

location = Point(lat, lon)

print("start fetch")
df = ms.hourly(location, start, end).fetch()
print("done", df.shape)
print(df.head())