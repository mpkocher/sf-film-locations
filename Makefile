


Film_Locations_in_San_Francisco.json:
	wget "https://data.sfgov.org/api/views/yitu-d5am/rows.json?accessType=DOWNLOAD" --output-document Film_Locations_in_San_Francisco.json

reformat:
	black converter.py

convert:
	./converter.py -k "${API_KEY}" -f Film_Locations_in_San_Francisco.json -c geolocation-cache.json -o SF-Film-Locations.geojson --log-level=DEBUG --log-file=output.log --location-overrides location-overrides.json

convert-test:
	./converter.py -k "${API_KEY}" -f Film_Locations_in_San_Francisco.json -c geolocation-cache.json -o SF-Film-Locations.geojson --max-records=5 --log-level=DEBUG --location-overrides location-overrides.json

mypy:
	mypy --strict converter.py
