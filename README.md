# SF Film Locations

SF Film locations conversion tool to GeoJson using data from sfgov.org via SF Film commission [filmsf.org](https://filmsf.org/).

Raw Data: https://data.sfgov.org/Culture-and-Recreation/Film-Locations-in-San-Francisco/yitu-d5am 

The raw locations are loose text descriptions of the film location site and is not easily or robustly consumed for plotting purposes. 

Examples of raw film site locations:

- "Bayshore Blvd near Cesar Chavez (Bayview)"
- "420 Jones St. at Ellis St."
- "Hayes Street at Laguna"
- "City Hall"
- "Leavenworth from Filbert & Francisco St"


This tool will convert the 1600+ raw locations from the RDF-ish JSON format from SFgov.org into [GeoJSON](https://geojson.org/) format. The locations are looked up using Google Cloud Platform (GCP) geolocation service. Even using the GCP gelocation server, there's still 10-15 locations that have be edited by hand to correctly resolve the location successfully. See `LOCATION_OVERRIDES` in `converter.py` for details.


## Conversion Tool

Pulling the raw data from sfgov.


```bash
wget "https://data.sfgov.org/api/views/yitu-d5am/rows.json?accessType=DOWNLOAD" --output-document Film_Locations_in_San_Francisco.json
```

### Running the Conversion Tool

Note, this requires a Google Cloud Platform (GCP) API key. The GCP service is used to take the raw location description to resolve to lat/long and well formatted address values. 

```bash
./converter.py -k "${API_KEY}" --sf-json Film_Locations_in_San_Francisco.json -c geolocation-cache.json -o SF-Film-Locations.geojson --log-level=DEBUG --log-file=output.log
```

The output (`SF-Film-Locations.geojson`) is using the open [GeoJSON](https://geojson.org/) standard as a `FeatureCollection`.

Github provides a [view of GeoJSON](https://github.com/mpkocher/sf-film-locations/blob/master/SF-Film-Locations.geojson). I've also checked in the [output CSV](https://github.com/mpkocher/sf-film-locations/blob/master/SF-Film-Locations.csv) which provides a high level overview of the data in tabular form.


## Example GeoJson Feature instances

One of the many locations from `Bullitt` (1968).


```javascript
{
  "geometry":{
    "coordinates":[
      37.7502159,
      -122.3839432
    ],
    "type":"Point"
  },
  "id":"C51901DF-25CE-4B50-A369-6BE660D38C2B",
  "properties":{
    "Director":"Peter Yates",
    "Distributor":"Warner Brothers",
    "Fun Facts":null,
    "Locations":"Bayshore Blvd near Cesar Chavez (Bayview)",
    "Production Company":"Warner Brothers / Seven Arts\nSeven Arts",
    "Release Year":1968,
    "Title":"Bullitt",
    "Writer":"Alan R. Trustman",
    "actors":[
      "Steve McQueen",
      "Jacqueline Bisset",
      "Robert Vaughn"
    ],
    "created_at":1509143469,
    "created_meta":"881420",
    "formatted_address":"601 Cesar Chavez, San Francisco, CA 94124, USA",
    "id":"C51901DF-25CE-4B50-A369-6BE660D38C2B",
    "meta":null,
    "place_id":"ChIJawtxxrp_j4AR22luIA2ccZ4",
    "plus_code":{
      "compound_code":"QJ28+3C Bayview, San Francisco, CA, United States",
      "global_code":"849VQJ28+3C"
    },
    "position":211,
    "sid":211,
    "updated_at":1509143469,
    "updated_meta":"881420"
  },
  "type":"Feature"
}
```

One of the locations from a more recent film, `Ant-Man` (2015).

```javascript
{
  "geometry":{
    "coordinates":[
      37.7852042,
      -122.412723
    ],
    "type":"Point"
  },
  "id":"7807E6F8-0428-41C4-92FA-3EA8A1782113",
  "properties":{
    "Director":"Peyton Reed",
    "Distributor":"Walt Disney Studios Motion Pictures",
    "Fun Facts":null,
    "Locations":"420 Jones St. at Ellis St.",
    "Production Company":"PYM Particles Productions, LLC",
    "Release Year":2015,
    "Title":"Ant-Man",
    "Writer":"Gabriel Ferrari ",
    "actors":[
      "Michael Douglas",
      "Paul Rudd"
    ],
    "created_at":1509143469,
    "created_meta":"881420",
    "formatted_address":"420 Jones St, San Francisco, CA 94102, USA",
    "id":"7807E6F8-0428-41C4-92FA-3EA8A1782113",
    "meta":null,
    "place_id":"ChIJmQ0U34-AhYARQODKYn8PKzU",
    "plus_code":{
      "compound_code":"QHPP+3W Tenderloin, San Francisco, CA, United States",
      "global_code":"849VQHPP+3W"
    },
    "position":78,
    "sid":78,
    "updated_at":1509143469,
    "updated_meta":"881420"
  },
  "type":"Feature"
}
```
