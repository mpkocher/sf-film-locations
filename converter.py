#!/usr/bin/env python3
"""
Convert the SF Film's data to GeoJson format.
"""
import argparse
import datetime
import functools
import json
import logging
import operator
import os
import sys
import time

import geojson
from geojson import Feature, FeatureCollection, Point
from googlemaps import Client

__version__ = "0.1.0"

log = logging.getLogger("sf_movies.converter")


class Constants:
    DEFAULT_FILE = "Film_Locations_in_San_Francisco.json"
    GCP_KEY_FILE = "gcp-key"
    # UUID -> georesult
    GEO_CACHE_FILE = "geolocation-cache.json"


# This is very commonly misunderstood by google
LOCATION_EMBARCO = 'The Embarcadero & Ferry Building'
LOCATION_EMBARCO_MARKET = '399 Embarcadero'
LOCATION_CANDLESTICK = 'Candlestick Park'

# Manual Fixes/Improvements for specific locations
# that Google's geolocation service won't be able to
# resolve. E.g., "Leavenworth from Filbert & Francisco St"
# Filbert and Fransisco are parallel and are 4 blocks away
LOCATION_OVERRIDES = {
    "E52DC90F-9FEA-416C-85FC-D8F059A93FB6": "Leavenworth & Filbert",
    "E4005BA5-6E32-4BD8-AE7E-496EB6B9E5BC": "California at Mason Street",
    "DE489975-BC16-4518-A013-4E08251A19DD": "Illinois St & 20th St",
    "F8473537-8CBE-46CA-AA5E-55033E948FC6": "Lombard St & Columbus Ave",
    "2556C889-30DA-40BC-B6C2-93346CDBE1E": "Jack Kerouac Alley & Grant Ave",
    "39C0FD32-0A65-410F-985B-3F5119CE198A": "Spofford St & Washington St",
    "CE5A834B-B8AC-41C0-8002-93897E50499C": "Sansome St & Pine St",
    # Not sure if this is correct
    "16F9105D-0658-4A4B-B972-20F8E34656E3": "23rd St & Illinois St",
    "42C4A4CD-3E6B-424F-BBD0-7ECB435AD9DF": "2nd & Mariposa Street",
    "2556C889-30DA-40BC-B6C2-93346CDBE1EE": "Jack Kerouac Alley & Grant Ave",
    "9C5B6F02-9DD6-49D0-B7EF-9F5D74F540EA": "847 Montgomery Street",
    # the Embarcadero freeway in this film doesn't exist
    "B28D2C31-7463-43E4-8922-9AEC3C0BCBA6": LOCATION_EMBARCO_MARKET,
    "6E393A2-665C-40B1-B357-62D86FC0D57F": LOCATION_EMBARCO,
    "46E393A2-665C-40B1-B357-62D86FC0D57F": LOCATION_EMBARCO,
    "3B4AD4CC-29D8-4BAF-8800-3F81D2E3DBB5": LOCATION_EMBARCO,
    "36F5D3B6-1D7D-4F28-A711-478A53182E8E": LOCATION_EMBARCO,
    "340BFF5F-7D3F-4FA2-9CA3-CF96DAAF2914": LOCATION_EMBARCO,
    "7FC183C0-A150-48A9-B113-380403981800": LOCATION_EMBARCO,
    "EB708888-D5B3-4594-BDA1-68C2AD63EA71": LOCATION_EMBARCO,

    "736BEA89-092D-40A3-8CCB-67996C2FDD58": "1332 Grant Ave",

    # It's odd that Google can't figure this out
    # 'Candlestick Park Exit, Highway 101'
    "2A13BB99-7F65-4F3D-83FD-5E39D2EF48D2": LOCATION_CANDLESTICK,
    "44722700-FB67-4E7F-A1F6-4C6A5FE9ACD5": LOCATION_CANDLESTICK,
    '18F0ED6C-3561-4ABB-ABCA-7F2299D42BD7': LOCATION_CANDLESTICK,
    '5C8E4FDC-4994-4D2B-91E7-50BBBA277FFA': LOCATION_CANDLESTICK,
    '928394DF-EC7B-496C-8DC6-2A59243E5874': LOCATION_CANDLESTICK,
    '36AC33AC-6455-4797-9849-6579DC9F051B': LOCATION_CANDLESTICK,
    'B212265D-08CD-44F4-B156-318699A3D142': LOCATION_CANDLESTICK,
    '0D8F9C14-80D0-43E3-A61B-4E9C3D312D3B': LOCATION_CANDLESTICK,
    '9F88777C-1968-49B4-99A9-6EC4B724BF2A': LOCATION_CANDLESTICK,
    'DE2158E9-7DA8-4310-A306-C65C4F5EE89F': LOCATION_CANDLESTICK,
    'A4F445AB-A0BF-4466-A9AB-A1B2C3FFD1E6': LOCATION_CANDLESTICK,

    '089F05BF-FDDA-473E-87CD-CB7FB97C47A0': 'spofford and washington',
    '7B5DB347-473B-4360-85F8-6CBEFCD42A9A': 'american can company',
    '6C61AA14-3213-478E-A0E8-D5CA503F6696': 'Columbus and Filbert St',
    '8D726183-F542-4B8E-9B13-EA2BDCD5298C': 'Bush and Kerney St',
    '274355E9-42CA-4FCB-945A-8F466D5EFE2F': 'grant and sacramento street',

    '4E745A1A-6FAE-4D93-B5CE-2046FD28F973': LOCATION_EMBARCO,
    '1E1A2DF9-1E46-427E-84DF-E09DB68F7E03': LOCATION_EMBARCO,
    '39A950CF-0F78-4C3A-90CD-8852F368C886': LOCATION_EMBARCO,
    '07D46736-F34F-470A-B6C2-248289FD580C': LOCATION_EMBARCO,
    '23114F2F-DE88-402E-995F-B5B367AF80FA': LOCATION_EMBARCO,
    '98D96283-0764-4838-BC97-C822F97F691B': LOCATION_EMBARCO,
    '07F59C2E-8B78-4A37-8E30-8A8CDB0ABEF3': LOCATION_EMBARCO

}


def load_gcp_key(path):
    with open(path, "r") as f:
        s = f.read().strip()
    return s


def _is_not_null(x):
    return x is not None


def _custom_converter(d):
    # Minor tweaks to raw d from data from sfdata

    dx = d.copy()

    ix = dx["id"]

    if ix in LOCATION_OVERRIDES:
        dx["Locations"] = LOCATION_OVERRIDES[ix]

    def f(i):
        return f"Actor {i}"

    # this should be done with operator
    def fx(k):
        return dx[k]

    author_keys = list(map(f, range(1, 4)))

    p0 = map(fx, author_keys)
    p1 = filter(_is_not_null, p0)

    dx["actors"] = list(p1)
    dx["Release Year"] = int(dx["Release Year"])
    for key in author_keys:
        del dx[key]

    return dx


def _to_fields(d):
    """Extract fields from RDF-ish file"""
    cs = d["meta"]["view"]["columns"]
    return list(map(lambda x: x["name"], cs))


def to_simple_d(d):
    """collapse the raw structured RDF-ish metadata structure to simple dict"""

    fields = _to_fields(d)

    def to_d(jx):
        return dict(zip(fields, jx))

    items = d["data"]
    return map(to_d, items)


def to_geojson_feature(dx, ix=None, properties=None):
    # More of this raw data from google should be pushed down
    g = dx["geometry"]["location"]
    coords = g["lng"], g["lat"]
    pt = Point(coords)

    keys = ("formatted_address", "place_id", "plus_code")
    if properties is not None:
        for key in keys:
            properties[key] = dx.get(key)
    return Feature(id=ix, geometry=pt, properties=properties)


def feature_to_simple_d(f0):
    """Create a simple/terse dict from a Feature"""
    d = {}
    p = f0["properties"]
    # d["coordinates"] = f0["geometry"]["coordinates"]
    d["geo_lat"] = f0["geometry"]["coordinates"][1]
    d["geo_lng"] = f0["geometry"]["coordinates"][0]
    d["id"] = p["id"]
    d["title"] = p["Title"]
    d["director"] = p["Director"]
    d["release_year"] = p["Release Year"]
    #d["raw_location"] = p["Locations"]
    d["location"] = p["formatted_address"]
    # d['global_code'] = p.get('global_code')
    return d


def load_raw_data(f):
    """Load raw SF data and convert to 'simple' dict form"""
    with open(f, "r") as reader:
        raw_d = json.load(reader)
    return to_simple_d(raw_d)


def _to_sf_location(lx):
    """Append SF specific info to location string to
    improve GeoLocation lookup"""
    return lx + ", San Francisco, CA"


def lookup_location(client, location, throttle_sec=None):
    # Not clear what errors can occur here at the GCP level
    log.info(f"Looking up Location {location}")
    result = client.geocode(location)
    if throttle_sec is not None:
        time.sleep(throttle_sec)
    return result


def write_features_to_geojson(features, output_geojson):

    feature_collection = FeatureCollection(features)

    with open(output_geojson, "w+") as f:
        geojson.dump(feature_collection, f, indent=True)
    log.info("Wrote {} features to {}".format(len(features), output_geojson))


def write_features_to_csv(features, output_csv):

    import pandas as pd

    dx = list(map(feature_to_simple_d, features))
    df = pd.DataFrame(dx)

    f0 = df.set_index('id')
    f1 = df.sort_values(['release_year', 'title'])
    f1.to_csv(output_csv, index=False)


class GeoLocationCacheIO:
    def __init__(self, file_name, records=None):
        self.records = {} if records is None else records
        self.file_name = file_name

    def __repr__(self):
        _d = dict(k=self.__class__.__name__,
                  n=len(self.records),
                  f=self.file_name)
        return "<{k} num-records:{n} file:{f} >".format(**_d)

    @staticmethod
    def load_from(file_name):
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                records = json.load(f)
        else:
            records = {}

        return GeoLocationCacheIO(file_name, records=records)

    def write(self):
        with open(self.file_name, "w+") as f:
            json.dump(self.records, f, indent=2)


class GeoLocationCacheNullIO(GeoLocationCacheIO):
    def __init__(self, records=None):
        super(GeoLocationCacheNullIO, self).__init__(os.devnull, records=records)

    def write(self):
        pass


GEO_CACHE_NULL = GeoLocationCacheNullIO()


def converter(client, raw_records, geo_cache=GEO_CACHE_NULL, throttle_sec=None):
    def fx(dx):
        loc = dx["Locations"]
        if loc is None:
            msg = "Location is not defined. Skipping {} for Title `{}`".format(
                dx["id"], dx["Title"]
            )
            log.warning(msg)
        return loc

    log.info(
        "Starting converting with {} cached locations".format(len(geo_cache.records))
    )
    features = []
    for record in filter(fx, raw_records):
        r = _custom_converter(record)
        ix = r["id"]
        title = r["Title"]
        location = r["Locations"]

        lx = geo_cache.records.get(ix)

        # dirty hack to force the cache to get updated
        # when manually changing labels
        # if ix in LOCATION_OVERRIDES:
        #    lx = None

        if lx is None:
            results = lookup_location(
                client, _to_sf_location(location), throttle_sec=throttle_sec
            )

            # why is this a list? If it can't resolve the address it just returns an empty list?
            if results:
                result = results[0]
                geo_cache.records[ix] = result
                log.info("Resolved raw location `{}` to `{}`".format(location, result['formatted_address']))

                feature = to_geojson_feature(result, ix=ix, properties=r)
                features.append(feature)
            else:
                log.error(
                    f"UNABLE TO RESOLVE LOCATION `{location}` for Title {title} for {ix}"
                )
        else:
            log.debug(f"Loading GEO Location from cache. id:{ix}")
            result = lx
            feature = to_geojson_feature(result, ix=ix, properties=r)
            features.append(feature)

    log.debug("Converted {} GeoJson features".format(len(features)))
    geo_cache.write()
    log.debug("Wrote cache to {}".format(geo_cache))
    return features, geo_cache


def setup_logger(level=logging.INFO, file_name=None, stream=sys.stdout):
    # this is an odd interface. stream and filename are mutually exclusive
    formatter = "[%(levelname)s] %(asctime)s [%(pathname)s:%(lineno)d] - %(message)s"
    if file_name is None:
        logging.basicConfig(level=level, stream=stream, format=formatter)
    else:
        logging.basicConfig(level=level, filename=file_name, format=formatter)


def converter_io(
    client_key,
    raw_record_json,
    geo_cache_json=None,
    output_geojson=None,
    output_csv=None,
    max_records=None,
    throttle_sec=None
):

    client = Client(key=client_key)
    raw_records = list(load_raw_data(raw_record_json))
    log.info("Loaded {} raw records".format(len(raw_records)))

    if max_records is not None:
        records = raw_records[0: min(max_records, len(raw_records) + 1)]
    else:
        records = raw_records

    gcache = (
        GEO_CACHE_NULL
        if geo_cache_json is None
        else GeoLocationCacheIO.load_from(geo_cache_json)
    )

    features, cache = converter(client, records, gcache, throttle_sec=throttle_sec)

    if output_geojson is not None:
        write_features_to_geojson(features, output_geojson)

    if output_csv is not None:
        write_features_to_csv(features, output_csv)

    return features, cache


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)

    f = p.add_argument

    f("-k", "--api-key", help="Google API Key", required=True)
    f("-f", "--sf-json", help="Path to SF Org JSON file", required=True)
    f("-o", "--output-geojson", help="Output GeoJSON file", default="output.geojson")
    f("-c", "--geolocation-cache", help="Cached GeoLocation", default=None)
    f("--output-csv", help="Output 'Slim' CSV with minimal metadata", default=None)
    f("--max-records", help="Max Number of records to process", type=int)
    f("--throttle-sec", help="Throttling time between requests", type=int, default=1)

    f("--log-level", help="Logging Level", default=logging.INFO)
    f("--log-file", help="Output Logging file", default=None)
    f("--version", action="version", version=__version__)

    return p


def run_main(argv):
    p = get_parser()
    pargs = p.parse_args(argv)

    now = datetime.datetime.now
    exit_code = 1

    started_at = now()

    try:
        setup_logger(level=pargs.log_level, file_name=pargs.log_file)
        log.debug("Parsed args {}".format(pargs))

        _ = converter_io(
            pargs.api_key,
            pargs.sf_json,
            pargs.geolocation_cache,
            output_geojson=pargs.output_geojson,
            output_csv=pargs.output_csv,
            max_records=pargs.max_records,
            throttle_sec=pargs.throttle_sec,
        )
        exit_code = 0
    except Exception as e:
        log.exception(e, exc_info=True)

    dt = now() - started_at
    msg = "Completed running in {:.2f} sec with exit code {}".format(
        dt.total_seconds(), exit_code
    )
    log.info(msg)
    return exit_code


if __name__ == "__main__":
    sys.exit(run_main(sys.argv[1:]))
