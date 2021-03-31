#!/usr/bin/env python3
"""
Convert the SF Film's data to GeoJson format.
"""
import argparse
import datetime
import json
import logging
import os
import sys
import time

from typing import Optional, Dict, List, Any, Iterator, Tuple, TextIO

import geojson
from geojson import Feature, FeatureCollection, Point
from googlemaps import Client

__version__ = "0.1.0"

log = logging.getLogger("sf_movies.converter")

# This program is essentially a big json munging tool.
# Adding this type is motivated by pragmatism
DictAny = Dict[str, Any]


class Constants:
    DEFAULT_FILE = "Film_Locations_in_San_Francisco.json"
    GCP_KEY_FILE = "gcp-key"
    # UUID -> georesult
    GEO_CACHE_FILE = "geolocation-cache.json"
    # Manual overrides of locations that google has trouble resolving correctly
    LOCATION_OVERRIDES = 'location-overrides.json'


def load_gcp_key(path: str) -> str:
    with open(path, "r") as f:
        s = f.read().strip()
    return s


def _is_not_null(x: str) -> bool:
    return x is not None


def _custom_converter(d: DictAny, override_locations:DictAny) -> DictAny:
    # Minor tweaks to raw d from data from sfdata

    dx = d.copy()

    raw_location = dx["Locations"]

    if raw_location in override_locations:
        dx["Locations"] = override_locations[raw_location]

    def f(i: int) -> str:
        return f"Actor {i}"

    # this should be done with operator
    def fx(k: str) -> Any:
        return dx[k]

    author_keys = list(map(f, range(1, 4)))

    p0 = map(fx, author_keys)
    p1 = filter(_is_not_null, p0)

    dx["actors"] = list(p1)
    dx["Release Year"] = int(dx["Release Year"])
    for key in author_keys:
        del dx[key]

    return dx


def _to_fields(d: DictAny) -> List[str]:
    """Extract fields from RDF-ish file"""
    cs: List[DictAny] = d["meta"]["view"]["columns"]

    # written this way to get type annotations to work
    def f(x: DictAny) -> str:
        sx: str = x['name']
        return sx

    return list(map(f, cs))


def to_simple_d(d: DictAny) -> Iterator[DictAny]:
    """collapse the raw structured RDF-ish metadata structure to simple dict"""

    fields = _to_fields(d)

    def to_d(jx: List[DictAny]) -> DictAny:
        dx: DictAny = dict(zip(fields, jx))
        return dx

    items = d["data"]
    return map(to_d, items)


def to_geojson_feature(dx: DictAny, properties: Optional[DictAny]=None) -> Feature:
    # More of this raw data from google should be pushed down
    # this ix is perhaps an issue given that the data source isn't
    # persisting UUIDs across updates.
    g = dx["geometry"]["location"]
    coords = g["lng"], g["lat"]
    pt = Point(coords)

    keys = ("formatted_address", "place_id", "plus_code")
    if properties is not None:
        for key in keys:
            properties[key] = dx.get(key)
    return Feature(geometry=pt, properties=properties)


def feature_to_simple_d(f0: DictAny) -> DictAny:
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


def load_json(f: str) -> DictAny:
    with open(f, "r") as reader:
        raw_d: DictAny = json.load(reader)
    return raw_d


def load_raw_data(f: str) -> Iterator[DictAny]:
    """Load raw SF data and convert to 'simple' dict form"""
    return to_simple_d(load_json(f))


def _to_sf_location(lx: str) -> str:
    """Append SF specific info to location string to
    improve GeoLocation lookup"""
    return lx + ", San Francisco, CA"


def lookup_location(client: Client, location: str, throttle_sec: Optional[float] =None) -> List[DictAny]:
    # Not clear what errors can occur here at the GCP level
    log.debug(f"Looking up Location {location}")
    result: List[DictAny] = client.geocode(location)
    if throttle_sec is not None:
        time.sleep(throttle_sec)
    return result


def write_features_to_geojson(features: List[Feature], output_geojson: str) -> None:

    feature_collection = FeatureCollection(features)

    with open(output_geojson, "w+") as f:
        geojson.dump(feature_collection, f, indent=True)
    log.info("Wrote {} features to {}".format(len(features), output_geojson))


def write_features_to_csv(features: List[Feature], output_csv: str) -> None:

    import pandas as pd

    dx = list(map(feature_to_simple_d, features))
    df = pd.DataFrame(dx)

    df.set_index('id')
    f1 = df.sort_values(['release_year', 'title'])
    f1.to_csv(output_csv, index=False)


class GeoLocationCacheIO:
    def __init__(self, file_name: str, records: Optional[DictAny]=None):
        self.records = {} if records is None else records
        self.file_name = file_name

    def __repr__(self) -> str:
        _d = dict(k=self.__class__.__name__,
                  n=len(self.records),
                  f=self.file_name)
        return "<{k} num-records:{n} file:{f} >".format(**_d)

    @staticmethod
    def load_from(file_name: str) -> 'GeoLocationCacheIO':
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                records = json.load(f)
        else:
            records = {}

        return GeoLocationCacheIO(file_name, records=records)

    def write(self) -> None:
        with open(self.file_name, "w+") as f:
            json.dump(self.records, f, indent=2)


class GeoLocationCacheNullIO(GeoLocationCacheIO):
    def __init__(self, records:Optional[DictAny]=None):
        super(GeoLocationCacheNullIO, self).__init__(os.devnull, records=records)

    def write(self) -> None:
        pass


GEO_CACHE_NULL = GeoLocationCacheNullIO()


def converter(client: Client, raw_records: List[DictAny], geo_cache:GeoLocationCacheIO, location_overrides:Optional[DictAny]=None, throttle_sec:Optional[float]=None) -> Tuple[List[Feature], GeoLocationCacheIO]:

    loc_overrides: DictAny = {} if location_overrides is None else location_overrides

    def fx(dx: DictAny) -> Optional[str]:
        loc: Optional[str] = dx["Locations"]
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
        r = _custom_converter(record, loc_overrides)
        # they changed the UUID in the Sept 6, 2019 update for some reason.
        # Therefore, it's no longer safe to rely on that to persist across updates.
        # Going forward using the location as the id (this is not without it's
        # own set of issues).
        ix = r["id"]

        title = r["Title"]
        raw_location = r["Locations"]

        lx: Optional[DictAny] = geo_cache.records.get(raw_location)

        # dirty hack to force the cache to get updated
        # when manually changing labels
        #if raw_location in location_overrides:
        #    lx = None

        if lx is None:
            results = lookup_location(
                client, _to_sf_location(raw_location), throttle_sec=throttle_sec
            )

            # why is this a list? If it can't resolve the address it just returns an empty list?
            if results:
                result: DictAny = results[0]
                geo_cache.records[raw_location] = result
                log.info("Resolved record `{}` raw location `{}` to `{}`".format(ix, raw_location, result['formatted_address']))

                feature = to_geojson_feature(result, properties=r)
                features.append(feature)
            else:
                log.error(
                    f"UNABLE TO RESOLVE LOCATION `{raw_location}` for Title {title} for {ix}"
                )
        else:
            log.debug(f"Loading GEO Location from cache. Location `{raw_location}`")
            result = lx
            feature = to_geojson_feature(result, properties=r)
            features.append(feature)

    log.debug("Converted {} GeoJson features".format(len(features)))
    geo_cache.write()
    log.debug("Wrote cache to {}".format(geo_cache))
    return features, geo_cache


def setup_logger(level:str = "INFO", file_name: Optional[str]=None, stream:TextIO=sys.stdout) -> None:
    # this is an odd interface. stream and filename are mutually exclusive
    formatter = "[%(levelname)s] %(asctime)s [%(pathname)s:%(lineno)d] - %(message)s"
    if file_name is None:
        logging.basicConfig(level=level, stream=stream, format=formatter)
    else:
        logging.basicConfig(level=level, filename=file_name, format=formatter)


def converter_io(
    client_key:str,
    raw_record_json:str,
    geo_cache_json: Optional[str]=None,
    location_overrides: Optional[str]=None,
    output_geojson: Optional[str]=None,
    output_csv: Optional[str]=None,
    max_records: Optional[int]=None,
    throttle_sec: Optional[float]=None,
) -> Tuple[List[Feature], GeoLocationCacheIO]:

    client = Client(key=client_key)
    raw_records = list(load_raw_data(raw_record_json))
    log.info("Loaded {} raw records".format(len(raw_records)))

    if max_records is not None:
        records = raw_records[0: min(max_records, len(raw_records) + 1)]
    else:
        records = raw_records

    loc_overrides = {} if location_overrides is None else load_json(location_overrides)
    log.info("Loaded {} location overrides".format(len(loc_overrides)))

    gcache = (
        GEO_CACHE_NULL
        if geo_cache_json is None
        else GeoLocationCacheIO.load_from(geo_cache_json)
    )

    features, cache = converter(client, records, gcache, location_overrides=loc_overrides, throttle_sec=throttle_sec)

    if output_geojson is not None:
        write_features_to_geojson(features, output_geojson)

    if output_csv is not None:
        write_features_to_csv(features, output_csv)

    return features, cache


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    f = p.add_argument

    f("-k", "--api-key", help="Google API Key", required=True)
    f("-f", "--sf-json", help="Path to SF Org JSON file", required=True)
    f("-o", "--output-geojson", help="Output GeoJSON file", default="output.geojson")
    f("-c", "--geolocation-cache", help="Cached GeoLocation", default=None)
    f('--location-overrides', help="Location overrides JSON file for locations that google will have trouble resolving", default=None)
    f("--output-csv", help="Output 'Slim' CSV with minimal metadata", default=None)
    f("--max-records", help="Max Number of records to process", type=int)
    f("--throttle-sec", help="Throttling time between requests", type=float, default=1)

    f("--log-level", help="Logging Level", default=logging.INFO)
    f("--log-file", help="Output Logging file", default=None)
    f("--version", action="version", version=__version__)

    return p


def run_main(argv: List[str]) -> int:
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
            location_overrides=pargs.location_overrides,
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
