import pickle
import sys
import os
from functools import lru_cache
import hashlib
import gzip


def disk_cache(function):
    def wrapper(*args, **kwargs):
        if not os.path.exists("cache/"):
            print("[DISK_CACHE] creating cache dir")
            os.makedirs("cache/")

        fid = function.__name__
        if fid == "eval_option_on_mdp":
            args = tuple((args[0], args[1], args[2], list(args[3]), list(args[4])))
        cache_file = "cache/{}".format(fid)
        if args:
            if not os.path.exists(cache_file):
                os.makedirs(cache_file)
            args_filtered = []
            for arg in args:
                arg = str(arg)
                if len(arg) > 100:
                    arg = hashlib.sha1(arg.encode('utf-8')).hexdigest()
                args_filtered.append(arg)

            fid = fid + "/" + "::".join(args_filtered).replace("/", "_")
            cache_file = "cache/{}".format(fid)
        cache_file += ".pkl.gz"

        try:
            with gzip.open(cache_file, "rb") as fin:
                retr = pickle.load(fin)
        except FileNotFoundError:
            retr = function(*args, **kwargs)

            with gzip.open(cache_file, "wb") as fout:
                pickle.dump(retr, fout)
        return retr
    return wrapper
