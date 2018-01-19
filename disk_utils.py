import pickle
import sys
import os
# from functools import lru_cache
# @lru_cache(maxsize=1024)
import hashlib
import gzip


def disk_cache(function):
    # @lru_cache(maxsize=1024)
    def wrapper(*args, **kwargs):
        if not os.path.exists("cache/"):
            print("[DISK_CACHE] creating cache dir")
            os.makedirs("cache/")

        fid = function.__name__
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
            try:
                # OLD STYLE
                with open(cache_file[:-3], "rb") as fin:
                    retr = pickle.load(fin)
            except (FileNotFoundError, OSError):
                retr = function(*args, **kwargs)

            with gzip.open(cache_file, "wb") as fout:
                pickle.dump(retr, fout)

            try:
                os.remove(cache_file[:-3])
                print("removed old file", cache_file[:-3])
            except Exception:
                pass
        return retr
    return wrapper
