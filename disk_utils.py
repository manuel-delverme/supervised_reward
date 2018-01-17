import pickle
import sys
import os
# from functools import lru_cache
# @lru_cache(maxsize=1024)
import hashlib


def disk_cache(f):
    # @lru_cache(maxsize=1024)
    def wrapper(*args, **kwargs):
        if not os.path.exists("cache/"):
            print("[DISK_CACHE] creating cache dir")
            os.makedirs("cache/")

        fid = f.__name__
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
        cache_file += ".pkl"
        try:
            with open(cache_file, "rb") as fin:
                retr = pickle.load(fin)
        except FileNotFoundError:
            # OLD STYLE
            old_fid = f.__name__
            if args:
                old_fid += "::".join(str(arg) for arg in args)
            old_cache_file = "cache/{}.pkl".format(old_fid)

            try:
                with open(old_cache_file, "rb") as fin:
                    retr = pickle.load(fin)
            except (FileNotFoundError, OSError):
                retr = f(*args, **kwargs)

            with open(cache_file, "wb") as fout:
                pickle.dump(retr, fout)
        return retr

    return wrapper
