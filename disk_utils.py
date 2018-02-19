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
            args_filtered = []
            for arg in args:
                arg = str(arg)
                if len(arg) > 100:
                    arg = hashlib.sha1(arg.encode('utf-8')).hexdigest()
                args_filtered.append(arg.replace("/", "_"))
            if function.__name__ == "eval_option_on_mdp":
                option_hash = args_filtered[-2]
                del args_filtered[-2]
                new_hash = list(option_hash[:6]) + [option_hash[6:]]
                args_filtered = args_filtered[:-1] + new_hash + args_filtered[-1:]

            fid = fid + "/" + "/".join(args_filtered)
            cache_file = "cache/{}".format(fid)
        cache_file += ".pkl.gz"

        try:
            with gzip.open(cache_file, "rb") as fin:
                retr = pickle.load(fin)
        except FileNotFoundError:
            retr = function(*args, **kwargs)
            if not os.path.exists(cache_file):
                cache_folder = cache_file[:cache_file.rindex("/")]
                os.makedirs(cache_folder, exist_ok=True)
            with gzip.open(cache_file, "wb") as fout:
                pickle.dump(retr, fout)
        return retr

    return wrapper
