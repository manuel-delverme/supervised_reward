import pickle
import sys
import os
import hashlib
import gzip


def disk_cache(function):
    def wrapper(*args, **kwargs):
        fid = function.__name__
        if fid == "eval_option_on_mdp":
            args = tuple((args[0], args[1], args[2], list(args[3]), list(args[4])))
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
            if function.__name__ == "gather_stats":
                storage_fn = open
            else:
                storage_fn = gzip.open
            with storage_fn(cache_file, "rb") as fin:
                retr = pickle.load(fin)
        except FileNotFoundError:
            retr = function(*args, **kwargs)
            if not os.path.exists(cache_file):
                cache_folder = cache_file[:cache_file.rindex("/")]
                os.makedirs(cache_folder, exist_ok=True)
            if function.__name__ == "gather_stats":
                storage_fn = open
            else:
                storage_fn = gzip.open
            with storage_fn(cache_file, "wb") as fout:
                pickle.dump(retr, fout)
        return retr

    return wrapper
