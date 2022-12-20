import os
import argparse

from cbp_translate.pipeline import Config, run


if __name__ == "__main__":

    if os.getenv("MODAL_RUN_LOCALLY", None) != "1":
        raise RuntimeError(
            "This script can only be used locally, please set the environment variable MODAL_RUN_LOCALLY=1"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-in", type=str, required=True)
    parser.add_argument("--path-out", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--path-tmp", type=str, default="/tmp")

    args = parser.parse_args()
    config = Config(target_lang=args.language, speaker_markers=True)

    run(
        path_in=args.path_in,
        path_out=args.path_out,
        path_tmp=args.path_tmp,
        config=config,
    )
