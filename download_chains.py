#Download the large MCMC chain files hosted on Zenodo.
import argparse
import sys
import urllib.request
from pathlib import Path

ZENODO_RECORD_ID = "24322863"


FILES = [
    ("d18O_top_chain.h5",  "d18O_top/chain.h5",  682),
    ("d18O_full_chain.h5", "d18O_full/chain.h5", 356),
    ("D47_top_chain.h5",   "D47_top/chain.h5",   448),
    ("D47_full_chain.h5",  "D47_full/chain.h5",  423),
]

BASE_URL = "https://zenodo.org/records/{record}/files/{name}?download=1"


def progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100 * done / total_size
    sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:.0f} / {total_size / 1e6:.0f} MB)")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Download Zenodo-hosted data files.")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that already exist")
    args = parser.parse_args()

    if ZENODO_RECORD_ID == "XXXXXXX":
        sys.exit("Set ZENODO_RECORD_ID at the top of this script first.")

    root = Path(__file__).resolve().parent
    total_mb = sum(size for _, _, size in FILES)
    print(f"Downloading up to {len(FILES)} files (~{total_mb} MB total)\n")

    for remote_name, local_path, size_mb in FILES:
        dest = root / local_path
        if dest.exists() and not args.force:
            print(f"Skipping {local_path} (already exists)")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        url = BASE_URL.format(record=ZENODO_RECORD_ID, name=remote_name)
        tmp = dest.with_suffix(dest.suffix + ".part")

        print(f"Downloading {local_path} (~{size_mb} MB)")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=progress)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            sys.exit(f"\nFailed to download {remote_name}: {exc}")
        tmp.replace(dest)  # only appears under its final name once complete
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
