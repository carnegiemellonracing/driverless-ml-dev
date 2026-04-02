from multiprocessing import freeze_support

import dataset


def main() -> None:
    dataset.prepare()


if __name__ == "__main__":
    freeze_support()
    main()
