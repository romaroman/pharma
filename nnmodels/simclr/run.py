from nnmodels.simclr import SimCLR
from nnmodels.dataset import Dataset


def main():
    dataset = Dataset()

    simclr = SimCLR(dataset)
    simclr.train()


if __name__ == "__main__":
    main()
