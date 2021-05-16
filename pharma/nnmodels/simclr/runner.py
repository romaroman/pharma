from pharma.nnmodels.simclr import SimCLR
from pharma.nnmodels.datasets import Dataset


def main():
    dataset = Dataset()

    simclr = SimCLR(dataset)
    simclr.train()


if __name__ == "__main__":
    main()
