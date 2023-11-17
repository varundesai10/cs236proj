import pandas as pd
import matplotlib.pyplot as plt

from certification_pic import *


sigmas = ["0.25", "0.5", "0.75", "1"]
sigmas_5 = ["0.25"]
colors = ["red", "blue", "green", "orange"]

plot_certified_accuracy(
    "cifar10_test", "", 2.5, 
    [Line(ApproximateAccuracy(f"cifar10_{s}_merge_results.txt"), f"$\sigma = {s}$", plot_fmt=c) for s,c in zip(sigmas, colors)]
)

plot_certified_accuracy(
    "cifar10_test_5", "", 2.5, 
    [Line(ApproximateAccuracy(f"cifar10_5_{s}_merge_results.txt"), f"$\sigma = {s}$", plot_fmt=c) for s,c in zip(sigmas_5, colors)]
)

latex_table_certified_accuracy(
    "cifar10_test_table", 0.0, 1.0, 0.25,  
    [Line(ApproximateAccuracy(f"cifar10_{s}_merge_results.txt"), f"$\sigma = {s}$", plot_fmt=c) for s,c in zip(sigmas, colors)]
)