import pandas as pd
import matplotlib.pyplot as plt
 
plotdata = pd.DataFrame({
    "Baseline":[40, 12, 10],
    "CoorDL":[19, 8, 30],
    "JOADER":[10, 10, 42],
    "SUPER":[10, 10, 42]

    }, 
    index=["VGG11", "SqueezeNet", "DenseNet121"]
)

plotdata.plot(kind="bar")
#plt.title("Mince Pie Consumption Study")
#plt.xlabel("Family Member")
plt.ylabel("Training Time (s)")
 
# Display the chart
plt.show()

print