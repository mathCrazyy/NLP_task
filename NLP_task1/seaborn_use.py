import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

tips=sns.load_dataset("tips")
print(tips.head())
g=sns.lmplot(x="total_bill",y="tip",data=tips,legend=True)
plt.show()