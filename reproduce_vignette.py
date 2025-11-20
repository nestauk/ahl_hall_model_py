import numpy as np
import matplotlib.pyplot as plt
from bw_model import adult_weight

# --- Vignette Section: Usage in R ---

# "As an example consider a 40 year old 'female' weighting 80 kg
# with a height of 1.8 metres:"
print("Running Model 1 (Basic)...")
female_model1 = adult_weight(bw=80, ht=1.8, age=40, sex="female")

# Output shape is (Individuals, Time).
# Access [0, -1] for (1st Individual, Last Time Step)
final_w1 = female_model1["Body_Weight"][0, -1]
print(f"Final Weight (Model 1): {final_w1:.2f} kg")

# "For example, this female can reduce her energy consumption by -250 kcals
# and her sodium intake by 20 mg"
print("\nRunning Model 2 (Diet)...")
days = 365
EI_change = np.full(days, -250)  # rep(-250, 365)
NA_change = np.full(days, -20)  # rep(-20, 365)

female_model2 = adult_weight(
    bw=80,
    ht=1.8,
    age=40,
    sex="female",
    EIchange=EI_change,
    NAchange=NA_change,
    days=days,
)

final_w2 = female_model2["Body_Weight"][0, -1]
print(f"Final Weight (Model 2): {final_w2:.2f} kg")

# --- Vignette Section: Plots ---
# "Result plots can be obtained by model_plot function"

time = female_model2["Time"]  # Shape (366,)
bw = female_model2["Body_Weight"][0]  # Shape (366,) - Extract 1st individual

plt.figure(figsize=(7, 4))
plt.plot(time, bw, label="Body Weight", color="black")
plt.title("Body Weight Change (Female, -250kcal/day)")
plt.xlabel("Time (days)")
plt.ylabel("Weight (kg)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Save verification image
plt.savefig("vignette_reproduction.png")
print("\nPlot saved to 'vignette_reproduction.png'")
