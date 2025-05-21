## 📊 Scenario 3: Distance-Based Failures — Experimental Analysis

This scenario evaluates how varying the **distance threshold** in the `DistanceModel` affects agent performance and step time. A lower threshold limits communication to nearby agents, while a higher threshold allows more communication links to be preserved.

### **Distance Thresholds Tested**
- 0.2
- 0.4
- 0.6
- 0.8

### **Summary of Results**

| Threshold | Avg Reward ± Std    | Avg Step Time ± Std (ms) |
|-----------|---------------------|---------------------------|
| 0.2       | -75.43 ± 25.11      | 1.06 ± 0.05               |
| 0.4       | -75.09 ± 24.55      | 1.08 ± 0.07               |
| 0.6       | -73.73 ± 22.51      | 1.06 ± 0.06               |
| 0.8       | -80.12 ± 24.50      | 1.08 ± 0.07               |

---

### **Interpretation of Results**

#### 🔹 **Avg Reward vs Distance Threshold**

- **0.2 → 0.6**: Gradual improvement in average reward, suggesting that **moderate communication range** supports better coordination among agents.
- **0.6** yielded the **best performance**, indicating it may be an optimal balance between visibility and noise.
- Surprisingly, **threshold = 0.8** led to a **drop in performance**, despite increased connectivity. This could be due to:
  - Overcommunication creating **redundant or noisy observations**
  - Agents possibly reacting to too many other agents simultaneously, reducing efficiency
  - Communication overload without strategic filtering

#### 🔹 **Avg Step Time vs Distance Threshold**

- The per-step computational time remained **very stable** across thresholds (within ~0.02 ms difference).
- Slightly higher step time at thresholds 0.4 and 0.8 may result from denser communication matrices being masked.
- Overall, this confirms that the `DistanceModel` wrapper incurs **minimal performance overhead** even under full communication.

---

### ✅ **Conclusion**

This experiment demonstrates that distance-based failures introduce **realistic degradation** in communication, and agent performance is **sensitive** to the distance threshold. However, **more communication is not always better**. Moderate thresholds (like 0.6) may yield optimal performance due to balanced visibility.

Future scenarios (e.g., Probabilistic or Delay-Based models) will help further investigate trade-offs between communication fidelity and computational efficiency.

