Exhaustive List of Factors and Hyperparameters to Experiment With
Model Architecture:

Network Depth/Width: Increase or decrease the number of layers and the number of neurons per layer in both shared networks and the policy and value heads.

Recurrent vs. Feed‑Forward: Use a recurrent architecture (LSTM, GRU) to capture temporal dependencies versus a feed-forward architecture.

Feature Extraction Layers: Experiment with different preprocessing networks (e.g., convolutional layers if you have temporal/spatial data).

Reward Function:

Scale: Adjust the magnitude of the rewards so that they are neither too small (vanishing gradients) nor excessively large.

Shaping: Include dense signals such as intermediate rewards (e.g., unrealized profit or deviation from a buy-and‑hold baseline) and not only binary trade outcomes.

Counterfactual Rewards: Provide feedback even when actions are not executed, if that aligns with your simulation.

Baselines: Add a relative component (e.g., comparison with an optimal or buy‑and‑hold strategy).

Optimization and Learning Parameters:

Learning Rate: Vary the initial learning rate and experiment with different decay schedules.

Clip Range: Modify the PPO clipping parameter to control policy update magnitude.

Batch Size: Increase or decrease the batch size to affect gradient variance.

Number of Epochs: Adjust the number of epochs per update. Fewer epochs can prevent overfitting to on‑policy data.

Gradient Clipping Norm: Tune the maximum gradient norm to mitigate exploding gradients.

Entropy Coefficient: Modify to balance exploration vs. exploitation (if entropy is too high, the policy may not converge).

Rollout Buffer & Timesteps:

Episode Lengths: Modify the interval lengths; shorter episodes give more frequent feedback, while longer episodes may capture more complex long‑term behavior.

Number of Timesteps: Experiment with total training timesteps to assess whether the model eventually improves.

Vectorization: Ensure the number of parallel environments is set appropriately (as it affects how recurrent states are handled).

Normalization:

Input Features: Ensure that your features are normalized (e.g., standardized) to allow for more stable training.

Rewards: Consider normalizing or scaling rewards, particularly if they are sparse.

Randomness and Overfitting Checks:

Fixed vs. Random Starting Points: Run benchmarks with a fixed starting point to check for overfitting capability, and then gradually introduce variability.

Data Augmentation: Experiment with different time periods or additional features if your current data is too narrow.

State Representation:

Additional Inputs: Incorporate technical indicators or memory of past trades, if you believe your state lacks critical information.

Temporal Window: Adjust the length of the input sequence (how many past timesteps the LSTM receives) which can affect the amount of temporal context.

Recurrent State Handling:

LSTM Initialization: Ensure that recurrent states are reset appropriately when episodes end, and experiment with different numbers of layers or hidden dimensions.