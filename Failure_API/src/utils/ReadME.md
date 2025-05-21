### Usage

For DistanceModel and SignalModel:
```python
pos_fn = make_position_fn(env)  # Default: single-agent
model = DistanceModel(..., pos_fn=pos_fn)



pos_fn = make_position_fn(env, return_batch=True)  # Batch mode
model = SignalBasedModel(..., pos_fn=pos_fn)
