

print(f"Data Information:")
print(f"- Type: {type(data)}")
print(f"- Shape: {data.shape}")
print(f"- Data Type: {data.dtype}")
print(f"- Minimum Value: {data.min()}")
print(f"- Maximum Value: {data.max()}")
print(f"- Mean: {data.mean()}")
print(f"- Standard Deviation: {data.std()}")
print(f"\nFirst 5 samples:")
print(data[:5])