import torch


a = torch.randint(0, 10, size=(3, 8), dtype=torch.float)
b = torch.randint(0, 10, size=(3, 8), dtype=torch.float)

print(len(a))

a_times_a = a.unsqueeze(1).bmm(a.unsqueeze(2)).reshape(-1, 1)
b_times_b = a.unsqueeze(1).bmm(a.unsqueeze(2)).reshape(1, -1)
a_times_b = a.mm(a.T)

distance_matrix = (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()
print("\n---distance_matrix---")
print(distance_matrix)

coreset_anchor_distances = torch.norm(distance_matrix, dim=1)
print("\n---coreset_anchor_distances---")
print(coreset_anchor_distances)

select_idx = torch.argmax(coreset_anchor_distances).item()
coreset_select_distance = distance_matrix[:, select_idx : select_idx + 1]  # noqa E203


coreset_anchor_distances = torch.cat(
    [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
)
print("\n---coreset_anchor_distances---")
print(coreset_anchor_distances)

coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values
